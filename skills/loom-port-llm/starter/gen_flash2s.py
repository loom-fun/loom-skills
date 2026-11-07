"""Prefill flash attention for head_dim 256 (gen_flash2.py hands this case over: head_dim 256, no sinks, >= 4
query heads per KV head, no FLASH2_SCALE/NAME/DO/KT customization; FLASH2_SPLIT=0 keeps the generic kernel).

Same contract and transposed form as gen_flash2.py (S^T = K Q^T: each lane's result column is one query,
its 8 elements keys 2i + lane/16; softmax is in-lane plus one xor-16 shuffle; O^T += V^T P^T), but each
query head is split over TWO waves that own 128 dims each. A wave keeps its half of Q^T in registers for
the whole key loop (gen_flash2.py reloads all of Q from global memory every key tile at head_dim 256,
because 256 dims of Q plus 256 dims of output do not fit 256 VGPRs) and accumulates its 128 output dims.
Per 32-key tile:
  1. stage K [32 x 256] and V^T [256 x 32] in workgroup memory: lane (workitem % 32) owns one key and
     (workitem / 32) picks a 256/(waves) dim segment, so the V^T writes of a wave hit 32 consecutive
     columns (conflict-free) and each lane does one paged-KV lookup per tile;
  2. each wave computes S^T partial over its 128 dims, publishes it, and adds its partner's (wave ^ 1):
     both waves add the same two values (a + b == b + a), so their S, softmax and P are bit-identical;
  3. softmax, P^T, O^T += V^T[own dims] P^T.
Workgroup: (16-token tile, KV head), 64 * G lanes. Masks: causal, and keys > position - window on
sliding layers. The output is scaled by sigmoid(gate) when the model has an attention output gate.
"""
import json
import os

_c = json.load(open(os.environ["LLM_CONFIG"]))
P = "g"
KVH, G, D = _c["kv_heads"], _c["heads"] // _c["kv_heads"], _c["head_dim"]
QH = KVH * G
WINDOW = _c.get("sliding_window") or 262144
GATE = bool(_c.get("attn_gate", False))
GATE_SIG = _c.get("attn_gate_act", "sigmoid") == "sigmoid"   # else the gate holds the activated value (softplus)
LAYERS = _c["layers"] - 1
assert D == 256 and not _c.get("attn_sinks"), "gen_flash2s: head_dim 256 without attention sinks"
HALF = D // 2                     # dims per wave
NFH = HALF // 16                  # 16-dim fragments per wave (8)
KT = 32
DP8 = D + 8                       # padded K row
VTC = KT + 8                      # padded V^T row
# Query-head groups per KV head: one workgroup per (tile, KV head, group) when the G heads' S partials don't fit
# next to the K and V^T tiles in 64 KiB of workgroup memory (G = 8: 70 KiB; Qwen3.5/3.6 MoE). Each group stages
# K/V itself (the staging is repeated per group; the score/softmax/output work is not).
NG = next(n for n in (1, 2, 4) if G % n == 0 and (KT * DP8 * 2 + D * VTC * 2 + (2 * G // n) * 2 * 16 * 16 * 4 <= 65536
                                                  or 2 * G // n == 8))
GW = G // NG                      # query heads per workgroup
WAVES = 2 * GW
LANES = 32 * WAVES
SEGS = WAVES                      # staging: lane = key, workitem/32 = dim segment
DPT = D // SEGS if D % SEGS == 0 else None
# 256 dims over 2G segments: G = 6 -> 12 segments of 21.3: use 16 segments of 16 dims on the first 16*32 = 512
# lanes when there are that many, else 8 segments of 32 dims (any G >= 4 has >= 8 waves).
SEG_N = 16 if WAVES >= 16 else 8
SEG_W = D // SEG_N
assert WAVES >= SEG_N, "need at least 8 waves (G >= 4)"
SCALE = repr(D ** -0.5)

L = []
A = L.append
A(f'''kernel.def export("{P}_attention_flash2") @{P}_attention_flash2(%tokens: index) {{
  %c1 = index.constant 1 : index
  %ckv = index.constant {KVH * NG} : index
  %c16 = index.constant 16 : index
  %clanes = index.constant {LANES} : index
  %tiles = index.div %tokens, %c16 : index
  kernel.launch.config workgroups(%tiles, %ckv, %c1) workgroup_size(%clanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %sliding: i32, %context_arg: index, %pool_arg: index, %q: buffer, %gate: buffer, %info: buffer, %origins: buffer, %arena: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {LAYERS})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c0_i32 = scalar.constant 0 : i32
  %tile_count = index.div %tokens, %c16 : index
  %tile_id = kernel.workgroup.id<x> : index
  %tile = index.assume %tile_id [lt(%tile_id, %tile_count)] : index
  %kv_id = kernel.workgroup.id<y> : index
''' + (f'''  %kv = index.assume %kv_id [range(%kv_id, 0, {KVH - 1})] : index''' if NG == 1 else f'''  %cng = index.constant {NG} : index
  %kv_value = index.div %kv_id, %cng : index
  %kv = index.assume %kv_value [range(%kv_value, 0, {KVH - 1})] : index
  %hgroup = index.rem %kv_id, %cng : index''') + f'''
  %first_token = index.mul %tile, %c16 : index
  %uniform, %row32, %pos0_32 = func.call @{P}_tile_uniform(%info, %tokens, %first_token) : (buffer, index, index) -> (i1, i32, i32)
  scf.if %uniform {{
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {LANES - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {WAVES - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %lane_col = index.rem %lane, %c16 : index
  %lane_half = index.div %lane, %c16 : index
  %hi_half = index.cmp ne, %lane_half, %c0 : index
  %head_in_group = index.div %wave, %c2 : index
  %dim_half = index.rem %wave, %c2 : index
  %partner = scalar.xori_index_placeholder
  %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
  %row = index.cast %row_bound : i32 to index
  %pos0_bound = scalar.assume %pos0_32 [range(%pos0_32, 0, 262128)] : i32
  %pos0 = index.cast %pos0_bound : i32 to index
  %pos_last = index.add %pos0, %c15 : index
  %padded = index.add %context, %c15 : index
  %blocks = index.div %padded, %c16 : index
  %k_plane = index.mul %layer, %c2 : index
  %is_sliding = scalar.cmpi ne, %sliding, %c0_i32 : i32
  %window = index.constant {WINDOW} : index
  %window_less = index.constant {WINDOW - 1} : index
  %far = index.cmp uge, %pos0, %window : index
  %use_window = scalar.andi %is_sliding, %far : i1
  %window_start = index.sub %pos0, %window_less : index
  %key_low_raw = scf.select %use_window, %window_start, %c0 : index
  // Key tiles start on a 16-key page boundary (one page lookup covers 16 consecutive keys).
  %key_low_page = index.div %key_low_raw, %c16 : index
  %key_low = index.mul %key_low_page, %c16 : index
  %key_span = index.sub %pos_last, %key_low : index
  %key_count_value = index.add %key_span, %c1 : index
  %key_count = index.assume %key_count_value [range(%key_count_value, 1, 262144)] : index
  %ckt_less = index.constant {KT - 1} : index
  %ckt = index.constant {KT} : index
  %key_tiles_round = index.add %key_count, %ckt_less : index
  %key_tiles = index.div %key_tiles_round, %ckt : index
  %cg = index.constant {G} : index
''' + ("  %head_value = index.madd %kv, %cg, %head_in_group : index" if NG == 1 else f'''  %cgw = index.constant {GW} : index
  %head_first = index.madd %kv, %cg, %head_in_group : index
  %head_value = index.madd %hgroup, %cgw, %head_first : index''') + f'''
  %head = index.assume %head_value [range(%head_value, 0, {QH - 1})] : index
  %chalf = index.constant {HALF} : index
  %dim_base = index.mul %dim_half, %chalf : index

  %k_bytes = index.constant {KT * DP8 * 2} : offset
  %k_tile = buffer.alloca<workgroup> align(16) %k_bytes : buffer
  %k_view = buffer.view %k_tile[%origin] : buffer -> view<{KT}x{DP8}xf16>
  %vt_bytes = index.constant {D * VTC * 2} : offset
  %vt_tile = buffer.alloca<workgroup> align(16) %vt_bytes : buffer
  %vt_view = buffer.view %vt_tile[%origin] : buffer -> view<{D}x{VTC}xf16>
  // S^T partials: [wave][key block][16 keys x 16 tokens] f32.
  %s_bytes = index.constant {WAVES * 2 * 16 * 16 * 4} : offset
  %s_tile = buffer.alloca<workgroup> align(16) %s_bytes : buffer
  %s_view = buffer.view %s_tile[%origin] : buffer -> view<{WAVES * 2 * 16}x16xf32>

  // This wave's half of Q^T as MMA rhs, held for the whole key loop: lane column = token, 16 dims each.
  %q_view = buffer.view %q[%origin] : buffer -> view<[%tokens]x{QH}x{D}xf32>
  %q_token_value = index.add %first_token, %lane_col : index
  %q_token = index.assume %q_token_value [lt(%q_token_value, %tokens)] : index''')
for c in range(NFH):
    A(f'''  %qd{c}_o = index.constant {c * 16} : index
  %qd{c}_v = index.add %dim_base, %qd{c}_o : index
  %qd{c} = index.assume %qd{c}_v [range(%qd{c}_v, 0, {D - 16})] : index
  %qraw{c} = vector.load %q_view[%q_token, %head, %qd{c}] : view<[%tokens]x{QH}x{D}xf32> -> vector<16xf32>
  %qh{c} = vector.fptrunc %qraw{c} : vector<16xf32> to vector<16xf16>
  %qf{c} = vector.fragment<rhs> %qh{c} shape [%c16, %c16] : vector<16xf16>''')
A(f'''  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %scale = scalar.constant {SCALE} : f32
  %scale_v = vector.splat %scale : vector<8xf32>
  %neg_v = vector.splat %neg_big : vector<8xf32>
  %exchange = scalar.constant 16 : i32
  %width = scalar.constant 32 : i32
  %two_i32 = scalar.constant 2 : i32
  %cwin_i32 = scalar.constant {WINDOW - 1} : i32
  %true_const = scalar.constant true : i1
  %not_sliding = scalar.xori %is_sliding, %true_const : i1
  %ns_v = vector.splat %not_sliding : vector<8xi1>
  %query_pos = index.add %pos0, %lane_col : index
  %qp32 = index.cast %query_pos : index to i32
  %qp_v = vector.splat %qp32 : vector<8xi32>
  %low32 = scalar.subi %qp32, %cwin_i32 : i32
  %low_v = vector.splat %low32 : vector<8xi32>
  %half32 = index.cast %lane_half : index to i32
  %row_off = vector.iota %half32 step %two_i32 : vector<8xi32>

  // Staging slots: key = workitem % 32, dim segment = workitem / 32 ({SEG_N} segments of {SEG_W} dims).
  %st_key = index.rem %workitem, %c32 : index
  %st_seg_value = index.div %workitem, %c32 : index
  %cseg_n = index.constant {SEG_N} : index
  %st_live = index.cmp ult, %st_seg_value, %cseg_n : index
  %cseg_last = index.constant {SEG_N - 1} : index
  %st_seg = index.min %st_seg_value, %cseg_last : index
  %cseg_w = index.constant {SEG_W} : index
  %st_dim_value = index.mul %st_seg, %cseg_w : index
  %st_dim = index.assume %st_dim_value [range(%st_dim_value, 0, {D - SEG_W})] : index
  %kz = vector.constant 0.0 : vector<{SEG_W}xf16>
  %plane_bytes_i = index.mul %pool, %c1 : index
  %pos_bytes = index.constant {KVH * D * 2} : offset
  %v_plane_delta = index.scale %plane_bytes_i, %pos_bytes : index, offset -> offset
  // S exchange slots: mine and my partner's (wave ^ 1).
  %my_s = index.mul %wave, %c32 : index
  %partner_odd = index.cmp eq, %dim_half, %c0 : index
  %partner_add = scf.select %partner_odd, %c1, %c0 : index
  %partner_id_value = index.madd %head_in_group, %c2, %partner_add : index
  %partner_id = index.assume %partner_id_value [range(%partner_id_value, 0, {WAVES - 1})] : index
  %their_s = index.mul %partner_id, %c32 : index
  %kb1 = index.constant 16 : index
  %my_s1 = index.add %my_s, %kb1 : index
  %their_s1 = index.add %their_s, %kb1 : index
''')
acc = ", ".join(f"%a{d} = %init : vector<8xf32>" for d in range(NFH))
A(f'''  %m_final, %l_final, {", ".join(f"%o{d}" for d in range(NFH))} = scf.for %kt = [%c0 to %key_tiles step %c1](%m = %neg_big : f32, %l = %zero_f : f32, {acc}) -> (f32, f32, {", ".join(["vector<8xf32>"] * NFH)}) {{
    %tile_first = index.madd %kt, %ckt, %key_low : index
    scf.if %st_live {{
      %key_pos = index.add %tile_first, %st_key : index
      %key_valid = index.cmp ule, %key_pos, %pos_last : index
      %kvals, %vvals = scf.if %key_valid -> (vector<{SEG_W}xf16>, vector<{SEG_W}xf16>) {{
        %kp = index.assume %key_pos [range(%key_pos, 0, 262143)] : index
        %k_offset = func.call @{P}_kv_offset(%origins, %row, %kp, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
        %v_offset = index.add %k_offset, %v_plane_delta : offset
        %k_cache = buffer.view %arena[%k_offset] : buffer -> view<{KVH}x{D}xf16>
        %v_cache = buffer.view %arena[%v_offset] : buffer -> view<{KVH}x{D}xf16>
        %kh = vector.load %k_cache[%kv, %st_dim] : view<{KVH}x{D}xf16> -> vector<{SEG_W}xf16>
        %vh = vector.load %v_cache[%kv, %st_dim] : view<{KVH}x{D}xf16> -> vector<{SEG_W}xf16>
        scf.yield %kh, %vh : vector<{SEG_W}xf16>, vector<{SEG_W}xf16>
      }} else {{
        scf.yield %kz, %kz : vector<{SEG_W}xf16>, vector<{SEG_W}xf16>
      }}
      vector.store %kvals, %k_view[%st_key, %st_dim] : vector<{SEG_W}xf16>, view<{KT}x{DP8}xf16>
      scf.for %j = [%c0 to %cseg_w step %c1] unroll {{
        %vj = vector.extract %vvals[%j] : vector<{SEG_W}xf16> -> f16
        %vt_row_value = index.add %st_dim, %j : index
        %vt_row = index.assume %vt_row_value [range(%vt_row_value, 0, {D - 1})] : index
        view.store %vj, %vt_view[%vt_row, %st_key] : f16, view<{D}x{VTC}xf16>
      }}
    }}
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %tf32 = index.cast %tile_first : index to i32''')
for kb in range(2):
    accn = "%init"
    for c in range(NFH):
        A(f'''    %kc{kb}_{c} = index.add %dim_base, %qd{c}_o : index
    %kl{kb}_{c} = vector.fragment.load<lhs> %k_view[%kb{kb}_row, %kc{kb}_{c}] shape [%c16, %c16] : view<{KT}x{DP8}xf16> -> vector<16xf16>
    %s{kb}_{c} = vector.mma %kl{kb}_{c}, %qf{c}, {accn} : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
        accn = f"%s{kb}_{c}"
    mine = "%my_s" if kb == 0 else "%my_s1"
    A(f"    vector.fragment.store<result> %s{kb}_{NFH - 1}, %s_view[{mine}, %c0] shape [%c16, %c16] : vector<8xf32>, view<{WAVES * 2 * 16}x16xf32>")
A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
for kb in range(2):
    theirs = "%their_s" if kb == 0 else "%their_s1"
    A(f'''    %sp{kb} = vector.fragment.load<result> %s_view[{theirs}, %c0] shape [%c16, %c16] : view<{WAVES * 2 * 16}x16xf32> -> vector<8xf32>
    %st{kb} = vector.addf %s{kb}_{NFH - 1}, %sp{kb} : vector<8xf32>
    %sc{kb} = vector.mulf %st{kb}, %scale_v : vector<8xf32>
    %base{kb} = scalar.addi %tf32, %kbo{kb} : i32
    %base{kb}_v = vector.splat %base{kb} : vector<8xi32>
    %kpos{kb} = vector.addi %base{kb}_v, %row_off : vector<8xi32>
    %causal{kb} = vector.cmpi sle, %kpos{kb}, %qp_v : vector<8xi32> -> vector<8xi1>
    %inwin{kb} = vector.cmpi sge, %kpos{kb}, %low_v : vector<8xi32> -> vector<8xi1>
    %winok{kb} = vector.ori %inwin{kb}, %ns_v : vector<8xi1>
    %valid{kb} = vector.andi %causal{kb}, %winok{kb} : vector<8xi1>
    %masked{kb} = vector.select %valid{kb}, %sc{kb}, %neg_v : vector<8xf32>
    %lmax{kb} = vector.reduce<maxnumf> %masked{kb}, %neg_big : vector<8xf32>, f32''')
A('''    %lmax = scalar.maxnumf %lmax0, %lmax1 : f32
    %omax, %omax_u = kernel.subgroup.shuffle<xor> %lmax, %exchange, %width : f32, i32, i32
    %tmax = scalar.maxnumf %lmax, %omax : f32
    %m_new = scalar.maxnumf %m, %tmax : f32
    %m_shift = scalar.subf %m, %m_new : f32
    %alpha = scalar.expf<afn> %m_shift : f32
    %m_new_v = vector.splat %m_new : vector<8xf32>''')
for kb in range(2):
    A(f'''    %sh{kb} = vector.subf %masked{kb}, %m_new_v : vector<8xf32>
    %pe{kb} = vector.expf<afn> %sh{kb} : vector<8xf32>
    %p{kb} = vector.select %valid{kb}, %pe{kb}, %zero8 : vector<8xf32>
    %ls{kb} = vector.reduce<addf> %p{kb}, %zero_f : vector<8xf32>, f32''')
    for i in range(8):
        A(f'''    %pi{kb}_{i} = vector.extract %p{kb}[%e{i}] : vector<8xf32> -> f32
    %po{kb}_{i}, %pu{kb}_{i} = kernel.subgroup.shuffle<xor> %pi{kb}_{i}, %exchange, %width : f32, i32, i32''')
    A(f'''    %partner{kb} = vector.from_elements {", ".join(f"%po{kb}_{i}" for i in range(8))} : vector<8xf32>
    %even{kb} = scf.select %hi_half, %partner{kb}, %p{kb} : vector<8xf32>
    %odd{kb} = scf.select %hi_half, %p{kb}, %partner{kb} : vector<8xf32>
    %pt{kb}_f = vector.interleave<0> %even{kb}, %odd{kb} : vector<8xf32>, vector<8xf32> -> vector<16xf32>
    %pt{kb}_h = vector.fptrunc %pt{kb}_f : vector<16xf32> to vector<16xf16>
    %pt{kb} = vector.fragment<rhs> %pt{kb}_h shape [%c16, %c16] : vector<16xf16>''')
A('''    %lsum = scalar.addf %ls0, %ls1 : f32
    %osum, %osum_u = kernel.subgroup.shuffle<xor> %lsum, %exchange, %width : f32, i32, i32
    %tsum = scalar.addf %lsum, %osum : f32
    %l_scaled = scalar.mulf %l, %alpha : f32
    %l_new = scalar.addf %l_scaled, %tsum : f32
    %alpha_v = vector.splat %alpha : vector<8xf32>''')
for d in range(NFH):
    A(f'''    %os{d} = vector.mulf %a{d}, %alpha_v : vector<8xf32>
    %vr{d} = index.add %dim_base, %qd{d}_o : index
    %vl{d}_0 = vector.fragment.load<lhs> %vt_view[%vr{d}, %kb0_row] shape [%c16, %c16] : view<{D}x{VTC}xf16> -> vector<16xf16>
    %on{d}_0 = vector.mma %vl{d}_0, %pt0, %os{d} : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %vl{d}_1 = vector.fragment.load<lhs> %vt_view[%vr{d}, %kb1_row] shape [%c16, %c16] : view<{D}x{VTC}xf16> -> vector<16xf16>
    %on{d} = vector.mma %vl{d}_1, %pt1, %on{d}_0 : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
A('''    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield %m_new, %l_new, ''' + ", ".join(f"%on{d}" for d in range(NFH)) + " : f32, f32, " + ", ".join(["vector<8xf32>"] * NFH) + f'''
  }}
  // O^T / l, gate, store through a transposed strided view: (dim row, token col) -> out[token, head*D + dim].
  %one_f = scalar.constant 1.0 : f32
  %inv_l = scalar.divf %one_f, %l_final : f32
  %inv_v = vector.splat %inv_l : vector<8xf32>
  %cqd = index.constant {QH * D} : index
  %out_layout = encoding.layout.strided [%c1, %cqd] : encoding<layout>
  %out_first_a = index.mul %first_token, %cqd : index
  %cdim = index.constant {D} : index
  %head_col = index.mul %head, %cdim : index
  %head_half = index.add %head_col, %dim_base : index
  %out_first = index.add %out_first_a, %head_half : index
  %four = index.constant 4 : offset
  %out_offset = index.scale %out_first, %four : index, offset -> offset
  %out_view = buffer.view %out[%out_offset] : buffer -> view<{HALF}x16xf32, %out_layout>''' + (f'''
  %gate_view = buffer.view %gate[%out_offset] : buffer -> view<{HALF}x16xf32, %out_layout>''' if GATE else ""))
for d in range(NFH):
    act = (f"\n  %gs{d} = vector.logisticf<afn> %gf{d} : vector<8xf32>" if GATE_SIG
           else f"\n  %gs{d} = vector.addf %gf{d}, %zero8 : vector<8xf32>")   # softplus: the gate holds the value
    gate_txt = f'''
  %gf{d} = vector.fragment.load<result> %gate_view[%qd{d}_o, %c0] shape [%c16, %c16] : view<{HALF}x16xf32, %out_layout> -> vector<8xf32>{act}
  %of{d} = vector.mulf %ou{d}, %gs{d} : vector<8xf32>''' if GATE else f'''
  %of{d} = vector.addf %ou{d}, %zero8 : vector<8xf32>'''
    A(f'''  %ou{d} = vector.mulf %o{d}, %inv_v : vector<8xf32>{gate_txt}
  vector.fragment.store<result> %of{d}, %out_view[%qd{d}_o, %c0] shape [%c16, %c16] : vector<8xf32>, view<{HALF}x16xf32, %out_layout>''')
A("  }\n  kernel.return\n}\n")
text = "\n".join(L)
consts = "".join(f"  %e{i} = index.constant {i} : index\n" for i in range(8))
consts += "  %kb0_row = index.constant 0 : index\n  %kb1_row = index.constant 16 : index\n  %kbo0 = scalar.constant 0 : i32\n  %kbo1 = scalar.constant 16 : i32\n"
text = text.replace("  %tile_count = index.div %tokens, %c16 : index\n", consts + "  %tile_count = index.div %tokens, %c16 : index\n", 1)
text = text.replace("  %partner = scalar.xori_index_placeholder\n", "")
out_path = os.environ.get("FLASH2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash2.loom"))
open(out_path, "w", newline="\n").write(text)
