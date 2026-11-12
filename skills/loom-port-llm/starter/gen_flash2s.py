"""Prefill flash attention for head_dim 256 and 512 (gen_flash2.py hands these over: no sinks, head_dim x query
heads per KV head >= 1024, no FLASH2_DO/KT customization; FLASH2_SPLIT=0 keeps the generic kernel).

Same contract and transposed form as gen_flash2.py (S^T = K Q^T: each lane's result column is one query,
its 8 elements keys 2i + lane/16; softmax is in-lane plus one xor-16 shuffle; O^T += V^T P^T), but each
query head is split over head_dim/128 waves (2, or 4 at head_dim 512) that own 128 dims each. A wave keeps
its part of Q^T in registers for the whole key loop (gen_flash2.py reloads all of Q from global memory every
key tile at head_dim > 128, because all of Q plus the output do not fit 256 VGPRs) and accumulates its 128
output dims. Per key tile (32 keys; 16 at head_dim 512, where K and V^T of 32 keys exceed 64 KiB):
  1. stage K [KT x D] and V^T [D x KT] in workgroup memory: lane (workitem % KT) owns one key and
     (workitem / KT) picks a dim segment, so the V^T writes of a wave hit consecutive columns
     (conflict-free) and each lane does one paged-KV lookup per tile;
  2. each wave computes S^T partial over its 128 dims and publishes it; at head_dim 256 it adds its partner's
     (wave ^ 1): both waves add the same two values (a + b == b + a); at 512 every wave of the head sums the
     four partials in wave order. Either way their S, softmax and P are bit-identical;
  3. softmax, P^T, O^T += V^T[own dims] P^T.
Workgroup: (listed 16-token tile, KV head[, head group]), 32 * head_dim/128 lanes per query head. Masks: causal, and
keys > position - window on sliding layers. The output is scaled by sigmoid(gate) when the model has an
attention output gate. FLASH2_SCALE (score scale), FLASH2_NAME (kernel name) and FLASH2_POS_BYTES (bytes per
KV position when a wider layer kind sets the stride) customize it.
"""
import json
import os
import subprocess
import sys

_c = json.load(open(os.environ["LLM_CONFIG"]))
P = "g"
KVH, G, D = _c["kv_heads"], _c["heads"] // _c["kv_heads"], _c["head_dim"]
QH = KVH * G
WINDOW = _c.get("sliding_window") or 262144
GATE = bool(_c.get("attn_gate", False))
GATE_SIG = _c.get("attn_gate_act", "sigmoid") == "sigmoid"   # else the gate holds the activated value (softplus)
LAYERS = _c["layers"] - 1
assert D in (256, 512) and not _c.get("attn_sinks"), "gen_flash2s: head_dim 256 or 512 without attention sinks"
HALF = 128                        # dims per wave
NFH = HALF // 16                  # 16-dim fragments per wave (8)
WPH = D // HALF                   # waves per query head (2, or 4 at head_dim 512)
KT = 32 if D == 256 else 16       # keys per tile (head_dim 512: K and V^T of 32 keys exceed 64 KiB)
KB = KT // 16                     # 16-key blocks per tile
DP8 = D + 8                       # padded K row
VTC = KT + 8                      # padded V^T row
# Query-head groups per KV head: one workgroup per (tile, KV head, group) when the G heads' S partials don't fit
# next to the K and V^T tiles in 64 KiB of workgroup memory (G = 8: 70 KiB; Qwen3.5/3.6 MoE). Each group stages
# K/V itself (the staging is repeated per group; the score/softmax/output work is not).
NG = next(n for n in (1, 2, 4) if G % n == 0 and (KT * DP8 * 2 + D * VTC * 2 + (WPH * G // n) * KB * 16 * 16 * 4 <= 65536
                                                  or WPH * G // n == 8))
GW = G // NG                      # query heads per workgroup
WAVES = WPH * GW
LANES = 32 * WAVES
# Staging: lane = key (workitem % KT), workitem / KT = dim segment. 256 dims over 2G segments: G = 6 -> 12
# segments of 21.3: use 16 segments of 16 dims on the first 16*32 = 512 lanes when there are that many, else 8
# segments of 32 dims (any G >= 4 has >= 8 waves). Head_dim 512 (16-key tiles): 32 segments of 16 dims.
SEG_N = 16 if WAVES >= 16 else 8
if KT == 16:
    SEG_N = 2 * SEG_N
SEG_W = D // SEG_N
assert WPH == 2 or KB == 1, "head_dim 512: 16-key tiles"
assert LANES >= KT * SEG_N, "need at least 8 waves (head_dim * G >= 1024)"
SCALE = os.environ.get("FLASH2_SCALE") or repr(D ** -0.5)
POS_BYTES = int(os.environ.get("FLASH2_POS_BYTES") or KVH * D * 2)   # bytes per KV position (widest layer kind)

POSHI = """  // The last live position of the block: keys past it are not written yet (stale memory, possibly NaN), so they are
  // staged as zeros, which the causal mask already gives zero weight.
  %hi_c16_i32 = scalar.constant 16 : i32
  %hi_true = scalar.constant true : i1
  %hi_one_i32 = scalar.constant 1 : i32
  %hi_block_end = scalar.addi %pos0_32, %hi_c16_i32 : i32
  %hi_alive, %hi_cnt = scf.for %k = [%c1 to %c16 step %c1](%alive = %hi_true : i1, %n = %c0_i32 : i32) -> (i1, i32) {
    %ahead = index.add %first_token, %k : index
    %room = index.cmp ult, %ahead, %tokens : index
    %at_value = scf.select %room, %ahead, %first_token : index
    %at = index.assume %at_value [lt(%at_value, %tokens)] : index
    %m_row = view.load %info_view[%at, %c0] : view<[%tokens]x4xi32> -> i32
    %m_pos = view.load %info_view[%at, %c1] : view<[%tokens]x4xi32> -> i32
    %m_live = view.load %info_view[%at, %c2] : view<[%tokens]x4xi32> -> i32
    %k32 = index.cast %k : index to i32
    %m_want = scalar.addi %posf_32, %k32 : i32
    %m_same_row = scalar.cmpi eq, %m_row, %row32 : i32
    %m_at_pos = scalar.cmpi eq, %m_pos, %m_want : i32
    %m_is_live = scalar.cmpi ne, %m_live, %c0_i32 : i32
    %m_in_block = scalar.cmpi slt, %m_want, %hi_block_end : i32
    %m_a = scalar.andi %m_same_row, %m_at_pos : i1
    %m_b = scalar.andi %m_a, %m_is_live : i1
    %m_c = scalar.andi %m_b, %m_in_block : i1
    %m_d = scalar.andi %m_c, %room : i1
    %next_alive = scalar.andi %alive, %m_d : i1
    %step = scf.select %next_alive, %hi_one_i32, %c0_i32 : i32
    %n_next = scalar.addi %n, %step : i32
    scf.yield %next_alive, %n_next : i1, i32
  }
  %pos_hi_i32 = scalar.addi %posf_32, %hi_cnt : i32
  %pos_hi = index.cast %pos_hi_i32 : i32 to index
"""
PARTIAL = os.environ.get("FLASH2_ROLE") == "partial"   # the tiles a run covers only in part (see gen_flash2.py)
QTOKEN = """%q_token_value = index.add %first_token, %lane_col : index
  %q_token = index.assume %q_token_value [lt(%q_token_value, %tokens)] : index"""
if PARTIAL:   # lane column c is the token ENTRY + c - OFF when it continues the run; other columns read the entry's Q
    QTOKEN = """%col_ge = index.cmp uge, %lane_col, %off : index
  %col_rel = index.sub %lane_col, %off : index
  %col_rel_c = scf.select %col_ge, %col_rel, %c0 : index
  %q_tok_raw = index.add %first_token, %col_rel_c : index
  %q_in = index.cmp ult, %q_tok_raw, %tokens : index
  %q_tok_c = scf.select %q_in, %q_tok_raw, %first_token : index
  %q_token = index.assume %q_tok_c [lt(%q_tok_c, %tokens)] : index
  %slot_row = view.load %info_view[%q_token, %c0] : view<[%tokens]x4xi32> -> i32
  %slot_pos = view.load %info_view[%q_token, %c1] : view<[%tokens]x4xi32> -> i32
  %slot_live = view.load %info_view[%q_token, %c2] : view<[%tokens]x4xi32> -> i32
  %lane_col32 = index.cast %lane_col : index to i32
  %slot_want = scalar.addi %pos0_32, %lane_col32 : i32
  %slot_same_row = scalar.cmpi eq, %slot_row, %row32 : i32
  %slot_at_pos = scalar.cmpi eq, %slot_pos, %slot_want : i32
  %slot_is_live = scalar.cmpi ne, %slot_live, %c0_i32 : i32
  %slot_a = scalar.andi %slot_same_row, %slot_at_pos : i1
  %slot_b = scalar.andi %slot_a, %slot_is_live : i1
  %slot_c = scalar.andi %slot_b, %q_in : i1
  %slot_ok = scalar.andi %slot_c, %col_ge : i1"""
LISTED = f"""  %tiles_view = buffer.view %tiles[%origin] : buffer -> view<[%tokens]xi32>
  %listed_count32 = view.load %tiles_view[%c0] : view<[%tokens]xi32> -> i32
  %listed_count = index.cast %listed_count32 : i32 to index
  %listed = index.cmp ult, %tile, %listed_count : index
  %entry_slot_value = index.add %tile, %c1 : index
  %entry_slot = index.assume %entry_slot_value [lt(%entry_slot_value, %tokens)] : index
  %entry32 = view.load %tiles_view[%entry_slot] : view<[%tokens]xi32> -> i32
  %entry = index.cast %entry32 : i32 to index
  %first_token = index.assume %entry [range(%entry, 0, 8191), lt(%entry, %tokens)] : index
  %info_view = buffer.view %info[%origin] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%first_token, %c0] : view<[%tokens]x4xi32> -> i32
  %pos0_32 = view.load %info_view[%first_token, %c1] : view<[%tokens]x4xi32> -> i32
"""
if PARTIAL:   # tiles[tokens / 2] counts the partial tiles, the entries follow: the first live token of each block
    LISTED = """  %c2h = index.constant 2 : index
  %half = index.div %tokens, %c2h : index
  %half_bound = index.assume %half [lt(%half, %tokens)] : index
  %tiles_view = buffer.view %tiles[%origin] : buffer -> view<[%tokens]xi32>
  %listed_count32 = view.load %tiles_view[%half_bound] : view<[%tokens]xi32> -> i32
  %listed_count = index.cast %listed_count32 : i32 to index
  %listed = index.cmp ult, %tile, %listed_count : index
  %entry_base = index.add %half, %c1 : index
  %entry_slot_value = index.add %entry_base, %tile : index
  %last_slot = index.sub %tokens, %c1 : index
  %entry_slot_clamped = index.min %entry_slot_value, %last_slot : index
  %entry_slot = index.assume %entry_slot_clamped [lt(%entry_slot_clamped, %tokens)] : index
  %entry32 = view.load %tiles_view[%entry_slot] : view<[%tokens]xi32> -> i32
  %entry_raw = index.cast %entry32 : i32 to index
  // entries past the count are not initialized: a workgroup without a tile reads token 0
  %entry = scf.select %listed, %entry_raw, %c0 : index
  %first_token = index.assume %entry [range(%entry, 0, 8191), lt(%entry, %tokens)] : index
  %info_view = buffer.view %info[%origin] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%first_token, %c0] : view<[%tokens]x4xi32> -> i32
  %posf_32 = view.load %info_view[%first_token, %c1] : view<[%tokens]x4xi32> -> i32
  %c15_i32 = scalar.constant 15 : i32
  %off32 = scalar.andi %posf_32, %c15_i32 : i32
  %pos0_32 = scalar.subi %posf_32, %off32 : i32
  %off = index.cast %off32 : i32 to index
"""
L = []
A = L.append
A(f'''kernel.def export("{P}_attention_flash2") @{P}_attention_flash2(%tokens: index) {{
  %c1 = index.constant 1 : index
  %ckv = index.constant {KVH * NG} : index
  %c16 = index.constant 16 : index
  %clanes = index.constant {LANES} : index
  %tiles = index.div %tokens, %c16 : index
{"  %c4 = index.constant 4 : index" + chr(10) + "  %tiles4 = index.mul %tiles, %c4 : index" + chr(10) if PARTIAL else ""}  kernel.launch.config workgroups({"%tiles4" if PARTIAL else "%tiles"}, %ckv, %c1) workgroup_size(%clanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %sliding: i32, %context_arg: index, %pool_arg: index, %q: buffer, %gate: buffer, %info: buffer, %tiles: buffer, %origins: buffer, %arena: buffer, %out: buffer) {{
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
{"  %c4t = index.constant 4 : index" + chr(10) + "  %tile_count4 = index.mul %tile_count, %c4t : index" + chr(10) if PARTIAL else ""}  %tile_id = kernel.workgroup.id<x> : index
  %tile = index.assume %tile_id [lt(%tile_id, {"%tile_count4" if PARTIAL else "%tile_count"})] : index
  %kv_id = kernel.workgroup.id<y> : index
''' + (f'''  %kv = index.assume %kv_id [range(%kv_id, 0, {KVH - 1})] : index''' if NG == 1 else f'''  %cng = index.constant {NG} : index
  %kv_value = index.div %kv_id, %cng : index
  %kv = index.assume %kv_value [range(%kv_value, 0, {KVH - 1})] : index
  %hgroup = index.rem %kv_id, %cng : index''') + f'''
{LISTED}  scf.if %listed {{
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {LANES - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {WAVES - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %lane_col = index.rem %lane, %c16 : index
  %lane_half = index.div %lane, %c16 : index
  %hi_half = index.cmp ne, %lane_half, %c0 : index
  %head_in_group = index.div %wave, %c{WPH} : index
  %dim_half = index.rem %wave, %c{WPH} : index
  %partner = scalar.xori_index_placeholder
  %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
  %row = index.cast %row_bound : i32 to index
  %pos0_bound = scalar.assume %pos0_32 [range(%pos0_32, 0, 262128)] : i32
  %pos0 = index.cast %pos0_bound : i32 to index
  %pos_last = index.add %pos0, %c15 : index
{POSHI if PARTIAL else ""}  %padded = index.add %context, %c15 : index
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
  %s_bytes = index.constant {WAVES * KB * 16 * 16 * 4} : offset
  %s_tile = buffer.alloca<workgroup> align(16) %s_bytes : buffer
  %s_view = buffer.view %s_tile[%origin] : buffer -> view<{WAVES * KB * 16}x16xf32>

  // This wave's half of Q^T as MMA rhs, held for the whole key loop: lane column = token, 16 dims each.
  %q_view = buffer.view %q[%origin] : buffer -> view<[%tokens]x{QH}x{D}xf32>
  {QTOKEN}''')
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
  %st_key = index.rem %workitem, %c{KT} : index
  %st_seg_value = index.div %workitem, %c{KT} : index
  %cseg_n = index.constant {SEG_N} : index
  %st_live = index.cmp ult, %st_seg_value, %cseg_n : index
  %cseg_last = index.constant {SEG_N - 1} : index
  %st_seg = index.min %st_seg_value, %cseg_last : index
  %cseg_w = index.constant {SEG_W} : index
  %st_dim_value = index.mul %st_seg, %cseg_w : index
  %st_dim = index.assume %st_dim_value [range(%st_dim_value, 0, {D - SEG_W})] : index
  %kz = vector.constant 0.0 : vector<{SEG_W}xf16>
  %plane_bytes_i = index.mul %pool, %c1 : index
  %pos_bytes = index.constant {POS_BYTES} : offset
  %v_plane_delta = index.scale %plane_bytes_i, %pos_bytes : index, offset -> offset''')
if WPH == 2:
    A(f'''  // S exchange slots: mine and my partner's (wave ^ 1).
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
else:   # 16-key tiles (one key block): every wave of a head reads all WPH partials and sums them in wave order
    A(f'''  // S exchange slots: mine, and those of every wave of my head.
  %my_s = index.mul %wave, %c16 : index
  %cparts = index.constant {WPH * 16} : index
  %head_s = index.mul %head_in_group, %cparts : index
''' + "".join(f'''  %part{j}_o = index.constant {16 * j} : index
  %part{j}_s = index.add %head_s, %part{j}_o : index
''' for j in range(WPH)))
acc = ", ".join(f"%a{d} = %init : vector<8xf32>" for d in range(NFH))
A(f'''  %m_final, %l_final, {", ".join(f"%o{d}" for d in range(NFH))} = scf.for %kt = [%c0 to %key_tiles step %c1](%m = %neg_big : f32, %l = %zero_f : f32, {acc}) -> (f32, f32, {", ".join(["vector<8xf32>"] * NFH)}) {{
    %tile_first = index.madd %kt, %ckt, %key_low : index
    scf.if %st_live {{
      %key_pos = index.add %tile_first, %st_key : index
      %key_valid = index.cmp ule, %key_pos, {"%pos_hi" if PARTIAL else "%pos_last"} : index
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
SV = f"view<{WAVES * KB * 16}x16xf32>"
for kb in range(KB):
    accn = "%init"
    for c in range(NFH):
        A(f'''    %kc{kb}_{c} = index.add %dim_base, %qd{c}_o : index
    %kl{kb}_{c} = vector.fragment.load<lhs> %k_view[%kb{kb}_row, %kc{kb}_{c}] shape [%c16, %c16] : view<{KT}x{DP8}xf16> -> vector<16xf16>
    %s{kb}_{c} = vector.mma %kl{kb}_{c}, %qf{c}, {accn} : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
        accn = f"%s{kb}_{c}"
    mine = "%my_s" if kb == 0 else "%my_s1"
    A(f"    vector.fragment.store<result> %s{kb}_{NFH - 1}, %s_view[{mine}, %c0] shape [%c16, %c16] : vector<8xf32>, {SV}")
A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
for kb in range(KB):
    if WPH == 2:   # mine + my partner's: a + b == b + a, so both waves get the same S
        theirs = "%their_s" if kb == 0 else "%their_s1"
        A(f'''    %sp{kb} = vector.fragment.load<result> %s_view[{theirs}, %c0] shape [%c16, %c16] : {SV} -> vector<8xf32>
    %st{kb} = vector.addf %s{kb}_{NFH - 1}, %sp{kb} : vector<8xf32>''')
    else:          # all of the head's partials, in wave order
        A("\n".join(f"    %sp{kb}_{j} = vector.fragment.load<result> %s_view[%part{j}_s, %c0] shape [%c16, %c16] : {SV} -> vector<8xf32>"
                    for j in range(WPH)))
        A("\n".join(f"    %st{kb}{'' if j == WPH - 1 else f'_{j}'} = vector.addf {f'%st{kb}_{j - 1}' if j > 1 else f'%sp{kb}_0'}, %sp{kb}_{j} : vector<8xf32>"
                    for j in range(1, WPH)))
    A(f'''    %sc{kb} = vector.mulf %st{kb}, %scale_v : vector<8xf32>
    %base{kb} = scalar.addi %tf32, %kbo{kb} : i32
    %base{kb}_v = vector.splat %base{kb} : vector<8xi32>
    %kpos{kb} = vector.addi %base{kb}_v, %row_off : vector<8xi32>
    %causal{kb} = vector.cmpi sle, %kpos{kb}, %qp_v : vector<8xi32> -> vector<8xi1>
    %inwin{kb} = vector.cmpi sge, %kpos{kb}, %low_v : vector<8xi32> -> vector<8xi1>
    %winok{kb} = vector.ori %inwin{kb}, %ns_v : vector<8xi1>
    %valid{kb} = vector.andi %causal{kb}, %winok{kb} : vector<8xi1>
    %masked{kb} = vector.select %valid{kb}, %sc{kb}, %neg_v : vector<8xf32>
    %lmax{kb} = vector.reduce<maxnumf> %masked{kb}, %neg_big : vector<8xf32>, f32''')
A(f'''    %lmax = scalar.maxnumf %lmax0, %lmax{KB - 1} : f32
    %omax, %omax_u = kernel.subgroup.shuffle<xor> %lmax, %exchange, %width : f32, i32, i32
    %tmax = scalar.maxnumf %lmax, %omax : f32
    %m_new = scalar.maxnumf %m, %tmax : f32
    %m_shift = scalar.subf %m, %m_new : f32
    %alpha = scalar.expf<afn> %m_shift : f32
    %m_new_v = vector.splat %m_new : vector<8xf32>''')
for kb in range(KB):
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
A(f'''    %lsum = scalar.addf %ls0, {'%ls1' if KB == 2 else '%zero_f'} : f32
    %osum, %osum_u = kernel.subgroup.shuffle<xor> %lsum, %exchange, %width : f32, i32, i32
    %tsum = scalar.addf %lsum, %osum : f32
    %l_scaled = scalar.mulf %l, %alpha : f32
    %l_new = scalar.addf %l_scaled, %tsum : f32
    %alpha_v = vector.splat %alpha : vector<8xf32>''')
# O^T += V^T P^T per 16-key block. A weight of exactly 0 does not hide its key's V in the WMMA sum (see gen_flash2.py), so
# a block with masked keys (the diagonal block; the window's lower edge) runs once per query column with V zeroed at that
# column's masked keys, keeping that column's result.
for d in range(NFH):
    A(f'''    %os{d} = vector.mulf %a{d}, %alpha_v : vector<8xf32>
    %vr{d} = index.add %dim_base, %qd{d}_o : index''')
A('''    %pos0_i32 = index.cast %pos0 : index to i32
    %pos15_i32 = scalar.addi %pos0_i32, %c15_i32m : i32
    %lane_col32m = index.cast %lane_col : index to i32
    %win_lo15 = scalar.subi %pos15_i32, %cwin_i32 : i32
    %zero16h = vector.constant 0.0 : vector<16xf16>
    %iota16 = vector.iota %c0_i32 step %c1_i32m : vector<16xi32>''')
# A block that starts past the tile's last position holds no visible key: its weights are exactly 0 and its V is staged as
# zeros, so adding it changes nothing (the plain path does it at the cost of one set of MMAs; it never takes the masked path).
# FLASH2_EDGE_ZERO=1: a sliding window's lower-edge block also takes the masked path (V zeroed per query column). Default off:
# its masked keys are keys of the run whatever the cut (older than the window, written long before), so a token's output
# cannot depend on how the scheduler cut the prefill.
EDGE_ZERO = os.environ.get("FLASH2_EDGE_ZERO", "0") == "1"
NEED_OP = ("scalar.ori", "%nb_edge") if EDGE_ZERO else ("scalar.andi", "%nb_live")
for kb in range(KB):
    last = kb == KB - 1
    ins = [f"%os{d}" if kb == 0 else f"%on{d}_{kb - 1}" for d in range(NFH)]
    outs = [f"%on{d}" if last else f"%on{d}_{kb}" for d in range(NFH)]
    vt = ", ".join(["vector<8xf32>"] * NFH)
    A(f"""    %nb_last{kb} = scalar.addi %base{kb}, %c15_i32m : i32
    %nb_diag{kb} = scalar.cmpi sgt, %nb_last{kb}, %pos0_i32 : i32
    %nb_win{kb} = scalar.cmpi slt, %base{kb}, %win_lo15 : i32
    %nb_wins{kb} = scalar.andi %nb_win{kb}, %is_sliding : i1
    %nb_dead{kb} = scalar.cmpi sgt, %base{kb}, %pos15_i32 : i32
    %nb_live{kb} = scalar.xori %nb_dead{kb}, %true_const : i1
    %nb_diag_live{kb} = scalar.andi %nb_diag{kb}, %nb_live{kb} : i1
    %nb_edge{kb} = scalar.andi %nb_wins{kb}, %nb_live{kb} : i1
    %need{kb} = {NEED_OP[0]} %nb_diag_live{kb}, {NEED_OP[1]}{kb} : i1
    {", ".join(outs)} = scf.if %need{kb} -> ({vt}) {{""")
    for d in range(NFH):
        A(f"""      %ac{d}_{kb} = scf.for %mc_{d}_{kb} = [%c0 to %c16 step %c1](%acc{d}_{kb} = {ins[d]} : vector<8xf32>) -> (vector<8xf32>) {{
        %mci{d}_{kb} = index.cast %mc_{d}_{kb} : index to i32
        %qc{d}_{kb} = scalar.addi %pos0_i32, %mci{d}_{kb} : i32
        %thr{d}_{kb} = scalar.subi %qc{d}_{kb}, %base{kb} : i32
        %thr_v{d}_{kb} = vector.splat %thr{d}_{kb} : vector<16xi32>
        %mhi{d}_{kb} = vector.cmpi sle, %iota16, %thr_v{d}_{kb} : vector<16xi32> -> vector<16xi1>
        %lo{d}_{kb} = scalar.subi %thr{d}_{kb}, %cwin_i32 : i32
        %loe{d}_{kb} = scf.select %is_sliding, %lo{d}_{kb}, %far_below : i32
        %lo_v{d}_{kb} = vector.splat %loe{d}_{kb} : vector<16xi32>
        %mlo{d}_{kb} = vector.cmpi sge, %iota16, %lo_v{d}_{kb} : vector<16xi32> -> vector<16xi1>
        %m{d}_{kb} = vector.andi %mhi{d}_{kb}, %mlo{d}_{kb} : vector<16xi1>
        %isc{d}_{kb} = scalar.cmpi eq, %lane_col32m, %mci{d}_{kb} : i32
        %vl{d}_{kb} = vector.fragment.load<lhs> %vt_view[%vr{d}, %kb{kb}_row] shape [%c16, %c16] : view<{D}x{VTC}xf16> -> vector<16xf16>
        %vs{d}_{kb} = vector.select %m{d}_{kb}, %vl{d}_{kb}, %zero16h : vector<16xf16>
        %vm{d}_{kb} = vector.fragment<lhs> %vs{d}_{kb} shape [%c16, %c16] : vector<16xf16>
        %rm{d}_{kb} = vector.mma %vm{d}_{kb}, %pt{kb}, {ins[d]} : vector<16xf16>, vector<16xf16>, vector<8xf32>
        %acn{d}_{kb} = scf.select %isc{d}_{kb}, %rm{d}_{kb}, %acc{d}_{kb} : vector<8xf32>
        scf.yield %acn{d}_{kb} : vector<8xf32>
      }}""")
    yv = ", ".join(f"%ac{d}_{kb}" for d in range(NFH))
    A(f"      scf.yield {yv} : {vt}" + chr(10) + "    } else {")
    for d in range(NFH):
        A(f"      %vl{d}_{kb} = vector.fragment.load<lhs> %vt_view[%vr{d}, %kb{kb}_row] shape [%c16, %c16] : view<{D}x{VTC}xf16> -> vector<16xf16>\n      %pl{d}_{kb} = vector.mma %vl{d}_{kb}, %pt{kb}, {ins[d]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
    A(f"      scf.yield {', '.join(f'%pl{d}_{kb}' for d in range(NFH))} : {vt}" + chr(10) + "    }")
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
consts += "".join(f"  %kb{kb}_row = index.constant {16 * kb} : index\n" for kb in range(KB))
consts += "".join(f"  %kbo{kb} = scalar.constant {16 * kb} : i32\n" for kb in range(KB))
consts += "  %c15_i32m = scalar.constant 15 : i32\n  %c1_i32m = scalar.constant 1 : i32\n  %far_below = scalar.constant -1000000 : i32\n"
if WPH != 2:
    consts += f"  %c{WPH} = index.constant {WPH} : index\n"
text = text.replace("  %tile_count = index.div %tokens, %c16 : index\n", consts + "  %tile_count = index.div %tokens, %c16 : index\n", 1)
text = text.replace("  %partner = scalar.xori_index_placeholder\n", "")
if PARTIAL:
    i0 = text.index("  // O^T / l, gate, store through"); i1 = text.rindex("  }\n  kernel.return")
    E = [f"""  // Partial tile: O^T / l, gate, and a store per live column (a fragment store would write the dead columns' tokens).
  %one_f = scalar.constant 1.0 : f32
  %inv_l = scalar.divf %one_f, %l_final : f32
  %inv_v = vector.splat %inv_l : vector<8xf32>
  %cqd = index.constant {QH * D} : index
  %out_flat = buffer.view %out[%origin] : buffer -> view<[%tokens]x{QH * D}xf32>
""" + (f"  %gate_flat = buffer.view %gate[%origin] : buffer -> view<[%tokens]x{QH * D}xf32>\n" if GATE else "") + f"""  %cdim = index.constant {D} : index
  %head_col = index.mul %head, %cdim : index
  %head_half = index.add %head_col, %dim_base : index
  %col_base = index.add %head_half, %lane_half : index
  scf.if %slot_ok {{"""]
    for d in range(NFH):
        E.append(f"    %ou{d} = vector.mulf %o{d}, %inv_v : vector<8xf32>")
        if GATE:
            for i in range(8):
                E.append(f"    %dc{d}_{i} = index.constant {d * 16 + 2 * i} : index\n    %dm{d}_{i} = index.add %col_base, %dc{d}_{i} : index\n"
                         f"    %dn{d}_{i} = index.assume %dm{d}_{i} [lt(%dm{d}_{i}, %cqd)] : index\n"
                         f"    %gl{d}_{i} = view.load %gate_flat[%q_token, %dn{d}_{i}] : view<[%tokens]x{QH * D}xf32> -> f32")
            E.append(f"    %gf{d} = vector.from_elements {', '.join(f'%gl{d}_{i}' for i in range(8))} : vector<8xf32>")
            E.append(f"    %gs{d} = vector.logisticf<afn> %gf{d} : vector<8xf32>" if GATE_SIG else f"    %gs{d} = vector.addf %gf{d}, %zero8 : vector<8xf32>")
            E.append(f"    %of{d} = vector.mulf %ou{d}, %gs{d} : vector<8xf32>")
        else:
            E.append(f"    %of{d} = vector.addf %ou{d}, %zero8 : vector<8xf32>")
        for i in range(8):
            E.append(f"    %ov{d}_{i} = vector.extract %of{d}[%e{i}] : vector<8xf32> -> f32\n"
                     + ("" if GATE else f"    %dc{d}_{i} = index.constant {d * 16 + 2 * i} : index\n    %dm{d}_{i} = index.add %col_base, %dc{d}_{i} : index\n    %dn{d}_{i} = index.assume %dm{d}_{i} [lt(%dm{d}_{i}, %cqd)] : index\n")
                     + f"    view.store %ov{d}_{i}, %out_flat[%q_token, %dn{d}_{i}] : f32, view<[%tokens]x{QH * D}xf32>")
    E.append("  }\n")
    text = text[:i0] + "\n".join(E) + text[i1:]
    text = text.replace(f'"{P}_attention_flash2") @{P}_attention_flash2(', f'"{P}_attention_flash2_partial") @{P}_attention_flash2_partial(', 1)
if os.environ.get("FLASH2_PEEL", "1") == "1" and not EDGE_ZERO:   # the masked path only in the last key tile (flash2_peel.py)
    from flash2_peel import peel_last_tile
    text = peel_last_tile(text)
if os.environ.get("FLASH2_ENTRY") == "bcast" and os.environ.get("FLASH2_ROLE") != "partial":   # experiment: the listed entry broadcast from lane 0
    _old = "  %entry32 = view.load %tiles_view[%entry_slot] : view<[%tokens]xi32> -> i32\n  %entry = index.cast %entry32 : i32 to index\n"
    assert _old in text, "entry load not found"
    text = text.replace(_old, "  %entry32r = view.load %tiles_view[%entry_slot] : view<[%tokens]xi32> -> i32\n  %lane0 = scalar.constant 0 : i32\n  %entry32 = kernel.subgroup.broadcast %entry32r from %lane0 : i32, i32\n  %entry = index.cast %entry32 : i32 to index\n")
out_path = os.environ.get("FLASH2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash2.loom"))
if not PARTIAL:
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_flash2s.py")], check=True,
                   env=dict({k: v for k, v in os.environ.items() if k != "FLASH2_NAME"}, FLASH2_ROLE="partial", FLASH2_OUT=out_path + ".partial"))   # renamed below with the rest
    text += "\n" + open(out_path + ".partial").read()
    os.remove(out_path + ".partial")
if os.environ.get("FLASH2_NAME"):
    text = text.replace(f"{P}_attention_flash2", os.environ["FLASH2_NAME"])
open(out_path, "w", newline="\n").write(text)
