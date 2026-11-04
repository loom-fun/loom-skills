"""Split-K decode attention on WMMA for a segment of same-row tokens (gen_llm.py attention_text calls kernel()).

{P}_attention_split_mma has the per-token split kernel's ABI and partial-output layout (acc, m, l per token, query
head and split; {P}_attention_combine folds them), but one workgroup (segment head token, KV head, split) serves the
segment's up to 16 tokens as the 16 MMA columns: a decode step or a speculative span's verify tokens. Wave w owns
query head G*kv + w. Per 32-key tile, K and V^T are staged once in workgroup memory; S^T = K Q^T and O^T += V^T P^T
run on WMMA (F16 inputs, F32 accumulation), the softmax in registers (RDNA3 result layout: lane column = token
lane%16, its 8 elements are keys 2i + lane/16), as in gen_flash2.py.

Exactness: a column's result depends only on its own query, the keys of its own range and the tile grid. Every
token's split range is the per-token kernel's ([start, stop) of its visible keys over s_eff splits), keys outside it
are masked to p = 0, the tiles are 32-aligned absolute positions and a tile with no valid key leaves the state
exactly unchanged (alpha = 1 when the max does not move). So a token computes the same bits alone (1-token decode)
as inside a segment (verify), and decode and verify round alike.

Segments (attn_segment in gen_llm.py): consecutive tokens of one row at consecutive positions, broken at every
position that is a multiple of 16; the workgroup of a non-head token exits at once.
"""
TT = 16
KT = 32


def kernel(P, S, D, KVH, G, scale, window, segment, pos_bytes):
    """Kernel text. segment(TT, tok0, info_view, tokens, ind) -> membership text (gen_llm.attn_segment)."""
    assert D in (64, 128, 256) and KT == 32
    NF = D // 16
    QH = KVH * G
    LANES = 32 * G
    DP8, VTC = D + 8, KT + 8
    POW2 = G & (G - 1) == 0
    GP = G if POW2 else 1 << (G.bit_length() - 1)
    DPT = D * 32 // (32 * GP)
    TPK = D // DPT
    SLOTS = 32 * TPK
    PASSES = -(-SLOTS // LANES)
    W = window or 262144
    Q_IN_LOOP = D > 128
    L = []
    A = L.append
    A(f"""
kernel.def export("{P}_attention_split_mma") @{P}_attention_split_mma(%tokens: index) {{
  %c1 = index.constant 1 : index
  %ckvh = index.constant {KVH} : index
  %cS = index.constant {S} : index
  %clanes = index.constant {LANES} : index
  kernel.launch.config workgroups(%tokens, %ckvh, %cS) workgroup_size(%clanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %sliding: i32, %context_arg: index, %pool_arg: index, %q: buffer, %info: buffer, %origins: buffer, %arena: buffer, %part_acc: buffer, %part_ml: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 255)] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c31 = index.constant 31 : index
  %c32 = index.constant 32 : index
  %c0_i32 = scalar.constant 0 : i32
  %big_idx = index.constant 1048576 : index
  %token_id = kernel.workgroup.id<x> : index
  %tok0 = index.assume %token_id [lt(%token_id, %tokens)] : index
  %kv_id = kernel.workgroup.id<y> : index
  %kv = index.assume %kv_id [range(%kv_id, 0, {KVH - 1})] : index
  %split_id = kernel.workgroup.id<z> : index
  %split = index.assume %split_id [range(%split_id, 0, {S - 1})] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {LANES - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {G - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %lane_col = index.rem %lane, %c16 : index
  %lane_half = index.div %lane, %c16 : index
  %hi_half = index.cmp ne, %lane_half, %c0 : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
{segment(TT, "%tok0", "%info_view", "%tokens", "  ")}
  scf.if %seg_head {{
  %row_bound = scalar.assume %row32_0 [range(%row32_0, 0, 2147483646)] : i32
  %row = index.cast %row_bound : i32 to index
  %padded = index.add %context, %c15 : index
  %blocks = index.div %padded, %c16 : index
  %k_plane = index.mul %layer, %c2 : index
  %position_bytes_v = index.constant {pos_bytes} : offset
  %plane_stride = index.scale %pool, %position_bytes_v : index, offset -> offset
  %is_sliding = scalar.cmpi ne, %sliding, %c0_i32 : i32
  %window = index.constant {W} : index
  %window_less = index.constant {W - 1} : index""")
    # per member (uniform scalars): the per-token kernel's split range
    for j in range(TT):
        A(f"""  %position32_{j} = view.load %info_view[%tk{j}, 1] : view<[%tokens]x4xi32> -> i32
  %position_bound_{j} = scalar.assume %position32_{j} [range(%position32_{j}, 0, 262143)] : i32
  %position_{j} = index.cast %position_bound_{j} : i32 to index
  %far_{j} = index.cmp uge, %position_{j}, %window : index
  %use_window_{j} = scalar.andi %is_sliding, %far_{j} : i1
  %window_start_{j} = index.sub %position_{j}, %window_less : index
  %window_low_{j} = scf.select %use_window_{j}, %window_start_{j}, %c0 : index
  %end_{j} = index.add %position_{j}, %c1 : index
  %visible_value_{j} = index.sub %end_{j}, %window_low_{j} : index
  %visible_{j} = index.assume %visible_value_{j} [range(%visible_value_{j}, 1, 262144)] : index
  %s_eff_{j} = func.call @{P}_split_count(%visible_{j}) : (index) -> (index)
  %act_s{j} = index.cmp ult, %split, %s_eff_{j} : index
  %act_{j} = scalar.andi %act_s{j}, %mem{j} : i1
  %s_less_{j} = index.sub %s_eff_{j}, %c1 : index
  %per_padded_{j} = index.add %visible_{j}, %s_less_{j} : index
  %per_{j} = index.div %per_padded_{j}, %s_eff_{j} : index
  %start_raw_{j} = index.mul %split, %per_{j} : index
  %start_{j} = index.min %start_raw_{j}, %visible_{j} : index
  %stop_raw_{j} = index.add %start_raw_{j}, %per_{j} : index
  %stop_{j} = index.min %stop_raw_{j}, %visible_{j} : index
  %lo_a{j} = index.add %window_low_{j}, %start_{j} : index
  %hi_a{j} = index.add %window_low_{j}, %stop_{j} : index
  %lo_{j} = scf.select %act_{j}, %lo_a{j}, %big_idx : index
  %hi_{j} = scf.select %act_{j}, %hi_a{j}, %c0 : index""")
    lo, hi = "%big_idx", "%c0"
    for j in range(TT):
        A(f"""  %lmin{j} = index.min {lo}, %lo_{j} : index
  %hmax{j} = index.max {hi}, %hi_{j} : index""")
        lo, hi = f"%lmin{j}", f"%hmax{j}"
    # this lane's column: token tok0 + lane_col
    sel = ["  %my_lo_init = index.constant 0 : index", "  %my_act_init = scalar.constant false : i1"]
    plo, phi, pact = "%my_lo_init", "%my_lo_init", "%my_act_init"
    for j in range(TT):
        sel.append(f"""  %cj{j} = index.constant {j} : index
  %isc{j} = index.cmp eq, %lane_col, %cj{j} : index
  %mlo{j} = scf.select %isc{j}, %lo_{j}, {plo} : index
  %mhi{j} = scf.select %isc{j}, %hi_{j}, {phi} : index
  %mact{j} = scf.select %isc{j}, %act_{j}, {pact} : i1""")
        plo, phi, pact = f"%mlo{j}", f"%mhi{j}", f"%mact{j}"
    A(f"""  %any_keys = index.cmp ugt, {hi}, %c0 : index
  scf.if %any_keys {{
  %lo_u = index.min {lo}, {hi} : index
  %lo_all = index.assume %lo_u [range(%lo_u, 0, 262143)] : index
  %hi_all = index.assume {hi} [range({hi}, 1, 262144)] : index
""" + "\n".join(sel) + f"""
  %col_tok_raw = index.add %tok0, %lane_col : index
  %tokens_last = index.sub %tokens, %c1 : index
  %col_tok_c = index.min %col_tok_raw, %tokens_last : index
  %col_tok = index.assume %col_tok_c [lt(%col_tok_c, %tokens)] : index
  %my_lo32 = index.cast {plo} : index to i32
  %my_hi32 = index.cast {phi} : index to i32
  %lo_v = vector.splat %my_lo32 : vector<8xi32>
  %hi_v = vector.splat %my_hi32 : vector<8xi32>
  %my_act = scalar.andi {pact}, %true_m : i1
  %cg = index.constant {G} : index
  %head_value = index.madd %kv, %cg, %wave : index
  %head = index.assume %head_value [range(%head_value, 0, {QH - 1})] : index
  %b0_q = index.div %lo_all, %c32 : index
  %b0 = index.mul %b0_q, %c32 : index
  %key_span = index.sub %hi_all, %b0 : index
  %key_span_b = index.assume %key_span [range(%key_span, 1, 262144)] : index
  %key_span_r = index.add %key_span_b, %c31 : index
  %key_tiles = index.div %key_span_r, %c32 : index
  %k_bytes = index.constant {KT * DP8 * 2} : offset
  %k_tile = buffer.alloca<workgroup> align(16) %k_bytes : buffer
  %k_view = buffer.view %k_tile[%origin] : buffer -> view<{KT}x{DP8}xf16>
  %vt_bytes = index.constant {D * VTC * 2} : offset
  %vt_tile = buffer.alloca<workgroup> align(16) %vt_bytes : buffer
  %vt_view = buffer.view %vt_tile[%origin] : buffer -> view<{D}x{VTC}xf16>
  %q_view = buffer.view %q[%origin] : buffer -> view<[%tokens]x{QH}x{NF}x16xf32>
  %zero16 = vector.constant 0.0 : vector<16xf32>""")
    QLOAD = "\n".join(f"""  %qc{c} = index.constant {c} : index
  %qraw{c} = scf.if %my_act -> (vector<16xf32>) {{
    %qv = vector.load %q_view[%col_tok, %head, %qc{c}, %c0] : view<[%tokens]x{QH}x{NF}x16xf32> -> vector<16xf32>
    scf.yield %qv : vector<16xf32>
  }} else {{
    scf.yield %zero16 : vector<16xf32>
  }}
  %qh{c} = vector.fptrunc %qraw{c} : vector<16xf32> to vector<16xf16>
  %qf{c} = vector.fragment<rhs> %qh{c} shape [%c16, %c16] : vector<16xf16>""" for c in range(NF))
    if not Q_IN_LOOP:
        A(QLOAD)
    A(f"""  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %zero_f = scalar.constant 0.0 : f32
  %one_f = scalar.constant 1.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %scale = scalar.constant {scale} : f32
  %scale_v = vector.splat %scale : vector<8xf32>
  %neg_v = vector.splat %neg_big : vector<8xf32>
  %zero_v8 = vector.constant 0.0 : vector<8xf32>
  %exchange = scalar.constant 16 : i32
  %width = scalar.constant 32 : i32
  %two_i32 = scalar.constant 2 : i32
  %half32 = index.cast %lane_half : index to i32
  %row_off = vector.iota %half32 step %two_i32 : vector<8xi32>
  %ctpk = index.constant {TPK} : index
  %cdpt = index.constant {DPT} : index
  %kz = vector.constant 0.0 : vector<{DPT}xf16>""")
    for c in range(NF):
        A(f"  %d{c} = index.constant {c * 16} : index")
    for i in range(8):
        A(f"  %d{i}i = index.constant {i} : index")
    A("  %kb0_row = index.constant 0 : index\n  %kbo0 = scalar.constant 0 : i32\n  %kb1_row = index.constant 16 : index\n  %kbo1 = scalar.constant 16 : i32")
    # staging slots (gen_flash2.py's scheme)
    if POW2:
        A(f"""  %st_key_value = index.div %workitem, %ctpk : index
  %st_key = index.assume %st_key_value [range(%st_key_value, 0, 31)] : index
  %st_seg = index.rem %workitem, %ctpk : index
  %st_dim_value = index.mul %st_seg, %cdpt : index
  %st_dim = index.assume %st_dim_value [range(%st_dim_value, 0, {D - DPT})] : index""")
    else:
        A(f"  %c_slots = index.constant {SLOTS} : index\n  %c_slots_less = index.constant {SLOTS - 1} : index")
        for p in range(PASSES):
            A(f"""  %st_add_p{p} = index.constant {p * LANES} : index
  %st_slot_p{p} = index.add %workitem, %st_add_p{p} : index
  %st_live_p{p} = index.cmp ult, %st_slot_p{p}, %c_slots : index
  %st_slot_cl_p{p} = index.min %st_slot_p{p}, %c_slots_less : index
  %st_key_value_p{p} = index.div %st_slot_cl_p{p}, %ctpk : index
  %st_key_p{p} = index.assume %st_key_value_p{p} [range(%st_key_value_p{p}, 0, 31)] : index
  %st_seg_p{p} = index.rem %st_slot_cl_p{p}, %ctpk : index
  %st_dim_value_p{p} = index.mul %st_seg_p{p}, %cdpt : index
  %st_dim_p{p} = index.assume %st_dim_value_p{p} [range(%st_dim_value_p{p}, 0, {D - DPT})] : index""")
    acc = ", ".join(f"%a{d} = %init : vector<8xf32>" for d in range(NF))
    A(f"""  %m_final, %l_final, {", ".join(f"%o{d}" for d in range(NF))} = scf.for %kt = [%c0 to %key_tiles step %c1](%m = %neg_big : f32, %l = %zero_f : f32, {acc}) -> (f32, f32, {", ".join(["vector<8xf32>"] * NF)}) {{
    %tile_first = index.madd %kt, %c32, %b0 : index""")

    def stage(sfx, key, dim):
        return f"""    %key_pos{sfx} = index.add %tile_first, {key} : index
    %kv_lo{sfx} = index.cmp uge, %key_pos{sfx}, %lo_all : index
    %kv_hi{sfx} = index.cmp ult, %key_pos{sfx}, %hi_all : index
    %key_valid{sfx} = scalar.andi %kv_lo{sfx}, %kv_hi{sfx} : i1
    %kvals{sfx}, %vvals{sfx} = scf.if %key_valid{sfx} -> (vector<{DPT}xf16>, vector<{DPT}xf16>) {{
      %kp{sfx} = index.assume %key_pos{sfx} [range(%key_pos{sfx}, 0, 262143)] : index
      %k_offset{sfx} = func.call @g_kv_offset(%origins, %row, %kp{sfx}, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %v_offset{sfx} = index.add %k_offset{sfx}, %plane_stride : offset
      %k_cache{sfx} = buffer.view %arena[%k_offset{sfx}] : buffer -> view<{KVH}x{D}xf16>
      %v_cache{sfx} = buffer.view %arena[%v_offset{sfx}] : buffer -> view<{KVH}x{D}xf16>
      %kh{sfx} = vector.load %k_cache{sfx}[%kv, {dim}] : view<{KVH}x{D}xf16> -> vector<{DPT}xf16>
      %vh{sfx} = vector.load %v_cache{sfx}[%kv, {dim}] : view<{KVH}x{D}xf16> -> vector<{DPT}xf16>
      scf.yield %kh{sfx}, %vh{sfx} : vector<{DPT}xf16>, vector<{DPT}xf16>
    }} else {{
      scf.yield %kz, %kz : vector<{DPT}xf16>, vector<{DPT}xf16>
    }}
    vector.store %kvals{sfx}, %k_view[{key}, {dim}] : vector<{DPT}xf16>, view<{KT}x{DP8}xf16>
    scf.for %j{sfx} = [%c0 to %cdpt step %c1] unroll {{
      %vj{sfx} = vector.extract %vvals{sfx}[%j{sfx}] : vector<{DPT}xf16> -> f16
      %vt_row_value{sfx} = index.add {dim}, %j{sfx} : index
      %vt_row{sfx} = index.assume %vt_row_value{sfx} [range(%vt_row_value{sfx}, 0, {D - 1})] : index
      view.store %vj{sfx}, %vt_view[%vt_row{sfx}, {key}] : f16, view<{D}x{VTC}xf16>
    }}"""
    if POW2:
        A(stage("", "%st_key", "%st_dim"))
    else:
        for p in range(PASSES):
            A(f"    scf.if %st_live_p{p} {{\n" + stage(f"_p{p}", f"%st_key_p{p}", f"%st_dim_p{p}").replace("\n    ", "\n      ") + "\n    }")
    A("""    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %tf32 = index.cast %tile_first : index to i32""")
    # S^T for both 16-key halves; with Q reloaded per tile (D > 128) each Q fragment is live for its two MMAs only
    qparts = QLOAD.split("\n  %qc")
    qparts = [qparts[0]] + ["  %qc" + x for x in qparts[1:]]
    prev = ["%init", "%init"]
    for c in range(NF):
        if Q_IN_LOOP:
            A("\n".join("  " + line for line in qparts[c].split("\n")))
        for kb in range(2):
            A(f"""    %kl{kb}_{c} = vector.fragment.load<lhs> %k_view[%kb{kb}_row, %d{c}] shape [%c16, %c16] : view<{KT}x{DP8}xf16> -> vector<16xf16>
    %s{kb}_{c} = vector.mma %kl{kb}_{c}, %qf{c}, {prev[kb]} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
            prev[kb] = f"%s{kb}_{c}"
    for kb in range(2):
        A(f"""    %sc{kb} = vector.mulf %s{kb}_{NF - 1}, %scale_v : vector<8xf32>
    %base{kb} = scalar.addi %tf32, %kbo{kb} : i32
    %base{kb}_v = vector.splat %base{kb} : vector<8xi32>
    %kpos{kb} = vector.addi %base{kb}_v, %row_off : vector<8xi32>
    %vlo{kb} = vector.cmpi sge, %kpos{kb}, %lo_v : vector<8xi32> -> vector<8xi1>
    %vhi{kb} = vector.cmpi slt, %kpos{kb}, %hi_v : vector<8xi32> -> vector<8xi1>
    %valid{kb} = vector.andi %vlo{kb}, %vhi{kb} : vector<8xi1>
    %masked{kb} = vector.select %valid{kb}, %sc{kb}, %neg_v : vector<8xf32>
    %lmax{kb} = vector.reduce<maxnumf> %masked{kb}, %neg_big : vector<8xf32>, f32""")
    A("""    %lmax = scalar.maxnumf %lmax0, %lmax1 : f32
    %omax, %omax_u = kernel.subgroup.shuffle<xor> %lmax, %exchange, %width : f32, i32, i32
    %tmax = scalar.maxnumf %lmax, %omax : f32
    %m_new = scalar.maxnumf %m, %tmax : f32
    %m_shift = scalar.subf %m, %m_new : f32
    %alpha_e = scalar.expf<afn> %m_shift : f32
    %same = scalar.cmpf oeq, %m_shift, %zero_f : f32
    %alpha = scf.select %same, %one_f, %alpha_e : f32
    %m_new_v = vector.splat %m_new : vector<8xf32>""")
    for kb in range(2):
        A(f"""    %sh{kb} = vector.subf %masked{kb}, %m_new_v : vector<8xf32>
    %pe{kb} = vector.expf<afn> %sh{kb} : vector<8xf32>
    %p{kb} = vector.select %valid{kb}, %pe{kb}, %zero_v8 : vector<8xf32>
    %ls{kb} = vector.reduce<addf> %p{kb}, %zero_f : vector<8xf32>, f32""")
        for i in range(8):
            A(f"""    %pi{kb}_{i} = vector.extract %p{kb}[%d{i}i] : vector<8xf32> -> f32
    %po{kb}_{i}, %pu{kb}_{i} = kernel.subgroup.shuffle<xor> %pi{kb}_{i}, %exchange, %width : f32, i32, i32""")
        A(f"""    %partner{kb} = vector.from_elements {", ".join(f"%po{kb}_{i}" for i in range(8))} : vector<8xf32>
    %even{kb} = scf.select %hi_half, %partner{kb}, %p{kb} : vector<8xf32>
    %odd{kb} = scf.select %hi_half, %p{kb}, %partner{kb} : vector<8xf32>
    %pt{kb}_f = vector.interleave<0> %even{kb}, %odd{kb} : vector<8xf32>, vector<8xf32> -> vector<16xf32>
    %pt{kb}_h = vector.fptrunc %pt{kb}_f : vector<16xf32> to vector<16xf16>
    %pt{kb} = vector.fragment<rhs> %pt{kb}_h shape [%c16, %c16] : vector<16xf16>""")
    A("""    %lsum = scalar.addf %ls0, %ls1 : f32
    %osum, %osum_u = kernel.subgroup.shuffle<xor> %lsum, %exchange, %width : f32, i32, i32
    %tsum = scalar.addf %lsum, %osum : f32
    %l_scaled = scalar.mulf %l, %alpha : f32
    %l_new = scalar.addf %l_scaled, %tsum : f32
    %alpha_v = vector.splat %alpha : vector<8xf32>""")
    for d in range(NF):
        A(f"    %os{d} = vector.mulf %a{d}, %alpha_v : vector<8xf32>")
        prev = f"%os{d}"
        for kb in range(2):
            name = f"%on{d}" if kb == 1 else f"%on{d}_{kb}"
            A(f"""    %vl{d}_{kb} = vector.fragment.load<lhs> %vt_view[%d{d}, %kb{kb}_row] shape [%c16, %c16] : view<{D}x{VTC}xf16> -> vector<16xf16>
    {name} = vector.mma %vl{d}_{kb}, %pt{kb}, {prev} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
            prev = name
    A("""    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield %m_new, %l_new, """ + ", ".join(f"%on{d}" for d in range(NF)) + " : f32, f32, " + ", ".join(["vector<8xf32>"] * NF) + """
  }""")
    # partials of this lane's column (token), per-token kernel layout: acc [token, head, split, D], ml [.., 2]
    A(f"""  %acc_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{QH}x{S}x{D}xf32>
  %ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{QH}x{S}x2xf32>
  scf.if %my_act {{
    %lo_half = index.cmp eq, %lane_half, %c0 : index
    scf.if %lo_half {{
      view.store %m_final, %ml_view[%col_tok, %head, %split, 0] : f32, view<[%tokens]x{QH}x{S}x2xf32>
      view.store %l_final, %ml_view[%col_tok, %head, %split, 1] : f32, view<[%tokens]x{QH}x{S}x2xf32>
    }}""")
    for d in range(NF):
        for i in range(8):
            A(f"""    %ov{d}_{i} = vector.extract %o{d}[%d{i}i] : vector<8xf32> -> f32
    %odim{d}_{i}_a = index.constant {d * 16 + 2 * i} : index
    %odim{d}_{i} = index.add %odim{d}_{i}_a, %lane_half : index
    %odb{d}_{i} = index.assume %odim{d}_{i} [range(%odim{d}_{i}, 0, {D - 1})] : index
    view.store %ov{d}_{i}, %acc_view[%col_tok, %head, %split, %odb{d}_{i}] : f32, view<[%tokens]x{QH}x{S}x{D}xf32>""")
    A("""  }
  }
  }
  kernel.return
}
""")
    return "\n".join(L)
