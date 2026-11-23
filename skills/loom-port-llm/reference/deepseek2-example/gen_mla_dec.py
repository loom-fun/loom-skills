"""MLA decode attention on WMMA (gen_llm.py, MLA_DEC=wmma, the default for cfg "mla"; MLA_DEC=split keeps the per-head
split-K kernels of attention_text()).

Absorbed MLA is MQA with ONE 576-dim latent cache row per token shared by all 32 query heads (V = its first 512 dims).
The per-head split-K kernel gives every head its own wave, so every key row is read 32 times (through the cache) and
scored with fp32 VALU dot products: ~70 KFLOP per key per layer, VALU-bound already at 2k keys. Here a workgroup scores
a 16-key block against all 32 heads with WMMA, in the transposed form of gen_flash2s.py where the 16 lane columns are
16 query HEADS of the one decode token instead of 16 tokens of one head:
  S^T [16 keys x 16 heads] = K [keys x 576] Q^T,  O^T [512 dims x 16 heads] += V^T [dims x keys] P^T.
Workgroup (token, split) of 8 waves: wave w = (head tile ht = w / 4: heads 16 ht .. +15, quarter qd = w % 4).
  - S: wave (ht, qd) owns score dims 144 qd .. +143 (9 MMA k-steps; its Q^T fragments stay in registers), loads its
    K fragments straight from the paged cache (a 16-key block is one 16-position page: 16 contiguous rows), publishes
    its S^T partial in workgroup memory; each wave of the head tile sums the four partials in wave order (identical S
    in all four), applies the scale and the causal mask (keys past the token's position: zero K/V, weight 0).
  - V^T is staged in workgroup memory from the same K fragments (the chunks of dims < 512, split between the two head
    tiles' waves by chunk parity).
  - softmax per column (in-lane + one xor-16 shuffle), P^T (f16) as the MMA rhs, O^T += V^T[128 qd + 16 d ..] P^T, 8 MMAs.
Splits: 16-key blocks 0 .. position/16, about MLA_DEC_BMIN blocks or more per split, at most MLA_DEC_S splits.
Partials: part_acc [tokens][S][heads][512] f32 (unnormalized), part_ml [tokens][S][heads][2] (max, sum);
g_mla_dcombine folds them into o[token][head][0:512] (o keeps the 576-dim row stride; dims 512.. are not written:
v_b reads the first 512). Numerics: Q and P rounded to f16 (as the flash prefill), f32 accumulation.
"""
import os

SMAX = int(os.environ.get("MLA_DEC_S", "64"))      # splits per token at most (grid y)
BMIN = int(os.environ.get("MLA_DEC_BMIN", "4"))    # 16-key blocks per split at least
VTC = 16 + int(os.environ.get("MLA_DEC_VTPAD", "8"))   # padded V^T row (halves)


def on():
    return os.environ.get("MLA_DEC", "wmma") == "wmma"


def kernels(cfg, layer_max=51, kvo="g_kv_offset"):
    m = cfg["mla"]
    QH, KL, D = cfg["heads"], m["kv_lora"], m["kv_lora"] + m["rope_dims"]
    assert QH == 32 and KL == 512 and D == 576, "g_mla_dsplit: 32 heads, 512 + 64 latent dims"
    scale = repr(m["scale"])
    NJ = D // 4 // 16          # 9 score chunks of 16 dims per wave
    A = []
    a = A.append
    a(f"""
// MLA decode: splits for nblk 16-key blocks (at least {BMIN} blocks each, at most {SMAX}).
func.def inline @g_mla_dsplits(%nblk: index) -> (index) {{
  %c1 = index.constant 1 : index
  %cb = index.constant {BMIN} : index
  %cbl = index.constant {BMIN - 1} : index
  %cmax = index.constant {SMAX} : index
  %padded = index.add %nblk, %cbl : index
  %target = index.div %padded, %cb : index
  %capped = index.min %target, %cmax : index
  %count_value = index.max %capped, %c1 : index
  %count = index.assume %count_value [range(%count_value, 1, {SMAX})] : index
  func.return %count : index
}}

kernel.def export("g_mla_dsplit") @g_mla_dsplit(%tokens: index) {{
  %c1 = index.constant 1 : index
  %cS = index.constant {SMAX} : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %cS, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %sliding: i32, %context_arg: index, %pool_arg: index, %q: buffer, %info: buffer, %origins: buffer, %arena: buffer, %part_acc: buffer, %part_ml: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {layer_max})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2147483584)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %split_id = kernel.workgroup.id<y> : index
  %split = index.assume %split_id [range(%split_id, 0, {SMAX - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c9 = index.constant 9 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c128 = index.constant 128 : index
  %c144 = index.constant 144 : index
  %c0_i32 = scalar.constant 0 : i32
""" + "".join(f"  %ex{i} = index.constant {i} : index\n" for i in range(16)) + f"""  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  scf.if %live {{
    %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
    %row = index.cast %row_bound : i32 to index
    %position_bound = scalar.assume %position32 [range(%position32, 0, 262143)] : i32
    %position = index.cast %position_bound : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %last_blk = index.div %position, %c16 : index
    %nblk_value = index.add %last_blk, %c1 : index
    %nblk = index.assume %nblk_value [range(%nblk_value, 1, 16384)] : index
    %s_eff = func.call @g_mla_dsplits(%nblk) : (index) -> (index)
    %active = index.cmp ult, %split, %s_eff : index
    scf.if %active {{
    %s_less = index.sub %s_eff, %c1 : index
    %per_padded = index.add %nblk, %s_less : index
    %per = index.div %per_padded, %s_eff : index
    %b0_raw = index.mul %split, %per : index
    %b0 = index.min %b0_raw, %nblk : index
    %b1_raw = index.add %b0_raw, %per : index
    %b1 = index.min %b1_raw, %nblk : index
    %ht = index.div %wave, %c4 : index
    %qd = index.rem %wave, %c4 : index
    %lane_col = index.rem %lane, %c16 : index
    %lane_half = index.div %lane, %c16 : index
    %hi_half = index.cmp ne, %lane_half, %c0 : index
    %head_value = index.madd %ht, %c16, %lane_col : index
    %head = index.assume %head_value [range(%head_value, 0, {QH - 1})] : index
    %qbase = index.mul %qd, %c144 : index
    %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>""")
    for j in range(NJ):
        a(f"""    %qo{j} = index.constant {16 * j} : index
    %qd{j}_v = index.add %qbase, %qo{j} : index
    %qd{j} = index.assume %qd{j}_v [range(%qd{j}_v, 0, {D - 16})] : index
    %qraw{j} = vector.load %q_view[%token, %head, %qd{j}] : view<[%tokens]x{QH}x{D}xf32> -> vector<16xf32>
    %qh{j} = vector.fptrunc %qraw{j} : vector<16xf32> to vector<16xf16>
    %qf{j} = vector.fragment<rhs> %qh{j} shape [%c16, %c16] : vector<16xf16>""")
        # global 16-dim chunk index of this wave's chunk j (qd * 9 + j) and whether this wave stages it into V^T
        a(f"""    %cj{j} = index.constant {j} : index
    %gc{j} = index.madd %qd, %c9, %cj{j} : index
    %gcin{j} = index.cmp ult, %gc{j}, %c32 : index
    %gcpar{j} = index.rem %gc{j}, %c2 : index
    %gcmine{j} = index.cmp eq, %gcpar{j}, %ht : index
    %vst{j} = scalar.andi %gcin{j}, %gcmine{j} : i1
    %vrow{j}_v = index.mul %gc{j}, %c16 : index
    %vrow{j}_h = index.madd %lane_half, %c8, %vrow{j}_v : index""")
    a(f"""    %zero8 = vector.splatf 0.0 : vector<8xf32>
    %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
    %zero16h = vector.splatf 0.0 : vector<16xf16>
    %zero_f = scalar.constant 0.0 : f32
    %neg_big = scalar.constant -3.0e38 : f32
    %scale = scalar.constant {scale} : f32
    %scale_v = vector.splat %scale : vector<8xf32>
    %neg_v = vector.splat %neg_big : vector<8xf32>
    %exchange = scalar.constant 16 : i32
    %width = scalar.constant 32 : i32
    %two_i32 = scalar.constant 2 : i32
    %half32 = index.cast %lane_half : index to i32
    %row_off = vector.iota %half32 step %two_i32 : vector<8xi32>
    %pos32 = index.cast %position : index to i32
    %pos_v = vector.splat %pos32 : vector<8xi32>
    %vt_bytes = index.constant {KL * VTC * 2} : offset
    %vt_tile = buffer.alloca<workgroup> align(16) %vt_bytes : buffer
    %vt_view = buffer.view %vt_tile[%zero] : buffer -> view<{KL}x{VTC}xf16>
    %s_bytes = index.constant {8 * 16 * 16 * 4} : offset
    %s_tile = buffer.alloca<workgroup> align(16) %s_bytes : buffer
    %s_view = buffer.view %s_tile[%zero] : buffer -> view<128x16xf32>
    %my_s = index.mul %wave, %c16 : index
    %ht_s = index.mul %ht, %c4 : index""")
    for p in range(4):
        a(f"""    %pw{p}_c = index.constant {p} : index
    %pw{p} = index.add %ht_s, %pw{p}_c : index
    %ps{p} = index.mul %pw{p}, %c16 : index""")
    a("""    %vbase = index.mul %qd, %c128 : index""")
    for d in range(8):
        a(f"""    %vd{d}_o = index.constant {16 * d} : index
    %vr{d} = index.add %vbase, %vd{d}_o : index""")
    acc = ", ".join(f"%a{d} = %init : vector<8xf32>" for d in range(8))
    a(f"""    %m_final, %l_final, {", ".join(f"%o{d}" for d in range(8))} = scf.for %b = [%b0 to %b1 step %c1](%m = %neg_big : f32, %l = %zero_f : f32, {acc}) -> (f32, f32, {", ".join(["vector<8xf32>"] * 8)}) {{
      %tile_first = index.mul %b, %c16 : index
      %k_off = func.call @{kvo}(%origins, %row, %tile_first, %blocks, %layer, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %kv = buffer.view %arena[%k_off] : buffer -> view<16x{D}xf16>
      %key_pos = index.add %tile_first, %lane_col : index
      %kvalid = index.cmp ule, %key_pos, %position : index""")
    prev = "%init"
    for j in range(NJ):
        a(f"""      %kr{j} = vector.load %kv[%lane_col, %qd{j}] : view<16x{D}xf16> -> vector<16xf16>
      %kh{j} = scf.select %kvalid, %kr{j}, %zero16h : vector<16xf16>
      %kf{j} = vector.fragment<lhs> %kh{j} shape [%c16, %c16] : vector<16xf16>
      %s{j} = vector.mma %kf{j}, %qf{j}, {prev} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
        prev = f"%s{j}"
    for j in range(NJ):
        st = []
        for i in range(8):
            st.append(f"""        %ea{j}_{i} = vector.extract %kh{j}[%ex{i}] : vector<16xf16> -> f16
        %eb{j}_{i} = vector.extract %kh{j}[%ex{i + 8}] : vector<16xf16> -> f16
        %ev{j}_{i} = scf.select %hi_half, %eb{j}_{i}, %ea{j}_{i} : f16
        %vro{j}_{i}_c = index.constant {i} : index
        %vro{j}_{i}_v = index.add %vrow{j}_h, %vro{j}_{i}_c : index
        %vro{j}_{i} = index.assume %vro{j}_{i}_v [range(%vro{j}_{i}_v, 0, {KL - 1})] : index
        view.store %ev{j}_{i}, %vt_view[%vro{j}_{i}, %lane_col] : f16, view<{KL}x{VTC}xf16>""")
        a(f"""      scf.if %vst{j} {{
{chr(10).join(st)}
      }}""")
    a(f"""      vector.fragment.store<result> %s{NJ - 1}, %s_view[%my_s, %c0] shape [%c16, %c16] : vector<8xf32>, view<128x16xf32>
      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
      %sp0 = vector.fragment.load<result> %s_view[%ps0, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
      %sp1 = vector.fragment.load<result> %s_view[%ps1, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
      %sp2 = vector.fragment.load<result> %s_view[%ps2, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
      %sp3 = vector.fragment.load<result> %s_view[%ps3, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
      %st01 = vector.addf %sp0, %sp1 : vector<8xf32>
      %st012 = vector.addf %st01, %sp2 : vector<8xf32>
      %st = vector.addf %st012, %sp3 : vector<8xf32>
      %sc = vector.mulf %st, %scale_v : vector<8xf32>
      %tf32 = index.cast %tile_first : index to i32
      %base_v = vector.splat %tf32 : vector<8xi32>
      %kpos = vector.addi %base_v, %row_off : vector<8xi32>
      %valid = vector.cmpi sle, %kpos, %pos_v : vector<8xi32> -> vector<8xi1>
      %masked = vector.select %valid, %sc, %neg_v : vector<8xf32>
      %lmax = vector.reduce<maxnumf> %masked, %neg_big : vector<8xf32>, f32
      %omax, %omax_u = kernel.subgroup.shuffle<xor> %lmax, %exchange, %width : f32, i32, i32
      %tmax = scalar.maxnumf %lmax, %omax : f32
      %m_new = scalar.maxnumf %m, %tmax : f32
      %m_shift = scalar.subf %m, %m_new : f32
      %alpha = scalar.expf<afn> %m_shift : f32
      %m_new_v = vector.splat %m_new : vector<8xf32>
      %sh = vector.subf %masked, %m_new_v : vector<8xf32>
      %pe = vector.expf<afn> %sh : vector<8xf32>
      %p = vector.select %valid, %pe, %zero8 : vector<8xf32>
      %ls = vector.reduce<addf> %p, %zero_f : vector<8xf32>, f32""")
    for i in range(8):
        a(f"""      %pi{i} = vector.extract %p[%ex{i}] : vector<8xf32> -> f32
      %po{i}, %pu{i} = kernel.subgroup.shuffle<xor> %pi{i}, %exchange, %width : f32, i32, i32""")
    a(f"""      %partner = vector.from_elements {", ".join(f"%po{i}" for i in range(8))} : vector<8xf32>
      %even = scf.select %hi_half, %partner, %p : vector<8xf32>
      %odd = scf.select %hi_half, %p, %partner : vector<8xf32>
      %pt_f = vector.interleave<0> %even, %odd : vector<8xf32>, vector<8xf32> -> vector<16xf32>
      %pt_h = vector.fptrunc %pt_f : vector<16xf32> to vector<16xf16>
      %pt = vector.fragment<rhs> %pt_h shape [%c16, %c16] : vector<16xf16>
      %osum, %osum_u = kernel.subgroup.shuffle<xor> %ls, %exchange, %width : f32, i32, i32
      %tsum = scalar.addf %ls, %osum : f32
      %l_scaled = scalar.mulf %l, %alpha : f32
      %l_new = scalar.addf %l_scaled, %tsum : f32
      %alpha_v = vector.splat %alpha : vector<8xf32>""")
    for d in range(8):
        a(f"""      %os{d} = vector.mulf %a{d}, %alpha_v : vector<8xf32>
      %vl{d} = vector.fragment.load<lhs> %vt_view[%vr{d}, %c0] shape [%c16, %c16] : view<{KL}x{VTC}xf16> -> vector<16xf16>
      %an{d} = vector.mma %vl{d}, %pt, %os{d} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
    a(f"""      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
      scf.yield %m_new, %l_new, {", ".join(f"%an{d}" for d in range(8))} : f32, f32, {", ".join(["vector<8xf32>"] * 8)}
    }}
    // partials: O^T (dim row, head col) through a transposed strided view of part_acc[token][split][head][512]
    %c512 = index.constant {KL} : index
    %acc_layout = encoding.layout.strided [%c1, %c512] : encoding<layout>
    %cqh = index.constant {QH} : index
    %ts = index.madd %token, %cS_, %split : index
    %tsh = index.madd %ts, %cqh, %head_tile0 : index
    %acc_first_a = index.mul %tsh, %c512 : index
    %acc_first = index.add %acc_first_a, %vbase : index
    %four = index.constant 4 : offset
    %acc_off = index.scale %acc_first, %four : index, offset -> offset
    %acc_view = buffer.view %part_acc[%acc_off] : buffer -> view<128x16xf32, %acc_layout>""")
    for d in range(8):
        a(f"""    vector.fragment.store<result> %o{d}, %acc_view[%vd{d}_o, %c0] shape [%c16, %c16] : vector<8xf32>, view<128x16xf32, %acc_layout>""")
    a(f"""    %q0 = index.cmp eq, %qd, %c0 : index
    %lo_half = index.cmp eq, %lane_half, %c0 : index
    %ml_lane = scalar.andi %q0, %lo_half : i1
    scf.if %ml_lane {{
      %ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{SMAX}x{QH}x2xf32>
      view.store %m_final, %ml_view[%token, %split, %head, 0] : f32, view<[%tokens]x{SMAX}x{QH}x2xf32>
      view.store %l_final, %ml_view[%token, %split, %head, 1] : f32, view<[%tokens]x{SMAX}x{QH}x2xf32>
    }}
    }}
  }}
  kernel.return
}}

// o[t, head, 0:512] = sum_s e^(m_s - M) acc_s / sum_s e^(m_s - M) l_s over the token's splits.
kernel.def export("g_mla_dcombine") @g_mla_dcombine(%tokens: index) {{
  %c1 = index.constant 1 : index
  %cqh = index.constant {QH} : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%tokens, %cqh, %c1) workgroup_size(%c128, %c1, %c1) : index
}} launch(%tokens_arg: index, %sliding: i32, %info: buffer, %part_acc: buffer, %part_ml: buffer, %gate: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %head = index.assume %head_id [range(%head_id, 0, {QH - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 127)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %zero4 = vector.splat %zero_f : vector<4xf32>
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{QH}x{D // 4}x4xf32>
  scf.if %live {{
    %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
    %position_bound = scalar.assume %position32 [range(%position32, 0, 262143)] : i32
    %position = index.cast %position_bound : i32 to index
    %last_blk = index.div %position, %c16 : index
    %nblk_value = index.add %last_blk, %c1 : index
    %nblk = index.assume %nblk_value [range(%nblk_value, 1, 16384)] : index
    %s_eff = func.call @g_mla_dsplits(%nblk) : (index) -> (index)
    %ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{SMAX}x{QH}x2xf32>
    %acc_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{SMAX}x{QH}x128x4xf32>
    %maximum = scf.for %s = [%c0 to %s_eff step %c1](%mx = %neg_big : f32) -> (f32) {{
      %bs = index.assume %s [range(%s, 0, {SMAX - 1})] : index
      %ms = view.load %ml_view[%token, %bs, %head, 0] : view<[%tokens]x{SMAX}x{QH}x2xf32> -> f32
      %next = scalar.maxnumf %mx, %ms : f32
      scf.yield %next : f32
    }}
    %den, %total = scf.for %s = [%c0 to %s_eff step %c1](%d = %zero_f : f32, %acc = %zero4 : vector<4xf32>) -> (f32, vector<4xf32>) {{
      %bs = index.assume %s [range(%s, 0, {SMAX - 1})] : index
      %ms = view.load %ml_view[%token, %bs, %head, 0] : view<[%tokens]x{SMAX}x{QH}x2xf32> -> f32
      %ls = view.load %ml_view[%token, %bs, %head, 1] : view<[%tokens]x{SMAX}x{QH}x2xf32> -> f32
      %shift = scalar.subf %ms, %maximum : f32
      %w = scalar.expf<afn> %shift : f32
      %wl = scalar.mulf %w, %ls : f32
      %d_next = scalar.addf %d, %wl : f32
      %part = vector.load %acc_view[%token, %bs, %head, %lane, %c0] : view<[%tokens]x{SMAX}x{QH}x128x4xf32> -> vector<4xf32>
      %w_v = vector.splat %w : vector<4xf32>
      %acc_next = vector.fmaf %w_v, %part, %acc : vector<4xf32>
      scf.yield %d_next, %acc_next : f32, vector<4xf32>
    }}
    %one = scalar.constant 1.0 : f32
    %inv = scalar.divf %one, %den : f32
    %inv_v = vector.splat %inv : vector<4xf32>
    %result = vector.mulf %total, %inv_v : vector<4xf32>
    vector.store %result, %out_view[%token, %head, %lane, %c0] : vector<4xf32>, view<[%tokens]x{QH}x{D // 4}x4xf32>
  }} else {{
    vector.store %zero4, %out_view[%token, %head, %lane, %c0] : vector<4xf32>, view<[%tokens]x{QH}x{D // 4}x4xf32>
  }}
  kernel.return
}}
""")
    t = "\n".join(A)
    # constants used before their natural place
    t = t.replace("    %ts = index.madd %token, %cS_, %split : index\n",
                  f"    %cS_ = index.constant {SMAX} : index\n    %head_tile0 = index.mul %ht, %c16 : index\n    %ts = index.madd %token, %cS_, %split : index\n")
    return t


if __name__ == "__main__":
    import json
    import sys
    print(kernels(json.load(open(sys.argv[1]))))
