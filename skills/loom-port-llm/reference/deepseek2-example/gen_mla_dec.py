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
import re

SMAX = int(os.environ.get("MLA_DEC_S", "32"))      # splits per token at most (grid y)
BMIN = int(os.environ.get("MLA_DEC_BMIN", "2"))    # 16-key blocks per split at least
# QL=1: the Q^T fragments live in workgroup memory (36 KB) instead of 72 VGPRs, read back per block, so PF 2 (next
# block's K during this block's softmax / PV) fits in registers; needs MLA_DEC_VTPAD <= 2 to stay within 64 KB
QL = int(os.environ.get("MLA_DEC_QL", "0"))
VTC = 16 + int(os.environ.get("MLA_DEC_VTPAD", "2" if QL else "8"))   # padded V^T row (halves)
# timing ablations (kbench only, results wrong): empty (no key blocks), noqk (no S MMAs), nostv (no V^T stores),
# nopv (no PV MMAs), nosx (no S exchange: own partial, no first barrier), nopart (no partial O stores)
# prefetch: 0 = g_kv_offset per block (3 dependent loads), 1 = page map hoisted + next page index prefetched (1 load
# round trip per block), 2 = 1 + the next block's K fragments loaded during this block's softmax / PV
PF = int(os.environ.get("MLA_DEC_PF", "0"))
NPF = int(os.environ.get("MLA_DEC_PFN", "9"))   # PF 2: score chunks of the next block prefetched (the rest loaded in the block)
# partial layout: 0 = part_acc[token][split][head][512] (transposed fragment stores: 8 scattered 4 B stores per lane),
# 1 = part_acc[token][split][512][head] (16 lanes = 16 heads write 64 contiguous bytes per row); g_mla_dcombine follows
PL = int(os.environ.get("MLA_DEC_PL", "1"))
HG = int(os.environ.get("MLA_DEC_HG", "1"))   # 2: the 32 heads split over 2 workgroups (twice the workgroups, half the serial chain each)
PH = int(os.environ.get("MLA_DEC_PH", "0"))   # 1 (PL 1, CU > 1): partial O normalized by its own l and stored as f16 (half the partial traffic)
# KLDS=1 (PF 0): K blocks staged in workgroup memory, double-buffered: the next block's global loads are issued before
# this block's QK / softmax / PV and stored to the other buffer after it (one barrier per block); V^T fragments are read
# straight from the staged K block (no separate V^T tile). Rows past the token's position are stored as zeros.
KLDS = int(os.environ.get("MLA_DEC_KLDS", "0"))
QK2 = int(os.environ.get("MLA_DEC_QK2", "0"))   # 1: the score MMA chain split into two accumulators (even / odd chunks), summed at the end
STI = int(os.environ.get("MLA_DEC_STI", "1" if HG == 2 else "0"))   # V^T staging of each chunk right after its score MMA
CU = int(os.environ.get("MLA_DEC_CU", "4"))   # g_mla_dcombine (PL 1): splits whose loads are issued together
ABL = set(x for x in os.environ.get("MLA_DEC_ABL", "").split(",") if x)


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
    %b1 = index.min %b1_raw, %nblk : index""" + ("""
    %b1e = index.add %b0, %c0 : index""" if "empty" in ABL else "") + f"""
    %ht = index.div %wave, %c4 : index
    %qd = index.rem %wave, %c4 : index
    %lane_col = index.rem %lane, %c16 : index
    %lane_half = index.div %lane, %c16 : index
    %hi_half = index.cmp ne, %lane_half, %c0 : index
    %head_value = index.madd %ht, %c16, %lane_col : index
    %head = index.assume %head_value [range(%head_value, 0, {QH - 1})] : index
    %qbase = index.mul %qd, %c144 : index
    %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>""" + (f"""
    %ql_bytes = index.constant {8 * NJ * 16 * 16 * 2} : offset
    %ql_tile = buffer.alloca<workgroup> align(16) %ql_bytes : buffer
    %ql_view = buffer.view %ql_tile[%zero] : buffer -> view<8x{NJ}x16x16xf16>""" if QL else ""))
    for j in range(NJ):
        a(f"""    %qo{j} = index.constant {16 * j} : index
    %qd{j}_v = index.add %qbase, %qo{j} : index
    %qd{j} = index.assume %qd{j}_v [range(%qd{j}_v, 0, {D - 16})] : index
    %qraw{j} = vector.load %q_view[%token, %head, %qd{j}] : view<[%tokens]x{QH}x{D}xf32> -> vector<16xf32>
    %qh{j} = vector.fptrunc %qraw{j} : vector<16xf32> to vector<16xf16>""" + (f"""
    %cqj{j} = index.constant {j} : index
    vector.store %qh{j}, %ql_view[%wave, %cqj{j}, %lane_col, %c0] : vector<16xf16>, view<8x{NJ}x16x16xf16>""" if QL else f"""
    %qf{j} = vector.fragment<rhs> %qh{j} shape [%c16, %c16] : vector<16xf16>"""))
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
""" + (f"""    %kb_bytes = index.constant {2 * 16 * D * 2} : offset
    %kbuf = buffer.alloca<workgroup> align(16) %kb_bytes : buffer
    %kb_view = buffer.view %kbuf[%zero] : buffer -> view<2x16x{D}xf16>
    %kslot_bytes = index.constant {16 * D * 2} : offset""" if KLDS else f"""    %vt_bytes = index.constant {KL * VTC * 2} : offset
    %vt_tile = buffer.alloca<workgroup> align(16) %vt_bytes : buffer
    %vt_view = buffer.view %vt_tile[%zero] : buffer -> view<{KL}x{VTC}xf16>""") + f"""
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
    acc_pf = ""
    ty_pf = ""
    if PF:
        a(f"""    %rec_stride = index.constant 40 : offset
    %rec_off = index.scale %row, %rec_stride : index, offset -> offset
    %records = buffer.view %origins[%rec_off] : buffer -> view<5xi64>
    %map64 = view.load %records[4] : view<5xi64> -> i64
    %map_bound = scalar.assume %map64 [range(%map64, 0, 9223372036854775807)] : i64
    %map_origin = index.cast %map_bound : i64 to offset
    %page_map = buffer.view %origins[%map_origin] : buffer -> view<[%blocks]xi32>
    %blk_last = index.sub %nblk, %c1 : index
    %row_bytes = index.constant {D * 2} : offset
    %blk_max = index.sub %blocks, %c1 : index
    %pb0_v = index.min %b0, %blk_last : index
    %pb0 = index.assume %pb0_v [range(%pb0_v, 0, %blk_max), lt(%pb0_v, %blocks)] : index
    %ph0 = view.load %page_map[%pb0] : view<[%blocks]xi32> -> i32""")
        acc_pf = ", %ph = %ph0 : i32"
        ty_pf = ", i32"
        if PF == 2:
            a(f"""    %pb1_v = index.add %b0, %c1 : index
    %pb1_m = index.min %pb1_v, %blk_last : index
    %pb1 = index.assume %pb1_m [range(%pb1_m, 0, %blk_max), lt(%pb1_m, %blocks)] : index
    %ph1 = view.load %page_map[%pb1] : view<[%blocks]xi32> -> i32
    %phb0 = scalar.assume %ph0 [range(%ph0, 0, 134217727)] : i32
    %phys0 = index.cast %phb0 : i32 to index
    %pbase0 = index.mul %phys0, %c16 : index
    %pslot0 = index.madd %layer, %pool, %pbase0 : index
    %koff0 = index.scale %pslot0, %row_bytes : index, offset -> offset
    %kv0 = buffer.view %arena[%koff0] : buffer -> view<16x{D}xf16>""")
            for j in range(min(NPF, NJ)):
                a(f"""    %kq0_{j} = vector.load %kv0[%lane_col, %qd{j}] : view<16x{D}xf16> -> vector<16xf16>""")
            acc_pf = ", %ph = %ph1 : i32" + "".join(f", %kc{j} = %kq0_{j} : vector<16xf16>" for j in range(min(NPF, NJ)))
            ty_pf = ", i32" + ", vector<16xf16>" * min(NPF, NJ)
    def kload(tag, blk, guard):
        """global loads of block BLK's 16 x D K rows, cooperatively (lane = (row, 36-value segment)), rows past the
        position as zeros; returns the 9 vector<4xf16> names"""
        a(f"""    %{tag}_tf = index.mul {blk}, %c16 : index
    %{tag}_r_v = index.div %kwi, %c16 : index
    %{tag}_r = index.assume %{tag}_r_v [range(%{tag}_r_v, 0, 15)] : index
    %{tag}_seg = index.rem %kwi, %c16 : index
    %{tag}_kp = index.add %{tag}_tf, %{tag}_r : index
    %{tag}_ok = index.cmp ule, %{tag}_kp, %position : index
    %{tag}_ld = scalar.andi %{tag}_ok, {guard} : i1
    {", ".join(f"%{tag}_v{m}" for m in range(9))} = scf.if %{tag}_ld -> ({", ".join(["vector<4xf16>"] * 9)}) {{
      %{tag}_off = func.call @{kvo}(%origins, %row, %{tag}_tf, %blocks, %layer, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %{tag}_g = buffer.view %arena[%{tag}_off] : buffer -> view<16x{D}xf16>""")
        for m in range(9):
            a(f"""      %{tag}_d{m}_c = index.constant {4 * m} : index
      %{tag}_d{m}_v = index.madd %{tag}_seg, %c36k, %{tag}_d{m}_c : index
      %{tag}_d{m} = index.assume %{tag}_d{m}_v [range(%{tag}_d{m}_v, 0, {D - 4})] : index
      %{tag}_l{m} = vector.load %{tag}_g[%{tag}_r, %{tag}_d{m}] : view<16x{D}xf16> -> vector<4xf16>""")
        a(f"""      scf.yield {", ".join(f"%{tag}_l{m}" for m in range(9))} : {", ".join(["vector<4xf16>"] * 9)}
    }} else {{
      scf.yield {", ".join(["%zero4h"] * 9)} : {", ".join(["vector<4xf16>"] * 9)}
    }}""")

    def kstore(tag, slot, guard):
        a(f"""    scf.if {guard} {{""")
        for m in range(9):
            a(f"""      %{tag}_e{m}_c = index.constant {4 * m} : index
      %{tag}_e{m}_v = index.madd %{tag}_seg, %c36k, %{tag}_e{m}_c : index
      %{tag}_e{m} = index.assume %{tag}_e{m}_v [range(%{tag}_e{m}_v, 0, {D - 4})] : index
      vector.store %{tag}_v{m}, %kb_view[{slot}, %{tag}_r, %{tag}_e{m}] : vector<4xf16>, view<2x16x{D}xf16>""")
        a("    }")
    if KLDS:
        assert PF == 0 and HG == 1
        a(f"""    %kwi_id = kernel.workitem.id<x> : index
    %kwi = index.assume %kwi_id [range(%kwi_id, 0, 255)] : index
    %c36k = index.constant 36 : index
    %zero4h = vector.splatf 0.0 : vector<4xf16>
    %true_k = scalar.constant true : i1
    %kslot0 = index.constant 0 : index""")
        kload("kp0", "%b0", "%true_k")
        kstore("kp0", "%kslot0", "%true_k")
        a("""    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)""")
    a(f"""    %m_final, %l_final, {", ".join(f"%o{d}" for d in range(8))}{", %ph_fin" if PF else ""}{"".join(f", %kc_fin{j}" for j in range(min(NPF, NJ))) if PF == 2 else ""} = scf.for %b = [%b0 to {"%b1e" if "empty" in ABL else "%b1"} step %c1](%m = %neg_big : f32, %l = %zero_f : f32, {acc}{acc_pf}) -> (f32, f32, {", ".join(["vector<8xf32>"] * 8)}{ty_pf}) {{
      %tile_first = index.mul %b, %c16 : index""")
    if PF == 0 and KLDS:
        a(f"""      %kb_rel = index.sub %b, %b0 : index
      %kb_cur = index.rem %kb_rel, %c2 : index
      %kb_nxt = index.sub %c1, %kb_cur : index
      %kb_off = index.scale %kb_cur, %kslot_bytes : index, offset -> offset
      %kv = buffer.view %kbuf[%kb_off] : buffer -> view<16x{D}xf16>
      %kb_bn = index.add %b, %c1 : index
      %kb_has = index.cmp ult, %kb_bn, %b1 : index""")
        kload("kn", "%kb_bn", "%kb_has")
    elif PF == 0:
        a(f"""      %k_off = func.call @{kvo}(%origins, %row, %tile_first, %blocks, %layer, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %kv = buffer.view %arena[%k_off] : buffer -> view<16x{D}xf16>""")
    elif PF == 1:
        a(f"""      %bn_v = index.add %b, %c1 : index
      %bn_m = index.min %bn_v, %blk_last : index
      %bn = index.assume %bn_m [range(%bn_m, 0, %blk_max), lt(%bn_m, %blocks)] : index
      %ph_next = view.load %page_map[%bn] : view<[%blocks]xi32> -> i32
      %phb = scalar.assume %ph [range(%ph, 0, 134217727)] : i32
      %phys = index.cast %phb : i32 to index
      %pbase = index.mul %phys, %c16 : index
      %pslot = index.madd %layer, %pool, %pbase : index
      %k_off = index.scale %pslot, %row_bytes : index, offset -> offset
      %kv = buffer.view %arena[%k_off] : buffer -> view<16x{D}xf16>""")
    else:
        a(f"""      %bn_v = index.add %b, %c2 : index
      %bn_m = index.min %bn_v, %blk_last : index
      %bn = index.assume %bn_m [range(%bn_m, 0, %blk_max), lt(%bn_m, %blocks)] : index
      %ph_next = view.load %page_map[%bn] : view<[%blocks]xi32> -> i32""" + (f"""
      %bc = index.assume %b [range(%b, 0, %blk_max), lt(%b, %blocks)] : index
      %phc = view.load %page_map[%bc] : view<[%blocks]xi32> -> i32
      %phcb = scalar.assume %phc [range(%phc, 0, 134217727)] : i32
      %physc = index.cast %phcb : i32 to index
      %pbasec = index.mul %physc, %c16 : index
      %pslotc = index.madd %layer, %pool, %pbasec : index
      %k_offc = index.scale %pslotc, %row_bytes : index, offset -> offset
      %kv = buffer.view %arena[%k_offc] : buffer -> view<16x{D}xf16>""" if NPF < NJ else ""))
    a(f"""      %key_pos = index.add %tile_first, %lane_col : index
      %kvalid = index.cmp ule, %key_pos, %position : index""")
    def stage(j):
        st = []
        for i in range(8):
            st.append(f"""        %ea{j}_{i} = vector.extract %kh{j}[%ex{i}] : vector<16xf16> -> f16
        %eb{j}_{i} = vector.extract %kh{j}[%ex{i + 8}] : vector<16xf16> -> f16
        %ev{j}_{i} = scf.select %hi_half, %eb{j}_{i}, %ea{j}_{i} : f16
        %vro{j}_{i}_c = index.constant {i} : index
        %vro{j}_{i}_v = index.add %vrow{j}_h, %vro{j}_{i}_c : index
        %vro{j}_{i} = index.assume %vro{j}_{i}_v [range(%vro{j}_{i}_v, 0, {KL - 1})] : index
        view.store %ev{j}_{i}, %vt_view[%vro{j}_{i}, %lane_col] : f16, view<{KL}x{VTC}xf16>""")
        if "nostv" in ABL:
            return
        a(f"""      scf.if %vst{j} {{
{chr(10).join(st)}
      }}""")
    prev = "%init"
    prev2 = ["%init", "%init"]
    for j in range(NJ):
        pre = PF == 2 and j < NPF
        a((f"""      %kr{j} = vector.load %kv[%lane_col, %qd{j}] : view<16x{D}xf16> -> vector<16xf16>
""" if not pre else "") + f"""      %kh{j} = scf.select %kvalid, {"%kc" if pre else "%kr"}{j}, %zero16h : vector<16xf16>
      %kf{j} = vector.fragment<lhs> %kh{j} shape [%c16, %c16] : vector<16xf16>
""" + (f"""      %qv{j} = vector.load %ql_view[%wave, %cqj{j}, %lane_col, %c0] : view<8x{NJ}x16x16xf16> -> vector<16xf16>
      %qf{j} = vector.fragment<rhs> %qv{j} shape [%c16, %c16] : vector<16xf16>
""" if QL else "") + (f"""      %{"sq" if QK2 else "s"}{j} = vector.mma %kf{j}, %qf{j}, {(prev2[j % 2] if QK2 else prev)} : vector<16xf16>, vector<16xf16>, vector<8xf32>""" if "noqk" not in ABL else f"""      %s{j} = vector.addf {prev}, {prev} : vector<8xf32>"""))
        prev = f"%s{j}"
        prev2[j % 2] = f"%sq{j}"
        if STI and not KLDS:
            stage(j)
    if not STI and not KLDS:
        for j in range(NJ):
            stage(j)
    if QK2 and "noqk" not in ABL:
        a(f"""      %s{NJ - 1} = vector.addf {prev2[0]}, {prev2[1]} : vector<8xf32>""")
    if PF == 2:
        a(f"""      %phb = scalar.assume %ph [range(%ph, 0, 134217727)] : i32
      %phys = index.cast %phb : i32 to index
      %pbase = index.mul %phys, %c16 : index
      %pslot = index.madd %layer, %pool, %pbase : index
      %k_off = index.scale %pslot, %row_bytes : index, offset -> offset
      %kvn = buffer.view %arena[%k_off] : buffer -> view<16x{D}xf16>""")
        for j in range(min(NPF, NJ)):
            a(f"""      %kn{j} = vector.load %kvn[%lane_col, %qd{j}] : view<16x{D}xf16> -> vector<16xf16>""")
    if "nosx" in ABL:
        a(f"""      %st = vector.addf %s{NJ - 1}, %s{NJ - 1} : vector<8xf32>
      %sc = vector.mulf %st, %scale_v : vector<8xf32>""")
    else:
        a(f"""      vector.fragment.store<result> %s{NJ - 1}, %s_view[%my_s, %c0] shape [%c16, %c16] : vector<8xf32>, view<128x16xf32>
      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
      %sp0 = vector.fragment.load<result> %s_view[%ps0, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
      %sp1 = vector.fragment.load<result> %s_view[%ps1, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
      %sp2 = vector.fragment.load<result> %s_view[%ps2, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
      %sp3 = vector.fragment.load<result> %s_view[%ps3, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
      %st01 = vector.addf %sp0, %sp1 : vector<8xf32>
      %st012 = vector.addf %st01, %sp2 : vector<8xf32>
      %st = vector.addf %st012, %sp3 : vector<8xf32>
      %sc = vector.mulf %st, %scale_v : vector<8xf32>""")
    a(f"""      %tf32 = index.cast %tile_first : index to i32
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
""" + (f"""      %vl{d} = vector.fragment.load<lhs> %vt_view[%vr{d}, %c0] shape [%c16, %c16] : view<{KL}x{VTC}xf16> -> vector<16xf16>
""" if not KLDS else f"""      %vdim{d}_v = index.add %vr{d}, %lane_col : index
      %vdim{d} = index.assume %vdim{d}_v [range(%vdim{d}_v, 0, {KL - 1})] : index
""" + "".join(f"""      %vk{d}_{k} = view.load %kv[%ex{k}, %vdim{d}] : view<16x{D}xf16> -> f16
""" for k in range(16)) + f"""      %vle{d} = vector.from_elements {", ".join(f"%vk{d}_{k}" for k in range(16))} : vector<16xf16>
      %vl{d} = vector.fragment<lhs> %vle{d} shape [%c16, %c16] : vector<16xf16>
""") + (f"""      %an{d} = vector.mma %vl{d}, %pt, %os{d} : vector<16xf16>, vector<16xf16>, vector<8xf32>""" if "nopv" not in ABL else f"""      %an{d} = vector.addf %os{d}, %os{d} : vector<8xf32>"""))
    if KLDS:
        kstore("kn", "%kb_nxt", "%kb_has")
    a(f"""      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
      scf.yield %m_new, %l_new, {", ".join(f"%an{d}" for d in range(8))}{", %ph_next" if PF else ""}{"".join(f", %kn{j}" for j in range(min(NPF, NJ))) if PF == 2 else ""} : f32, f32, {", ".join(["vector<8xf32>"] * 8)}{ty_pf}
    }}
    // partials: O^T (dim row, head col) through a transposed strided view of part_acc[token][split][head][512]
    %c512 = index.constant {KL} : index
    %cqh = index.constant {QH} : index
    %ts = index.madd %token, %cS_, %split : index""" + ("""
    %acc_layout = encoding.layout.strided [%c1, %c512] : encoding<layout>
    %tsh = index.madd %ts, %cqh, %head_tile0 : index
    %acc_first_a = index.mul %tsh, %c512 : index
    %acc_first = index.add %acc_first_a, %vbase : index
""" if PL == 0 else """
    %acc_layout = encoding.layout.strided [%c32, %c1] : encoding<layout>
    %tsd = index.madd %ts, %c512, %vbase : index
    %acc_first = index.madd %tsd, %c32, %head_tile0 : index
""") + f"""    %four = index.constant 4 : offset
    %acc_off = index.scale %acc_first, %four : index, offset -> offset
    %acc_view = buffer.view %part_acc[%acc_off] : buffer -> view<128x16xf32, %acc_layout>""")
    if PH and "nopart" not in ABL:
        assert PL == 1
        a(f"""    %ph_one = scalar.constant 1.0 : f32
    %ph_l_gt = scalar.cmpf ogt, %l_final, %zero_f : f32
    %ph_linv = scalar.divf %ph_one, %l_final : f32
    %ph_inv = scf.select %ph_l_gt, %ph_linv, %zero_f : f32
    %ph_invv = vector.splat %ph_inv : vector<8xf32>
    %ph_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{SMAX}x{KL}x{QH}xf16>""")
        for d in range(8):
            a(f"""    %on{d} = vector.mulf %o{d}, %ph_invv : vector<8xf32>
    %oh{d} = vector.fptrunc %on{d} : vector<8xf32> to vector<8xf16>""")
            for i in range(8):
                a(f"""    %phe{d}_{i} = vector.extract %oh{d}[%ex{i}] : vector<8xf16> -> f16
    %phr{d}_{i}_c = index.constant {16 * d + 2 * i} : index
    %phr{d}_{i}_b = index.add %vbase, %phr{d}_{i}_c : index
    %phr{d}_{i}_v = index.add %phr{d}_{i}_b, %lane_half : index
    %phr{d}_{i} = index.assume %phr{d}_{i}_v [range(%phr{d}_{i}_v, 0, {KL - 1})] : index
    view.store %phe{d}_{i}, %ph_view[%token, %split, %phr{d}_{i}, %head] : f16, view<[%tokens]x{SMAX}x{KL}x{QH}xf16>""")
    for d in range(8 if "nopart" not in ABL and not PH else 0):
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

""")
    if PL == 0:
        a(f"""
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
    else:
        a(combine_pl1(QH, KL, D))
    t = "\n".join(A)
    # constants used before their natural place
    t = t.replace("    %ts = index.madd %token, %cS_, %split : index\n",
                  f"    %cS_ = index.constant {SMAX} : index\n    %head_tile0 = index.mul %ht, %c16 : index\n    %ts = index.madd %token, %cS_, %split : index\n")
    if HG == 2:   # head groups: workgroup (token, split, head tile) of 4 waves (one 16-head tile, the 4 dim quarters)
        for a_, b_ in (("kernel.launch.config workgroups(%tokens, %cS, %c1) workgroup_size(%c256, %c1, %c1)",
                        "%c2hg = index.constant 2 : index\n  %c128hg = index.constant 128 : index\n  kernel.launch.config workgroups(%tokens, %cS, %c2hg) workgroup_size(%c128hg, %c1, %c1)"),
                       ("%wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index", "%wave = index.assume %wave_id [range(%wave_id, 0, 3)] : index"),
                       ("    %ht = index.div %wave, %c4 : index", "    %ht_id = kernel.workgroup.id<z> : index\n    %ht = index.assume %ht_id [range(%ht_id, 0, 1)] : index"),
                       ("    %qd = index.rem %wave, %c4 : index", "    %qd = index.add %wave, %c0 : index"),
                       ("    %ht_s = index.mul %ht, %c4 : index", "    %ht_s = index.add %c0, %c0 : index")):
            assert t.count(a_) == 1, a_
            t = t.replace(a_, b_)
        t = re.sub(r"%gcmine(\d+) = index.cmp eq, %gcpar\1, %ht : index", r"%gcmine\1 = index.cmp ult, %gcpar\1, %c2 : index", t)
    return t


def combine_loops_unrolled(QH, KL, U):
    """the two split loops of combine_pl1, U splits per iteration (loads in flight together; slots past s_eff are read
    in bounds and masked)"""
    ML = f"view<[%tokens]x{SMAX}x{QH}x2xf32>"
    AV = f"view<[%tokens]x{SMAX}x{KL}x{QH}x{'f16' if PH else 'f32'}>"
    t = [f"""    %cU = index.constant {U} : index
    %cUl = index.constant {U - 1} : index
    %smax_l = index.constant {SMAX - 1} : index
    %g_pad = index.add %s_eff, %cUl : index
    %groups = index.div %g_pad, %cU : index
    %maximum = scf.for %g = [%c0 to %groups step %c1](%mx = %neg_big : f32) -> (f32) {{
      %s0 = index.mul %g, %cU : index"""]
    for i in range(U):
        t.append(f"""      %si{i}_c = index.constant {i} : index
      %si{i} = index.add %s0, %si{i}_c : index
      %vi{i} = index.cmp ult, %si{i}, %s_eff : index
      %bi{i}_m = index.min %si{i}, %smax_l : index
      %bi{i} = index.assume %bi{i}_m [range(%bi{i}_m, 0, {SMAX - 1})] : index
      %mr{i} = view.load %ml_view[%token, %bi{i}, %head, 0] : {ML} -> f32
      %mm{i} = scf.select %vi{i}, %mr{i}, %neg_big : f32""")
    prev = "%mx"
    for i in range(U):
        t.append(f"""      %mx{i} = scalar.maxnumf {prev}, %mm{i} : f32""")
        prev = f"%mx{i}"
    t.append(f"""      scf.yield {prev} : f32
    }}
    %den, %total = scf.for %g = [%c0 to %groups step %c1](%d = %zero_f : f32, %acc = %zero4 : vector<4xf32>) -> (f32, vector<4xf32>) {{
      %s0 = index.mul %g, %cU : index""")
    for i in range(U):
        t.append(f"""      %si{i}_c = index.constant {i} : index
      %si{i} = index.add %s0, %si{i}_c : index
      %vi{i} = index.cmp ult, %si{i}, %s_eff : index
      %bi{i}_m = index.min %si{i}, %smax_l : index
      %bi{i} = index.assume %bi{i}_m [range(%bi{i}_m, 0, {SMAX - 1})] : index
      %ms{i} = view.load %ml_view[%token, %bi{i}, %head, 0] : {ML} -> f32
      %lr{i} = view.load %ml_view[%token, %bi{i}, %head, 1] : {ML} -> f32""" + "".join(f"""
      %pr{i}_{k} = view.load %acc_view[%token, %bi{i}, %dm{k}, %head] : {AV} -> f32""" if not PH else f"""
      %prh{i}_{k} = view.load %acc_view[%token, %bi{i}, %dm{k}, %head] : {AV} -> f16
      %pr{i}_{k} = scalar.extf %prh{i}_{k} : f16 to f32""" for k in range(4)))
    d, acc = "%d", "%acc"
    for i in range(U):
        t.append(f"""      %sh{i} = scalar.subf %ms{i}, %maximum : f32
      %we{i} = scalar.expf<afn> %sh{i} : f32
      %w{i} = scf.select %vi{i}, %we{i}, %zero_f : f32
      %ls{i} = scf.select %vi{i}, %lr{i}, %zero_f : f32
      %wl{i} = scalar.mulf %w{i}, %ls{i} : f32
      %dn{i} = scalar.addf {d}, %wl{i} : f32""" + "".join(f"""
      %pz{i}_{k} = scf.select %vi{i}, %pr{i}_{k}, %zero_f : f32""" for k in range(4)) + f"""
      %part{i} = vector.from_elements %pz{i}_0, %pz{i}_1, %pz{i}_2, %pz{i}_3 : vector<4xf32>
      %wv{i} = vector.splat {"%wl" if PH else "%w"}{i} : vector<4xf32>
      %an{i} = vector.fmaf %wv{i}, %part{i}, {acc} : vector<4xf32>""")
        d, acc = f"%dn{i}", f"%an{i}"
    t.append(f"""      scf.yield {d}, {acc} : f32, vector<4xf32>
    }}
""")
    return "\n".join(t)


def combine_pl1(QH, KL, D):
    """g_mla_dcombine for partial layout 1."""
    return f"""
// o[t, head, 0:512] = sum_s e^(m_s - M) acc_s / sum_s e^(m_s - M) l_s over the token's splits.
// partial layout 1: workgroup (token, 16-dim chunk), lane = (dim quad, head): each partial load is 32 heads x 4 B contiguous.
kernel.def export("g_mla_dcombine") @g_mla_dcombine(%tokens: index) {{
  %c1 = index.constant 1 : index
  %cch = index.constant {KL // 16} : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%tokens, %cch, %c1) workgroup_size(%c128, %c1, %c1) : index
}} launch(%tokens_arg: index, %sliding: i32, %info: buffer, %part_acc: buffer, %part_ml: buffer, %gate: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %ch_id = kernel.workgroup.id<y> : index
  %chunk = index.assume %ch_id [range(%ch_id, 0, {KL // 16 - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 127)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %zero4 = vector.splat %zero_f : vector<4xf32>
  %head_v = index.rem %lane, %c32 : index
  %head = index.assume %head_v [range(%head_v, 0, {QH - 1})] : index
  %dq = index.div %lane, %c32 : index
  %quad_v = index.madd %chunk, %c4, %dq : index
  %quad = index.assume %quad_v [range(%quad_v, 0, {KL // 4 - 1})] : index
  %dim0 = index.mul %quad, %c4 : index
""" + "".join(f"""  %dm{i}_c = index.constant {i} : index
  %dm{i}_v = index.add %dim0, %dm{i}_c : index
  %dm{i} = index.assume %dm{i}_v [range(%dm{i}_v, 0, {KL - 1})] : index
""" for i in range(4)) + f"""  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
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
    %acc_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{SMAX}x{KL}x{QH}x{'f16' if PH else 'f32'}>
""" + (combine_loops_unrolled(QH, KL, CU) if CU > 1 else f"""    %maximum = scf.for %s = [%c0 to %s_eff step %c1](%mx = %neg_big : f32) -> (f32) {{
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
""" + "".join(f"""      %pe{i} = view.load %acc_view[%token, %bs, %dm{i}, %head] : view<[%tokens]x{SMAX}x{KL}x{QH}xf32> -> f32
""" for i in range(4)) + f"""      %part = vector.from_elements %pe0, %pe1, %pe2, %pe3 : vector<4xf32>
      %w_v = vector.splat %w : vector<4xf32>
      %acc_next = vector.fmaf %w_v, %part, %acc : vector<4xf32>
      scf.yield %d_next, %acc_next : f32, vector<4xf32>
    }}
""") + f"""    %one = scalar.constant 1.0 : f32
    %inv = scalar.divf %one, %den : f32
    %inv_v = vector.splat %inv : vector<4xf32>
    %result = vector.mulf %total, %inv_v : vector<4xf32>
    vector.store %result, %out_view[%token, %head, %quad, %c0] : vector<4xf32>, view<[%tokens]x{QH}x{D // 4}x4xf32>
  }} else {{
    vector.store %zero4, %out_view[%token, %head, %quad, %c0] : vector<4xf32>, view<[%tokens]x{QH}x{D // 4}x4xf32>
  }}
  kernel.return
}}
"""


if __name__ == "__main__":
    import json
    import sys
    print(kernels(json.load(open(sys.argv[1]))))
