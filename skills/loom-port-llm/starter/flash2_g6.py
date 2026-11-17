"""One query head per wave, all query heads of a KV head per workgroup: the gfx12 whole-tile flash kernel for head_dim 256
(gen_flash2s.py, FLASH2_B=6; choice prefill.attn.flash2_b2 = "6"; the partial-tile kernel keeps the key-split loop of flash2_b2.py).

The key-split loop (flash2_b2.py) splits a head pair's keys and output dims over four waves, so every 64 keys it exchanges the column maxima,
P^T and the row sums through workgroup memory (two barriers) and every wave redoes the page lookups of the block: ~400 VALU and ~280 SALU
instructions per 64 WMMAs (local ISA), and VALU does not overlap WMMA on RDNA4. Here a wave owns ONE query head of the 16-token tile and
ALL 256 dims and ALL keys (FlashAttention-2 per wave): Q^T as 16 rhs fragments (64 VGPRs, loaded once), O^T as 16 result fragments
(128 VGPRs), S^T of 16 keys (one paged-KV page) from 16 WMMAs, the softmax in-lane plus one xor-16 shuffle, P^T as it is the rhs fragment,
O^T += V^T P^T from 16 transposed V^T fragment loads. No workgroup memory, no barrier. The G waves of a workgroup are the G query heads of
one KV head on the same 16 tokens: they read the same K / V lines at about the same time (the vector caches serve the repeats).
- exp2 domain: S is scaled by scale * log2(e) (one multiply) and P = exp2(S - m) (v_exp_f32 is exp2).
- Lazy rescale: the running max m moves only when a block max exceeds it by more than TAU (log2 units; P <= 2^TAU stays exact in f16
  relative terms and O / l are unchanged in exact arithmetic), and the 128-value O^T rescale runs only when some lane of the wave moved
  (subgroup vote). TAU 0: move on any increase (the classic online softmax with the rescale skipped when nothing moved).
- The row sum l is kept per lane (its 8 keys of each page) and the lane pair is added once at the end (alpha is the same for both lanes).
- Masks: only the diagonal page (the tile's own 16 positions) is causally masked; it is peeled out of the loop. Whole tiles are fully live,
  so every key of the diagonal page is written. No sliding window (asserted by the caller).
"""
import math


def g6_kernel(KVH, G, D, QH, scale, gate, gate_sig, layers, pos_bytes, opts=None):
    """The whole-tile kernel text (gfx11-style `kernel.def export("g_attention_flash2")`; the caller's twin_names makes it the gfx12 twin)."""
    o = dict(chains=1, tau=-1.0, qlds=16, carry=0, qscale=1, fast=1, abl="")
    o.update(opts or {})
    assert D == 256 and G * 32 <= 1024
    NF = D // 16                     # 16 contraction fragments / 16 output fragments
    NQL = int(o["qlds"])             # Q^T fragments held in workgroup memory (the rest in registers)
    NQR = NF - NQL
    R = pos_bytes // 2               # f16 values per KV position
    abl = set(x for x in o["abl"].split(",") if x)
    c2 = scale * math.log2(math.e)
    L = []
    A = L.append
    A(f'''kernel.def export("g_attention_flash2") @g_attention_flash2(%tokens: index) {{
  %c1 = index.constant 1 : index
  %ckv = index.constant {KVH} : index
  %c16 = index.constant 16 : index
  %clanes = index.constant {32 * G} : index
  %tiles = index.div %tokens, %c16 : index
  kernel.launch.config workgroups(%tiles, %ckv, %c1) workgroup_size(%clanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %sliding: i32, %context_arg: index, %pool_arg: index, %q: buffer, %gate: buffer, %info: buffer, %tiles: buffer, %origins: buffer, %arena: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {layers})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2147483584)] : index
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c0_i32 = scalar.constant 0 : i32
  %tile_count = index.div %tokens, %c16 : index
  %tile_id = kernel.workgroup.id<x> : index
  %tile = index.assume %tile_id [lt(%tile_id, %tile_count)] : index
  %kv_id = kernel.workgroup.id<y> : index
  %kv = index.assume %kv_id [range(%kv_id, 0, {KVH - 1})] : index
  %tiles_view = buffer.view %tiles[%origin] : buffer -> view<[%tokens]xi32>
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
  scf.if %listed {{
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {G - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %lane_col = index.rem %lane, %c16 : index
  %lane_half = index.div %lane, %c16 : index
  %c8 = index.constant 8 : index
  %lane_h8_v = index.mul %lane_half, %c8 : index
  %lane_h8 = index.assume %lane_h8_v [range(%lane_h8_v, 0, 8)] : index
  %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
  %row = index.cast %row_bound : i32 to index
  %pos0_bound = scalar.assume %pos0_32 [range(%pos0_32, 0, 262128)] : i32
  %pos0 = index.cast %pos0_bound : i32 to index
  %padded = index.add %context, %c15 : index
  %blocks = index.div %padded, %c16 : index
  %k_plane = index.mul %layer, %c2 : index
  %cg = index.constant {G} : index
  %head_value = index.madd %kv, %cg, %wave : index
  %head = index.assume %head_value [range(%head_value, 0, {QH - 1})] : index
  // pages 0 .. pg_diag - 1 lie wholly below the tile (no mask); page pg_diag holds the tile's own positions
  %pg_diag_v = index.div %pos0, %c16 : index
  %kh_last = index.sub %blocks, %c1 : index
  %pg_diag = index.assume %pg_diag_v [range(%pg_diag_v, 0, %kh_last), lt(%pg_diag_v, %blocks)] : index
  // Q^T of this head as 16 rhs fragments: fragment (kw, j) holds dims 128 * half + 64 kw + 8 j .. + 7 of the lane's token
  %q_view = buffer.view %q[%origin] : buffer -> view<[%tokens]x{QH}x{D}xf32>
  %q_token_value = index.add %first_token, %lane_col : index
  %q_token = index.assume %q_token_value [lt(%q_token_value, %tokens)] : index
  %c128 = index.constant 128 : index
  %qh_base = index.mul %lane_half, %c128 : index''')
    if NQL:   # Q^T fragments NQR.. live in workgroup memory: each wave stages and reads back its own (no barrier)
        A(f'''  %ql_bytes = index.constant {G * NQL * 32 * 16} : offset
  %ql_tile = buffer.alloca<workgroup> align(16) %ql_bytes : buffer
  %ql_view = buffer.view %ql_tile[%origin] : buffer -> view<{G}x{NQL}x32x8xf16>''')
    QS = "s" if o["qscale"] else ""
    # the plane bases as opaque wave-uniform values (a broadcast): otherwise the compiler rebuilds the 64-bit address chain per load
    OPQ = ("\n  %kb_i = index.cast %kb_o0 : offset to i64\n  %vb_i = index.cast %vb_o0 : offset to i64\n  %opq_l0 = scalar.constant 0 : i32\n"
           "  %kb_b = kernel.subgroup.broadcast %kb_i from %opq_l0 : i64, i32\n  %vb_b = kernel.subgroup.broadcast %vb_i from %opq_l0 : i64, i32\n"
           "  %kb_a = scalar.assume %kb_b [range(%kb_b, 0, 9223372036854775807), multiple_of(%kb_b, 512)] : i64\n"
           "  %vb_a = scalar.assume %vb_b [range(%vb_b, 0, 9223372036854775807), multiple_of(%vb_b, 512)] : i64\n"
           "  %kb_o = index.cast %kb_a : i64 to offset\n" + ("  %vb_o = index.cast %vb_a : i64 to offset" if o.get("opq", 1) == 2 else "  %vb_o = index.add %vb_o0, %origin : offset")) if o.get("opq", 1) else \
        "\n  %kb_o = index.add %kb_o0, %origin : offset\n  %vb_o = index.add %vb_o0, %origin : offset"   # Q^T pre-scaled by scale * log2(e) (S^T comes out in log2 units)
    A(f"  %c2s = scalar.constant {c2!r} : f32\n  %c2_v = vector.splat %c2s : vector<8xf32>")
    for f in range(NF):
        kw, j = divmod(f, 8)
        if o["qscale"]:
            A(f"  %qo{f}_s = index.constant {64 * kw + 8 * j} : index\n  %qn{f}_sv = index.add %qh_base, %qo{f}_s : index\n"
              f"  %qn{f}_s = index.assume %qn{f}_sv [range(%qn{f}_sv, 0, {D - 8})] : index\n"
              f"  %qraw{f}_l = vector.load %q_view[%q_token, %head, %qn{f}_s] : view<[%tokens]x{QH}x{D}xf32> -> vector<8xf32>\n"
              f"  %qraw{f}s = vector.mulf %qraw{f}_l, %c2_v : vector<8xf32>")
        A(f'''  %qo{f} = index.constant {64 * kw + 8 * j} : index
  %qn{f}_v = index.add %qh_base, %qo{f} : index
  %qn{f} = index.assume %qn{f}_v [range(%qn{f}_v, 0, {D - 8})] : index
  %qraw{f} = vector.load %q_view[%q_token, %head, %qn{f}] : view<[%tokens]x{QH}x{D}xf32> -> vector<8xf32>
  %qh{f} = vector.fptrunc %qraw{f}{QS} : vector<8xf32> to vector<8xf16>
  %qf{f} = vector.fragment<rhs> %qh{f} shape [%c16, %c16] : vector<8xf16>''' if f < NQR else f'''  %qo{f} = index.constant {64 * kw + 8 * j} : index
  %qn{f}_v = index.add %qh_base, %qo{f} : index
  %qn{f} = index.assume %qn{f}_v [range(%qn{f}_v, 0, {D - 8})] : index
  %qraw{f} = vector.load %q_view[%q_token, %head, %qn{f}] : view<[%tokens]x{QH}x{D}xf32> -> vector<8xf32>
  %qh{f} = vector.fptrunc %qraw{f}{QS} : vector<8xf32> to vector<8xf16>
  %qsl{f} = index.constant {f - NQR} : index
  vector.store %qh{f}, %ql_view[%wave, %qsl{f}, %lane, %c0] : vector<8xf16>, view<{G}x{NQL}x32x8xf16>''')
    A(f'''  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %neg_v = vector.splat %neg_big : vector<8xf32>
  %tau = scalar.constant {float(o["tau"])!r} : f32
  %exchange = scalar.constant 16 : i32
  %width = scalar.constant 32 : i32
  %query_pos = index.add %pos0, %lane_col : index
  %qp32 = index.cast %query_pos : index to i32
  %qp_v = vector.splat %qp32 : vector<8xi32>
  %lane_h8_i32 = index.cast %lane_h8 : index to i32
  %one_row_i32 = scalar.constant 1 : i32
  %row_off = vector.iota %lane_h8_i32 step %one_row_i32 : vector<8xi32>
  // paged KV: K / V of position p of this row live at page map[p / 16], row p % 16, of the layer's two planes
  %pos_bytes = index.constant {pos_bytes} : offset
  %v_plane_delta = index.scale %pool, %pos_bytes : index, offset -> offset
  %vt_c1 = index.constant 1 : index
  %vt_crow = index.constant {R} : index
  %vt_layout = encoding.layout.strided [%vt_c1, %vt_crow] : encoding<layout>
  %k_layout = encoding.layout.strided [%vt_crow, %vt_c1] : encoding<layout>
  %cd = index.constant {D} : index
  %kvd = index.mul %kv, %cd : index
  %c2b = index.constant 2 : offset
  %kv_boff = index.scale %kvd, %c2b : index, offset -> offset
  %kh_lane0 = scalar.constant 0 : i32
  %kh_rec_stride = index.constant 40 : offset
  %kh_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %kh_rec_off = index.scale %kh_row, %kh_rec_stride : index, offset -> offset
  %kh_records = buffer.view %origins[%kh_rec_off] : buffer -> view<5xi64>
  %kh_map64 = view.load %kh_records[4] : view<5xi64> -> i64
  %kh_map_b = scalar.assume %kh_map64 [range(%kh_map64, 0, 9223372036854775807)] : i64
  %kh_map_origin = index.cast %kh_map_b : i64 to offset
  %kh_map = buffer.view %origins[%kh_map_origin] : buffer -> view<[%blocks]xi32>
  %kh_pp = index.mul %k_plane, %pool : index
  %kh_ppo = index.scale %kh_pp, %pos_bytes : index, offset -> offset
  %kb_o0 = index.add %kh_ppo, %kv_boff : offset
  %vb_o0 = index.add %kb_o0, %v_plane_delta : offset{OPQ}
  %pl_rows_m = index.sub %pool, %c16 : index
  %k_pv = buffer.view %arena[%kb_o] : buffer -> view<[%pool]x{D}xf16, %k_layout>
  %kq_h = index.mul %lane_half, %c128 : index''')
    for d in range(NF):
        A(f"  %vcol{d} = index.constant {16 * d} : index")
    for f in range(NF):
        A(f"  %kco{f} = index.constant {8 * f} : index\n  %kcn{f}_v = index.add %kq_h, %kco{f} : index\n"
          f"  %kcn{f} = index.assume %kcn{f}_v [range(%kcn{f}_v, 0, {D - 8})] : index")
    HK = int(o["carry"])   # K fragments carried into the next iteration (loaded during the previous page's PV)

    def lookup(pg, sfx):
        row0 = " " * 4
        return (f"""{row0}%ph{sfx}_r = view.load %kh_map[{pg}] : view<[%blocks]xi32> -> i32
{row0}%ph{sfx}_s = scalar.addi %ph{sfx}_r, %kh_lane0 : i32
{row0}%ph{sfx}_b = scalar.assume %ph{sfx}_s [range(%ph{sfx}_s, 0, 65535)] : i32
{row0}%ph{sfx}_i = index.cast %ph{sfx}_b : i32 to index
{row0}%rw{sfx}_v = index.mul %ph{sfx}_i, %c16 : index
{row0}%rw{sfx} = index.assume %rw{sfx}_v [range(%rw{sfx}_v, 0, 2147483568), lt(%rw{sfx}_v, %pl_rows_m)] : index""" if "kv0" not in abl else
                f"{row0}%rw{sfx} = index.assume %c0 [range(%c0, 0, 2147483568), lt(%c0, %pl_rows_m)] : index")

    kviews = set()

    def kload(rw, f, nm, sfx):
        """K fragment F (dims 128 half + 8 F .. + 7 of key lane_col) of the page at row RW, through a per-page view (uniform base, lane
        and fragment offsets only)."""
        out = ""
        kv_ = f"%kp{sfx}_{rw[1:]}"
        if kv_ not in kviews:
            kviews.add(kv_)
            out = (f"    {kv_}_r = index.scale {rw}, %pos_bytes : index, offset -> offset\n    {kv_}_o = index.add %kb_o, {kv_}_r : offset\n"
                   f"    {kv_} = buffer.view %arena[{kv_}_o] : buffer -> view<16x{D}xf16, %k_layout>\n")
        return out + f"    %{nm} = vector.load {kv_}[%lane_col, %kcn{f}] : view<16x{D}xf16, %k_layout> -> vector<8xf16>"

    def block(rw, kin, masked, pgd, ins_m, ins_l, ins_o, sfx, nxt):
        """One page (16 keys) at row RW whose K fragments 0..HK-1 are KIN: S^T, softmax, O^T += V^T P^T; NXT = (index of the next
        page, suffix) whose row is looked up and whose first K fragments are loaded during this page's PV. Returns (lines, m, l, o, kn, rwn)."""
        B = []
        a = B.append
        ks = list(kin)
        for f in range(HK, NF):
            a(kload(rw, f, f"k{sfx}_{f}", sfx))
            ks.append(f"%k{sfx}_{f}")
        if nxt:
            a(lookup(nxt[0], nxt[1]))
        nch = o["chains"]
        acc = ["%init"] * nch
        vj = f"%vj{sfx}"
        a(f"""    %vo{sfx}_r = index.scale {rw}, %pos_bytes : index, offset -> offset
    %vo{sfx} = index.add %vb_o, %vo{sfx}_r : offset
    {vj} = buffer.view %arena[%vo{sfx}] : buffer -> view<{D}x16xf16, %vt_layout>""")
        vs = []

        def vload(d):
            vrow = "%vcol0" if "novld" in abl else f"%vcol{d}"
            a(f"    %vc{sfx}_{d} = vector.fragment.load<lhs> {vj}[{vrow}, %c0] shape [%c16, %c16] : view<{D}x16xf16, %vt_layout> -> vector<8xf16>")
            vs.append(f"%vc{sfx}_{d}")
        for f in range(NF):
            ch = f % nch
            nm = f"%s{sfx}_{f}"
            if "noqk" in abl:
                a(f"    {nm} = vector.addf {acc[ch] if acc[ch] != '%init' else '%zero8'}, %zero8 : vector<8xf32>")
            else:
                if f >= NQR:
                    a(f"    %qx{sfx}_{f} = vector.load %ql_view[%wave, %qsl{f}, %lane, %c0] : view<{G}x{NQL}x32x8xf16> -> vector<8xf16>\n"
                      f"    %qf{sfx}_{f} = vector.fragment<rhs> %qx{sfx}_{f} shape [%c16, %c16] : vector<8xf16>")
                qn = f"%qf{f}" if f < NQR else f"%qf{sfx}_{f}"
                a(f"    %kf{sfx}_{f} = vector.fragment<lhs> {ks[f]} shape [%c16, %c16] : vector<8xf16>\n"
                  f"    {nm} = vector.mma %kf{sfx}_{f}, {qn}, {acc[ch]} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
            acc[ch] = nm
            if f == NF // 2 - 1:     # the V^T fragments of the first half while the second half of S^T runs
                for d in range(NF // 2):
                    vload(d)
        for d in range(NF // 2, NF):
            vload(d)
        s_ = acc[0]
        for ch in range(1, nch):
            a(f"    %sj{sfx}_{ch} = vector.addf {s_}, {acc[ch]} : vector<8xf32>")
            s_ = f"%sj{sfx}_{ch}"
        a(f"    %t{sfx} = vector.mulf {s_}, %c2_v : vector<8xf32>" if not o["qscale"] else f"    %t{sfx} = vector.addf {s_}, %zero8 : vector<8xf32>")
        t = f"%t{sfx}"
        if masked:
            a(f"""    %pgk{sfx} = index.mul {pgd}, %c16 : index
    %pgk{sfx}_i = index.cast %pgk{sfx} : index to i32
    %kb{sfx}_v = vector.splat %pgk{sfx}_i : vector<8xi32>
    %kpos{sfx} = vector.addi %kb{sfx}_v, %row_off : vector<8xi32>
    %valid{sfx} = vector.cmpi sle, %kpos{sfx}, %qp_v : vector<8xi32> -> vector<8xi1>
    %tm{sfx} = vector.select %valid{sfx}, %t{sfx}, %neg_v : vector<8xf32>""")
            t = f"%tm{sfx}"
        a(f"""    %lmax{sfx} = vector.reduce<maxnumf> {t}, %neg_big : vector<8xf32>, f32
    %omax{sfx}, %omax{sfx}_u = kernel.subgroup.shuffle<xor> %lmax{sfx}, %exchange, %width : f32, i32, i32
    %bm{sfx} = scalar.maxnumf %lmax{sfx}, %omax{sfx} : f32""")
        ty1 = ", ".join(["f32"] + ["vector<8xf32>"] * NF)
        if o["tau"] < 0:   # always rescale
            a(f"""    %mn{sfx} = scalar.maxnumf {ins_m}, %bm{sfx} : f32
    %msh{sfx} = scalar.subf {ins_m}, %mn{sfx} : f32
    %al{sfx} = scalar.exp2f<afn> %msh{sfx} : f32
    %al{sfx}_v = vector.splat %al{sfx} : vector<8xf32>""")
            outs = []
            for d in range(NF):
                if "noresc" in abl:
                    outs.append(ins_o[d])
                    continue
                a(f"    %os{sfx}_{d} = vector.mulf {ins_o[d]}, %al{sfx}_v : vector<8xf32>")
                outs.append(f"%os{sfx}_{d}")
            a(f"    %ls{sfx} = scalar.mulf {ins_l}, %al{sfx} : f32")
        else:
            a(f"""    %mt{sfx} = scalar.addf {ins_m}, %tau : f32
    %mv{sfx} = scalar.cmpf ogt, %bm{sfx}, %mt{sfx} : f32
    %mx{sfx} = scalar.maxnumf {ins_m}, %bm{sfx} : f32
    %mn{sfx} = scf.select %mv{sfx}, %mx{sfx}, {ins_m} : f32
    %any{sfx} = kernel.subgroup.vote.any %mv{sfx} : i1""")
            outs = [f"%os{sfx}_{d}" for d in range(NF)]
            a(f"    %ls{sfx}, {', '.join(outs)} = scf.if %any{sfx} -> ({ty1}) {{")
            a(f"""      %msh{sfx} = scalar.subf {ins_m}, %mn{sfx} : f32
      %al{sfx} = scalar.exp2f<afn> %msh{sfx} : f32
      %al{sfx}_v = vector.splat %al{sfx} : vector<8xf32>
      %lsi{sfx} = scalar.mulf {ins_l}, %al{sfx} : f32""")
            for d in range(NF):
                a(f"      %osi{sfx}_{d} = vector.mulf {ins_o[d]}, %al{sfx}_v : vector<8xf32>")
            a(f"      scf.yield %lsi{sfx}, {', '.join(f'%osi{sfx}_{d}' for d in range(NF))} : {ty1}\n    }} else {{\n"
              f"      scf.yield {ins_l}, {', '.join(ins_o)} : {ty1}\n    }}")
        a(f"""    %mn{sfx}_v = vector.splat %mn{sfx} : vector<8xf32>
    %sh{sfx} = vector.subf {t}, %mn{sfx}_v : vector<8xf32>""")
        a(f"    %p{sfx} = vector.addf %sh{sfx}, %zero8 : vector<8xf32>" if "noexp" in abl else f"    %p{sfx} = vector.exp2f<afn> %sh{sfx} : vector<8xf32>")
        a(f"""    %psum{sfx} = vector.reduce<addf> %p{sfx}, %zero_f : vector<8xf32>, f32
    %ln{sfx} = scalar.addf %ls{sfx}, %psum{sfx} : f32
    %ph16{sfx} = vector.fptrunc %p{sfx} : vector<8xf32> to vector<8xf16>
    %pf{sfx} = vector.fragment<rhs> %ph16{sfx} shape [%c16, %c16] : vector<8xf16>""")
        res, kn = [], []
        for d in range(NF):
            if "nopv" in abl:
                a(f"    %on{sfx}_{d} = vector.addf {outs[d]}, %zero8 : vector<8xf32>")
            else:
                a(f"    %on{sfx}_{d} = vector.mma {vs[d]}, %pf{sfx}, {outs[d]} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
            res.append(f"%on{sfx}_{d}")
            if nxt and d < HK:       # the next page's first K fragments, into the registers this page's V^T fragments free
                a(kload(f"%rw{nxt[1]}", d, f"kn{sfx}_{d}", sfx))
                kn.append(f"%kn{sfx}_{d}")
        return B, f"%mn{sfx}", f"%ln{sfx}", res, kn, (f"%rw{nxt[1]}" if nxt else None)

    def qk(rw, sfx, init, vfirst=None):
        """S^T of the page at row RW from 16 K fragments loaded here (in program order: the loads first), accumulated onto INIT;
        VFIRST(d) is called after the first half (the V^T loads of the page). Returns (lines, S name)."""
        B = []
        a = B.append
        ks = []
        for f in range(NF):
            a(kload(rw, f, f"k{sfx}_{f}", sfx))
            ks.append(f"%k{sfx}_{f}")
        acc = init
        for f in range(NF):
            nm = f"%s{sfx}_{f}"
            if "noqk" in abl:
                a(f"    {nm} = vector.addf {acc if acc != '%init' else '%zero8'}, %zero8 : vector<8xf32>")
            else:
                if f >= NQR:
                    a(f"    %qx{sfx}_{f} = vector.load %ql_view[%wave, %qsl{f}, %lane, %c0] : view<{G}x{NQL}x32x8xf16> -> vector<8xf16>\n"
                      f"    %qf{sfx}_{f} = vector.fragment<rhs> %qx{sfx}_{f} shape [%c16, %c16] : vector<8xf16>")
                qn = f"%qf{f}" if f < NQR else f"%qf{sfx}_{f}"
                a(f"    %kf{sfx}_{f} = vector.fragment<lhs> {ks[f]} shape [%c16, %c16] : vector<8xf16>\n"
                  f"    {nm} = vector.mma %kf{sfx}_{f}, {qn}, {acc} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
            acc = nm
            if f == NF // 2 - 1 and vfirst:
                B.extend(vfirst())
        return B, acc

    def vview(rw, sfx):
        return (f"    %vo{sfx}_r = index.scale {rw}, %pos_bytes : index, offset -> offset\n    %vo{sfx} = index.add %vb_o, %vo{sfx}_r : offset\n"
                f"    %vj{sfx} = buffer.view %arena[%vo{sfx}] : buffer -> view<{D}x16xf16, %vt_layout>")

    def vloads(sfx, ds):
        return [f"    %vc{sfx}_{d} = vector.fragment.load<lhs> %vj{sfx}[{'%vcol0' if 'novld' in abl else f'%vcol{d}'}, %c0] shape [%c16, %c16] : view<{D}x16xf16, %vt_layout> -> vector<8xf16>" for d in ds]

    def pv(sfx, pf, ins, B):
        res = []
        for d in range(NF):
            if "nopv" in abl:
                B.append(f"    %on{sfx}_{d} = vector.addf {ins[d]}, %zero8 : vector<8xf32>")
            else:
                B.append(f"    %on{sfx}_{d} = vector.mma %vc{sfx}_{d}, {pf}, {ins[d]} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
            res.append(f"%on{sfx}_{d}")
        return res

    def ind(lines):   # loop-body lines (4 spaces) at kernel-body depth (2 spaces)
        return [x.replace("\n    ", "\n  ")[2:] if x.startswith("    ") else x for x in lines]

    if not o["fast"]:   # the classic online softmax: the slow path alone
        slow_lines, mres, lres, ores = None, None, None, None
    # ---- the slow path: classic online softmax over every page (always rescales); also the fallback of the fast path
    def slow(pfx):
        S_ = []
        ty = ", ".join(["f32", "f32"] + ["vector<8xf32>"] * NF)
        inits = ", ".join([f"%{pfx}m = %neg_big : f32", f"%{pfx}l = %zero_f : f32"] + [f"%{pfx}a{d} = %init : vector<8xf32>" for d in range(NF)])
        S_.append(f"  %{pfx}mh, %{pfx}lh, {', '.join(f'%{pfx}h{d}' for d in range(NF))} = scf.for %{pfx}pg = [%c0 to %pg_diag step %c1]({inits}) -> ({ty}) {{")
        S_.append(f"    %{pfx}pga = index.assume %{pfx}pg [range(%{pfx}pg, 0, %kh_last), lt(%{pfx}pg, %blocks)] : index")
        S_.append(lookup(f"%{pfx}pga", f"{pfx}x"))
        B, m_, l_, o_, _, _ = block(f"%rw{pfx}x", [], False, None, f"%{pfx}m", f"%{pfx}l", [f"%{pfx}a{d}" for d in range(NF)], f"{pfx}", None)
        S_.extend(B)
        S_.append(f"    scf.yield {m_}, {l_}, {', '.join(o_)} : {ty}\n  }}")
        S_.append(lookup("%pg_diag", f"{pfx}dd").replace("\n    ", "\n  ")[2:])
        B, m_, l_, o_, _, _ = block(f"%rw{pfx}dd", [], True, "%pg_diag", f"%{pfx}mh", f"%{pfx}lh", [f"%{pfx}h{d}" for d in range(NF)], f"{pfx}d", None)
        S_.extend(ind(B))
        return S_, l_, o_

    if not o["fast"]:
        B, lres, ores = slow("w")
        L.extend(B)
    else:
        # ---- fast path. m is fixed for the whole tile: the max of page 0 and of the diagonal page (computed first, exactly). The other
        # pages run with S^T accumulated onto -m (the MMA init: S^T - m for free), P = exp2(.), no rescale and no running max; the loop
        # only tracks the largest exponent. P is f16: an exponent above 15 would overflow, so a tile that saw one (a key more than
        # 2^15 times heavier than every key of page 0 and of its own page) recomputes with the slow path. Exact either way.
        A("  %rw0 = index.assume %c0 [range(%c0, 0, 2147483568), lt(%c0, %pl_rows_m)] : index" if "kv0" in abl else lookup("%c0", "z").replace("\n    ", "\n  ")[2:])
        z = "%rw0" if "kv0" in abl else "%rwz"
        A(lookup("%pg_diag", "q").replace("\n    ", "\n  ")[2:])
        A(vview(z, "z").replace("\n    ", "\n  ")[2:]); A(vview("%rwq", "q").replace("\n    ", "\n  ")[2:])
        B, s0 = qk(z, "z", "%init"); L.extend(ind(B))
        B, sd = qk("%rwq", "q", "%init"); L.extend(ind(B))
        L.extend(ind(vloads("z", range(NF)) + vloads("q", range(NF))))
        A(f"""  %t0 = {"vector.addf " + s0 + ", %zero8" if o["qscale"] else "vector.mulf " + s0 + ", %c2_v"} : vector<8xf32>
  %td = {"vector.addf " + sd + ", %zero8" if o["qscale"] else "vector.mulf " + sd + ", %c2_v"} : vector<8xf32>
  %kpz_v = vector.splat %c0_i32 : vector<8xi32>
  %kpz = vector.addi %kpz_v, %row_off : vector<8xi32>
  %vz = vector.cmpi sle, %kpz, %qp_v : vector<8xi32> -> vector<8xi1>
  %t0m = vector.select %vz, %t0, %neg_v : vector<8xf32>
  %pgdk = index.mul %pg_diag, %c16 : index
  %pgdk_i = index.cast %pgdk : index to i32
  %kpd_v = vector.splat %pgdk_i : vector<8xi32>
  %kpd = vector.addi %kpd_v, %row_off : vector<8xi32>
  %vd0 = vector.cmpi sle, %kpd, %qp_v : vector<8xi32> -> vector<8xi1>
  %dz = index.cmp ne, %pg_diag, %c0 : index
  %dz_v = vector.splat %dz : vector<8xi1>
  %vd = vector.andi %vd0, %dz_v : vector<8xi1>
  %tdm = vector.select %vd, %td, %neg_v : vector<8xf32>
  %mx0 = vector.reduce<maxnumf> %t0m, %neg_big : vector<8xf32>, f32
  %mxd = vector.reduce<maxnumf> %tdm, %mx0 : vector<8xf32>, f32
  %mxo, %mxo_u = kernel.subgroup.shuffle<xor> %mxd, %exchange, %width : f32, i32, i32
  %mfix = scalar.maxnumf %mxd, %mxo : f32
  %mfix_v = vector.splat %mfix : vector<8xf32>
  %negm = scalar.subf %zero_f, %mfix : f32
  %negm_v = vector.splat %negm : vector<8xf32>
  %initm = vector.fragment<init> %negm_v shape [%c16, %c16] : vector<8xf32>
  %sh0 = vector.subf %t0m, %mfix_v : vector<8xf32>
  %p0 = vector.exp2f<afn> %sh0 : vector<8xf32>
  %shd = vector.subf %tdm, %mfix_v : vector<8xf32>
  %pd = vector.exp2f<afn> %shd : vector<8xf32>
  %ls0 = vector.reduce<addf> %p0, %zero_f : vector<8xf32>, f32
  %lsd = vector.reduce<addf> %pd, %ls0 : vector<8xf32>, f32
  %p0h = vector.fptrunc %p0 : vector<8xf32> to vector<8xf16>
  %p0f = vector.fragment<rhs> %p0h shape [%c16, %c16] : vector<8xf16>
  %pdh = vector.fptrunc %pd : vector<8xf32> to vector<8xf16>
  %pdf = vector.fragment<rhs> %pdh shape [%c16, %c16] : vector<8xf16>""")
        B = []
        o0 = pv("z", "%p0f", ["%init"] * NF, B)
        o1 = pv("q", "%pdf", o0, B)
        L.extend(ind(B))
        # hot loop: pages 1 .. pg_diag - 1; the next page's row is looked up one page ahead
        ty = ", ".join(["f32", "vector<8xf32>"] + ["vector<8xf32>"] * NF + ["index"])
        A("  %pg1_v = index.min %c1, %pg_diag : index\n  %pg1 = index.assume %pg1_v [range(%pg1_v, 0, %kh_last), lt(%pg1_v, %blocks)] : index")
        A(lookup("%pg1", "f1").replace("\n    ", "\n  ")[2:] if "kv0" not in abl else "  %rwf1 = index.assume %c0 [range(%c0, 0, 2147483568), lt(%c0, %pl_rows_m)] : index")
        inits = ", ".join(["%fl = %lsd : f32", "%ftm = %neg_v : vector<8xf32>"] + [f"%fa{d} = {o1[d]} : vector<8xf32>" for d in range(NF)] + ["%frw = %rwf1 : index"])
        A(f"  %fl_e, %ftm_e, {', '.join(f'%fo{d}' for d in range(NF))}, %frw_e = scf.for %fpg = [%c1 to %pg_diag step %c1]({inits}) -> ({ty}) {{")
        A(f"""    %fpgn_v = index.add %fpg, %c1 : index
    %fpgn_m = index.min %fpgn_v, %pg_diag : index
    %fpgn = index.assume %fpgn_m [range(%fpgn_m, 0, %kh_last), lt(%fpgn_m, %blocks)] : index
    %frwi = index.assume %frw [range(%frw, 0, 2147483568), lt(%frw, %pl_rows_m)] : index""")
        A(lookup("%fpgn", "fn"))
        A(vview("%frwi", "f"))
        B, sf = qk("%frwi", "f", "%initm", lambda: vloads("f", range(NF // 2)))
        L.extend(B)
        L.extend(vloads("f", range(NF // 2, NF)))
        A(f"""    %tf = {"vector.addf " + sf + ", %zero8" if o["qscale"] else "vector.mulf " + sf + ", %c2_v"} : vector<8xf32>
    %ftm_n = vector.maxnumf %ftm, %tf : vector<8xf32>
    %pff = vector.exp2f<afn> %tf : vector<8xf32>
    %fls = vector.reduce<addf> %pff, %fl : vector<8xf32>, f32
    %pfh = vector.fptrunc %pff : vector<8xf32> to vector<8xf16>
    %pfx = vector.fragment<rhs> %pfh shape [%c16, %c16] : vector<8xf16>""")
        B = []
        of = pv("f", "%pfx", [f"%fa{d}" for d in range(NF)], B)
        L.extend(B)
        A(f"    scf.yield %fls, %ftm_n, {', '.join(of)}, {'%rwfn' if 'kv0' not in abl else '%frwi'} : {ty}\n  }}")
        # overflow check, slow-path fallback
        A(f"""  %ftm_r = vector.reduce<maxnumf> %ftm_e, %neg_big : vector<8xf32>, f32
  %ovf_lim = scalar.constant 15.0 : f32
  %ovf = scalar.cmpf ogt, %ftm_r, %ovf_lim : f32
  %ovf_any = kernel.subgroup.vote.any %ovf : i1""")
        ty1 = ", ".join(["f32"] + ["vector<8xf32>"] * NF)
        A(f"  %rl, {', '.join(f'%ro{d}' for d in range(NF))} = scf.if %ovf_any -> ({ty1}) {{")
        B, l_, o_ = slow("w")
        L.extend("  " + x.replace("\n", "\n  ") for x in B)
        A(f"    scf.yield {l_}, {', '.join(o_)} : {ty1}\n  }} else {{\n    scf.yield %fl_e, {', '.join(f'%fo{d}' for d in range(NF))} : {ty1}\n  }}")
        lres, ores = "%rl", [f"%ro{d}" for d in range(NF)]
    # epilogue: l of the lane pair, O^T / l, gate, store through the transposed strided view (dim row, token col) -> out[token, head D + dim]
    A(f"""  %lo, %lo_u = kernel.subgroup.shuffle<xor> {lres}, %exchange, %width : f32, i32, i32
  %l_final = scalar.addf {lres}, %lo : f32
  %one_f = scalar.constant 1.0 : f32
  %inv_l = scalar.divf %one_f, %l_final : f32
  %inv_v = vector.splat %inv_l : vector<8xf32>
  %cqd = index.constant {QH * D} : index
  %out_layout = encoding.layout.strided [%c1, %cqd] : encoding<layout>
  %out_first_a = index.mul %first_token, %cqd : index
  %head_col = index.mul %head, %cd : index
  %out_first = index.add %out_first_a, %head_col : index
  %four = index.constant 4 : offset
  %out_offset = index.scale %out_first, %four : index, offset -> offset
  %out_view = buffer.view %out[%out_offset] : buffer -> view<{D}x16xf32, %out_layout>""" + (f"""
  %gate_view = buffer.view %gate[%out_offset] : buffer -> view<{D}x16xf32, %out_layout>""" if gate else ""))
    for d in range(NF):
        A(f"  %ou{d} = vector.mulf {ores[d]}, %inv_v : vector<8xf32>")
        if gate:
            act = f"vector.logisticf<afn> %gf{d} : vector<8xf32>" if gate_sig else f"vector.addf %gf{d}, %zero8 : vector<8xf32>"
            A(f"  %gf{d} = vector.fragment.load<result> %gate_view[%vcol{d}, %c0] shape [%c16, %c16] : view<{D}x16xf32, %out_layout> -> vector<8xf32>\n"
              f"  %gs{d} = {act}\n  %of{d} = vector.mulf %ou{d}, %gs{d} : vector<8xf32>")
        else:
            A(f"  %of{d} = vector.addf %ou{d}, %zero8 : vector<8xf32>")
        A(f"  vector.fragment.store<result> %of{d}, %out_view[%vcol{d}, %c0] shape [%c16, %c16] : vector<8xf32>, view<{D}x16xf32, %out_layout>")
    A("  }\n  kernel.return\n}\n")
    return "\n".join(L)
