"""FLASH2_B=2: the key-split loop of flash2_b.py with TWO query heads per wave (gfx12 head_dim 512 twin, opt-in).

A wave owns 16 query columns of each of two heads that share the KV head: the K fragments of its key chunk and the V^T fragments of its
dims are loaded ONCE and feed both heads' WMMAs (half the global fragment loads and address arithmetic per WMMA, and half the K/V
bytes requested per query row). A workgroup is the four dim/chunk waves of a head PAIR (128 lanes); the softmax, P^T exchange and
barriers are those of flash2_b.py, per head. The output epilogue of the first head is repeated for the second.
"""
import re


def rewrite_b2(text, kvh, d, partial):
    assert d == 512
    import os
    JU = " unroll" if os.environ.get("FLASH2_B2_UNROLL") == "1" else ""
    VOFF = False
    waves = 4
    R = kvh * d
    NF = d // 16
    QP = NF * 8 + 8
    pos_bytes = int(re.search(r"%pos_bytes = index\.constant (\d+) : offset", text).group(1))
    marker = "  // Partial tile: O^T / l" if partial else "  // O^T / l, gate, store through"
    text = re.sub(r"^ *%k_bytes = .*\n *%k_tile = .*\n *%k_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%vt_bytes = .*\n *%vt_tile = .*\n *%vt_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%s_bytes = .*\n *%s_tile = .*\n *%s_view = .*\n", "", text, flags=re.M)
    h0 = text.index("  %m_final, %l_final,")
    h1 = text.index(marker)
    epi = text[h1:]
    body_end = epi.index("  }\n  kernel.return")        # the epilogue ends at the scf.if's closing brace
    epilogue, tail = epi[:body_end], epi[body_end:]
    # --- second head's epilogue: every name the epilogue defines gets a B suffix; the loop results and the head index are renamed
    defined = set(re.findall(r"^\s*(%\w+)(?:, %\w+)* =", epilogue, flags=re.M))
    defined |= set(m for line in re.findall(r"^\s*((?:%\w+, )+%\w+) =", epilogue, flags=re.M) for m in line.split(", "))
    epi_b = epilogue
    for nm in sorted(defined, key=len, reverse=True):
        epi_b = re.sub(re.escape(nm) + r"\b(?!\w)", nm + "B", epi_b)
    epi_b = re.sub(r"%o(\d+)B?\b(?!\w)", r"%oB\1", epi_b) if False else epi_b
    for k in range(8):
        epi_b = re.sub(rf"%o{k}\b(?!\w)", f"%oB{k}", epi_b)
    epi_b = re.sub(r"%l_final\b(?!\w)", "%lB_final", epi_b)
    epi_b = re.sub(r"(?<![\w%])%head\b(?!\w)", "%head1", epi_b)
    epi_b = re.sub(r"%head\b(?!\w)", "%head1", epi_b)
    # Q^T staging of both heads
    qst = ("  %fb_c8 = index.constant 8 : index\n  %fb_c64 = index.constant 64 : index\n  %fb_c256 = index.constant 256 : index\n"
           "  %fb_dh8 = index.mul %dim_half, %fb_c8 : index\n  %fb_dh64 = index.mul %dim_half, %fb_c64 : index\n"
           "  %fb_h256 = index.mul %lane_half, %fb_c256 : index\n  %fb_qb = index.add %fb_h256, %fb_dh64 : index\n"
           "  %head1_v = index.add %head, %c1 : index\n  %head1 = index.assume %head1_v [range(%head1_v, 1, " + str(kvh * 8 - 1) + ")] : index\n")
    for hh, hn in ((0, "%head"), (1, "%head1")):
        for c in range(8):
            qst += (f"  %fqs{hh}_{c}_v = index.add %fb_dh8, %e{c} : index\n  %fqs{hh}_{c} = index.assume %fqs{hh}_{c}_v [range(%fqs{hh}_{c}_v, 0, {NF - 1})] : index\n"
                    f"  %fqe{hh}_{c}_v = index.mul %fqs{hh}_{c}, %fb_c8 : index\n  %fqe{hh}_{c} = index.assume %fqe{hh}_{c}_v [range(%fqe{hh}_{c}_v, 0, {QP - 8})] : index\n"
                    f"  %fqn{hh}_{c}_o = index.constant {8 * c} : index\n  %fqn{hh}_{c}_v = index.add %fb_qb, %fqn{hh}_{c}_o : index\n"
                    f"  %fqn{hh}_{c} = index.assume %fqn{hh}_{c}_v [range(%fqn{hh}_{c}_v, 0, {d - 8})] : index\n"
                    f"  %fqraw{hh}_{c} = vector.load %q_view[%q_token, {hn}, %fqn{hh}_{c}] : view<[%tokens]x{kvh * 8}x{d}xf32> -> vector<8xf32>\n"
                    f"  %fqh{hh}_{c} = vector.fptrunc %fqraw{hh}_{c} : vector<8xf32> to vector<8xf16>\n"
                    f"  vector.store %fqh{hh}_{c}, %ql_view[%c{hh}_i, %lane, %fqe{hh}_{c}] : vector<8xf16>, view<2x32x{QP}xf16>\n")
    qst += "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    alloc = (f"  %c0_i = index.constant 0 : index\n  %c1_i = index.constant 1 : index\n"
             f"  %ql_bytes = index.constant {2 * 32 * QP * 2} : offset\n  %ql_tile = buffer.alloca<workgroup> align(16) %ql_bytes : buffer\n"
             f"  %ql_view = buffer.view %ql_tile[%origin] : buffer -> view<2x32x{QP}xf16>\n"
             f"  %pl_bytes = index.constant {2 * waves * 32 * 16} : offset\n  %pl_tile = buffer.alloca<workgroup> align(16) %pl_bytes : buffer\n"
             f"  %pl_view = buffer.view %pl_tile[%origin] : buffer -> view<1x2x{waves}x32x8xf16>\n"
             f"  %mx_bytes = index.constant {2 * waves * 32 * 4} : offset\n  %mx_tile = buffer.alloca<workgroup> align(16) %mx_bytes : buffer\n"
             f"  %mx_view = buffer.view %mx_tile[%origin] : buffer -> view<1x2x{waves}x32xf32>\n"
             f"  %ls_tile = buffer.alloca<workgroup> align(16) %mx_bytes : buffer\n"
             f"  %ls_view = buffer.view %ls_tile[%origin] : buffer -> view<1x2x{waves}x32xf32>\n")
    defs = f"""  %vt_g_c1 = index.constant 1 : index
  %vt_g_crow = index.constant {R} : index
  %vt_g_layout = encoding.layout.strided [%vt_g_c1, %vt_g_crow] : encoding<layout>
  %k_g_layout = encoding.layout.strided [%vt_g_crow, %vt_g_c1] : encoding<layout>
  %vd_cd = index.constant {d} : index
  %vd_kvd = index.mul %kv, %vd_cd : index
  %vd_dk = index.add %vd_kvd, %dim_base : index
  %vd_c2b = index.constant 2 : offset
  %kv_boff = index.scale %vd_kvd, %vd_c2b : index, offset -> offset
  %vd_boff = index.scale %vd_dk, %vd_c2b : index, offset -> offset
  %kh_lane0 = scalar.constant 0 : i32
  %kh_rec_stride = index.constant 40 : offset
  %kh_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %kh_rec_off = index.scale %kh_row, %kh_rec_stride : index, offset -> offset
  %kh_records = buffer.view %origins[%kh_rec_off] : buffer -> view<5xi64>
  %kh_map64 = view.load %kh_records[4] : view<5xi64> -> i64
  %kh_map_b = scalar.assume %kh_map64 [range(%kh_map64, 0, 9223372036854775807)] : i64
  %kh_map_origin = index.cast %kh_map_b : i64 to offset
  %kh_map = buffer.view %origins[%kh_map_origin] : buffer -> view<[%blocks]xi32>
  %kh_last = index.sub %blocks, %c1 : index
  %kh_pp = index.mul %k_plane, %pool : index
  %kh_ppo = index.scale %kh_pp, %pos_bytes : index, offset -> offset
  %kb_o = index.add %kh_ppo, %kv_boff : offset
  %vb_o0 = index.add %kh_ppo, %vd_boff : offset
  %vb_o = index.add %vb_o0, %v_plane_delta : offset
  %pl_rows_m = index.sub %pool, %c16 : index
  %k_pv = buffer.view %arena[%kb_o] : buffer -> view<[%pool]x{d}xf16, %k_g_layout>
  %pg_live_v = index.div %pos0, %c16 : index
  %pg_live = index.assume %pg_live_v [range(%pg_live_v, 0, %kh_last), lt(%pg_live_v, %blocks)] : index
  %c64 = index.constant 64 : index
  %c63 = index.constant 63 : index
  %zv8 = vector.constant 0.0 : vector<8xf16>
  %nblk_round = index.add %key_count, %c63 : index
  %nblk = index.div %nblk_round, %c64 : index
  %dh16 = index.mul %dim_half, %c16 : index
  %dh16_i32 = index.cast %dh16 : index to i32
  %fb_is0 = index.cmp eq, %dim_half, %c0 : index
  %fb_is1 = index.cmp eq, %dim_half, %c1 : index
  %fb_is2 = index.cmp eq, %dim_half, %c2 : index
  %fb_wv = index.assume %wave [range(%wave, 0, 3)] : index
"""
    for f in range(NF):
        defs += f"  %fq{f} = index.constant {f} : index\n  %kcol{f} = index.constant {16 * f} : index\n"
    if partial:
        pass
    ty8 = ", ".join(["vector<8xf32>"] * 8)
    acc = ", ".join([f"%a{k} = %init : vector<8xf32>" for k in range(8)] + [f"%b{k} = %init : vector<8xf32>" for k in range(8)])
    outs = ", ".join([f"%o{k}" for k in range(8)] + [f"%oB{k}" for k in range(8)])
    ty16 = ", ".join(["vector<8xf32>"] * 16)
    L = []
    A = L.append
    A(f"  %m_final, %l_final, %mB_final, %lB_final, {outs} = scf.for %bt = [%c0 to %nblk step %c1](%m = %neg_big : f32, %l = %zero_f : f32, %mB = %neg_big : f32, %lB = %zero_f : f32, {acc}) -> (f32, f32, f32, f32, {ty16}) {{")
    A("    %bt64 = index.mul %bt, %c64 : index\n    %blk_first = index.add %bt64, %key_low : index\n    %blk_pg0 = index.div %blk_first, %c16 : index\n"
      "    %par_i = index.rem %bt, %c2 : index\n    %par = index.assume %par_i [range(%par_i, 0, 1)] : index")
    for j in range(4):
        A(f"""    %pg{j}_a = index.add %blk_pg0, %fq{j} : index
    %pg{j}_c = index.min %pg{j}_a, %pg_live : index
    %pg{j} = index.assume %pg{j}_c [range(%pg{j}_c, 0, %kh_last), lt(%pg{j}_c, %blocks)] : index
    %ph{j}_r = view.load %kh_map[%pg{j}] : view<[%blocks]xi32> -> i32
    %ph{j}_s = scalar.addi %ph{j}_r, %kh_lane0 : i32
    %ph{j}_b = scalar.assume %ph{j}_s [range(%ph{j}_s, 0, 65535)] : i32
    %ph{j}_i = index.cast %ph{j}_b : i32 to index
    %rw{j}_v = index.mul %ph{j}_i, %c16 : index
    %rw{j} = index.assume %rw{j}_v [range(%rw{j}_v, 0, 1048560), lt(%rw{j}_v, %pl_rows_m)] : index""")
    A("    %rk_a = scf.select %fb_is2, %rw2, %rw3 : index\n    %rk_b = scf.select %fb_is1, %rw1, %rk_a : index\n    %rk_c = scf.select %fb_is0, %rw0, %rk_b : index\n    %rk = index.assume %rk_c [range(%rk_c, 0, 1048560), lt(%rk_c, %pl_rows_m)] : index")
    A("    %kw_lr_v = index.add %rk, %lane_col : index\n    %kw_lr = index.assume %kw_lr_v [range(%kw_lr_v, 0, 1048575), lt(%kw_lr_v, %pool)] : index\n"
      "    %kw_c256 = index.constant 256 : index\n    %kw_h = index.mul %lane_half, %kw_c256 : index\n    %c64k = index.constant 64 : index\n    %fq8c = index.constant 8 : index")
    A(f"    %sA_f, %sB_f = scf.for %kw = [%c0 to %c4 step %c1](%sA = %init : vector<8xf32>, %sB = %init : vector<8xf32>) -> (vector<8xf32>, vector<8xf32>) {{")
    A(f"""      %kw_a = index.assume %kw [range(%kw, 0, 3)] : index
      %kw_o = index.mul %kw_a, %c64k : index
      %kw_c = index.add %kw_h, %kw_o : index
      %kw_ci = index.assume %kw_c [range(%kw_c, 0, {d - 64})] : index
      %kwv = vector.load %k_pv[%kw_lr, %kw_ci] : view<[%pool]x{d}xf16, %k_g_layout> -> vector<64xf16>
      %kw_f0 = index.mul %kw_a, %fq8c : index
      %kqo_v = index.mul %kw_f0, %fq8c : index
      %kqo = index.assume %kqo_v [range(%kqo_v, 0, {QP - 64})] : index
      %qrwA = vector.load %ql_view[%c0_i, %lane, %kqo] : view<2x32x{QP}xf16> -> vector<64xf16>
      %qrwB = vector.load %ql_view[%c1_i, %lane, %kqo] : view<2x32x{QP}xf16> -> vector<64xf16>""")
    pa, pb = "%sA", "%sB"
    for sl in range(8):
        A(f"""      %kfs{sl} = vector.slice %kwv[{8 * sl}] : vector<64xf16> -> vector<8xf16>
      %kf{sl} = vector.fragment<lhs> %kfs{sl} shape [%c16, %c16] : vector<8xf16>
      %qrA{sl} = vector.slice %qrwA[{8 * sl}] : vector<64xf16> -> vector<8xf16>
      %qrB{sl} = vector.slice %qrwB[{8 * sl}] : vector<64xf16> -> vector<8xf16>
      %qbA{sl} = vector.fragment<rhs> %qrA{sl} shape [%c16, %c16] : vector<8xf16>
      %qbB{sl} = vector.fragment<rhs> %qrB{sl} shape [%c16, %c16] : vector<8xf16>
      %sqA{sl} = vector.mma %kf{sl}, %qbA{sl}, {pa} : vector<8xf16>, vector<8xf16>, vector<8xf32>
      %sqB{sl} = vector.mma %kf{sl}, %qbB{sl}, {pb} : vector<8xf16>, vector<8xf16>, vector<8xf32>""")
        pa, pb = f"%sqA{sl}", f"%sqB{sl}"
    A(f"      scf.yield {pa}, {pb} : vector<8xf32>, vector<8xf32>\n    }}")
    A(f"""    %tf32 = index.cast %blk_first : index to i32
    %kbase32 = scalar.addi %tf32, %dh16_i32 : i32
    %base_v = vector.splat %kbase32 : vector<8xi32>
    %kpos = vector.addi %base_v, %row_off : vector<8xi32>
    %causal = vector.cmpi sle, %kpos, %qp_v : vector<8xi32> -> vector<8xi1>
    %inwin = vector.cmpi sge, %kpos, %low_v : vector<8xi32> -> vector<8xi1>
    %winok = vector.ori %inwin, %ns_v : vector<8xi1>
    %valid = vector.andi %causal, %winok : vector<8xi1>""")
    for hh, s_ in ((0, ""), (1, "B")):
        A(f"""    %sc{s_} = vector.mulf %s{'A' if hh == 0 else 'B'}_f, %scale_v : vector<8xf32>
    %masked{s_} = vector.select %valid, %sc{s_}, %neg_v : vector<8xf32>
    %lmax{s_} = vector.reduce<maxnumf> %masked{s_}, %neg_big : vector<8xf32>, f32
    %omax{s_}, %omax{s_}_u = kernel.subgroup.shuffle<xor> %lmax{s_}, %exchange, %width : f32, i32, i32
    %tmax{s_} = scalar.maxnumf %lmax{s_}, %omax{s_} : f32
    view.store %tmax{s_}, %mx_view[%c0_i, %c{hh}_i, %fb_wv, %lane] : f32, view<1x2x{waves}x32xf32>""")
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    for hh, s_ in ((0, ""), (1, "B")):
        for i in range(4):
            A(f"    %mx{s_}{i} = view.load %mx_view[%c0_i, %c{hh}_i, %fq{i}, %lane] : view<1x2x{waves}x32xf32> -> f32")
        A(f"""    %mxa{s_} = scalar.maxnumf %mx{s_}0, %mx{s_}1 : f32
    %mxb{s_} = scalar.maxnumf %mx{s_}2, %mx{s_}3 : f32
    %bm{s_} = scalar.maxnumf %mxa{s_}, %mxb{s_} : f32
    %m{s_}_new = scalar.maxnumf %m{s_}, %bm{s_} : f32
    %m{s_}_shift = scalar.subf %m{s_}, %m{s_}_new : f32
    %alpha{s_} = scalar.expf<afn> %m{s_}_shift : f32
    %m{s_}_new_v = vector.splat %m{s_}_new : vector<8xf32>
    %sh{s_} = vector.subf %masked{s_}, %m{s_}_new_v : vector<8xf32>
    %pe{s_} = vector.expf<afn> %sh{s_} : vector<8xf32>
    %p{s_} = vector.select %valid, %pe{s_}, %zero8 : vector<8xf32>
    %ls{s_} = vector.reduce<addf> %p{s_}, %zero_f : vector<8xf32>, f32
    %osum{s_}, %osum{s_}_u = kernel.subgroup.shuffle<xor> %ls{s_}, %exchange, %width : f32, i32, i32
    %tsum{s_} = scalar.addf %ls{s_}, %osum{s_} : f32
    %pt_h{s_} = vector.fptrunc %p{s_} : vector<8xf32> to vector<8xf16>
    vector.store %pt_h{s_}, %pl_view[%c0_i, %c{hh}_i, %fb_wv, %lane, %c0] : vector<8xf16>, view<1x2x{waves}x32x8xf16>
    view.store %tsum{s_}, %ls_view[%c0_i, %c{hh}_i, %fb_wv, %lane] : f32, view<1x2x{waves}x32xf32>""")
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    for hh, s_ in ((0, ""), (1, "B")):
        for j in range(4):
            A(f"    %lv{s_}{j} = view.load %ls_view[%c0_i, %c{hh}_i, %fq{j}, %lane] : view<1x2x{waves}x32xf32> -> f32")
        A(f"""    %lt01{s_} = scalar.addf %lv{s_}0, %lv{s_}1 : f32
    %lt012{s_} = scalar.addf %lt01{s_}, %lv{s_}2 : f32
    %lt{s_} = scalar.addf %lt012{s_}, %lv{s_}3 : f32
    %l_scaled{s_} = scalar.mulf %l{s_}, %alpha{s_} : f32
    %l{s_}_new = scalar.addf %l_scaled{s_}, %lt{s_} : f32
    %alpha{s_}_v = vector.splat %alpha{s_} : vector<8xf32>""")
    if partial:
        A("    %pos_hi_v = vector.splat %pos_hi_i32 : vector<8xi32>")
    for k in range(8):
        A(f"    %os{k} = vector.mulf %a{k}, %alpha_v : vector<8xf32>\n    %osB{k} = vector.mulf %b{k}, %alphaB_v : vector<8xf32>")
    if VOFF:
        for j in range(4):
            A(f"""    %vjr{j}_o = index.scale %rw{j}, %pos_bytes : index, offset -> offset
    %vjo{j} = index.add %vb_o, %vjr{j}_o : offset
    %vjo{j}_i = index.cast %vjo{j} : offset to i64
    view.store %vjo{j}_i, %vo_view[%fb_wv, %fq{j}, %lane] : i64, view<{waves}x4x32xi64>""")
    pvin = ", ".join([f"%pa{k} = %os{k} : vector<8xf32>" for k in range(8)] + [f"%pb{k} = %osB{k} : vector<8xf32>" for k in range(8)])
    pvout = ", ".join([f"%on{k}" for k in range(8)] + [f"%onB{k}" for k in range(8)])
    A(f"    {pvout} = scf.for %jj = [%c0 to %c4 step %c1]({pvin}) -> ({ty16}){JU} {{")
    VJ = ("      %vj_o_i = view.load %vo_view[%fb_wv, %jj_a, %lane] : view<%WV%x4x32xi64> -> i64\n      %vj_o = index.cast %vj_o_i : i64 to offset\n".replace("%WV%", str(waves)) if VOFF else
          """      %rj_a = scf.select %jj_c2, %rw2, %rw3 : index
      %rj_b = scf.select %jj_c1, %rw1, %rj_a : index
      %rj_c = scf.select %jj_c0, %rw0, %rj_b : index
      %rj = index.assume %rj_c [range(%rj_c, 0, 1048560), lt(%rj_c, %pl_rows_m)] : index
      %vj_ro = index.scale %rj, %pos_bytes : index, offset -> offset
      %vj_o = index.add %vb_o, %vj_ro : offset
""")
    A(f"""      %jj_a = index.assume %jj [range(%jj, 0, 3)] : index
      %jj_c1 = index.cmp eq, %jj, %c1 : index
      %jj_c2 = index.cmp eq, %jj, %c2 : index
      %jj_c0 = index.cmp eq, %jj, %c0 : index
{VJ}      %vj = buffer.view %arena[%vj_o] : buffer -> view<128x16xf16, %vt_g_layout>
      %prA = vector.load %pl_view[%c0_i, %c0_i, %jj_a, %lane, %c0] : view<1x2x{waves}x32x8xf16> -> vector<8xf16>
      %pfA = vector.fragment<rhs> %prA shape [%c16, %c16] : vector<8xf16>
      %prB = vector.load %pl_view[%c0_i, %c1_i, %jj_a, %lane, %c0] : view<1x2x{waves}x32x8xf16> -> vector<8xf16>
      %pfB = vector.fragment<rhs> %prB shape [%c16, %c16] : vector<8xf16>""")
    if partial:
        A("""      %jj16 = index.mul %jj_a, %c16 : index
      %jj16_i = index.cast %jj16 : index to i32
      %vkb = scalar.addi %tf32, %jj16_i : i32
      %vkb_v = vector.splat %vkb : vector<8xi32>
      %vkp = vector.addi %vkb_v, %row_off : vector<8xi32>
      %vok = vector.cmpi sle, %vkp, %pos_hi_v : vector<8xi32> -> vector<8xi1>""")
    for k in range(8):
        A(f"      %vc{k} = vector.fragment.load<lhs> %vj[%kcol{k}, %c0] shape [%c16, %c16] : view<128x16xf16, %vt_g_layout> -> vector<8xf16>")
    for k in range(8):
        vname = f"%vc{k}"
        if partial:
            A(f"      %vs{k} = vector.select %vok, %vc{k}, %zv8 : vector<8xf16>\n      %vm{k} = vector.fragment<lhs> %vs{k} shape [%c16, %c16] : vector<8xf16>")
            vname = f"%vm{k}"
        A(f"      %bn{k} = vector.mma {vname}, %pfA, %pa{k} : vector<8xf16>, vector<8xf16>, vector<8xf32>\n"
          f"      %bnB{k} = vector.mma {vname}, %pfB, %pb{k} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    A("      scf.yield " + ", ".join([f"%bn{k}" for k in range(8)] + [f"%bnB{k}" for k in range(8)]) + f" : {ty16}")
    A("    }")
    A("    scf.yield %m_new, %l_new, %mB_new, %lB_new, " + ", ".join([f"%on{k}" for k in range(8)] + [f"%onB{k}" for k in range(8)]) + f" : f32, f32, f32, f32, {ty16}\n  }}\n")
    loop = "".join(x if x.endswith("\n") else x + "\n" for x in L)
    text = text[:h0] + alloc + defs + qst + loop + epilogue + epi_b + tail
    return text
