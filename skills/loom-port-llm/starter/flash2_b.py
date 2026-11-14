"""Key-split flash loop for the gfx12 head_dim 512 twin (gen_flash2s.py, FLASH2_B=1; gfx12 twin only, opt-in).

The four waves of a query head used to compute the S^T tile of every 16-key tile together (each over its 128 dims, then an f32
partial exchange and four redundant softmaxes). Here a block is 64 keys and the waves split the work the other way, as llama.cpp's
Vulkan flash_attn_cm1 does:
  S:   wave j computes the S^T tile of keys 16 j .. 16 j + 15 of the block over ALL 512 dims (32 WMMAs; K fragments straight from the
       paged KV arena, Q^T fragments from workgroup memory where the head's waves staged them once),
  softmax: ONE softmax per S tile (not four). The column maxima of the four tiles are exchanged (one barrier), every wave derives the
       same running max, scales and exponentiates its own S tile, and publishes P^T (f16, the rhs fragment as it is) and its row sum,
  PV:  every wave reads the four P^T tiles (one more barrier) and accumulates O^T for ITS 128 dims over all 64 keys (32 WMMAs; V^T
       fragments with global_load_tr_b128 straight from the arena).
Two barriers per 64 keys, 64 WMMAs per wave and block. Keys past the tile's last written page re-read the last page (finite, fully
masked, weight 0); the partial kernel zeroes V^T for positions past its last written key.
"""
import re


def rewrite_b(text, kvh, d, partial, gw):
    assert d == 512
    wph = 4
    waves = wph * gw
    R = kvh * d
    NF = d // 16     # 32 fragments over the head dim
    QP = NF * 8 + 8  # elements per lane in the Q^T staging (8 per fragment, padded: the 16-byte reads of a wave's lanes hit different banks)
    pos_bytes = int(re.search(r"%pos_bytes = index\.constant (\d+) : offset", text).group(1))
    h0 = text.index("  %m_final, %l_final,")
    marker = "  // Partial tile: O^T / l" if partial else "  // O^T / l, gate, store through"
    h1 = text.index(marker)
    # allocas of the old loop go (K / V^T staging tiles, the S partial exchange)
    text = re.sub(r"^ *%k_bytes = .*\n *%k_tile = .*\n *%k_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%vt_bytes = .*\n *%vt_tile = .*\n *%vt_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%s_bytes = .*\n *%s_tile = .*\n *%s_view = .*\n", "", text, flags=re.M)
    h0 = text.index("  %m_final, %l_final,")
    h1 = text.index(marker)
    # Q^T fragments to workgroup memory (lane layout), after the last Q fragment
    qlast = re.search(r"^ *%qf7 = vector\.fragment<rhs> .*\n", text, re.M)
    assert qlast, "flash2_b: q fragments not found"
    qst = ("  %fb_c8 = index.constant 8 : index\n  %fb_c64 = index.constant 64 : index\n  %fb_c256 = index.constant 256 : index\n"
           "  %fb_dh8 = index.mul %dim_half, %fb_c8 : index\n  %fb_dh64 = index.mul %dim_half, %fb_c64 : index\n"
           "  %fb_h256 = index.mul %lane_half, %fb_c256 : index\n  %fb_qb = index.add %fb_h256, %fb_dh64 : index\n")
    for c in range(8):
        qst += (f"  %fqs{c}_v = index.add %fb_dh8, %e{c} : index\n  %fqs{c} = index.assume %fqs{c}_v [range(%fqs{c}_v, 0, {NF - 1})] : index\n"
                f"  %fqe{c}_v = index.mul %fqs{c}, %fb_c8 : index\n  %fqe{c} = index.assume %fqe{c}_v [range(%fqe{c}_v, 0, {QP - 8})] : index\n"
                f"  %fb_hg{c} = index.assume %head_in_group [range(%head_in_group, 0, {gw - 1})] : index\n"
                f"  %fqn{c}_o = index.constant {8 * c} : index\n  %fqn{c}_v = index.add %fb_qb, %fqn{c}_o : index\n"
                f"  %fqn{c} = index.assume %fqn{c}_v [range(%fqn{c}_v, 0, {d - 8})] : index\n"
                f"  %fqraw{c} = vector.load %q_view[%q_token, %head, %fqn{c}] : view<[%tokens]x{kvh * 8}x{d}xf32> -> vector<8xf32>\n"
                f"  %fqh{c} = vector.fptrunc %fqraw{c} : vector<8xf32> to vector<8xf16>\n"
                f"  vector.store %fqh{c}, %ql_view[%fb_hg{c}, %lane, %fqe{c}] : vector<8xf16>, view<{gw}x32x{QP}xf16>\n")
    alloc = (f"  %ql_bytes = index.constant {gw * 32 * QP * 2} : offset\n  %ql_tile = buffer.alloca<workgroup> align(16) %ql_bytes : buffer\n"
             f"  %ql_view = buffer.view %ql_tile[%origin] : buffer -> view<{gw}x32x{QP}xf16>\n"
             f"  %pl_bytes = index.constant {2 * waves * 32 * 16} : offset\n  %pl_tile = buffer.alloca<workgroup> align(16) %pl_bytes : buffer\n"
             f"  %pl_view = buffer.view %pl_tile[%origin] : buffer -> view<2x{waves}x32x8xf16>\n"
             f"  %mx_bytes = index.constant {2 * waves * 32 * 4} : offset\n  %mx_tile = buffer.alloca<workgroup> align(16) %mx_bytes : buffer\n"
             f"  %mx_view = buffer.view %mx_tile[%origin] : buffer -> view<2x{waves}x32xf32>\n"
             f"  %ls_tile = buffer.alloca<workgroup> align(16) %mx_bytes : buffer\n"
             f"  %ls_view = buffer.view %ls_tile[%origin] : buffer -> view<2x{waves}x32xf32>\n")
    hi = "  "
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
  %v_pv = buffer.view %arena[%vb_o] : buffer -> view<128x[%pool]xf16, %vt_g_layout>
  %pg_live_v = index.div %pos0, %c16 : index
  %pg_live = index.assume %pg_live_v [range(%pg_live_v, 0, %kh_last), lt(%pg_live_v, %blocks)] : index
  %c64 = index.constant 64 : index
  %c63 = index.constant 63 : index
  %c3 = index.constant 3 : index
  %zv8 = vector.constant 0.0 : vector<8xf16>
  %nblk_round = index.add %key_count, %c63 : index
  %nblk = index.div %nblk_round, %c64 : index
  %dh16 = index.mul %dim_half, %c16 : index
  %dh16_i32 = index.cast %dh16 : index to i32
  %fb_hw_v = index.mul %head_in_group, %c4 : index
  %fb_hw = index.assume %fb_hw_v [range(%fb_hw_v, 0, {waves - 4})] : index
  %fb_is0 = index.cmp eq, %dim_half, %c0 : index
  %fb_is1 = index.cmp eq, %dim_half, %c1 : index
  %fb_is2 = index.cmp eq, %dim_half, %c2 : index
"""
    for f in range(NF):
        defs += f"  %fq{f} = index.constant {f} : index\n  %kcol{f} = index.constant {16 * f} : index\n"
    for j in range(1, 4):
        defs += f"  %fb_hw{j}_v = index.add %fb_hw, %fq{j} : index\n  %fb_hw{j} = index.assume %fb_hw{j}_v [range(%fb_hw{j}_v, 0, {waves - 1})] : index\n"
    defs += "  %fb_hw0 = index.assume %fb_hw [range(%fb_hw, 0, " + str(waves - 4) + ")] : index\n"
    defs += "  %fb_wv = index.assume %wave [range(%wave, 0, " + str(waves - 1) + ")] : index\n"
    acc = ", ".join(f"%a{k} = %init : vector<8xf32>" for k in range(8))
    outs = ", ".join(f"%o{k}" for k in range(8))
    ty8 = ", ".join(["vector<8xf32>"] * 8)
    L = []
    A = L.append
    A(f"  %m_final, %l_final, {outs} = scf.for %bt = [%c0 to %nblk step %c1](%m = %neg_big : f32, %l = %zero_f : f32, {acc}) -> (f32, f32, {ty8}) {{")
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
      "    %kw_c256 = index.constant 256 : index\n    %kw_h = index.mul %lane_half, %kw_c256 : index\n")
    A(f"    %sq31 = scf.for %kw = [%c0 to %c4 step %c1](%sacc = %init : vector<8xf32>) -> (vector<8xf32>) {{")
    A(f"""      %kw_a = index.assume %kw [range(%kw, 0, 3)] : index
      %kw_o = index.mul %kw_a, %c64 : index
      %kw_c = index.add %kw_h, %kw_o : index
      %kw_ci = index.assume %kw_c [range(%kw_c, 0, {d - 64})] : index
      %kwv = vector.load %k_pv[%kw_lr, %kw_ci] : view<[%pool]x{d}xf16, %k_g_layout> -> vector<64xf16>
      %kw_f0 = index.mul %kw_a, %fq8 : index""")
    A(f"""      %kqo_v = index.mul %kw_f0, %fb_c8 : index
      %kqo = index.assume %kqo_v [range(%kqo_v, 0, {QP - 64})] : index
      %qrw = vector.load %ql_view[%head_in_group, %lane, %kqo] : view<{gw}x32x{QP}xf16> -> vector<64xf16>""")
    for sl in range(8):
        A(f"      %qr{sl} = vector.slice %qrw[{8 * sl}] : vector<64xf16> -> vector<8xf16>")
    prev = "%sacc"
    for sl in range(8):
        A(f"""      %kfs{sl} = vector.slice %kwv[{8 * sl}] : vector<64xf16> -> vector<8xf16>
      %kf{sl} = vector.fragment<lhs> %kfs{sl} shape [%c16, %c16] : vector<8xf16>
      %qb{sl} = vector.fragment<rhs> %qr{sl} shape [%c16, %c16] : vector<8xf16>
      %sq{sl} = vector.mma %kf{sl}, %qb{sl}, {prev} : vector<8xf16>, vector<8xf16>, vector<8xf32>""")
        prev = f"%sq{sl}"
    A(f"      scf.yield {prev} : vector<8xf32>\n    }}")
    prev = "%sq31"
    A(f"""    %tf32 = index.cast %blk_first : index to i32
    %kbase32 = scalar.addi %tf32, %dh16_i32 : i32
    %sc = vector.mulf {prev}, %scale_v : vector<8xf32>
    %base_v = vector.splat %kbase32 : vector<8xi32>
    %kpos = vector.addi %base_v, %row_off : vector<8xi32>
    %causal = vector.cmpi sle, %kpos, %qp_v : vector<8xi32> -> vector<8xi1>
    %inwin = vector.cmpi sge, %kpos, %low_v : vector<8xi32> -> vector<8xi1>
    %winok = vector.ori %inwin, %ns_v : vector<8xi1>
    %valid = vector.andi %causal, %winok : vector<8xi1>
    %masked = vector.select %valid, %sc, %neg_v : vector<8xf32>
    %lmax = vector.reduce<maxnumf> %masked, %neg_big : vector<8xf32>, f32
    %omax, %omax_u = kernel.subgroup.shuffle<xor> %lmax, %exchange, %width : f32, i32, i32
    %tmax = scalar.maxnumf %lmax, %omax : f32
    view.store %tmax, %mx_view[%par, %fb_wv, %lane] : f32, view<2x{waves}x32xf32>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %mx0 = view.load %mx_view[%par, %fb_hw0, %lane] : view<2x{waves}x32xf32> -> f32
    %mx1 = view.load %mx_view[%par, %fb_hw1, %lane] : view<2x{waves}x32xf32> -> f32
    %mx2 = view.load %mx_view[%par, %fb_hw2, %lane] : view<2x{waves}x32xf32> -> f32
    %mx3 = view.load %mx_view[%par, %fb_hw3, %lane] : view<2x{waves}x32xf32> -> f32
    %mxa = scalar.maxnumf %mx0, %mx1 : f32
    %mxb = scalar.maxnumf %mx2, %mx3 : f32
    %bm = scalar.maxnumf %mxa, %mxb : f32
    %m_new = scalar.maxnumf %m, %bm : f32
    %m_shift = scalar.subf %m, %m_new : f32
    %alpha = scalar.expf<afn> %m_shift : f32
    %m_new_v = vector.splat %m_new : vector<8xf32>
    %sh = vector.subf %masked, %m_new_v : vector<8xf32>
    %pe = vector.expf<afn> %sh : vector<8xf32>
    %p = vector.select %valid, %pe, %zero8 : vector<8xf32>
    %ls = vector.reduce<addf> %p, %zero_f : vector<8xf32>, f32
    %osum, %osum_u = kernel.subgroup.shuffle<xor> %ls, %exchange, %width : f32, i32, i32
    %tsum = scalar.addf %ls, %osum : f32
    %pt_h = vector.fptrunc %p : vector<8xf32> to vector<8xf16>
    vector.store %pt_h, %pl_view[%par, %fb_wv, %lane, %c0] : vector<8xf16>, view<2x{waves}x32x8xf16>
    view.store %tsum, %ls_view[%par, %fb_wv, %lane] : f32, view<2x{waves}x32xf32>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)""")
    for j in range(4):
        A(f"""    %lv{j} = view.load %ls_view[%par, %fb_hw{j}, %lane] : view<2x{waves}x32xf32> -> f32""")
    A("""    %lt01 = scalar.addf %lv0, %lv1 : f32
    %lt012 = scalar.addf %lt01, %lv2 : f32
    %lt = scalar.addf %lt012, %lv3 : f32
    %l_scaled = scalar.mulf %l, %alpha : f32
    %l_new = scalar.addf %l_scaled, %lt : f32
    %alpha_v = vector.splat %alpha : vector<8xf32>""")
    if partial:
        A("    %pos_hi_v = vector.splat %pos_hi_i32 : vector<8xi32>")
    for k in range(8):
        A(f"    %os{k} = vector.mulf %a{k}, %alpha_v : vector<8xf32>")
    pvin = ", ".join(f"%b{k} = %os{k} : vector<8xf32>" for k in range(8))
    pvout = ", ".join(f"%on{k}" for k in range(8))
    A(f"    {pvout} = scf.for %jj = [%c0 to %c4 step %c1]({pvin}) -> ({ty8}) {{")
    A(f"""      %jj_a = index.assume %jj [range(%jj, 0, 3)] : index
      %jj_c1 = index.cmp eq, %jj, %c1 : index
      %jj_c2 = index.cmp eq, %jj, %c2 : index
      %jj_c0 = index.cmp eq, %jj, %c0 : index
      %rj_a = scf.select %jj_c2, %rw2, %rw3 : index
      %rj_b = scf.select %jj_c1, %rw1, %rj_a : index
      %rj_c = scf.select %jj_c0, %rw0, %rj_b : index
      %rj = index.assume %rj_c [range(%rj_c, 0, 1048560), lt(%rj_c, %pl_rows_m)] : index
      %vj_ro = index.scale %rj, %pos_bytes : index, offset -> offset
      %vj_o = index.add %vb_o, %vj_ro : offset
      %vj = buffer.view %arena[%vj_o] : buffer -> view<128x16xf16, %vt_g_layout>
      %hwj_v = index.add %fb_hw, %jj_a : index
      %hwj = index.assume %hwj_v [range(%hwj_v, 0, {waves - 1})] : index
      %pr = vector.load %pl_view[%par, %hwj, %lane, %c0] : view<2x{waves}x32x8xf16> -> vector<8xf16>
      %pf = vector.fragment<rhs> %pr shape [%c16, %c16] : vector<8xf16>""")
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
            A(f"      %vs{k} = vector.select %vok, %vc{k}, %zv8 : vector<8xf16>\n      %vm{k} = vector.fragment<lhs> %vs{k} shape [%c16, %c16] : vector<8xf16>".replace("\\n", "\n"))
            vname = f"%vm{k}"
        A(f"      %bn{k} = vector.mma {vname}, %pf, %b{k} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    A("      scf.yield " + ", ".join(f"%bn{k}" for k in range(8)) + f" : {ty8}")
    A("    }")
    A("    scf.yield %m_new, %l_new, " + ", ".join(f"%on{k}" for k in range(8)) + f" : f32, f32, {ty8}\n  }}\n")
    loop = "".join(x if x.endswith("\n") else x + "\n" for x in L)
    # Q stores + allocas + defs go before the loop; Q fragments themselves are not kept in registers any more
    text = text[:h0] + alloc + defs + qst_barrier(qst) + loop + text[h1:]
    # the Q stores must follow the Q loads: they sit right before the loop, after the loads
    return text


def qst_barrier(qst):
    return qst + "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
