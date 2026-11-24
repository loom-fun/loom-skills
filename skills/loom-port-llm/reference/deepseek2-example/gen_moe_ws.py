"""MOE_CW_WS (needs MOE_CW + MOE_CW_SWAP + MOE_CW_DQ): weight-stationary pair tiles over gen_moe_cw's column waves.

Same name, launch arguments and 64-row expert buckets as gen_moe_cw / gen_moe.grouped_gemm. A workgroup takes the bucket tiles
2y and 2y + 1: when both belong to one expert it runs ONE pass over their up-to-128 rows (nf = 5..8 live 16-row fragments), so each
lane dequantizes its column's IQ4_XS group once for up to 8 WMMA row fragments (cws: once per 64-row tile, nf <= 4); otherwise it
runs the two tiles as two passes. To fit 8 fragments' accumulators, a wave owns ONE 16-column weight fragment:
  swiglu: waves 0-3 own gate columns out*64 + 16w, waves 4-7 the up columns of the same output columns (64-column tiles); the up
          waves hand u * inv to the gate waves through LDS (the A stage, free after the K loop) for silu(g * inv) * (u * inv);
  store:  wave w owns columns out*128 + 16w (128-column tiles).
Values and K order are gen_moe_cw's (MOE_CW_DQ dequant; MOE_CW_PAIR K order when set), the epilogue does the same f32 ops, so live
rows are bit-identical to MOE_CW_DQ (+PAIR)."""
import gen_moe_cw as cw

KC = 128   # K per staged chunk
R = 128    # rows per pass region (two 64-row tiles)


def grouped_gemm_ws(fmt, mode, N, K, E, KU, tile=64):
    assert fmt == "iq4_xs" and mode in ("swiglu", "store") and cw.SWAP and cw.DQ, (fmt, mode)
    assert K % 256 == 0 and tile == 64
    sw = mode == "swiglu"
    TN = 64 if sw else 128
    assert N % TN == 0, (N, TN)
    rb = K // 256 * 136
    eb = rb * N
    name = f"g_moe_g{'gu' if sw else 'dn'}_{fmt}_k{K}"
    nch = K // KC
    max_rows = 8192 * KU + tile * E
    wargs = "%w: buffer, %w2: buffer" if sw else "%w: buffer"
    L = [f'''kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c64 = index.constant 64 : index
  %cTN = index.constant {TN} : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c64 : index
  %row_tiles1 = index.add %row_tiles, %c1 : index
  %pair_tiles = index.div %row_tiles1, %c2 : index
  %out_tiles = index.div %n, %cTN : index
  kernel.launch.config workgroups(%out_tiles, %pair_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xrows_arg: index, %tile_info: buffer, %rowmap: buffer, %x: buffer, %scales: buffer, {wargs}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 64, {max_rows}), multiple_of(%rows_arg, 64)] : index
  %k = index.assume %k_arg [range(%k_arg, {K}, {K})] : index
  %n = index.assume %n_arg [range(%n_arg, {N}, {N})] : index
  %origin = index.constant 0 : offset
  %xrows = index.assume %xrows_arg [range(%xrows_arg, 1, 1048576)] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %c256 = index.constant 256 : index
  %cTN = index.constant {TN} : index
  %c0_i32 = scalar.constant 0 : i32
  %cm1_i32 = scalar.constant -1 : i32
  %row_tile_count = index.div %rows, %c64 : index
  %rtc1 = index.add %row_tile_count, %c1 : index
  %pair_count = index.div %rtc1, %c2 : index
  %out_tile_count = index.div %n, %cTN : index
  %pair_id = kernel.workgroup.id<y> : index
  %out_group_id = kernel.workgroup.id<x> : index
  %pair = index.assume %pair_id [lt(%pair_id, %pair_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %ti_view = buffer.view %tile_info[%origin] : buffer -> view<[%row_tile_count]x2xi32>
  %t0_v = index.mul %pair, %c2 : index
  %t0 = index.assume %t0_v [lt(%t0_v, %row_tile_count)] : index
  %t1_v = index.add %t0, %c1 : index
  %t1_in = index.cmp ult, %t1_v, %row_tile_count : index
  %t1_c = index.min %t1_v, %row_tile_count : index
  %rtc_m1 = index.sub %row_tile_count, %c1 : index
  %t1_m = index.min %t1_c, %rtc_m1 : index
  %t1 = index.assume %t1_m [lt(%t1_m, %row_tile_count)] : index
  %e0 = view.load %ti_view[%t0, 0] : view<[%row_tile_count]x2xi32> -> i32
  %e1_raw = view.load %ti_view[%t1, 0] : view<[%row_tile_count]x2xi32> -> i32
  %e1 = scf.select %t1_in, %e1_raw, %cm1_i32 : i32
  %r1_raw = view.load %ti_view[%t1, 1] : view<[%row_tile_count]x2xi32> -> i32
  %e0_used = scalar.cmpi sge, %e0, %c0_i32 : i32
  %same = scalar.cmpi eq, %e0, %e1 : i32
  %joint = scalar.andi %same, %e0_used : i1
  %npass = scf.select %joint, %c1, %c2 : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %wave = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane16 = index.rem %lane, %c16 : index
  %cw_hi = index.cmp uge, %lane, %c16 : index
  %cw_x16 = scalar.constant 16 : i32
  %cw_true = scalar.constant true : i1
  %cw_w32 = scalar.constant 32 : i32{cw.DQ_CONSTS}
  %out_origin = index.mul %out_group, %cTN : index''']
    if sw:
        L.append('''  %is_up = index.cmp uge, %wave, %c4 : index
  %wave4 = index.rem %wave, %c4 : index
  %wave_col_v = index.mul %wave4, %c16 : index
  %wsel = scf.select %is_up, %w2, %w : buffer''')
    else:
        L.append('''  %wave_col_v = index.mul %wave, %c16 : index''')
    L.append(f'''  %wave_col = index.assume %wave_col_v [range(%wave_col_v, 0, {TN - 16})] : index
  %block_bytes = index.constant 136 : offset
  %row_bytes = index.constant {rb} : offset
  %expert_bytes = index.constant {eb} : offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%xrows]x2xf32>
  %stage_bytes = index.constant {R * (KC + 8) * 2} : offset
  %a_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %a_view = buffer.view %a_stage[%origin] : buffer -> view<{R}x{KC + 8}xf16>
{f"  %x_xview = buffer.view %a_stage[%origin] : buffer -> view<{R}x{TN}xf32>" + chr(10) if sw else ""}  %inv_bytes = index.constant {R * 16 * 4} : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<{R}x16xf32>
  // thread t stages activation row t/2, K columns (t%2)*64 .. +64 of each chunk
  %stage_row = index.div %workitem, %c2 : index
  %stage_q = index.rem %workitem, %c2 : index
  %stage_col0 = index.mul %stage_q, %c64 : index
  %stage_col1 = index.add %stage_col0, %c16 : index
  %stage_col2 = index.add %stage_col0, %c32 : index
  %c48 = index.constant 48 : index
  %stage_col3 = index.add %stage_col0, %c48 : index
  %rm_rows = index.mul %row_tile_count, %c64 : index
  %rm_last = index.sub %rm_rows, %c1 : index
  %rm_view = buffer.view %rowmap[%origin] : buffer -> view<[%rm_rows]xi32>
  %cKU = index.constant {KU} : index
  %xrows_m1 = index.sub %xrows, %c1 : index
  %zero16h = vector.splatf 0.0 : vector<16xf16>
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %nch = index.constant {nch} : index
  %ch_last = index.constant {nch - 1} : index
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %four = index.constant 4 : offset
  %cEm1 = scalar.constant {E - 1} : i32
  %c64_i32 = scalar.constant 64 : i32
  %wcol_v = index.add %out_origin, %wave_col : index
  %wcol_l = index.add %wcol_v, %lane16 : index
  %wcol0 = index.assume %wcol_l [lt(%wcol_l, %n)] : index
  %wro0_0 = index.scale %wcol0, %row_bytes : index, offset -> offset
  scf.for %p = [%c0 to %npass step %c1] {{
  %pt_s = index.add %t0, %p : index
  %pt_v = index.min %pt_s, %rtc_m1 : index
  %pt = index.assume %pt_v [lt(%pt_v, %row_tile_count)] : index
  %pe_raw = view.load %ti_view[%pt, 0] : view<[%row_tile_count]x2xi32> -> i32
  %p_is1 = index.cmp uge, %p, %c1 : index
  %pe = scf.select %p_is1, %e1, %pe_raw : i32
  %pr_own = view.load %ti_view[%pt, 1] : view<[%row_tile_count]x2xi32> -> i32
  %pr_joint = scalar.addi %r1_raw, %c64_i32 : i32
  %pr32 = scf.select %joint, %pr_joint, %pr_own : i32
  %p_used = scalar.cmpi sge, %pe, %c0_i32 : i32
  scf.if %p_used {{
  %e_lo = scalar.minsi %pe, %cEm1 : i32
  %e_b = scalar.assume %e_lo [range(%e_lo, 0, {E - 1})] : i32
  %expert = index.cast %e_b : i32 to index
  %expert_off = index.scale %expert, %expert_bytes : index, offset -> offset
  %wro0 = index.add %wro0_0, %expert_off : offset
  %pr_b = scalar.assume %pr32 [range(%pr32, 0, {R})] : i32
  %tile_rows = index.cast %pr_b : i32 to index
  %row_origin = index.mul %pt, %c64 : index
  %stage_tile_row_v = index.add %row_origin, %stage_row : index
  %stage_tile_row = index.min %stage_tile_row_v, %rm_last : index''')
    if sw:
        L.append('''  %st_prow = index.assume %stage_tile_row [lt(%stage_tile_row, %rm_rows)] : index
  %st_a32 = view.load %rm_view[%st_prow] : view<[%rm_rows]xi32> -> i32
  %st_valid = scalar.cmpi sge, %st_a32, %c0_i32 : i32
  %st_ac = scalar.maxsi %st_a32, %c0_i32 : i32
  %st_ab = scalar.assume %st_ac [range(%st_ac, 0, 1048575)] : i32
  %st_a = index.cast %st_ab : i32 to index
  %st_t = index.div %st_a, %cKU : index
  %x_row = index.assume %st_t [lt(%st_t, %xrows)] : index''')
    else:
        L.append('''  %x_rowc = index.min %stage_tile_row, %xrows_m1 : index
  %x_row = index.assume %x_rowc [lt(%x_rowc, %xrows)] : index''')
    # inverse row scales of the pass region (result layout) for the epilogue
    L.append(f'''  %writer = index.cmp ult, %workitem, %c128 : index
  scf.if %writer {{
    %iv_tr_v = index.add %row_origin, %workitem : index
    %iv_tr = index.min %iv_tr_v, %rm_last : index''')
    if sw:
        L.append('''    %iv_prow = index.assume %iv_tr [lt(%iv_tr, %rm_rows)] : index
    %iv_a32 = view.load %rm_view[%iv_prow] : view<[%rm_rows]xi32> -> i32
    %iv_ac = scalar.maxsi %iv_a32, %c0_i32 : i32
    %iv_ab = scalar.assume %iv_ac [range(%iv_ac, 0, 1048575)] : i32
    %iv_a = index.cast %iv_ab : i32 to index
    %iv_t = index.div %iv_a, %cKU : index
    %iv_row = index.assume %iv_t [lt(%iv_t, %xrows)] : index''')
    else:
        L.append('''    %iv_rc = index.min %iv_tr, %xrows_m1 : index
    %iv_row = index.assume %iv_rc [lt(%iv_rc, %xrows)] : index''')
    L.append(f'''    %inverse = view.load %s_view[%iv_row, %c1] : view<[%xrows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    %bw = index.assume %workitem [range(%workitem, 0, {R - 1})] : index
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<{R}x16xf32>
  }}
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<{R}x{TN}xf32, %y_layout>''')
    WB = "%wsel" if sw else "%w"

    def wload(ch, sfx):
        s = [f"    %sb{sfx} = index.div {ch}, %c2 : index", f"    %half{sfx} = index.rem {ch}, %c2 : index",
             f"    %sbo{sfx} = index.scale %sb{sfx}, %block_bytes : index, offset -> offset",
             f"    %hq{sfx} = index.constant 64 : offset", f"    %hqo{sfx} = index.scale %half{sfx}, %hq{sfx} : index, offset -> offset",
             f"""    %bo0{sfx} = index.add %wro0, %sbo{sfx} : offset
    %hv0{sfx} = buffer.view {WB}[%bo0{sfx}] : buffer -> view<2xi32>
    %h0{sfx} = vector.load %hv0{sfx}[0] : view<2xi32> -> vector<2xi32>
    %qb0{sfx}_0 = index.constant 8 : offset
    %qb0{sfx}_1 = index.add %bo0{sfx}, %qb0{sfx}_0 : offset
    %qb0{sfx} = index.add %qb0{sfx}_1, %hqo{sfx} : offset
    %qv0{sfx} = buffer.view {WB}[%qb0{sfx}] : buffer -> view<64xi8>"""]
        outs = [(f"%h0{sfx}", "vector<2xi32>")]
        for g in range(4):
            s.append(f"    %q0{g}{sfx} = vector.load %qv0{sfx}[{g * 16}] : view<64xi8> -> vector<16xi8>")
            outs.append((f"%q0{g}{sfx}", "vector<16xi8>"))
        return "\n".join(s), outs

    def xload(ch, sfx):
        s = [f"    %xk{sfx} = index.mul {ch}, %c128 : index"]
        outs = []
        for j in range(4):
            s.append(f"    %xc{j}{sfx} = index.add %xk{sfx}, %stage_col{j} : index")
            s.append(f"    %xr{j}{sfx} = vector.load %x_view[%x_row, %xc{j}{sfx}] : view<[%xrows]x[%k]xf16> -> vector<16xf16>")
            if sw and not cw.ZS:
                s.append(f"    %xs{j}{sfx} = scf.select %st_valid, %xr{j}{sfx}, %zero16h : vector<16xf16>")
                outs.append((f"%xs{j}{sfx}", "vector<16xf16>"))
            else:
                outs.append((f"%xr{j}{sfx}", "vector<16xf16>"))
        return "\n".join(s), outs

    def body(nf):
        accs = [f"%acc0_{f}" for f in range(nf)]
        s = [f"    // nf = {nf}", f"    %nfrows = index.constant {nf * 16} : index\n    %stage_live = index.cmp ult, %stage_row, %nfrows : index"]
        live = "%stage_live"
        if sw and cw.ZS:
            s.append("    %zs_not = scalar.xori %st_valid, %cw_true : i1\n    %zs_pad = scalar.andi %stage_live, %zs_not : i1\n    scf.if %zs_pad {")
            for j in range(4):
                s.append(f"      vector.store %zero16h, %a_view[%stage_row, %stage_col{j}] : vector<16xf16>, view<{R}x{KC + 8}xf16>")
            s.append("    }\n    %stage_ok = scalar.andi %stage_live, %st_valid : i1")
            live = "%stage_ok"
        wt, wouts = wload("%c0", "p")
        xt, xouts = xload("%c0", "p")
        s += [wt, xt]
        carried = wouts + xouts
        args = ", ".join([f"{a} = %init : vector<8xf32>" for a in accs] + [f"%cr{j} = {v} : {t}" for j, (v, t) in enumerate(carried)])
        types = ", ".join(["vector<8xf32>"] * len(accs) + [t for _, t in carried])
        res = [a.replace("%acc", "%fin") for a in accs] + [f"%unused_{j}" for j in range(len(carried))]
        s.append(f"    {', '.join(res)} = scf.for %ch = [%c0 to %nch step %c1]({args}) -> ({types}) {{")
        cr = {v: f"%cr{j}" for j, (v, _) in enumerate(carried)}
        s.append(f"    scf.if {live} {{")
        for j in range(4):
            s.append(f"      vector.store {cr[xouts[j][0]]}, %a_view[%stage_row, %stage_col{j}] : vector<16xf16>, view<{R}x{KC + 8}xf16>")
        s.append(f"""    }}
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %ch1 = index.add %ch, %c1 : index
    %chm = index.min %ch1, %ch_last : index
    %chn = index.assume %chm [lt(%chm, %nch)] : index""")
        wt, wouts_n = wload("%chn", "n")
        xt, xouts_n = xload("%chn", "n")
        s += [wt, xt]
        s.append("    %half_c = index.rem %ch, %c2 : index\n    %gbase = index.mul %half_c, %c4 : index")
        cur = {a: a for a in accs}
        for g in range(4):
            s.append(f"    %cg{g} = index.constant {g} : index\n    %grp{g}_v = index.add %gbase, %cg{g} : index")
            s.append(f"    %grp{g} = index.assume %grp{g}_v [range(%grp{g}_v, 0, 7)] : index")
            s.append(cw._dequant(f"d0{g}", cr["%h0p"], cr[f"%q0{g}p"], f"%grp{g}", swap=True))
            for kk in range(2):
                col = g * 32 + (0 if cw.PAIR else kk * 16)
                s.append(f"    %fc{g}{kk} = index.constant {col} : index")
                s.append(f"    %rhs0{g}{kk} = vector.fragment<rhs> %d0{g}_v{kk} shape [%c16, %c16] : vector<16xf16>")
                for f in range(nf):
                    s.append(f"    %fr{f}{g}{kk} = index.constant {f * 16} : index")
                    if cw.PAIR:
                        s.append(f"    %pr{f}{g}{kk} = index.add %fr{f}{g}{kk}, %lane16 : index\n"
                                 f"    %pc{f}{g}{kk} = index.add %fc{g}{kk}, %cw_ko{kk} : index\n"
                                 f"    %plv{f}{g}{kk} = vector.load %a_view[%pr{f}{g}{kk}, %pc{f}{g}{kk}] : view<{R}x{KC + 8}xf16> -> vector<16xf16>\n"
                                 f"    %lhs{f}{g}{kk} = vector.fragment<lhs> %plv{f}{g}{kk} shape [%c16, %c16] : vector<16xf16>")
                    else:
                        s.append(f"    %lhs{f}{g}{kk} = vector.fragment.load<lhs> %a_view[%fr{f}{g}{kk}, %fc{g}{kk}] shape [%c16, %c16] : view<{R}x{KC + 8}xf16> -> vector<16xf16>")
                    a = accs[f]
                    nv = f"%m0_{f}_{g}{kk}"
                    s.append(f"    {nv} = vector.mma %lhs{f}{g}{kk}, %rhs0{g}{kk}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
                    cur[a] = nv
        s.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        s.append(f"    scf.yield {', '.join([cur[a] for a in accs] + [v for v, _ in wouts_n + xouts_n])} : {types}\n    }}")
        for f in range(nf):
            s.append(f"    %er{f} = index.constant {f * 16} : index\n"
                     f"    %inv{f} = vector.fragment.load<result> %inv_view[%er{f}, %c0] shape [%c16, %c16] : view<{R}x16xf32> -> vector<8xf32>\n"
                     f"    %o{f} = vector.mulf %fin0_{f}, %inv{f} : vector<8xf32>")
        if sw:   # up waves: u * inv to LDS; gate waves: silu(g * inv) * u
            s.append("    scf.if %is_up {")
            for f in range(nf):
                s.append(f"      vector.fragment.store<result> %o{f}, %x_xview[%er{f}, %wave_col] shape [%c16, %c16] : vector<8xf32>, view<{R}x{TN}xf32>")
            s.append("    }\n    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n    %is_gate = index.cmp ult, %wave, %c4 : index\n    scf.if %is_gate {")
            for f in range(nf):
                s.append(f"""      %u{f} = vector.fragment.load<result> %x_xview[%er{f}, %wave_col] shape [%c16, %c16] : view<{R}x{TN}xf32> -> vector<8xf32>
      %sg{f} = vector.siluf<afn> %o{f} : vector<8xf32>
      %act{f} = vector.mulf %sg{f}, %u{f} : vector<8xf32>
      vector.fragment.store<result> %act{f}, %y_view[%er{f}, %wave_col] shape [%c16, %c16] : vector<8xf32>, view<{R}x{TN}xf32, %y_layout>""")
            s.append("    }")
        else:
            for f in range(nf):
                s.append(f"    vector.fragment.store<result> %o{f}, %y_view[%er{f}, %wave_col] shape [%c16, %c16] : vector<8xf32>, view<{R}x{TN}xf32, %y_layout>")
        return cw._suffix("\n".join(s), f"_n{nf}")

    # uniform dispatch on the live fragment count (1..8)
    def chain(lo, hi):
        if lo == hi:
            return body(lo)
        return (f"  %nfle{lo} = index.constant {lo * 16} : index\n  %nfc{lo} = index.cmp ule, %tile_rows, %nfle{lo} : index\n"
                f"  scf.if %nfc{lo} {{\n" + body(lo) + "\n  } else {\n" + chain(lo + 1, hi) + "\n  }")
    L.append(chain(1, 8))
    # the next pass reuses the A stage / inverse tile: everyone is done with them
    L.append("  }\n  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n  }\n  kernel.return\n}\n")
    return "\n".join(L)
