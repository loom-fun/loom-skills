"""GEMM_M256_WS (choice prefill.gemm.m256_ws; gfx11, q4_k): the m256 prefill GEMM with wave specialization.

A 384-thread workgroup: waves 0-7 are consumers (the m256 4 x 2 grid of 64 x 64 wave tiles, WMMA only), waves 8-11 are producers
(global loads, Q4_K dequant, LDS stores). They share a two-slot LDS ring of 32-K stages (activations 256 x 40 f16 + weights
128 x 40 f16 per slot, 61 KB with the flags: two workgroups per WGP) and synchronise through LDS counters instead of workgroup
barriers:
  full[s]  += 1 per producer lane after it stored slot s (release); consumers wait for 128 * generation (acquire)
  empty[s] += 1 per consumer lane after its MMAs read slot s (release); producers wait for 256 * (generation - 1) before refilling
A 32-K slot is exactly one Q4_K group, so each producer lane dequantizes one 32-value group of one weight row (one header decode).
Same launch interface and kernel name as gen_gemm.gemm_big256 (the program is unchanged)."""
import os
import gen_gemm as G


def ws_on():
    return os.environ.get("GEMM_M256_WS", "0").strip().lower() not in ("", "0", "false", "no", "off")


def ws_fmts():
    return ("q4_k",)


def gemm_ws(fmt, mode):
    assert fmt == "q4_k", fmt
    PD = int(os.environ.get("GEMM_M256_WS_PD", "1") or 1)
    f = G.FMT[fmt]
    two_w = mode == "swiglu"
    TW = 128
    TN = TW // 2 if two_w else TW
    cfr = 2 if two_w else 4
    RF = 4
    wave_cols = 16 * cfr
    NT = 384
    XS, WS = 256 * 80, TW * 80         # slot parts: activations 256 x 40 f16, weights 128 x 40 f16
    SLOT = XS + WS
    FLAGS = 2 * SLOT                   # 4 x i32 after the two slots
    LDS = FLAGS + 16
    name = f"{G.PREFIX}_gemm_{fmt}_{mode}_m256"
    weights = ["w", "w2"] if two_w else ["w"]
    WVR = TW - (TN - wave_cols) - 15
    L = []
    L.append(f'''kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cnt = index.constant {NT} : index
  %tn = index.constant {TN} : index
  %row_tiles = index.div %rows, %c256 : index
  %out_tiles = index.div %n, %tn : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%cnt, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, {", ".join(f"%{w}: buffer" for w in weights)}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 256, 8192), multiple_of(%rows_arg, 256)] : index
  %k = index.assume %k_arg [range(%k_arg, 256, 65536), multiple_of(%k_arg, 256)] : index
  %n = index.assume %n_arg [range(%n_arg, {TN}, 262144), multiple_of(%n_arg, {TN})] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %sixteen = index.constant 16 : offset
  %four = index.constant 4 : offset
  %block_bytes = index.constant {f["block_bytes"]} : offset
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
  %tn = index.constant {TN} : index
  %c80_off = index.constant 80 : offset
  %zero32 = scalar.constant 0 : i32
  %one32 = scalar.constant 1 : i32
  %row_tile_count = index.div %rows, %c256 : index
  %out_tile_count = index.div %n, %tn : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {NT - 1})] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, {NT // 32 - 1})] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %row_origin = index.mul %row_group, %c256 : index
  %out_origin = index.mul %out_group, %tn : index
  %blocks = index.div %k, %c256 : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %lds_bytes = index.constant {LDS} : offset
  %lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer
  %flags_off = index.constant {FLAGS} : offset
  %flags = buffer.view %lds[%flags_off] : buffer -> view<4xi32>
  %slot_bytes = index.constant {SLOT} : offset
  %xs_bytes = index.constant {XS} : offset
  %k32_steps = index.div %k, %c32 : index
  %k_limit = index.sub %k, %c32 : index
  %is_consumer = index.cmp ult, %bounded_subgroup, %c8 : index
  %first = index.cmp eq, %workitem, %c0 : index
  scf.if %first {{
    view.atomic.store %zero32, %flags[%c0] {{ordering = relaxed, scope = workgroup}} : i32, view<4xi32>
    view.atomic.store %zero32, %flags[%c1] {{ordering = relaxed, scope = workgroup}} : i32, view<4xi32>
    view.atomic.store %zero32, %flags[%c2] {{ordering = relaxed, scope = workgroup}} : i32, view<4xi32>
    %c3f = index.constant 3 : index
    view.atomic.store %zero32, %flags[%c3f] {{ordering = relaxed, scope = workgroup}} : i32, view<4xi32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>''')
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(RF) for c in range(cfr)]
    V8 = "vector<8xf32>"
    L.append("  " + ", ".join(f"%o{a}" for a in accs) + " = scf.if %is_consumer -> (" + ", ".join(V8 for _ in accs) + ") {")
    # ---------------- consumer ----------------
    L.append(f'''    %wave_row_group = index.div %bounded_subgroup, %c2 : index
    %wave_col_group = index.rem %bounded_subgroup, %c2 : index
    %wave_row_value = index.mul %wave_row_group, %c64 : index
    %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 192)] : index
    %wave_cols = index.constant {wave_cols} : index
    %wave_col_value = index.mul %wave_col_group, %wave_cols : index
    %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - wave_cols})] : index''')
    for r in range(1, RF):
        L.append(f"    %wave_row{r}_c = index.constant {16 * r} : index\n    %wave_row{r} = index.add %wave_row, %wave_row{r}_c : index")
    L.append('''    %wb_row = index.add %wave_col, %lane_row : index
    %wb_off = index.scale %wb_row, %c80_off : index, offset -> offset
    %need_unit = index.constant 128 : index''')
    L.append("    " + ", ".join(f"%co{a}" for a in accs) + " = scf.for %cstep = [%c0 to %k32_steps step %c1](" +
             ", ".join(f"%i{a} = %init : {V8}" for a in accs) + ") -> (" + ", ".join(V8 for _ in accs) + ") {")
    L.append(f'''      %cslot = index.rem %cstep, %c2 : index
      %cgen_h = index.div %cstep, %c2 : index
      %cgen = index.add %cgen_h, %c1 : index
      %cneed_i = index.mul %cgen, %need_unit : index
      %cneed = index.cast %cneed_i : index to i32
      %cseen = scf.while(%cs = %zero32 : i32) -> (i32) {{
        %cwait = scalar.cmpi slt, %cs, %cneed : i32
        scf.condition %cwait, %cs : i1, i32
      }} do(%cs2: i32) {{
        %ccur = view.atomic.load %flags[%cslot] {{ordering = acquire, scope = workgroup}} : view<4xi32> -> i32
        scf.yield %ccur : i32
      }}
      %cbase = index.scale %cslot, %slot_bytes : index, offset -> offset
      %cxv = buffer.view %lds[%cbase] : buffer -> view<256x40xf16>
      %cwo_a = index.add %cbase, %xs_bytes : offset
      %cwo = index.add %cwo_a, %wb_off : offset
      %cwv = buffer.view %lds[%cwo] : buffer -> view<{WVR}x40xf16>''')
    cur = {a: f"%i{a}" for a in accs}
    rowv = ["%wave_row"] + [f"%wave_row{r}" for r in range(1, RF)]
    for kk in range(2):
        L.append(f"      %fk{kk} = index.constant {kk * 16} : index")
        for r in range(RF):
            L.append(f"      %l{r}_{kk} = vector.fragment.load<lhs> %cxv[{rowv[r]}, %fk{kk}] shape [%c16, %c16] : view<256x40xf16> -> vector<16xf16>")
        for wi in range(len(weights)):
            for c in range(cfr):
                L.append(f'''      %wr{wi}{c}_{kk} = index.constant {wi * TN + c * 16} : index
      %rv{wi}{c}_{kk} = vector.load %cwv[%wr{wi}{c}_{kk}, %fk{kk}] : view<{WVR}x40xf16> -> vector<16xf16>
      %rf{wi}{c}_{kk} = vector.fragment<rhs> %rv{wi}{c}_{kk} shape [%c16, %c16] : vector<16xf16>''')
        for wi in range(len(weights)):
            for c in range(cfr):
                for r in range(RF):
                    a = f"{wi}r{r}c{c}"
                    L.append(f"      %m{a}_{kk} = vector.mma %l{r}_{kk}, %rf{wi}{c}_{kk}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
                    cur[a] = f"%m{a}_{kk}"
    L.append('''      %c_e = index.add %cslot, %c2 : index
      %cdone = view.atomic.rmw<addi> %one32, %flags[%c_e] {ordering = release, scope = workgroup} : i32, view<4xi32> -> i32''')
    L.append("      scf.yield " + ", ".join(cur[a] for a in accs) + " : " + ", ".join(V8 for _ in accs) + "\n    }")
    L.append("    scf.yield " + ", ".join(f"%co{a}" for a in accs) + " : " + ", ".join(V8 for _ in accs))
    # ---------------- producer ----------------
    L.append("  } else {")
    L.append(f'''    %pt_v = index.sub %workitem, %c256 : index
    %pt = index.assume %pt_v [range(%pt_v, 0, 127)] : index
    %x_row_value = index.add %row_origin, %pt : index
    %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index
    %x_row1_value = index.add %x_row, %c128 : index
    %x_row1 = index.assume %x_row1_value [lt(%x_row1_value, %rows)] : index
    %pt1 = index.add %pt, %c128 : index''')
    if two_w:
        L.append(f'''    %w_is_up = index.cmp uge, %bounded_subgroup, %c10 : index
    %w_local_up = index.sub %pt, %tn : index
    %w_local = scf.select %w_is_up, %w_local_up, %pt : index
    %w_out_row_value = index.add %out_origin, %w_local : index
    %wsel = scf.select %w_is_up, %w2, %w : buffer''')
        L[-1] = L[-1].replace("%c10 : index\n", "%c10 : index\n", 1)
        L.insert(-1, "    %c10 = index.constant 10 : index")
    else:
        L.append("    %w_out_row_value = index.add %out_origin, %pt : index")
    wsrc = "%wsel" if two_w else "%w"
    L.append(f'''    %w_out_row = index.assume %w_out_row_value [lt(%w_out_row_value, %n)] : index
    %w_rb_bb = index.constant {f["block_bytes"]} : index
    %w_rb = index.mul %blocks, %w_rb_bb : index
    %w_ro = index.mul %w_out_row, %w_rb : index
    %w_roa = index.assume %w_ro [range(%w_ro, 0, 2147000000)] : index
    %w_one = index.constant 1 : offset
    %w_row_origin = index.scale %w_roa, %w_one : index, offset -> offset
    %m15 = vector.splati 252645135 : vector<8xi32>''')

    def loads(sfx, kv):
        return f'''    %pk{sfx}_super = index.div {kv}, %c256 : index
    %pk{sfx}_within = index.rem {kv}, %c256 : index
    %pk{sfx}_page = index.div %pk{sfx}_within, %c64 : index
    %pk{sfx}_block_add = index.scale %pk{sfx}_super, %block_bytes : index, offset -> offset
    %pk{sfx}_block_origin = index.add %w_row_origin, %pk{sfx}_block_add : offset
    %pk{sfx}_header_view = buffer.view {wsrc}[%pk{sfx}_block_origin] : buffer -> view<4xi32>
    %pk{sfx}_payload = index.add %pk{sfx}_block_origin, %sixteen : offset
    %pk{sfx}_words = buffer.view {wsrc}[%pk{sfx}_payload] : buffer -> view<32xi32>
    %pk{sfx}_pw = index.mul %pk{sfx}_page, %c8 : index
    %pk{sfx}_header = vector.load %pk{sfx}_header_view[0] : view<4xi32> -> vector<4xi32>
    %pk{sfx}_packed = vector.load %pk{sfx}_words[%pk{sfx}_pw] : view<32xi32> -> vector<8xi32>
    %pk{sfx}_x0 = vector.load %x_view[%x_row, {kv}] : view<[%rows]x[%k]xf16> -> vector<32xf16>
    %pk{sfx}_x1 = vector.load %x_view[%x_row1, {kv}] : view<[%rows]x[%k]xf16> -> vector<32xf16>'''
    T4 = "vector<4xi32>, vector<8xi32>, vector<32xf16>, vector<32xf16>"
    L.append(loads("p", "%c0"))
    if PD == 2:   # GEMM_M256_WS_PD=2: producer loads two slots ahead
        L.append("    %pq_k = index.min %c32, %k_limit : index")
        L.append(loads("q", "%pq_k"))
        L.append("    %pd0, %pd1, %pd2, %pd3, %pd4, %pd5, %pd6, %pd7 = scf.for %pstep = [%c0 to %k32_steps step %c1](%ph = %pkp_header : vector<4xi32>, %pc = %pkp_packed : vector<8xi32>, "
                 "%px0 = %pkp_x0 : vector<32xf16>, %px1 = %pkp_x1 : vector<32xf16>, %qh = %pkq_header : vector<4xi32>, %qc = %pkq_packed : vector<8xi32>, "
                 f"%qx0 = %pkq_x0 : vector<32xf16>, %qx1 = %pkq_x1 : vector<32xf16>) -> ({T4}, {T4}) {{")
    else:
        L.append("    %pd0, %pd1, %pd2, %pd3 = scf.for %pstep = [%c0 to %k32_steps step %c1](%ph = %pkp_header : vector<4xi32>, %pc = %pkp_packed : vector<8xi32>, "
                 f"%px0 = %pkp_x0 : vector<32xf16>, %px1 = %pkp_x1 : vector<32xf16>) -> ({T4}) {{")
    L.append(f'''      %pslot = index.rem %pstep, %c2 : index
      %pgen_h = index.div %pstep, %c2 : index
      %pneed_i = index.mul %pgen_h, %c256 : index
      %pneed = index.cast %pneed_i : index to i32
      %p_e = index.add %pslot, %c2 : index
      %pseen = scf.while(%ps = %zero32 : i32) -> (i32) {{
        %pwait = scalar.cmpi slt, %ps, %pneed : i32
        scf.condition %pwait, %ps : i1, i32
      }} do(%ps2: i32) {{
        %pcur = view.atomic.load %flags[%p_e] {{ordering = acquire, scope = workgroup}} : view<4xi32> -> i32
        scf.yield %pcur : i32
      }}
      %pk = index.mul %pstep, %c32 : index
      %pgroup_v = index.div %pk, %c32 : index
      %pgroup_r = index.rem %pgroup_v, %c8 : index
      %pgroup = index.assume %pgroup_r [range(%pgroup_r, 0, 7)] : index
      %pnib = index.rem %pgroup, %c2 : index
      %pshift_i = index.mul %pnib, %c4 : index
      %pshift_32 = index.cast %pshift_i : index to i32
      %pshift = vector.splat %pshift_32 : vector<8xi32>
      %pscale, %pminimum = func.call {G.META}(%ph, %pgroup) : (vector<4xi32>, index) -> (f32, f32)
      %pshifted = vector.shrui %pc, %pshift : vector<8xi32>
      %pmasked = vector.andi %pshifted, %m15 : vector<8xi32>
      %pbytes = vector.bitcast %pmasked : vector<8xi32> to vector<32xi8>
      %pcodes = vector.uitofp %pbytes : vector<32xi8> to vector<32xf16>
      %pscale_h = scalar.fptrunc %pscale : f32 to f16
      %pneg_min = scalar.negf %pminimum : f32
      %pneg_min_h = scalar.fptrunc %pneg_min : f32 to f16
      %pscales = vector.splat %pscale_h : vector<32xf16>
      %pminima = vector.splat %pneg_min_h : vector<32xf16>
      %pw = vector.fmaf %pcodes, %pscales, %pminima : vector<32xf16>
      %pbase = index.scale %pslot, %slot_bytes : index, offset -> offset
      %pxv = buffer.view %lds[%pbase] : buffer -> view<256x40xf16>
      %pwo = index.add %pbase, %xs_bytes : offset
      %pwv = buffer.view %lds[%pwo] : buffer -> view<{TW}x40xf16>
      vector.store %px0, %pxv[%pt, %c0] : vector<32xf16>, view<256x40xf16>
      vector.store %px1, %pxv[%pt1, %c0] : vector<32xf16>, view<256x40xf16>
      vector.store %pw, %pwv[%pt, %c0] : vector<32xf16>, view<{TW}x40xf16>
      %pfull = view.atomic.rmw<addi> %one32, %flags[%pslot] {{ordering = release, scope = workgroup}} : i32, view<4xi32> -> i32
      %pkn_raw = index.add %pk, %c{32 * PD} : index
      %pkn_m = index.min %pkn_raw, %k_limit : index
      %pkn_c31 = index.constant 31 : index
      %pkn_lim = index.sub %k, %pkn_c31 : index
      %pkn = index.assume %pkn_m [lt(%pkn_m, %pkn_lim)] : index''')
    L.append(loads("n", "%pkn").replace("    ", "      ", 1).replace("\n    ", "\n      "))
    if PD == 2:
        L.append(f"      scf.yield %qh, %qc, %qx0, %qx1, %pkn_header, %pkn_packed, %pkn_x0, %pkn_x1 : {T4}, {T4}\n    }}")
    else:
        L.append(f"      scf.yield %pkn_header, %pkn_packed, %pkn_x0, %pkn_x1 : {T4}\n    }}")
    L.append("    scf.yield " + ", ".join("%init" for _ in accs) + " : " + ", ".join(V8 for _ in accs) + "\n  }")
    # ---------------- epilogue ----------------
    L.append(f'''  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_view = buffer.view %lds[%origin] : buffer -> view<256x16xf32>
  scf.if %is_consumer {{
    %inv_row_value = index.add %row_origin, %workitem : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    %inv_wi = index.assume %workitem [range(%workitem, 0, 255)] : index
    vector.store %inverse16, %inv_view[%inv_wi, %c0] : vector<16xf32>, view<256x16xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.if %is_consumer {{
  %e_wave_row_group = index.div %bounded_subgroup, %c2 : index
  %e_wave_col_group = index.rem %bounded_subgroup, %c2 : index
  %e_wave_row_value = index.mul %e_wave_row_group, %c64 : index
  %wave_row = index.assume %e_wave_row_value [range(%e_wave_row_value, 0, 192)] : index
  %e_wave_cols = index.constant {wave_cols} : index
  %e_wave_col_value = index.mul %e_wave_col_group, %e_wave_cols : index
  %wave_col = index.assume %e_wave_col_value [range(%e_wave_col_value, 0, {TN - wave_cols})] : index''')
    for r in range(1, RF):
        L.append(f"  %wave_row{r}_c = index.constant {16 * r} : index\n  %wave_row{r} = index.add %wave_row, %wave_row{r}_c : index")
    for r in range(RF):
        L.append(f"  %inv{r} = vector.fragment.load<result> %inv_view[{rowv[r]}, %c0] shape [%c16, %c16] : view<256x16xf32> -> vector<8xf32>")
    L.append(f'''  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<256x{TN}xf32, %y_layout>''')
    for r in range(RF):
        for c in range(cfr):
            row = rowv[r]
            L.append(f"  %oc{r}{c}_c = index.constant {c * 16} : index\n  %oc{r}{c} = index.add %wave_col, %oc{r}{c}_c : index")
            if two_w:
                L.append(f'''  %g{r}{c} = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>
  %u{r}{c} = vector.mulf %o1r{r}c{c}, %inv{r} : vector<8xf32>
  %sg{r}{c} = vector.siluf<afn> %g{r}{c} : vector<8xf32>
  %v{r}{c} = vector.mulf %sg{r}{c}, %u{r}{c} : vector<8xf32>''')
            else:
                L.append(f"  %v{r}{c}_s = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>")
                if mode == "acc":
                    L.append(f"  %old{r}{c} = vector.fragment.load<result> %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : view<256x{TN}xf32, %y_layout> -> vector<8xf32>\n  %v{r}{c} = vector.addf %old{r}{c}, %v{r}{c}_s : vector<8xf32>")
                else:
                    L.append(f"  %v{r}{c} = vector.addf %v{r}{c}_s, %zero8 : vector<8xf32>")
            L.append(f"  vector.fragment.store<result> %v{r}{c}, %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : vector<8xf32>, view<256x{TN}xf32, %y_layout>")
    L.append("  }\n  kernel.return\n}\n")
    t = "\n".join(L)
    abl = os.environ.get("GEMM_M256_WS_ABL", "")
    if "noprod" in abl:   # timing ablation: producers only signal (no loads, dequant or stores): the consumers' own bound
        import re as _re
        t = _re.sub(r"^      vector\.store %p[^\n]*\n", "", t, flags=_re.M)
    if "nospin" in abl:   # timing ablation: no waiting on the counters (wrong results): the synchronisation's cost
        t = t.replace("%cwait = scalar.cmpi slt, %cs, %cneed : i32", "%cwait = scalar.cmpi slt, %cs, %zero32 : i32")
        t = t.replace("%pwait = scalar.cmpi slt, %ps, %pneed : i32", "%pwait = scalar.cmpi slt, %ps, %zero32 : i32")
    return t
