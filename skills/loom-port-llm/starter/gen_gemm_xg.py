"""GEMM_M256_XG (choice prefill.gemm.m256_xg; gfx11): the m256 prefill GEMM with the activations read straight from global
memory into WMMA A fragments (no activation stage in LDS), which frees the LDS that capped the row tile:

  XG=1  256-row tiles, 8 waves (the m256 shape), A from global
  XG=2  512-row tiles, 16 waves of 64 x 64: each dequantized weight value feeds 512 rows (half the dequant VALU per MMA)

GEMM_M256_XG_DB=1 double-buffers the weight stage (2 x 18 KB): the next step's weights are dequantized into the other buffer
while this step's MMAs read the current one, one barrier per 64-K step instead of two.

The kernel keeps the m256 name and launch interface (rows, k, n, x, scales, w[, w2], y), so the program is unchanged.
A 512-row tile over an epoch of 256 * odd rows computes its upper half on clamped rows and does not store it.
Lane l of an A fragment holds row l % 16, 16 consecutive K values (the RDNA3 layout, same as the B fragments' columns)."""
import os
import re as _re
import gen_gemm as G


def _on(k):
    return os.environ.get(k, "0").strip().lower() not in ("", "0", "false", "no", "off")


def xg_mode():
    v = os.environ.get("GEMM_M256_XG", "0").strip().lower()
    return 0 if v in ("", "0", "false", "no", "off") else int(v)


def xg_fmts():
    return tuple(os.environ.get("GEMM_M256_XG_FMTS", "q4_k,q6_k,q8_0,q5_k").split(","))


def gemm_xg(fmt, mode, fmt2=None, M=None, db=None, g32=None):
    M = M or (512 if xg_mode() == 2 else 256)
    db = _on("GEMM_M256_XG_DB") if db is None else db
    f = G.FMT[fmt]
    two_w = mode == "swiglu"
    fmt2 = fmt2 if two_w and fmt2 not in (None, fmt) else None
    NT, TW = M, 128
    TN = TW // 2 if two_w else TW
    cfr = 2 if two_w else 4
    RF = 4
    WG = M // 64                       # wave row groups
    wave_cols = 16 * cfr
    WVR = TW - (TN - wave_cols) - 15
    VPT = TW * 64 // NT                # weight values staged per thread per step: 32 (M256) or 16 (M512)
    STAGE = TW * 144                   # bytes of one weight stage (128 x 72 f16)
    nbuf = 2 if db else 1
    lds = max(nbuf * STAGE, M * 64)    # the epilogue's inverse-scale tile (M x 16 f32) reuses the stage
    name = f"{G.PREFIX}_gemm_{fmt}_{mode}_m256" if fmt2 is None else f"{G.PREFIX}_gemm_{fmt}_{fmt2}_{mode}_m256"
    weights = ["w", "w2"] if two_w else ["w"]
    k_min = 256 if f["block_values"] == 256 else 64
    tail = M > 256                     # rows is a multiple of 256 only
    L = []
    rt = "%row_tiles_v = index.add %rows, %c256 : index\n  %row_tiles = index.div %row_tiles_v, %cm : index" if tail else "%row_tiles = index.div %rows, %cm : index"
    L.append(f'''kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cm = index.constant {M} : index
  %cnt = index.constant {NT} : index
  %tn = index.constant {TN} : index
  {rt}
  %out_tiles = index.div %n, %tn : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%cnt, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, {", ".join(f"%{w}: buffer" for w in weights)}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 256, 8192), multiple_of(%rows_arg, 256)] : index
  %k = index.assume %k_arg [range(%k_arg, {k_min}, 65536), multiple_of(%k_arg, {k_min})] : index
  %n = index.assume %n_arg [range(%n_arg, {TN}, 262144), multiple_of(%n_arg, {TN})] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %sixteen = index.constant 16 : offset
  %four = index.constant 4 : offset
  %block_bytes = index.constant {f["block_bytes"]} : offset
  %block_bytes2 = index.constant {G.FMT[fmt2 or fmt]["block_bytes"]} : offset
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
  %cm = index.constant {M} : index
{"  %c512 = index.constant 512 : index" if M == 512 else ""}
  %tn = index.constant {TN} : index
  %c144_off = index.constant 144 : offset
  %nibble_mask = vector.splati 252645135 : vector<4xi32>
  %row_tile_count_v = index.add %rows, %c{M - 256 if tail else 0} : index
  %row_tile_count = index.div %row_tile_count_v, %cm : index
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
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_col_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row_value = index.mul %wave_row_group, %c64 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, {M - 64})] : index''')
    if tail:
        L[-1] = L[-1].replace(f"%row_tile_count_v = index.add %rows, %c{M - 256} : index", f"%c_tail = index.constant {M - 256} : index\n  %row_tile_count_v = index.add %rows, %c_tail : index")
    else:
        L[-1] = L[-1].replace("  %row_tile_count_v = index.add %rows, %c0 : index\n  %row_tile_count = index.div %row_tile_count_v, %cm : index",
                              "  %row_tile_count = index.div %rows, %cm : index")
    for r in range(1, RF):
        L.append(f"  %wave_row{r}_c = index.constant {16 * r} : index\n  %wave_row{r} = index.add %wave_row, %wave_row{r}_c : index")
    L.append(f'''  %wave_cols = index.constant {wave_cols} : index
  %wave_col_value = index.mul %wave_col_group, %wave_cols : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - wave_cols})] : index
  %row_origin = index.mul %row_group, %c{M} : index
  %out_origin = index.mul %out_group, %tn : index
  %blocks = index.div %k, %c{f["block_values"]} : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %w_bytes = index.constant {lds} : offset
  %w_stage = buffer.alloca<workgroup> align(16) %w_bytes : buffer
  %rows_last = index.sub %rows, %c1 : index''')
    # A-fragment rows of this lane (clamped into the epoch for a 512-row tile's upper half over 256 * odd rows)
    for r in range(RF):
        rr = "%wave_row" if r == 0 else f"%wave_row{r}"
        L.append(f"  %xr{r}_a = index.add %row_origin, {rr} : index\n  %xr{r}_b = index.add %xr{r}_a, %lane_row : index")
        if tail:
            L.append(f"  %xr{r}_c = index.min %xr{r}_b, %rows_last : index\n  %xr{r} = index.assume %xr{r}_c [lt(%xr{r}_c, %rows)] : index")
        else:
            L.append(f"  %xr{r} = index.assume %xr{r}_b [lt(%xr{r}_b, %rows)] : index")
        # the lane's A row as its own view: the loop's load addresses are a fixed base + the step's K (no row * k per step)
        L.append(f"  %xro{r}_i = index.mul %xr{r}, %k : index\n  %xro{r} = index.scale %xro{r}_i, %two : index, offset -> offset\n"
                 f"  %xv{r} = buffer.view %x[%xro{r}] : buffer -> view<[%k]xf16>")
    tpr = 64 // VPT                     # threads per staged weight row: 2 (M256) or 4 (M512)
    L.append(f'''  %c_tpr = index.constant {tpr} : index
  %c_vpt = index.constant {VPT} : index
  %stage_row = index.div %workitem, %c_tpr : index
  %stage_ks = index.rem %workitem, %c_tpr : index
  %stage_col = index.mul %stage_ks, %c_vpt : index
  %stage_col1 = index.add %stage_col, %c16 : index''')
    if two_w:
        L.append(f'''  %half_tw = index.constant {TN} : index
  %w_is_up = index.cmp uge, %stage_row, %half_tw : index
  %w_local_up = index.sub %stage_row, %half_tw : index
  %w_local = scf.select %w_is_up, %w_local_up, %stage_row : index
  %w_out_row_value = index.add %out_origin, %w_local : index
  %wsel = scf.select %w_is_up, %w2, %w : buffer''')
    else:
        L.append("  %w_out_row_value = index.add %out_origin, %stage_row : index")
    L.append(f'''  %w_out_row = index.assume %w_out_row_value [lt(%w_out_row_value, %n)] : index
  %w_row_origin = index.scale %w_out_row, %row_bytes : index, offset -> offset
  %row_bytes2 = index.scale %blocks, %block_bytes2 : index, offset -> offset
  %w2_row_origin = index.scale %w_out_row, %row_bytes2 : index, offset -> offset
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index
  %k_limit = index.sub %k, %c64 : index
  %wb_row = index.add %wave_col, %lane_row : index
  %wb_off = index.scale %wb_row, %c144_off : index, offset -> offset
  %stage_bytes = index.constant {STAGE} : offset''')
    wsrc = "wsel" if two_w and not fmt2 else "w"

    def fmt_parts(fm, ws, pre, second):
        g = (VPT == 32) and ((_on("GEMM_M256_G32") if g32 is None else g32) and fm in G._g32_fmts())
        fix = (lambda t: t.replace("%w_row_origin", "%w2_row_origin").replace("%block_bytes", "%block_bytes2")) if second else (lambda t: t)

        def fixp(out, col, sc, res):
            lines, addr, loads = res
            return (out, col, sc, [fix(l) for l in lines], [fix(l) for l in addr], loads)
        if g:
            return g, [fixp(f"{pre}g", f"%{pre}colg", "%c0", G._g32_dequant(fm, ws, f"{pre}g", f"%{pre}colg"))]
        sl = (("a", "%stage_col"), ("b", "%stage_col1")) if VPT == 32 else (("a", "%stage_col"),)
        return g, [fixp(f"{pre}{x}", f"%{pre}col{x}", sc, G._split_dequant(fm, ws, f"{pre}{x}", f"%{pre}col{x}")) for x, sc in sl]
    g32, parts = fmt_parts(fmt, wsrc, "dq", False)
    groups = [(fmt, g32, parts)]
    if fmt2:
        g32b, parts2 = fmt_parts(fmt2, "w2", "du", True)
        groups.append((fmt2, g32b, parts2))
        parts = parts + parts2
    if any(g for _, g, _ in groups):
        L.append(G.G32_CONSTS)

    def addr_text(sfx, kvar):
        t = []
        for out, col, sc, lines, addr, loads in parts:
            c2 = col + sfx
            t.append(f"    {c2} = index.add {kvar}, {sc} : index")
            body = "\n".join(addr).replace(col + " ", c2 + " ").replace(col + ",", c2 + ",")
            body = _re.sub(rf"%{out}_", f"%{out}{sfx}_", body)
            t.append(body)
        return "\n".join(t)

    def compute(gparts, kvar, sfx, src):
        """dequant of the loads named src (a dict load name -> value) at K origin kvar"""
        t = []
        for out, col, sc, lines, addr, loads in gparts:
            c2 = col + sfx
            t.append(f"    {c2} = index.add {kvar}, {sc} : index")
            text = "\n".join(l for l in lines if ".load " not in l).replace(col + " ", c2 + " ").replace(col + ",", c2 + ",")
            text = _re.sub(rf"%{out}(?![A-Za-z0-9])", f"%{out}{sfx}", text)
            for nm, ty in loads:
                text = _re.sub(_re.escape(nm.replace(out + "_", out + sfx + "_", 1)) + r"\b", src[nm], text)
            t.append(text)
        return t

    def halves(fm, g, gparts, sfx):
        if not g:
            return [], [f"%{p[0]}{sfx}" for p in gparts]
        st = G._g32_stores(fm, gparts[0][0] + sfx)
        if len(st) == 2:
            return [], [st[0][0], st[1][0]]
        v = st[0][0]
        return [f"    {v}_h0 = vector.slice {v}[0] : vector<32xf16> -> vector<16xf16>",
                f"    {v}_h1 = vector.slice {v}[16] : vector<32xf16> -> vector<16xf16>"], [f"{v}_h0", f"{v}_h1"]

    def stage(kvar, sfx, src, view):
        """dequantize the weights loaded into src (step at K kvar) and store them into the stage view"""
        t = []
        cols = ["%stage_col", "%stage_col1"]
        st = lambda vals: [f"    vector.store {v}, {view}[%stage_row, {cols[i]}] : vector<16xf16>, view<{TW}x72xf16>" for i, v in enumerate(vals)]
        if fmt2:
            (fa, ga, pa), (fb, gb, pb) = groups
            sa, va = halves(fa, ga, pa, sfx)
            sb, vb = halves(fb, gb, pb, sfx)
            t.append("    scf.if %w_is_up {")
            t += compute(pb, kvar, sfx, src) + sb + st(vb)
            t.append("    } else {")
            t += compute(pa, kvar, sfx, src) + sa + st(va)
            t.append("    }")
        else:
            t += compute(parts, kvar, sfx, src)
            s, v = halves(fmt, g32, parts, sfx)
            t += s + st(v)
        return [x for x in t if x]

    load_names = [(nm, ty, out) for out, col, sc, lines, addr, loads in parts for nm, ty in loads]

    def loads_src(sfx):
        return {nm: nm.replace(out + "_", out + sfx + "_", 1) for nm, ty, out in load_names}
    carried = [(f"%cr_{nm[1:]}", ty, out, nm) for nm, ty, out in load_names]
    cr_src = {nm: c for c, ty, out, nm in carried}
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(RF) for c in range(cfr)]
    # prologue
    L.append(addr_text("p", "%c0").replace("    ", "  "))
    first_carry = "p"
    if db:
        L.append("  %s0_view = buffer.view %w_stage[%origin] : buffer -> view<" + f"{TW}x72xf16>")
        L += [l.replace("    ", "  ", 1) for l in stage("%c0", "s0", loads_src("p"), "%s0_view")]
        L.append("  %k_one_v = index.min %c64, %k_limit : index")
        L.append(addr_text("q", "%k_one_v").replace("    ", "  "))
        L.append("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        first_carry = "q"
    init_carry = [loads_src(first_carry)[nm] for c, ty, out, nm in carried]
    L.append("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(len(carried))) + " = scf.for %step = [%c0 to %k_steps step %c1](" +
             ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " + ", ".join(f"{c[0]} = {v} : {c[1]}" for c, v in zip(carried, init_carry)) + ") -> (" +
             ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[1] for c in carried) + ") {")
    L.append("    %k_origin = index.mul %step, %c64 : index")
    if db:
        L.append(f"""    %par = index.rem %step, %c2 : index
    %cur_off = index.scale %par, %stage_bytes : index, offset -> offset
    %nxt_par = index.sub %c1, %par : index
    %nxt_off = index.scale %nxt_par, %stage_bytes : index, offset -> offset
    %sn_view = buffer.view %w_stage[%nxt_off] : buffer -> view<{TW}x72xf16>
    %k_n_raw = index.add %k_origin, %c64 : index
    %k_n = index.min %k_n_raw, %k_limit : index
    %k_nn_raw = index.add %k_origin, %c128 : index
    %k_nn = index.min %k_nn_raw, %k_limit : index""")
        L += stage("%k_n", "s", cr_src, "%sn_view")
        L.append(addr_text("n", "%k_nn"))
        L.append("    %wv_off = index.add %wb_off, %cur_off : offset")
    else:
        L.append(f"    %s_view0 = buffer.view %w_stage[%origin] : buffer -> view<{TW}x72xf16>")
        L += stage("%k_origin", "s", cr_src, "%s_view0")
        L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        L.append("    %k_n_raw = index.add %k_origin, %c64 : index\n    %k_n = index.min %k_n_raw, %k_limit : index")
        L.append(addr_text("n", "%k_n"))
        L.append("    %wv_off = index.add %wb_off, %origin : offset")
    L.append(f"    %w_wave_view = buffer.view %w_stage[%wv_off] : buffer -> view<{WVR}x72xf16>")
    cur = {a: f"%i{a}" for a in accs}
    for kk in range(4):
        L.append(f"    %fk{kk} = index.constant {kk * 16} : index\n    %xk{kk} = index.add %k_origin, %fk{kk} : index")
        for r in range(RF):
            L.append(f"""    %lv{r}_{kk} = vector.load %xv{r}[%xk{kk}] : view<[%k]xf16> -> vector<16xf16>
    %l{r}_{kk} = vector.fragment<lhs> %lv{r}_{kk} shape [%c16, %c16] : vector<16xf16>""")
        for wi in range(len(weights)):
            for c in range(cfr):
                L.append(f'''    %wr{wi}{c}_{kk} = index.constant {wi * TN + c * 16} : index
    %rv{wi}{c}_{kk} = vector.load %w_wave_view[%wr{wi}{c}_{kk}, %fk{kk}] : view<{WVR}x72xf16> -> vector<16xf16>
    %rf{wi}{c}_{kk} = vector.fragment<rhs> %rv{wi}{c}_{kk} shape [%c16, %c16] : vector<16xf16>''')
        for wi in range(len(weights)):
            for c in range(cfr):
                for r in range(RF):
                    a = f"{wi}r{r}c{c}"
                    L.append(f"    %m{a}_{kk} = vector.mma %l{r}_{kk}, %rf{wi}{c}_{kk}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
                    cur[a] = f"%m{a}_{kk}"
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    nsrc = loads_src("n")
    L.append("    scf.yield " + ", ".join(cur[a] for a in accs) + ", " + ", ".join(nsrc[c[3]] for c in carried) + " : " +
             ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[1] for c in carried) + "\n  }")
    inv_row = "%inv_row_b" if not tail else "%inv_row_c"
    L.append(f'''  %inv_view = buffer.view %w_stage[%origin] : buffer -> view<{M}x16xf32>
  %inv_row_b = index.add %row_origin, %workitem : index''' + (f"\n  %inv_row_c = index.min %inv_row_b, %rows_last : index" if tail else "") + f'''
  %inv_row = index.assume {inv_row} [lt({inv_row}, %rows)] : index
  %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
  %inverse16 = vector.splat %inverse : vector<16xf32>
  vector.store %inverse16, %inv_view[%workitem, %c0] : vector<16xf32>, view<{M}x16xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)''')
    rowv = ["%wave_row"] + [f"%wave_row{r}" for r in range(1, RF)]
    for r in range(RF):
        L.append(f"  %inv{r} = vector.fragment.load<result> %inv_view[{rowv[r]}, %c0] shape [%c16, %c16] : view<{M}x16xf32> -> vector<8xf32>")
    L.append(f'''  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<{M}x{TN}xf32, %y_layout>''')
    if tail:
        L.append("  %wave_first = index.add %row_origin, %wave_row : index\n  %wave_live = index.cmp ult, %wave_first, %rows : index\n  scf.if %wave_live {")
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
                    L.append(f"  %old{r}{c} = vector.fragment.load<result> %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : view<{M}x{TN}xf32, %y_layout> -> vector<8xf32>\n  %v{r}{c} = vector.addf %old{r}{c}, %v{r}{c}_s : vector<8xf32>")
                else:
                    L.append(f"  %v{r}{c} = vector.addf %v{r}{c}_s, %zero8 : vector<8xf32>")
            L.append(f"  vector.store_marker %v{r}{c}, {row}, %oc{r}{c}")
    if tail:
        L.append("  }")
    L.append("  kernel.return\n}\n")
    t = "\n".join(L)
    t = _re.sub(r"  vector\.store_marker (%\S+), (%\S+), (%\S+)",
                lambda m: f"  vector.fragment.store<result> {m.group(1)}, %y_view[{m.group(2)}, {m.group(3)}] shape [%c16, %c16] : vector<8xf32>, view<{M}x{TN}xf32, %y_layout>", t)
    return t
