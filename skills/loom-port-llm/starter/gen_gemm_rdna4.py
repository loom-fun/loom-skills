"""RDNA4 (gfx12) twins of the WMMA GEMM kernels, derived from the gfx11 kernel text (gen_gemm.py, vision/gen_vision.py).

A gfx12 f16 16x16x16 WMMA takes lhs/rhs fragments of vector<8xf16> (gfx11: vector<16xf16>); the accumulator is
vector<8xf32> on both. The lhs comes from vector.fragment.load<lhs> (layout neutral: only the type changes). The rhs
fragment was built from a raw per-lane row load of the weight stage; on gfx12 it is vector.fragment.load<rhs> from a
transposed ([1, C]-strided) view of the same LDS bytes (B[k][n] = W[n][k]) at the tile's row base, the same for every lane.

twin(text) returns the `<name>_rdna4` kernel (target(@rdna4), same launch signature) of one gfx11 kernel text;
with_twins(text) appends it after every WMMA kernel of a generated file. LOOM_RDNA4=0 disables the twins (enabled()).
The dual_target.py post-pass tags the gfx11 kernels and routes each launch."""
import os, re


def enabled():
    return os.environ.get("LOOM_RDNA4", "1") not in ("", "0", "false", "no")


_DEF = re.compile(r'^kernel\.def export\("(\w+)"\) @(\w+)\(', re.M)
_LOAD = re.compile(r"^(\s*)(%\w+) = vector\.load (%\w+)\[(%\w+), (%\w+)\] : view<(\d+)x(\d+)xf16> -> vector<16xf16>$")
_FRAG = re.compile(r"^\s*(%\w+) = vector\.fragment<rhs> (%\w+) shape \[%c16, %c16\] : vector<16xf16>$")


def is_wmma(text):
    return "vector.mma " in text and "vector<16xf16>, vector<16xf16>, vector<8xf32>" in text


def kernel_text(part):
    """The kernel.def at the start of part, up to its closing brace (a part may carry text after the kernel)."""
    end = part.find("\n  kernel.return\n}\n")
    return part if end < 0 else part[:end + len("\n  kernel.return\n}\n")]


def twin(text):
    """The gfx12 twin of one gfx11 WMMA kernel text (a single kernel.def ... kernel.return block)."""
    m = _DEF.search(text)
    old = m.group(1)
    new = old + "_rdna4"
    t = text.replace(f'kernel.def export("{old}") @{old}(', f'kernel.def target(@rdna4) export("{new}") @{new}(', 1)
    t = re.sub(r"(vector\.fragment\.load<lhs> [^\n]*-> )vector<16xf16>", r"\1vector<8xf16>", t)
    views = {}   # %V -> (buffer, offset, R, C)
    for v in re.finditer(r"(%\w+) = buffer\.view (%\w+)\[(%\w+)\] : buffer -> view<(\d+)x(\d+)xf16>\n", t):
        views[v.group(1)] = v.groups()[1:]
    used, lines, out, i = {}, t.split("\n"), [], 0
    while i < len(lines):
        lm = _LOAD.match(lines[i])
        fm = _FRAG.match(lines[i + 1]) if i + 1 < len(lines) else None
        if lm and fm and fm.group(2) == lm.group(2):
            ind, rv, V, row, k, R, C = lm.groups()
            if V == "%w_wave_view":   # m256: a per-lane view at row (wave_col + lane%16); constant rows within it
                used["%w_stage_view"] = None
                out.append(f"{ind}{rv}_b = index.add %wave_col, {row} : index")
                out.append(f"{ind}{fm.group(1)} = vector.fragment.load<rhs> %w_stage_view_t[{k}, {rv}_b] shape [%c16, %c16] : view<72x{views['%w_stage_view'][2]}xf16, %w_stage_view_tl> -> vector<8xf16>")
            else:
                used[V] = None
                out.append(f"{ind}{rv}_b = index.sub {row}, %lane_row : index")
                out.append(f"{ind}{fm.group(1)} = vector.fragment.load<rhs> {V}_t[{k}, {rv}_b] shape [%c16, %c16] : view<{C}x{R}xf16, {V}_tl> -> vector<8xf16>")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    t = "\n".join(out)
    t = t.replace(": vector<16xf16>, vector<16xf16>, vector<8xf32>", ": vector<8xf16>, vector<8xf16>, vector<8xf32>")
    for V in used:
        buf, off, R, C = views[V]
        decl = f"  {V} = buffer.view {buf}[{off}] : buffer -> view<{R}x{C}xf16>\n"
        assert decl in t, (old, V)
        t = t.replace(decl, decl + f"  {V}_tl = encoding.layout.strided [1, {C}] : encoding<layout>\n"
                      f"  {V}_t = buffer.view {buf}[{off}] : buffer -> view<{C}x{R}xf16, {V}_tl>\n", 1)
    assert "%lane_row = " in t, old
    assert "vector<16xf16>, vector<16xf16>" not in t and "vector.fragment<rhs>" not in t, old
    return t


def with_twins(text):
    """text with `K_rdna4` appended right after every WMMA kernel K (no-op unless enabled())."""
    if not enabled():
        return text
    out = []
    # a kernel runs from its `kernel.def` to the closing "kernel.return\n}" line
    parts = re.split(r"(?m)^(?=kernel\.def )", text)
    for p in parts:
        out.append(p)
        if p.startswith("kernel.def "):
            k = kernel_text(p)
            if is_wmma(k):
                nat = native_twin(k)
                out.append((nat or twin(k)).rstrip("\n") + "\n\n")
    return "".join(out)


# ---- native gfx12 256-row kernel: double-buffered LDS, 32-value K steps -------------------------------------------------
# The gfx11 m256 kernel stages a 64-value K step into ONE LDS tile and needs a barrier before and after its MMAs, so the
# eight waves of a workgroup (and the two workgroups of a WGP, which start together) decode weights and run MMAs in
# lockstep: the matrix pipe idles during the decode (R9700 kbench: 58 % of the WMMA peak). This kernel (same name + _rdna4,
# same launch signature and grid) keeps two LDS tiles of 32 values: while the MMAs read tile s % 2, the waves dequantize
# and store step s + 1 into the other tile, one barrier per step. The loop is unrolled by two (static tile offsets).
def pipe_enabled():
    return enabled() and os.environ.get("RDNA4_GEMM_PIPE", "1") not in ("", "0", "false", "no")


def native_m256(fmt, mode):
    import gen_gemm as G
    f = G.FMT[fmt]
    two_w = mode == "swiglu"
    NT, TW = 256, 128
    TN = TW // 2 if two_w else TW
    cfr = 2 if two_w else 4
    RF = 4
    wave_cols = 16 * cfr
    name = f"{G.PREFIX}_gemm_{fmt}_{mode}_m256_rdna4"
    weights = ["w", "w2"] if two_w else ["w"]
    k_min = 256 if f["block_values"] == 256 else 64
    XS, XB, WB = 40, 256 * 40 * 2, TW * 40 * 2   # padded LDS row (halves), bytes of the x tile and of the weight tile
    STAGE = XB + WB
    wsrc = "wsel" if two_w else "w"
    L = []
    A = L.append
    A(f'''kernel.def target(@rdna4) export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cnt = index.constant {NT} : index
  %tn = index.constant {TN} : index
  %row_tiles = index.div %rows, %c256 : index
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
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
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
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_col_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row_value = index.mul %wave_row_group, %c64 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 192)] : index''')
    for r in range(1, RF):
        A(f"  %wave_row{r}_c = index.constant {16 * r} : index\n  %wave_row{r} = index.add %wave_row, %wave_row{r}_c : index")
    A(f'''  %wave_cols = index.constant {wave_cols} : index
  %wave_col_value = index.mul %wave_col_group, %wave_cols : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - wave_cols})] : index
  %row_origin = index.mul %row_group, %c256 : index
  %out_origin = index.mul %out_group, %tn : index
  %blocks = index.div %k, %c{f["block_values"]} : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %stage_bytes = index.constant {2 * STAGE} : offset
  %stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %wtl = encoding.layout.strided [1, {XS}] : encoding<layout>''')
    for b in range(2):
        A(f'''  %x_off{b} = index.constant {b * STAGE} : offset
  %w_off{b} = index.constant {b * STAGE + XB} : offset
  %xs{b} = buffer.view %stage[%x_off{b}] : buffer -> view<256x{XS}xf16>
  %ws{b} = buffer.view %stage[%w_off{b}] : buffer -> view<{TW}x{XS}xf16>
  %wt{b} = buffer.view %stage[%w_off{b}] : buffer -> view<{XS}x{TW}xf16, %wtl>''')
    # per-thread staging roles: x row = thread, weight row = thread / 2 (+ 16-value half = thread % 2)
    A(f'''  %x_row_value = index.add %row_origin, %workitem : index
  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index
  %stage_row = index.div %workitem, %c2 : index
  %stage_ks = index.rem %workitem, %c2 : index
  %stage_col = index.mul %stage_ks, %c16 : index''')
    if two_w:
        A(f'''  %half_tw = index.constant {TN} : index
  %w_is_up = index.cmp uge, %stage_row, %half_tw : index
  %w_local_up = index.sub %stage_row, %half_tw : index
  %w_local = scf.select %w_is_up, %w_local_up, %stage_row : index
  %w_out_row_value = index.add %out_origin, %w_local : index
  %wsel = scf.select %w_is_up, %w2, %w : buffer''')
    else:
        A("  %w_out_row_value = index.add %out_origin, %stage_row : index")
    A(f'''  %w_out_row = index.assume %w_out_row_value [lt(%w_out_row_value, %n)] : index
  %w_row_origin = index.scale %w_out_row, %row_bytes : index, offset -> offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index
  %k32 = index.div %k, %c32 : index
  %k_last = index.sub %k, %c32 : index''')

    # ---- raw loads / decode of one 32-value K step (global -> registers) ----
    def loads(tag, kvar, xidx=None, fallback=None):
        """loads of the K step starting at value index kvar: -> (text, [(name, type)]). xidx/fallback (loop): the x row
        load is guarded by the step index (the verifier proves its bound inside the guard) and yields fallback past the end"""
        _l, addr, lds = G._split_dequant(fmt, wsrc, f"dq{tag}", f"%wcol_{tag}")
        t = [f"  %wcol_{tag} = index.add {kvar}, %stage_col : index", "\n".join(addr)]
        if xidx is None:
            t.append(f"  %xr_{tag} = vector.load %x_view[%x_row, {kvar}] : view<[%rows]x[%k]xf16> -> vector<32xf16>")
        else:
            t.append(f"""  %x_blk_{tag} = index.add %step, %c1 : index
  %x_more_{tag} = index.cmp ult, %x_blk_{tag}, %k_steps : index
  %xr_{tag} = scf.if %x_more_{tag} -> (vector<32xf16>) {{
    %x_ia_{tag} = index.assume %x_blk_{tag} [lt(%x_blk_{tag}, %k_steps)] : index
    %x_ko_{tag} = index.mul %x_ia_{tag}, %c64 : index
    %x_col_{tag} = index.add %x_ko_{tag}, {xidx} : index
    %xv_{tag} = vector.load %x_view[%x_row, %x_col_{tag}] : view<[%rows]x[%k]xf16> -> vector<32xf16>
    scf.yield %xv_{tag} : vector<32xf16>
  }} else {{
    scf.yield {fallback} : vector<32xf16>
  }}""")
        return "\n".join(t), [(n, ty) for n, ty in lds] + [(f"%xr_{tag}", "vector<32xf16>")]
    def decode(tag, kvar, carried):
        """-> text defining %dq{tag} (vector<16xf16>) from the carried raw values (names in the order of loads())"""
        lines, addr, lds = G._split_dequant(fmt, wsrc, f"dq{tag}", f"%wcol_{tag}")
        body = "\n".join(l for l in lines if ".load " not in l)
        for (nm, ty), cr in zip(lds, carried):
            body = re.sub(re.escape(nm) + r"\b", cr, body)
        return f"  %wcol_{tag} = index.add {kvar}, %stage_col : index\n{body}"
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(RF) for c in range(cfr)]
    rowv = ["%wave_row"] + [f"%wave_row{r}" for r in range(1, RF)]
    # prologue: step 0 -> tile 0, raw step 1 -> carried
    txt, lds0 = loads("p0", "%c0")
    A(txt)
    A(decode("pd", "%c0", [n for n, _ in lds0[:-1]]))
    A(f"  vector.store %{'dqpd'}, %ws0[%stage_row, %stage_col] : vector<16xf16>, view<{TW}x{XS}xf16>\n  vector.store %xr_p0, %xs0[%workitem, %c0] : vector<32xf16>, view<256x{XS}xf16>")
    A("  %k_one = index.min %c32, %k_last : index")
    txt, lds1 = loads("p1", "%k_one")
    A(txt)
    A("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    ctys = [ty for _, ty in lds1]
    cn = len(ctys)
    A("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(cn)) + " = scf.for %step = [%c0 to %k_steps step %c1](" +
      ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " + ", ".join(f"%cr{i} = {lds1[i][0]} : {ctys[i]}" for i in range(cn)) + ") -> (" +
      ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(ctys) + ") {")
    A("    %k_origin = index.mul %step, %c64 : index")

    def mma_phase(buf, cur):
        out = []
        for kk in range(2):
            out.append(f"    %fk{kk}_{buf} = index.constant {kk * 16} : index")
            for r in range(RF):
                out.append(f"    %l{r}_{kk}_{buf} = vector.fragment.load<lhs> %xs{buf}[{rowv[r]}, %fk{kk}_{buf}] shape [%c16, %c16] : view<256x{XS}xf16> -> vector<8xf16>")
            for wi in range(len(weights)):
                for c in range(cfr):
                    out.append(f'''    %wn{wi}{c}_{kk}_{buf}_c = index.constant {wi * TN + c * 16} : index
    %wn{wi}{c}_{kk}_{buf} = index.add %wave_col, %wn{wi}{c}_{kk}_{buf}_c : index
    %rf{wi}{c}_{kk}_{buf} = vector.fragment.load<rhs> %wt{buf}[%fk{kk}_{buf}, %wn{wi}{c}_{kk}_{buf}] shape [%c16, %c16] : view<{XS}x{TW}xf16, %wtl> -> vector<8xf16>''')
            for wi in range(len(weights)):
                for c in range(cfr):
                    for r in range(RF):
                        a = f"{wi}r{r}c{c}"
                        nm = f"%m{a}_{kk}_{buf}"
                        out.append(f"    {nm} = vector.mma %l{r}_{kk}_{buf}, %rf{wi}{c}_{kk}_{buf}, {cur[a]} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
                        cur[a] = nm
        return "\n".join(out)

    cur = {a: f"%i{a}" for a in accs}
    A("    %c96 = index.constant 96 : index")
    nlds_prev = None
    for buf in range(2):
        # buf 0: compute step 2j from tile 0, decode step 2j+1 (carried raw) into tile 1, issue the raw loads of step 2j+2;
        # buf 1: compute step 2j+1 from tile 1, decode step 2j+2 into tile 0, issue the raw loads of step 2j+3.
        tg = "ab"[buf]
        dk = "%c32" if buf == 0 else "%c64"
        lk = "%c64" if buf == 0 else "%c96"
        A(f"    %k_{tg}d_raw = index.add %k_origin, {dk} : index\n    %k_{tg}d = index.min %k_{tg}d_raw, %k_last : index")
        A(f"    %k_{tg}l_raw = index.add %k_origin, {lk} : index\n    %k_{tg}l = index.min %k_{tg}l_raw, %k_last : index")
        fb = f"%cr{cn - 1}" if buf == 0 else nlds_prev[-1][0]
        txt, nlds = loads(f"{tg}n", f"%k_{tg}l", "%c0" if buf == 0 else "%c32", fb)
        A("\n".join("  " + l if l else l for l in txt.split("\n")))
        carried_in = [f"%cr{i}" for i in range(cn)] if buf == 0 else [n for n, _ in nlds_prev]
        A("\n".join("  " + l if l else l for l in decode(f"{tg}d", f"%k_{tg}d", carried_in).split("\n")))
        A(mma_phase(buf, cur))
        dst = 1 - buf
        A(f"    vector.store %dq{tg}d, %ws{dst}[%stage_row, %stage_col] : vector<16xf16>, view<{TW}x{XS}xf16>\n    vector.store {carried_in[-1]}, %xs{dst}[%workitem, %c0] : vector<32xf16>, view<256x{XS}xf16>")
        A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        nlds_prev = nlds
    A("    scf.yield " + ", ".join(cur[a] for a in accs) + ", " + ", ".join(n for n, _ in nlds_prev) + " : " + ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(ctys) + "\n  }")
    A(f'''  %inv_view = buffer.view %stage[%origin] : buffer -> view<256x16xf32>
  %inv_row_value = index.add %row_origin, %workitem : index
  %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
  %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
  %inverse16 = vector.splat %inverse : vector<16xf32>
  vector.store %inverse16, %inv_view[%workitem, %c0] : vector<16xf32>, view<256x16xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)''')
    for r in range(RF):
        A(f"  %inv{r} = vector.fragment.load<result> %inv_view[{rowv[r]}, %c0] shape [%c16, %c16] : view<256x16xf32> -> vector<8xf32>")
    A(f'''  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<256x{TN}xf32, %y_layout>''')
    for r in range(RF):
        for c in range(cfr):
            row = rowv[r]
            A(f"  %oc{r}{c}_c = index.constant {c * 16} : index\n  %oc{r}{c} = index.add %wave_col, %oc{r}{c}_c : index")
            if two_w:
                A(f'''  %g{r}{c} = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>
  %u{r}{c} = vector.mulf %o1r{r}c{c}, %inv{r} : vector<8xf32>
  %sg{r}{c} = vector.siluf<afn> %g{r}{c} : vector<8xf32>
  %v{r}{c} = vector.mulf %sg{r}{c}, %u{r}{c} : vector<8xf32>''')
            else:
                A(f"  %v{r}{c}_s = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>\n  %v{r}{c} = vector.addf %v{r}{c}_s, %zero8 : vector<8xf32>")
            A(f"  vector.fragment.store<result> %v{r}{c}, %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : vector<8xf32>, view<256x{TN}xf32, %y_layout>")
    A("  kernel.return\n}\n")
    return "\n".join(l for l in L if l != "")


_M256 = re.compile(r'^kernel\.def export\("(\w+)_gemm_(\w+?)_(store|swiglu)_m256"\) @', re.M)


def _skip_padding(text):
    """gen_gemm.py's GEMM_SKIP_PADDING transform (a row tile whose rows all have scale 0 does no work) for a 256-row kernel."""
    anchor = "  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>\n"
    assert anchor in text and "%row_origin = index.mul %row_group, %c256 : index" in text
    check = anchor + """  %skip_tid0 = kernel.workitem.id<x> : index
  %skip_tid = index.assume %skip_tid0 [range(%skip_tid0, 0, 511)] : index
  %skip_tm = index.constant 256 : index
  %skip_in = index.cmp ult, %skip_tid, %skip_tm : index
  %skip_row = index.add %row_origin, %skip_tid : index
  %skip_rc = scf.select %skip_in, %skip_row, %row_origin : index
  %tile_row = index.assume %skip_rc [lt(%skip_rc, %rows)] : index
  %tile_scale = view.load %s_view[%tile_row, %c0] : view<[%rows]x2xf32> -> f32
  %tile_abs = scalar.absf %tile_scale : f32
  %tile_any = kernel.workgroup.reduce<maxnumf> %tile_abs : f32
  %no_scale = scalar.constant 0.0 : f32
  %tile_live = scalar.cmpf one, %tile_any, %no_scale : f32
  scf.if %tile_live {
"""
    text = text.replace(anchor, check, 1)
    tail = "  kernel.return\n}\n"
    assert text.endswith(tail)
    return text[: -len(tail)] + "  }\n" + tail


def native_twin(k):
    """The native double-buffered twin of a gfx11 `g_gemm_<fmt>_<store|swiglu>_m256` kernel text, or None (any other
    kernel, RDNA4_GEMM_PIPE=0, or a kernel the padding-skip transform changed: those keep the text twin)."""
    m = _M256.match(k)
    if not m or not pipe_enabled():
        return None
    import gen_gemm as G
    if m.group(1) != G.PREFIX or m.group(2) in G.HELPER:
        return None
    t = native_m256(m.group(2), m.group(3))
    if "%tile_live" in k:   # GEMM_SKIP_PADDING (on in generated programs): the same live-row-tile guard as the gfx11 kernel
        t = _skip_padding(t)
    if "workgroups(%out_tiles, %row_tiles, %c1)" in k:   # GEMM_N_FASTEST: the same grid swap as the gfx11 kernel
        t = t.replace("workgroups(%row_tiles, %out_tiles, %c1)", "workgroups(%out_tiles, %row_tiles, %c1)")
        t = t.replace("  %row_group_id = kernel.workgroup.id<x> : index\n  %out_group_id = kernel.workgroup.id<y> : index",
                      "  %row_group_id = kernel.workgroup.id<y> : index\n  %out_group_id = kernel.workgroup.id<x> : index")
    return t
