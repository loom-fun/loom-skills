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


def f16out(twin_text):
    """The `<K>_f16o_rdna4` kernel of a swiglu twin `<K>_rdna4`: the f32 result is stored as f16 (an f32 fragment stored into an f16
    view narrows; no row scale: the down GEMM that reads it takes scale 1) and the kernel writes the scales its consumer reads:
    scales_out[row] = (1, 1) for a live row (input scale != 0), (0, 0) for a dead row, from the tile's n-tile 0. It replaces
    `<K>` + g_row_scale_f16[_live] of the f32 result (dual_target group, program.py): no f32 round trip and no second pass."""
    old = re.search(r'kernel\.def target\(@rdna4\) export\("(\w+)"\)', twin_text).group(1)
    assert old.endswith("_rdna4") and "swiglu" in old, old
    new = old[:-len("_rdna4")] + "_f16o_rdna4"
    t = twin_text.replace(f'export("{old}") @{old}(', f'export("{new}") @{new}(', 1)
    assert t.count(", %y: buffer) {") == 1
    t = t.replace(", %y: buffer) {", ", %y: buffer, %scales_out: buffer) {", 1)
    assert t.count("%tile_offset = index.scale %tile_first, %four : index, offset -> offset") == 1
    t = t.replace("%tile_offset = index.scale %tile_first, %four : index, offset -> offset", "%tile_offset = index.scale %tile_first, %two : index, offset -> offset")
    t, n1 = re.subn(r"(view<\d+x\d+x)f32(, %y_layout>)", r"\1f16\2", t)
    assert n1 >= 3, n1
    live = "  %tile_live = scalar.cmpf one, %tile_any, %no_scale : f32\n"
    assert t.count(live) == 1, "f16out needs GEMM_SKIP_PADDING=1 (gen_llm's default): the dead-row scales are written where the tile skip is decided"
    t = t.replace(live, live + """  %so_view = buffer.view %scales_out[%origin] : buffer -> view<[%rows]x2xf32>
  %so_n0 = index.cmp eq, %out_group, %c0 : index
  %so_act = scalar.andi %so_n0, %skip_in : i1
  scf.if %so_act {
    %so_one = scalar.constant 1.0 : f32
    %so_live = scalar.cmpf one, %tile_scale, %no_scale : f32
    %so_val = scf.select %so_live, %so_one, %no_scale : f32
    view.store %so_val, %so_view[%tile_row, %c0] : f32, view<[%rows]x2xf32>
    view.store %so_val, %so_view[%tile_row, %c1] : f32, view<[%rows]x2xf32>
  }
""")
    return t


F16O = os.environ.get("GEMM_F16O", "")   # "swiglu": also emit the f16-output swiglu twins (used through f16o_fuse.py's groups)


def f16o_swiglu(cfg=None):
    """Whether this build fuses the swiglu GEMM with its row scale on gfx12: env GEMM_F16O=swiglu, or the model's llm_config key
    `rdna4_f16_ffn` (GEMM_F16O=0 turns either off). Always False with LOOM_RDNA4=0."""
    e = os.environ.get("GEMM_F16O", "")
    if not enabled() or e.lower() in ("0", "off", "no", "false"):
        return False
    return "swiglu" in e or bool((cfg or {}).get("rdna4_f16_ffn"))


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
                tw = twin(k)
                out.append(tw.rstrip("\n") + "\n\n")
                if os.environ.get("GEMM_F16O_ON") == "swiglu" and "swiglu" in k.split("\n", 1)[0]:
                    out.append(f16out(tw).rstrip("\n") + "\n\n")
    return "".join(out)
