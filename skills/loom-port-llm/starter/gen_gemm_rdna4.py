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
                out.append(twin(k).rstrip("\n") + "\n\n")
    return "".join(out)


# ---- q4_k m256 twin: one 16-byte weight load per thread feeds both nibble groups of its page half -------------------
def q4k_m256_native(t, unroll=False):
    """Rewrites the dequant staging of a q4_k m256 twin text t (store or swiglu): a thread stages row t/2, 16-value
    half h = t%2 of BOTH groups of the step's 32-byte page (low nibbles = group 2p, high nibbles = 2p+1), so it loads
    the page half once (the mechanical twin loads 32 bytes and drops half) and decodes each group's scale/minimum once
    (it decoded the same group twice). Same arithmetic per value, so the result is bit-identical to the mechanical twin."""
    W = re.search(r"%dqap_header_view = buffer\.view (%\w+)\[", t).group(1)

    def loads(p, step):   # raw loads for weight step `step` (an index): superblock header + this thread's page half
        return f"""    {p}_sb = index.div {step}, %c4 : index
    {p}_page = index.rem {step}, %c4 : index
    {p}_bo_add = index.scale {p}_sb, %block_bytes : index, offset -> offset
    {p}_bo = index.add %w_row_origin, {p}_bo_add : offset
    {p}_hv = buffer.view {W}[{p}_bo] : buffer -> view<4xi32>
    {p}_hdr = vector.load {p}_hv[0] : view<4xi32> -> vector<4xi32>
    {p}_pay = index.add {p}_bo, %sixteen : offset
    {p}_wv = buffer.view {W}[{p}_pay] : buffer -> view<32xi32>
    {p}_pw8 = index.mul {p}_page, %c8 : index
    {p}_pw = index.add {p}_pw8, %pk_hw : index
    {p}_pk = vector.load {p}_wv[{p}_pw] : view<32xi32> -> vector<4xi32>
"""
    # prologue: step 0
    a = t.index("  %wcolap = ")
    b = t.index("    %x_colp = ")
    pro = "  %pk_hw = index.mul %stage_ks, %c4 : index\n  %pk_lo_col = index.mul %stage_ks, %c16 : index\n  %pk_hi_col = index.add %pk_lo_col, %c32 : index\n"
    pro += "  %step0 = index.constant 0 : index\n" + loads("%q0", "%step0").replace("\n    ", "\n  ").replace("    %q", "  %q", 1)
    t = t[:a] + pro + t[b:]
    # loop-carried values
    t = t.replace("%cr_dqa_header = %dqap_header : vector<4xi32>, %cr_dqa_packed = %dqap_packed : vector<4xi32>, %cr_dqb_header = %dqbp_header : vector<4xi32>, %cr_dqb_packed = %dqbp_packed : vector<4xi32>",
                  "%cr_hdr = %q0_hdr : vector<4xi32>, %cr_pk = %q0_pk : vector<4xi32>")
    t = t.replace("%unused_0, %unused_1, %unused_2, %unused_3, %unused_4, %unused_5 = scf.for", "%unused_0, %unused_1, %unused_2, %unused_3 = scf.for")
    t = t.replace("vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<32xf16>, vector<32xf16>) {", "vector<4xi32>, vector<4xi32>, vector<32xf16>, vector<32xf16>) {")
    t = t.replace("%dqan_header, %dqan_packed, %dqbn_header, %dqbn_packed, %x_rawn, %x_rawn1 : ", "%qn_hdr, %qn_pk, %x_rawn, %x_rawn1 : ")
    t = t.replace("vector<8xf32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<32xf16>, vector<32xf16>\n", "vector<8xf32>, vector<4xi32>, vector<4xi32>, vector<32xf16>, vector<32xf16>\n")
    # in-loop: dequantize and stage both slices
    a = t.index("    %wcola = ")
    b = t.index("    kernel.barrier<workgroup>", a)
    cp = "    %g0_page = index.rem %step, %c4 : index\n    %g0_v = index.mul %g0_page, %c2 : index\n    %g0 = index.assume %g0_v [range(%g0_v, 0, 6)] : index\n"
    cp += "    %g1_v = index.add %g0, %c1 : index\n    %g1 = index.assume %g1_v [range(%g1_v, 1, 7)] : index\n"
    cp += "    %s0, %m0 = func.call @g_k_metadata(%cr_hdr, %g0) : (vector<4xi32>, index) -> (f32, f32)\n"
    cp += "    %s1, %m1 = func.call @g_k_metadata(%cr_hdr, %g1) : (vector<4xi32>, index) -> (f32, f32)\n"
    cp += "    %c4_i32 = scalar.constant 4 : i32\n    %sh4 = vector.splat %c4_i32 : vector<4xi32>\n"
    cp += "    %pk_lo = vector.andi %cr_pk, %nibble_mask : vector<4xi32>\n    %pk_hs = vector.shrui %cr_pk, %sh4 : vector<4xi32>\n    %pk_hi = vector.andi %pk_hs, %nibble_mask : vector<4xi32>\n"
    for nm, g in (("lo", 0), ("hi", 1)):
        cp += f"""    %{nm}_bytes = vector.bitcast %pk_{nm} : vector<4xi32> to vector<16xi8>
    %{nm}_codes = vector.uitofp %{nm}_bytes : vector<16xi8> to vector<16xf16>
    %{nm}_scale_h = scalar.fptrunc %s{g} : f32 to f16
    %{nm}_neg_min = scalar.negf %m{g} : f32
    %{nm}_neg_min_h = scalar.fptrunc %{nm}_neg_min : f32 to f16
    %{nm}_scales = vector.splat %{nm}_scale_h : vector<16xf16>
    %{nm}_minima = vector.splat %{nm}_neg_min_h : vector<16xf16>
    %dq_{nm} = vector.fmaf %{nm}_codes, %{nm}_scales, %{nm}_minima : vector<16xf16>
    vector.store %dq_{nm}, %w_stage_view[%stage_row, %pk_{nm}_col] : vector<16xf16>, view<128x72xf16>
"""
    t = t[:a] + cp + t[b:]
    # next-stage raw loads
    a = t.index("    %wcolan = ")
    b = t.index("    %x_more = ")
    nx = "    %qn_s1 = index.add %step, %c1 : index\n    %qn_last = index.sub %k_steps, %c1 : index\n    %qn_step = index.min %qn_s1, %qn_last : index\n" + loads("%qn", "%qn_step")
    t = t[:a] + nx + t[b:]
    return t
