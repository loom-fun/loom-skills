"""RDNA4 (gfx12) twins of the starter's gfx11 WMMA kernels, as a text transform of the finished kernel.

gfx11 fragments are vector<16xf16>; gfx12's lhs/rhs fragments are vector<8xf16> (the accumulator is vector<8xf32> on
both). The lhs `vector.fragment.load<lhs>` only changes type. The rhs is a raw per-lane `vector.load` of one row of the
[row, k] staging view followed by `vector.fragment<rhs>`; on gfx12 it becomes `vector.fragment.load<rhs>` from a [1, C]-
strided transposed view of the same buffer bytes (B[k][n] = W[n][k]) at the wave's row base (row - lane % 16).
The epilogues use the layout-neutral fragment.store/load and stay as they are.

twin(text): the kernel `K` of `text` becomes `kernel.def target(@rdna4) export("K_rdna4") @K_rdna4(...)` with the same
launch signature (the dual-target post-pass, dual_target.py, adds target(@rdna3) to K and the template.apply launches).
Env LOOM_RDNA4=0 turns it off (`enabled()`): generators then emit exactly the gfx11 text.
"""
import os
import re


def enabled():
    return os.environ.get("LOOM_RDNA4", "1") not in ("", "0", "false", "no")


_VIEW = re.compile(r"(%\w+) = buffer\.view (%\w+)\[(%\w+)\] : buffer -> view<(\d+)x(\d+)xf16>\n")
_RHS_LOAD = re.compile(r"^(\s*)(%\w+) = vector\.load (%\w+)\[(%\w+), (%\w+)\] : view<(\d+)x(\d+)xf16> -> vector<16xf16>$")
_RHS_FRAG = re.compile(r"^\s*(%\w+) = vector\.fragment<rhs> (%\w+) shape \[%c16, %c16\] : vector<16xf16>$")


def twin(text, name=None):
    """The RDNA4 twin of the (single) kernel in `text`; `name` defaults to its export name."""
    k = name or re.search(r'kernel\.def export\("(\w+)"\) @', text).group(1)
    t = text.replace(f'kernel.def export("{k}") @{k}(', f'kernel.def target(@rdna4) export("{k}_rdna4") @{k}_rdna4(', 1)
    assert t != text, k
    t = re.sub(r"(vector\.fragment\.load<lhs> [^\n]*-> )vector<16xf16>", r"\1vector<8xf16>", t)
    views = {m.group(1): m.groups()[1:] for m in _VIEW.finditer(t)}
    used, out, lines, i = [], [], t.split("\n"), 0
    while i < len(lines):
        m = _RHS_LOAD.match(lines[i])
        m2 = _RHS_FRAG.match(lines[i + 1]) if i + 1 < len(lines) else None
        if m and m2 and m2.group(2) == m.group(2):
            ind, rv, V, row, col, R, C = m.groups()
            if V not in used:
                used.append(V)
            out.append(f"{ind}{rv}_b = index.sub {row}, %g12_l16 : index")
            out.append(f"{ind}{m2.group(1)} = vector.fragment.load<rhs> {V}_t[{col}, {rv}_b] shape [%c16, %c16] : view<{C}x{R}xf16, {V}_tl> -> vector<8xf16>")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    t = "\n".join(out)
    assert used, f"{k}: no rhs fragment to port"
    assert "vector<16xf16>, vector<16xf16>, vector<8xf32>" in t
    t = t.replace(": vector<16xf16>, vector<16xf16>, vector<8xf32>", ": vector<8xf16>, vector<8xf16>, vector<8xf32>")
    for V in used:
        buf, off, R, C = views[V]
        decl = f"  {V} = buffer.view {buf}[{off}] : buffer -> view<{R}x{C}xf16>\n"
        assert decl in t, (k, V)
        t = t.replace(decl, decl + f"  {V}_tl = encoding.layout.strided [1, {C}] : encoding<layout>\n"
                      f"  {V}_t = buffer.view {buf}[{off}] : buffer -> view<{C}x{R}xf16, {V}_tl>\n", 1)
    lane = re.search(r"(%\w+) = kernel\.subgroup\.lane\.id : index\n", t)
    assert lane, k
    t = t.replace(lane.group(0), lane.group(0) + f"  %g12_c16 = index.constant 16 : index\n  %g12_l16 = index.rem {lane.group(1)}, %g12_c16 : index\n", 1)
    return t
