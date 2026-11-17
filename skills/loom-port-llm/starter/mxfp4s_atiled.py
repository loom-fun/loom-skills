"""Fragment-contiguous FP8 activations, opt-in for the gfx12 MXFP4 GEMMs.

Design inspiration: GGZ14/vllm-mxfp4 d6162425, radiance_mxfp4_fp8.hip's
A-tiled prefill discussion. This implementation transforms Loom's own producers
and retains its original ascending-K WMMA order and arithmetic. No source copied.
"""
import os
import re

from flash2_rdna4 import twin_env


def enabled():
    # Keep the base literal default visible to the choice inventory scanner;
    # these gfx12-only kernels are emitted inside the shared generator process.
    env = dict(os.environ, GEMM_MXFP4S_A_TILED=os.environ.get("GEMM_MXFP4S_A_TILED", "0"))
    return twin_env(env).get("GEMM_MXFP4S_A_TILED", "0").lower() not in ("", "0", "false", "no", "off")


def offset(row, col, width):
    """Physical byte offset, also used by independent CPU layout tests."""
    return ((row // 16) * (width // 16) + col // 16) * 256 + ((col % 16) // 8) * 128 + (row % 16) * 8 + col % 8


def _address(prefix, row, col, width, nbytes):
    """A contiguous 4/8-byte write within one row's half fragment."""
    return f"""    %{prefix}_c16 = index.constant 16 : index
    %{prefix}_c8 = index.constant 8 : index
    %{prefix}_c256 = index.constant 256 : index
    %{prefix}_c128 = index.constant 128 : index
    %{prefix}_rt = index.div {row}, %{prefix}_c16 : index
    %{prefix}_ri = index.rem {row}, %{prefix}_c16 : index
    %{prefix}_kt = index.div {col}, %{prefix}_c16 : index
    %{prefix}_ki = index.rem {col}, %{prefix}_c16 : index
    %{prefix}_half = index.div %{prefix}_ki, %{prefix}_c8 : index
    %{prefix}_ki8 = index.rem %{prefix}_ki, %{prefix}_c8 : index
    %{prefix}_nk = index.div {width}, %{prefix}_c16 : index
    %{prefix}_frag = index.madd %{prefix}_rt, %{prefix}_nk, %{prefix}_kt : index
    %{prefix}_base = index.mul %{prefix}_frag, %{prefix}_c256 : index
    %{prefix}_hb = index.madd %{prefix}_half, %{prefix}_c128, %{prefix}_base : index
    %{prefix}_rb = index.madd %{prefix}_ri, %{prefix}_c8, %{prefix}_hb : index
    %{prefix}_byte = index.add %{prefix}_rb, %{prefix}_ki8 : index
    %{prefix}_one = index.constant 1 : offset
    %{prefix}_off = index.scale %{prefix}_byte, %{prefix}_one : index, offset -> offset
    %{prefix}_view = buffer.view %x8[%{prefix}_off] : buffer -> view<{nbytes}xf8E4M3>
    %{prefix}_zero = index.constant 0 : index
"""


def producer(text, row_scale):
    """Change only fp8 stores. Scales, norm reductions, fp8 rounding stay intact."""
    row, rows = ("%row", "%rows") if row_scale else ("%token", "%tokens")
    # Small/tail epochs have no tiled consumer. In particular, writing a 16-row
    # fragment layout into a one-row allocation would overrun it; don't write it.
    guard = f"""  %at_guard_c16 = index.constant 16 : index
  %at_guard_zero = index.constant 0 : index
  %at_rem = index.rem {rows}, %at_guard_c16 : index
  %at_on = index.cmp eq, %at_rem, %at_guard_zero : index
"""
    # The first kernel body declaration is a safe dominating insertion point:
    # rows/tokens is the assumed launch extent, defined before x8 conversion.
    anchor = re.search(rf"^  {re.escape(rows)} = index.assume[^\n]*\n", text, re.M)
    assert anchor, rows
    text = text[:anchor.end()] + guard + text[anchor.end():]
    if row_scale:
        pattern = r"    vector.store %q8, %x8_view\[%row, %bc, %c0\] : vector<4xf8E4M3>, view<\[%rows\]x\[%k4\]x4xf8E4M3>\n"
        code = "    scf.if %at_on {\n    %at_k4 = index.constant 4 : index\n    %at_col = index.mul %bc, %at_k4 : index\n    %at_width = index.mul %k4, %at_k4 : index\n"
        code += _address("at", row, "%at_col", "%at_width", 4)
        code += "    vector.store %q8, %at_view[%at_zero] : vector<4xf8E4M3>, view<4xf8E4M3>\n    }\n"
    else:
        pattern = r"    vector.store %h8v, %h8_view\[%token, %bl16, %c0\] : vector<32xf8E4M3>, view<\[%tokens\]x(\d+)x32xf8E4M3>\n"
        m = re.search(pattern, text)
        assert m
        width = int(m.group(1)) * 32
        code = f"    scf.if %at_on {{\n    %at_width = index.constant {width} : index\n    %at_c32 = index.constant 32 : index\n    %at_col0 = index.mul %bl16, %at_c32 : index\n"
        for j in range(4):
            p = f"at{j}"
            code += f"    %{p}_delta = index.constant {8*j} : index\n    %{p}_col = index.add %at_col0, %{p}_delta : index\n"
            code += _address(p, row, f"%{p}_col", "%at_width", 8)
            code += f"    %{p}_v = vector.slice %h8v[{8*j}] : vector<32xf8E4M3> -> vector<8xf8E4M3>\n    vector.store %{p}_v, %{p}_view[%{p}_zero] : vector<8xf8E4M3>, view<8xf8E4M3>\n"
        code += "    }\n"
    text, n = re.subn(pattern, lambda _: code, text)
    assert n == 1, n
    return text


def gemm(text):
    """Use one coalesced 8-byte load per original A fragment, same lane bytes."""
    # The original fragment lhs mapping is row=lane%16, K=lane//16*8.
    # With a packed fragment its entire wave is one contiguous 256-byte read.
    anchor = "  %c16 = index.constant 16 : index\n"
    setup = """  %at_lane = kernel.subgroup.lane.id : index
  %at_c256 = index.constant 256 : index
  %at_eight = index.constant 8 : index
  %at_lane8 = index.mul %at_lane, %at_eight : index
  %at_one = index.constant 1 : offset
"""
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + setup, 1)
    pat = re.compile(r"    %(l\d+_\d+) = vector.fragment.load<lhs> %x_g8\[(%xrow\d+), (%kg\d+)\] shape \[%c16, %c16\] : view<\[%rows\]x\[%k\]xf8E4M3, %a_storage> -> vector<2xi32>\n")
    def load(m):
        name, row, col = m.groups()
        return f"""    %{name}_rt = index.div {row}, %c16 : index
    %{name}_kt = index.div {col}, %c16 : index
    %{name}_nk = index.div %k, %c16 : index
    %{name}_frag = index.madd %{name}_rt, %{name}_nk, %{name}_kt : index
    %{name}_b = index.madd %{name}_frag, %at_c256, %at_lane8 : index
    %{name}_off = index.scale %{name}_b, %at_one : index, offset -> offset
    %{name}_view = buffer.view %x[%{name}_off] : buffer -> view<8xi8>
    %{name}_bytes = vector.load %{name}_view[%c0] : view<8xi8> -> vector<8xi8>
    %{name}_bits = vector.bitcast %{name}_bytes : vector<8xi8> to vector<2xi32>
    %{name} = vector.fragment<lhs> %{name}_bits shape [%c16, %c16] using {{schema = %a_schema : encoding<schema>}} : vector<2xi32>
"""
    text, n = pat.subn(load, text)
    assert n, "no original A fragments replaced"
    return text
