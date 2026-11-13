"""RDNA4 (gfx12) twin of a flash2 kernel text (used by gen_flash2.py and gen_flash2s.py with FLASH2_ARCH=rdna4).

The gfx11 kernels assume RDNA3's WMMA layouts: the 16x16 f32 result holds, in lane l, column l % 16 and rows
2i + l / 16 (element i), and the f16 operands are vector<16xf16> (a lane holds all 16 k, both lane halves alike).
gfx12 holds rows i + 8 * (l / 16) in the result and vector<8xf16> operands whose lane l holds k = 8 * (l / 16) + j.
The result layout of S^T (rows = keys) therefore IS the rhs layout of P^T (k = keys): the softmax weights of a lane
are its P^T fragment as they are, so the xor-16 shuffles and the even/odd interleave of the gfx11 kernel disappear.
Row reductions stay in-lane plus one xor-16 shuffle (the lane pair of a column), the key positions of a lane's
elements are 8 * half + i, and a lane's V^T keys (the masked path's zeroing) are 8 * half + j.
"""
import re


def to_rdna4(text):
    n = {}

    def sub(pat, rep, flags=re.M):
        nonlocal text
        text, c = re.subn(pat, rep, text, flags=flags)
        n[pat] = c
        return c

    # lane constants, right after the lane's half is known
    sub(r"^(  %lane_half = index\.div %lane, %c16 : index\n)",
        r"\1  %r4_c8 = index.constant 8 : index\n  %lane_h8_v = index.mul %lane_half, %r4_c8 : index\n"
        r"  %lane_h8 = index.assume %lane_h8_v [range(%lane_h8_v, 0, 8)] : index\n")
    # Q^T rhs: this lane's 8 of the 16 k (dims)
    sub(r"(%q_view\[%q_token, %head, %qc\d+, )%c0\]", r"\1%lane_h8]")
    text = re.sub(r"^( *)(%qraw(\d+) = vector\.load %q_view\[%q_token, %head, )(%qd\d+)\]", lambda m: (
        f"{m.group(1)}%qdh{m.group(3)}_v = index.add {m.group(4)}, %lane_h8 : index\n"
        f"{m.group(1)}%qdh{m.group(3)} = index.assume %qdh{m.group(3)}_v [range(%qdh{m.group(3)}_v, 0, 8191)] : index\n"
        f"{m.group(1)}{m.group(2)}%qdh{m.group(3)}]"), text, flags=re.M)
    # key rows of a lane's S elements: 8 * half + i
    sub(r"^  %row_off = vector\.iota %half32 step %two_i32 : vector<8xi32>\n",
        "  %lane_h8_i32 = index.cast %lane_h8 : index to i32\n  %one_row_i32 = scalar.constant 1 : i32\n"
        "  %row_off = vector.iota %lane_h8_i32 step %one_row_i32 : vector<8xi32>\n")
    # P^T: the softmax weights are the rhs fragment as they are
    sub(r"^ *%pi(\d+)_0 = vector\.extract .*?^ *%pt\1_f = vector\.interleave.*?\n", "", re.M | re.S)
    sub(r"(%pt(\d+)_h = vector\.fptrunc )%pt\2_f : vector<16xf32>", r"\1%p\2 : vector<8xf32>")
    # masked path: a lane's V^T keys are 8 * half + j
    sub(r"%iota16 = vector\.iota %c0_i32 step %c1_i32m : vector<16xi32>",
        "%iota16 = vector.iota %lane_h8_i32 step %c1_i32m : vector<8xi32>")
    for t in ("f16", "f32", "i32", "i1"):
        sub(rf"vector<16x{t}>", f"vector<8x{t}>")
    # partial tile epilogue: element i of this lane is row i + 8 * half
    sub(r"(%col_base = index\.add %\w+, )%lane_half : index", r"\1%lane_h8 : index")
    text = re.sub(r"(%dc(\d+)_(\d+) = index\.constant )(\d+)( : index)",
                  lambda m: f"{m.group(1)}{int(m.group(2)) * 16 + int(m.group(3))}{m.group(5)}", text)
    assert n[r"^(  %lane_half = index\.div %lane, %c16 : index\n)"] == 1 and "%pi0_0" not in text and "interleave" not in text, n
    return text


def twin_names(text):
    """kernel.def export("K") @K( -> kernel.def target(@rdna4) export("K_rdna4") @K_rdna4(  (same launch signature)."""
    return re.sub(r'kernel\.def export\("(\w+)"\) @(\w+)\(', r'kernel.def target(@rdna4) export("\1_rdna4") @\2_rdna4(', text)
