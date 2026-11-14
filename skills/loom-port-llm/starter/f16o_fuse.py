"""gfx12 fusion of a GEGLU/SwiGLU prefill GEMM with the row scale of its result (GEMM_F16O=swiglu, LOOM_RDNA4 on).

The prefill FFN is   swiglu GEMM -> f (f32, [t, ffn])  ->  g_row_scale_f16 (two reads of f, writes f16 + scales)  ->  down GEMM.
On gfx12 the swiglu twin `<K>_f16o_rdna4` (gen_gemm_rdna4.f16out) stores the result as f16 itself and writes the row scales its
consumer reads ((1, 1) for a live row, (0, 0) for a dead one): f never makes the f32 round trip. This pass rewrites
@llm_advance (and the other programs that have the pattern):

    scf.if %mm { scf.if %big { scf.if %huge { K256 } else { K128 } } else { K } } else { mv ... }
    scf.if %mm { g_row_scale_f16[_live][_lv](..., f, s, f16) } else { g_quant_q8 ... }

into the same launches with each GEMM leaf followed by the row-scale launch inside a `dual-group` (dual_target.py): the gfx11
provider launches both, the gfx12 provider launches the f16o twin; the row-scale launch leaves the second `scf.if`.
No f16 clamping beyond the f32 -> f16 conversion: a value above 65504 becomes inf (the row scale only kicks in at 8192).
"""
import re

LEAF = re.compile(r"^(?P<ind>\s*)kernel\.launch @(?P<k>g_gemm_\w*swiglu(?:_m128|_m256)?)\[(?P<grid>[^\]]*)\]\((?P<args>.*)\) : \[(?P<gt>[^\]]*)\]\((?P<at>.*)\)$")
RS = re.compile(r"^(?P<ind>\s*)kernel\.launch @g_row_scale_f16(?:_live)?(?:_lv)?\[%t\]\(%t, (?:%info, )?(?P<n>%?\w+), (?P<y>%\w+), (?P<s>%\w+), (?P<h16>%\w+)\) : .*$")


def _split(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<[(":
            depth += 1
        elif ch in ">])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def fuse(text):
    lines = text.split("\n")
    done = 0
    r = 0
    while r < len(lines):
        m = RS.match(lines[r] or "")
        if not (m and r >= 1 and lines[r - 1].strip() == "scf.if %mm {" and r + 3 < len(lines)
                and lines[r + 1].strip() == "} else {" and "g_quant_q8" in lines[r + 2] and lines[r + 3].strip() == "}"):
            r += 1
            continue
        y, s, f16 = m.group("y"), m.group("s"), m.group("h16")
        # the GEMM leaves of this row scale's input: the launches writing y in the lines before the `scf.if %mm {`
        leaves = []
        i = r - 2
        while i >= 0 and r - i < 80:
            lm = LEAF.match(lines[i] or "")
            if lm:
                args = _split(lm.group("args"))
                if args[-1] == y:
                    leaves.append((i, lm))
            if (lines[i] or "").startswith("command.program.def") or (RS.match(lines[i] or "") and i < r):
                break
            i -= 1
        if not leaves or len(leaves) > 3:
            r += 1
            continue
        for i, lm in leaves:
            args, at = _split(lm.group("args")), _split(lm.group("at"))
            ind, k = lm.group("ind"), lm.group("k")
            alt = (f"{ind}// dual-group: rdna4 kernel.launch @{k}_f16o_rdna4[{lm.group('grid')}]({', '.join(args[:-1] + [f16, s])}) "
                   f": [{lm.group('gt')}]({', '.join(at[:-1] + ['buffer', 'buffer'])})")
            rs = ind + lines[r].strip()
            lines[i] = "\n".join([alt, lines[i], rs, f"{ind}// dual-group: end"])
        lines[r] = None    # the row scale leaves the second scf.if (the then-branch stays empty)
        done += 1
        r += 1
    return "\n".join(ln for ln in lines if ln is not None), done
