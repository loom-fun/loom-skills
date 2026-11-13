"""Probe package: the WMMA kernels get target(@rdna3); each launch of one goes through template.apply with an rdna3
provider (the original kernel) and an rdna4 provider (a gfx12-tagged stub with the same launch config and an empty body).
usage: make_probe.py IN/llm.loom OUT/llm.loom [stub|none]"""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "stub"
WMMA = re.compile(r"^(g_gemm_q\d_k_(store|swiglu)(_m128|_m256)?|g_attention_flash2(_partial)?)$")
lines = open(src).read().split("\n")
out, stubs, kernels = [], [], set()
i = 0
while i < len(lines):
    l = lines[i]
    m = re.match(r'kernel\.def export\("([^"]+)"\) @(\S+?)\(', l)
    if m and WMMA.match(m.group(1)):
        k = m.group(1); kernels.add(k)
        block = [l]; i += 1
        while not (lines[i] == "}" ):
            block.append(lines[i]); i += 1
        block.append(lines[i]); i += 1
        out.append(block[0].replace("kernel.def export(", "kernel.def target(@rdna3) export(", 1))
        out.extend(block[1:])
        # stub: same header (launch config) and launch signature, empty body
        li = next(j for j, b in enumerate(block) if b.startswith("} launch("))
        st = [block[0].replace(f'kernel.def export("{k}") @{k}(', f'kernel.def target(@rdna4) export("{k}_rdna4") @{k}_rdna4(', 1)]
        st += block[1:li + 1] + ["  kernel.return", "}"]
        stubs.extend(st + [""])
        continue
    out.append(l); i += 1

def split_top(s):
    parts, d, cur = [], 0, ""
    for ch in s:
        if ch in "<[(": d += 1
        if ch in ">])": d -= 1
        if ch == "," and d == 0:
            parts.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip(): parts.append(cur.strip())
    return parts

decls, sigs, n = [], {}, 0
res = []
LAUNCH = re.compile(r"^(\s*)kernel\.launch @(\S+?)\[([^\]]*)\]\((.*)\) : \[([^\]]*)\]\((.*)\)\s*$")
for l in out:
    m = LAUNCH.match(l)
    if m and m.group(2) in kernels:
        ind, k, grid, args, gty, aty = m.groups()
        g, a, gt, at = split_top(grid), split_top(args), split_top(gty), split_top(aty)
        key = (k, tuple(gt), tuple(at))
        if key not in sigs:
            n += 1; name = f"sel{n}_{k}"; sigs[key] = name
            ps = [f"%g{j}: {t}" for j, t in enumerate(gt)] + [f"%a{j}: {t}" for j, t in enumerate(at)]
            gv = ", ".join(f"%g{j}" for j in range(len(gt))); av = ", ".join(f"%a{j}" for j in range(len(at)))
            decls.append(f"template.decl @{name}({', '.join(ps)})")
            for tgt, kk in (("rdna3", k), ("rdna4", k + "_rdna4")):
                if tgt == "rdna4" and mode == "none":
                    continue
                decls.append(f"template.def<@{name}> target(@{tgt}) @{name}_{tgt}({', '.join(ps)}) {{")
                decls.append(f"  kernel.launch @{kk}[{gv}]({av}) : [{', '.join(gt)}]({', '.join(at)})")
                decls.append("  template.return\n}")
        res.append(f"{ind}template.apply<@{sigs[key]}>({', '.join(g + a)}) : ({', '.join(gt + at)})")
    else:
        res.append(l)
head = ["target.decl @device", "amdgpu.target<gfx11-generic> @rdna3", "amdgpu.target<gfx12-generic> @rdna4", ""]
body = "\n".join(res)
# kernels and templates must precede the command program: insert before it
ci = body.index("command.program.def")
body = body.replace("command.program.def public @", "command.program.def public target(@device) @")
ci = body.index("command.program.def")
body = body[:ci] + ("\n".join(stubs) if mode == "stub" else "") + "\n" + "\n".join(decls) + "\n\n" + body[ci:]
open(dst, "w").write("\n".join(head) + "\n" + body)
print(f"{len(kernels)} kernels, {len(sigs)} launch signatures, {sum(1 for l in res if 'template.apply' in l)} launch sites")
