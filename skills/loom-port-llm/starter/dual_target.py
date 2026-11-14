#!/usr/bin/env python3
"""Dual-target post-pass: one program file that runs on RDNA3 (gfx11, Strix Halo) and RDNA4 (gfx12, Radeon R9700).

A generator that has a gfx12 version of a kernel K (WMMA fragments differ between the two) emits it next to K as
`K_rdna4`, with the same launch signature. This pass, applied to a program file, makes the file select between them:

  * header: `target.decl @device`, `amdgpu.target<gfx11-generic> @rdna3`, `amdgpu.target<gfx12-generic> @rdna4`;
  * K gets `target(@rdna3)` (nothing else of its text changes), K_rdna4 gets `target(@rdna4)` if it lacks a target;
  * every `kernel.launch @K[...]` becomes `template.apply<@selN_K>(...)`; one `template.decl` per launch signature, with
    a `template.def<@selN_K> target(@rdna3)` provider that launches K and a `target(@rdna4)` one that launches K_rdna4;
  * every `command.program.def` gets `target(@device)`.

  * a launch GROUP (gen_llm.py / make_kernel_tests.py, only on gfx12-specific fusions): the lines
        // dual-group: rdna4 kernel.launch @K_rdna4[...](...) : [...](...)
        kernel.launch @A[...](...) ...        (the gfx11 launches, in order)
        // dual-group: end
    become one `template.apply` whose @rdna3 provider is the gfx11 launches and whose @rdna4 provider is the single rdna4 launch
    (a fused kernel with no gfx11 counterpart). Operands are deduplicated by SSA name into the template's parameters.

A file without a twin is returned unchanged, as is every file when env LOOM_RDNA4 is "0". Applying the pass to its own
output is a no-op.

usage: dual_target.py FILE.loom [FILE.loom ...]     (rewrites in place; prints one line per file)
       from dual_target import apply; text = apply(text)
"""
import os
import re
import sys

SUFFIX = "_rdna4"
KERNEL_DEF = re.compile(r"^kernel\.def\b(?P<attrs>.*?)\s@(?P<sym>\w+)\(")
LAUNCH = re.compile(r"^(\s*)kernel\.launch @(\w+)\[([^\]]*)\]\((.*)\) : \[([^\]]*)\]\((.*)\)\s*$")
PROGRAM_DEF = re.compile(r"^(command\.program\.def\s+(?:public\s+)?)(?!target\()(@)", re.M)
HEADER = ("target.decl @device\n"
          "amdgpu.target<gfx11-generic> @rdna3\n"
          "amdgpu.target<gfx12-generic> @rdna4\n\n")


def enabled():
    return os.environ.get("LOOM_RDNA4", "1").strip().lower() not in ("0", "off", "no", "false")


def _split_top(s):
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


def twins(text):
    """Symbols K of the kernels that have a `K_rdna4` kernel in TEXT."""
    syms = {m.group("sym") for l in text.split("\n") if (m := KERNEL_DEF.match(l))}
    return sorted(k for k in syms if k + SUFFIX in syms)


GROUP_BEGIN = re.compile(r"^(\s*)// dual-group: rdna4 (kernel\.launch .*)$")
GROUP_END = re.compile(r"^\s*// dual-group: end\s*$")


def _groups(lines, decls, sigs):
    """Lines with every launch group replaced by a template.apply (its template appended to DECLS; SIGS keys the templates)."""
    out, i = [], 0
    while i < len(lines):
        m = GROUP_BEGIN.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        ind, alt = m.groups()
        j = i + 1
        body = []
        while not GROUP_END.match(lines[j]):
            body.append(lines[j])
            j += 1
        launches = []
        for l in body + [ind + alt]:
            lm = LAUNCH.match(l)
            if not lm:
                raise ValueError(f"dual-group: not a launch: {l.strip()}")
            _, k, grid, args, gty, aty = lm.groups()
            launches.append((k, _split_top(grid), _split_top(args), _split_top(gty), _split_top(aty)))
        # operands in first-seen order over the gfx11 launches (grid, then arguments), then the gfx12 one
        names, types, pat = [], [], []
        for k, g, a, gt, at in launches:
            for nm, ty in zip(g + a, gt + at):
                if nm not in names:
                    names.append(nm)
                    types.append(ty)
                pat.append(names.index(nm))
        key = ("group", tuple((k, tuple(gt), tuple(at)) for k, g, a, gt, at in launches), tuple(pat), tuple(types))
        if key not in sigs:
            name = f"grp{len([x for x in sigs.values() if x.startswith('grp')]) + 1}_{launches[-1][0]}"
            sigs[key] = name
            ps = ", ".join(f"%p{n}: {t}" for n, t in enumerate(types))
            decls.append(f"template.decl @{name}({ps})")
            pos = iter(pat)
            lines3 = []
            for k, g, a, gt, at in launches:
                gp = [f"%p{next(pos)}" for _ in g]
                ap = [f"%p{next(pos)}" for _ in a]
                lines3.append(f"  kernel.launch @{k}[{', '.join(gp)}]({', '.join(ap)}) : [{', '.join(gt)}]({', '.join(at)})")
            decls.append(f"template.def<@{name}> target(@rdna3) @{name}_rdna3({ps}) {{")
            decls += lines3[:-1]
            decls.append("  template.return\n}")
            decls.append(f"template.def<@{name}> target(@rdna4) @{name}_rdna4({ps}) {{")
            decls.append(lines3[-1])
            decls.append("  template.return\n}")
        out.append(f"{ind}template.apply<@{sigs[key]}>({', '.join(names)}) : ({', '.join(types)})")
        i = j + 1
    return out


def apply(text, enable=None):
    """TEXT made dual-target; unchanged when it has no twin kernels (or the pass is off)."""
    if enable is None:
        enable = enabled()
    if not enable or "target.decl @device" in text:
        return text
    tw = set(twins(text))
    if not tw:
        return text
    lines = text.split("\n")
    out, last_def_end = [], -1
    i = 0
    while i < len(lines):
        l = lines[i]
        m = KERNEL_DEF.match(l)
        if not m:
            out.append(l)
            i += 1
            continue
        sym = m.group("sym")
        base = sym[:-len(SUFFIX)] if sym.endswith(SUFFIX) and sym[:-len(SUFFIX)] in tw else sym
        if base in tw and "target(" not in m.group("attrs"):
            l = l.replace("kernel.def ", f"kernel.def target(@{'rdna4' if sym != base else 'rdna3'}) ", 1)
        out.append(l)
        i += 1
        while lines[i] != "}":
            out.append(lines[i])
            i += 1
        out.append("}")
        i += 1
        if base in tw or sym.endswith(SUFFIX):
            last_def_end = len(out)
    decls, sigs, res = [], {}, []
    out = _groups(out, decls, sigs)
    for idx, l in enumerate(out):
        m = LAUNCH.match(l)
        if m and m.group(2) in tw:
            if idx < last_def_end:
                raise ValueError(f"launch of {m.group(2)} before the end of its twin kernels: {l.strip()}")
            ind, k, grid, args, gty, aty = m.groups()
            g, a, gt, at = _split_top(grid), _split_top(args), _split_top(gty), _split_top(aty)
            key = (k, tuple(gt), tuple(at))
            if key not in sigs:
                name = f"sel{len(sigs) + 1}_{k}"
                sigs[key] = name
                ps = [f"%g{j}: {t}" for j, t in enumerate(gt)] + [f"%a{j}: {t}" for j, t in enumerate(at)]
                gv = ", ".join(f"%g{j}" for j in range(len(gt)))
                av = ", ".join(f"%a{j}" for j in range(len(at)))
                decls.append(f"template.decl @{name}({', '.join(ps)})")
                for tgt, kk in (("rdna3", k), ("rdna4", k + SUFFIX)):
                    decls.append(f"template.def<@{name}> target(@{tgt}) @{name}_{tgt}({', '.join(ps)}) {{")
                    decls.append(f"  kernel.launch @{kk}[{gv}]({av}) : [{', '.join(gt)}]({', '.join(at)})")
                    decls.append("  template.return\n}")
            res.append(f"{ind}template.apply<@{sigs[key]}>({', '.join(g + a)}) : ({', '.join(gt + at)})")
        else:
            res.append(l)
    # res has one line per line of out: the templates go right after the last twin kernel
    res = res[:last_def_end] + ([""] + decls if decls else []) + res[last_def_end:]
    body = PROGRAM_DEF.sub(r"\1target(@device) \2", "\n".join(res))
    return HEADER + encodings_first(body)


ENC_LIMIT = 60000   # SSA values (counted as below) before an encoding-typed use that hrx-loom 0.1.0 still resolves


ENC_BYTES = 4_000_000   # bytes before an encoding-typed line: hrx-loom 0.1.0 fails past 4 MiB (engineer sweep bisected 4,191,941 ok, 4,281,430 fails)


def _enc_depth(text):
    """The largest count of SSA values defined before a line that defines or uses an encoding in a type (or ENC_LIMIT + 1
    when such a line lies past ENC_BYTES of the file)."""
    c = worst = pos = 0
    for l in text.split("\n"):
        pos += len(l.encode()) + 1
        if pos > ENC_BYTES and ("encoding<layout>" in l or re.search(r"view<[^>]*, %", l)):
            return ENC_LIMIT + 1
        m = re.match(r"\s*((?:%[\w.]+\s*,\s*)*%[\w.]+)\s*=", l)
        if m:
            c += m.group(1).count("%")
        if "encoding<layout>" in l or re.search(r"view<[^>]*, %", l):
            worst = max(worst, c)
    return worst


def encodings_first(text):
    """hrx-loom 0.1.0 resolves an encoding referenced in a type wrongly once the module has defined about 2^16 values before
    it (ENCODING/004 '... references encoding value %x which is not defined at this point', naming an unrelated earlier
    value; seen on Laguna-XS-2.1's dual-target llm.loom, 94k lines). A big file whose encoding-typed kernels sit that deep
    gets its top-level definitions reordered, stably: funcs, then the kernels that use encodings, then the rest (templates
    and command programs stay after them). Kernel texts are unchanged; smaller files are returned as they are."""
    if _enc_depth(text) <= ENC_LIMIT:
        return text
    parts = re.split(r"(?m)^(?=kernel\.def |func\.def |template\.|command\.program\.def )", text)
    head, rest = parts[0], parts[1:]
    first_late = next((i for i, p in enumerate(rest) if p.startswith(("template.", "command.program.def"))), len(rest))
    defs, late = rest[:first_late], rest[first_late:]
    funcs = [p for p in defs if p.startswith("func.def")]
    enc = [p for p in defs if p.startswith("kernel.def") and ("encoding<layout>" in p or re.search(r"view<[^>]*, %", p))]
    other = [p for p in defs if not p.startswith("func.def") and p not in enc]
    return head + "".join(funcs + enc + other + late)


def apply_file(path):
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    new = apply(text)
    if new != text:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)
        return True
    return False


def apply_dir(program_dir):
    """The pass over every *.loom file of a package's program/ directory; the files rewritten. A kernel with a twin
    that a file launches without defining it (the twin is in another file) is an error: the pass only rewrites the
    launches of the file that defines both, and the others would launch the gfx11 kernel on a gfx12 device."""
    files = {n: open(os.path.join(program_dir, n), encoding="utf-8", newline="").read()
             for n in sorted(os.listdir(program_dir)) if n.endswith(".loom")}
    if enabled():
        owner = {k: n for n, t in files.items() for k in twins(t)}
        for n, t in files.items():
            if "target.decl @device" in t:
                continue
            for m in (LAUNCH.match(l) for l in t.split("\n")):
                if m and m.group(2) in owner and owner[m.group(2)] != n:
                    raise ValueError(f"program/{n} launches @{m.group(2)}, whose {SUFFIX} twin is defined in program/{owner[m.group(2)]}: "
                                     "a dual-target kernel must be launched from the file that defines it")
    return [n for n in files if apply_file(os.path.join(program_dir, n))]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        print(p, "rewritten" if apply_file(p) else "unchanged")
