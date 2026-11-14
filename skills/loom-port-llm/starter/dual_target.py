#!/usr/bin/env python3
"""Dual-target post-pass: one program file that runs on RDNA3 (gfx11, Strix Halo) and RDNA4 (gfx12, Radeon R9700).

A generator that has a gfx12 version of a kernel K (WMMA fragments differ between the two) emits it next to K as
`K_rdna4`, with the same launch signature. This pass, applied to a program file, makes the file select between them:

  * header: `target.decl @device`, `amdgpu.target<gfx11-generic> @rdna3`, `amdgpu.target<gfx12-generic> @rdna4`;
  * K gets `target(@rdna3)` (nothing else of its text changes), K_rdna4 gets `target(@rdna4)` if it lacks a target;
  * every `kernel.launch @K[...]` becomes `template.apply<@selN_K>(...)`; one `template.decl` per launch signature, with
    a `template.def<@selN_K> target(@rdna3)` provider that launches K and a `target(@rdna4)` one that launches K_rdna4;
  * every `command.program.def` gets `target(@device)`.

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


# hrx-loom 0.1.0 mis-verifies an `encoding.*` value (a strided layout used in a view type) defined beyond byte 4 MiB of
# the program file (ENCODING/004 "not defined at this point", reported on an unrelated value). The twins grow a big
# program past that (Laguna 5.0 -> 6.2 MB), so once a file is big its kernels that define encodings move ahead of the
# others (the kernel slots keep their places; only the order of the kernels changes).
ENCODING_LIMIT = 3 * 1024 * 1024


def hoist_encoding_kernels(text):
    """TEXT with the kernels that use `encoding.` ops first among the kernels (stable), when the file is big enough for
    a definition to land past the 4 MiB mark; otherwise unchanged."""
    if len(text) <= ENCODING_LIMIT:
        return text
    lines = text.split("\n")
    spans, i = [], 0
    while i < len(lines):
        if KERNEL_DEF.match(lines[i]):
            j = i
            while lines[j] != "}":
                j += 1
            spans.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    blocks = [lines[a:b] for a, b in spans]
    order = [b for b in blocks if any(" = encoding." in l for l in b)] + \
            [b for b in blocks if not any(" = encoding." in l for l in b)]
    if order == blocks:
        return text
    out, pos = [], 0
    for (a, b), blk in zip(spans, order):
        out += lines[pos:a] + blk
        pos = b
    return "\n".join(out + lines[pos:])


def apply(text, enable=None):
    """TEXT made dual-target; unchanged when it has no twin kernels (or the pass is off)."""
    if enable is None:
        enable = enabled()
    if not enable or "target.decl @device" in text:
        return text
    tw = set(twins(text))
    if not tw:
        return text
    lines = hoist_encoding_kernels(text).split("\n")
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
        if base in tw:
            last_def_end = len(out)
    decls, sigs, res = [], {}, []
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
    return HEADER + body


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
