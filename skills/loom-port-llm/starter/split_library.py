#!/usr/bin/env python3
"""Kernel-library split: move the encoding-typed kernels of a deep program library (program/llm.loom) into separate
library files, so no file holds an encoding-typed line deeper than hrx-loom's text parser resolves.

hrx-loom keeps the SSA value id of an encoding referenced in a type in 16 bits, numbered module-wide per parsed file
(topic/sdk.md; dual_target.encodings_first). encodings_first moves every encoding kernel to the front of the file, which
holds while those kernels alone fit (about 60,000 values / 4 MB). A package with more encoding kernels than that
(many formats x tile variants x the gfx12 twins: unsloth UD mixes, five or six GEMM formats) needs a second file.
loom-serve indexes every `libraries` file of the card as its own source and compiles each kernel from the closure of
its root, so a kernel parsed from llm_kernels_N.loom has small value ids; the program file keeps a `kernel.decl` with
the same signature for every moved kernel, which the linker resolves to the exported definition.

The split (choice target.library.split, env LOOM_LIBRARY_SPLIT):
  auto (default)  split only a file encodings_first cannot fit (every package below the limit stays byte-identical);
  always          split every file that has an encoding kernel (tests the path on a small model);
  off             never (a too-deep file raises in dual_target.encodings_first, as before).

Each kernel file: the dual-target header, the `func.def inline` helpers its kernels call (copied; they are private to
each module), then its kernels (a kernel and its `_rdna4` twin stay together), filled up to BUDGET_VALUES / BUDGET_BYTES.

usage: split_library.py PACKAGE_DIR          (rewrites program/llm.loom, writes program/llm_kernels_N.loom, model.json)
       from split_library import split_dir; split_dir(pkg)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dual_target  # noqa: E402

BUDGET_VALUES = 45000     # SSA values per kernel file (ENC_LIMIT 60,000 leaves room for growth of a kernel)
BUDGET_BYTES = 3_000_000  # bytes per kernel file (ENC_BYTES 4,000,000)
PREFIX = "llm_kernels_"
TOP = re.compile(r"(?m)^(?=kernel\.def |kernel\.decl |func\.def |template\.|command\.def |config\.)")
KDEF = re.compile(r"^kernel\.def\b[^\n]*?\s@([\w.]+)\(([^\n]*)\)\s*\{\s*$", re.M)
LAUNCH_HDR = re.compile(r"^\}\s*launch\((.*)\)\s*\{\s*$", re.M)
FUNC = re.compile(r"^func\.def\b[^\n@]*@([\w.]+)\(")


def mode():
    return os.environ.get("LOOM_LIBRARY_SPLIT", "auto").strip().lower()


def _values(text):
    c = 0
    for l in text.split("\n"):
        for m in dual_target._VALUE_DEFS.finditer(l):
            c += int(m.group(1)) if m.group(1) else 1
    return c


def needs_split(text):
    """True when the file has encoding kernels the split mode says to move."""
    m = mode()
    if m in ("off", "0", "no", "false") or not dual_target._ENC_LINE.search(text):
        return False
    if m in ("always", "1", "on", "yes", "true"):
        return True
    with_split_off = os.environ.get("LOOM_LIBRARY_SPLIT")
    os.environ["LOOM_LIBRARY_SPLIT"] = "off"   # encodings_first raises when it cannot fit the file
    try:
        dual_target.encodings_first(text)
        return False
    except ValueError:
        return True
    finally:
        os.environ.pop("LOOM_LIBRARY_SPLIT") if with_split_off is None else os.environ.__setitem__("LOOM_LIBRARY_SPLIT", with_split_off)


def _decl(part):
    """`kernel.decl @K(<workload params>) launch(<launch params>)` of a kernel.def part."""
    h = KDEF.match(part)
    lh = LAUNCH_HDR.search(part)
    if not h or not lh:
        raise ValueError(f"split_library: cannot read the signature of {part.split(chr(10), 1)[0][:120]}")
    return f"kernel.decl @{h.group(1)}({h.group(2)}) launch({lh.group(1)})\n"


def _name(part):
    return KDEF.match(part).group(1)


def split_text(text):
    """(program text with kernel.decl lines, [kernel file texts]) for a deep dual-target library."""
    parts = TOP.split(text)
    head, rest = parts[0], parts[1:]
    funcs = {FUNC.match(p).group(1): p for p in rest if p.startswith("func.def") and FUNC.match(p)}
    moved = [i for i, p in enumerate(rest) if p.startswith("kernel.def") and dual_target._is_enc(p)]
    names = {_name(rest[i]): i for i in moved}
    # a kernel's twin rides with it (the twin is an encoding kernel whenever its base is, but not always the reverse)
    order, seen = [], set()
    for i in moved:
        n = _name(rest[i])
        base = n[:-len(dual_target.SUFFIX)] if n.endswith(dual_target.SUFFIX) else n
        if base in seen:
            continue
        seen.add(base)
        order.append([names[k] for k in (base, base + dual_target.SUFFIX) if k in names])

    def helpers(group_text):   # inline funcs the kernels call, transitively, in file order
        need, todo = set(), set(re.findall(r"func\.call @([\w.]+)", group_text))
        while todo:
            f = todo.pop()
            if f in funcs and f not in need:
                need.add(f)
                todo |= set(re.findall(r"func\.call @([\w.]+)", funcs[f]))
        return [p for n, p in funcs.items() if n in need]

    files, cur, cur_v, cur_b = [], [], 0, 0
    for grp in order:
        gt = "".join(rest[i] for i in grp)
        v, b = _values(gt), len(gt.encode())
        if cur and (cur_v + v > BUDGET_VALUES or cur_b + b > BUDGET_BYTES):
            files.append(cur)
            cur, cur_v, cur_b = [], 0, 0
        cur += grp
        cur_v += v
        cur_b += b
    if cur:
        files.append(cur)
    out_files = []
    for idx in files:
        body = "".join(rest[i] for i in idx)
        fn = helpers(body)
        k = dual_target.HEADER + "// Kernel library split from llm.loom (split_library.py): encoding-typed kernels, launched\n" \
            "// through the kernel.decl lines of llm.loom.\n\n" + "".join(fn) + body
        if dual_target._enc_depth(k) > dual_target.ENC_LIMIT:
            raise ValueError("split_library: one kernel group alone is deeper than the encoding limit")
        out_files.append(k)
    mv = {i for f in files for i in f}
    prog = head + "".join(_decl(p) if i in mv else p for i, p in enumerate(rest))
    if dual_target._enc_depth(prog) > dual_target.ENC_LIMIT:
        prog = dual_target.encodings_first(prog)
    return prog, out_files


def split_dir(pkg, card=None, name="llm.loom"):
    """Split program/<name> of a built package when needs_split; list the kernel files in the card's libraries, ahead of
    program/<name> (a link that materializes a command's closure numbers the kernels first, as in one file after
    encodings_first). card: the card dict to update (else model.json is read and rewritten). The files written."""
    prog_dir = os.path.join(pkg, "program")
    path = os.path.join(prog_dir, name)
    for old in kernel_files(prog_dir):   # a rebuild into the same package starts clean
        os.remove(os.path.join(prog_dir, old))
    text = open(path, encoding="utf-8", newline="").read()
    if not needs_split(text):
        return []
    prog, kfiles = split_text(text)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(prog)
    written = []
    for n, k in enumerate(kfiles, 1):
        kn = f"{PREFIX}{n}.loom"
        with open(os.path.join(prog_dir, kn), "w", encoding="utf-8", newline="\n") as f:
            f.write(k)
        written.append(kn)
    card_path = os.path.join(pkg, "model.json")
    c = card if card is not None else json.load(open(card_path))
    for prog_set in c.get("programs", {}).values():
        libs = prog_set.get("libraries", [])
        if f"program/{name}" in libs:
            at = libs.index(f"program/{name}")
            prog_set["libraries"] = libs[:at] + [f"program/{k}" for k in written] + libs[at:]
    if card is None:
        json.dump(c, open(card_path, "w"), indent=1)
    return written


def library_text(program_dir, name="llm.loom"):
    """The text of program/<name> with the kernel files of a split appended (for tools that read kernel definitions
    by text: kernel tests, A/B of one kernel, profiles). The kernel.decl lines stay; a reader looks for kernel.def."""
    text = open(os.path.join(program_dir, name), encoding="utf-8", newline="").read()
    for k in kernel_files(program_dir):
        text += "\n" + open(os.path.join(program_dir, k), encoding="utf-8", newline="").read()
    return text


def kernel_files(program_dir):
    def key(n):
        return int(re.sub(r"\D", "", n) or 0)
    return sorted((n for n in os.listdir(program_dir) if n.startswith(PREFIX) and n.endswith(".loom")), key=key)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    w = split_dir(sys.argv[1])
    print("split:", ", ".join(w) if w else "not needed")
