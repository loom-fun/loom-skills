#!/usr/bin/env python3
"""Dual-target post-pass: one program file that runs on RDNA3 (gfx11, Strix Halo) and RDNA4 (gfx12, Radeon R9700).

A generator that has a gfx12 version of a kernel K (WMMA fragments differ between the two) emits it next to K as
`K_rdna4`, with the same launch signature. This pass, applied to a program file, makes the file select between them:

  * header: `target.decl @device`, `amdgpu.target<gfx11-generic> @rdna3`, `amdgpu.target<gfx12-generic> @rdna4`;
  * K gets `target(@rdna3)` (nothing else of its text changes), K_rdna4 gets `target(@rdna4)` if it lacks a target;
  * every `kernel.launch @K[...]` becomes `template.apply<@selN_K>(...)`; one `template.decl` per launch signature, with
    a `template.def<@selN_K> target(@rdna3)` provider that launches K and a `target(@rdna4)` one that launches K_rdna4;
  * every `command.def` gets `target(@device)`.

  * a launch GROUP (gen_llm.py / make_kernel_tests.py, only on gfx12-specific fusions): the lines
        // dual-group: rdna4 kernel.launch @K_rdna4[...](...) : [...](...)
        kernel.launch @A[...](...) ...        (the gfx11 launches, in order)
        // dual-group: end
    become one `template.apply` whose @rdna3 provider is the gfx11 launches and whose @rdna4 provider is the single rdna4 launch
    (a fused kernel with no gfx11 counterpart). Operands are deduplicated by SSA name into the template's parameters.

A file without a twin is returned unchanged, as is every file when env LOOM_RDNA4 is "0". Applying the pass to its own
output is a no-op.

Selection level (env LOOM_DUAL_SELECT, choice target.dual.select): "launch" (default) selects per launch as above. "program"
selects once per command program that launches a twin kernel: the program keeps its signature and becomes one
`template.apply<@NAME_body>(<its launch parameters>)` followed by `command.return`; the @rdna3 provider of `@NAME_body` is the
program's original body (plain `kernel.launch @K`, as in a gfx11-only build) and the @rdna4 provider the same body with every
twin launch on `K_rdna4` (a launch group: its single rdna4 launch). Launches outside such programs stay per-launch.

usage: dual_target.py FILE.loom [FILE.loom ...]     (rewrites in place; prints one line per file)
       from dual_target import apply; text = apply(text)
"""
import os
import re
import sys


def _rows():
    """(base target symbol, twin target symbol, twin suffix, header text, preset suffix of the twin family's device, {device: preset suffix}) from the device registry (skills/loom-kb/device/<family>/family.json).
    The registry is found next to the skills (LOOM_SKILLS, else relative to this file); a kit copied outside the skills tree falls back to the
    rows below, which tests/test_registry.py keeps equal to the registry's."""
    try:
        skills = os.environ.get("LOOM_SKILLS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        sys.path.insert(0, os.path.join(skills, "loom-kb"))
        try:
            import registry
        finally:
            sys.path.pop(0)
        r = registry.dual_target()
        suffixes = {k: d["preset_suffix"] for k, d in registry.devices().items() if d.get("preset_suffix")}
        twin = registry.families()[next(f for f, v in registry.families().items() if v.get("dual_target") and not v.get("base"))]["devices"][0]
        return r["base"], r["twins"][0], r["suffix"], r["header"], suffixes[twin], suffixes
    except (ImportError, OSError, KeyError):
        return ("rdna3", "rdna4", "_rdna4",
                "target.decl @device\namdgpu.target<gfx11-generic> @rdna3\namdgpu.target<gfx12-generic> @rdna4\n\n", "-r9700",
                {"radeon-rx7900xtx": "-rx7900xtx", "radeon-r9700": "-r9700"})


BASE, TWIN, SUFFIX, HEADER, R9700_PRESET_SUFFIX, PRESET_SUFFIXES = _rows()
# PRESET_SUFFIXES {device: suffix}: build.py's extra preset of a device `<preset><suffix>` (--prefill-device DEVICE=N); R9700_PRESET_SUFFIX is the twin
# family's (gfx12) first device's, the alias of --prefill-r9700
KERNEL_DEF = re.compile(r"^kernel\.def\b(?P<attrs>.*?)\s@(?P<sym>\w+)\(")
LAUNCH = re.compile(r"^(\s*)kernel\.launch @(\w+)\[([^\]]*)\]\((.*)\) : \[([^\]]*)\]\((.*)\)\s*$")
PROGRAM_DEF = re.compile(r"^(command\.def\s+(?:public\s+)?)(?!target\()(@)", re.M)


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


def select_mode():
    """"launch" (per-launch template.apply) or "program" (one template.apply per command program), from env LOOM_DUAL_SELECT."""
    v = os.environ.get("LOOM_DUAL_SELECT", "launch").strip().lower()
    if v not in ("launch", "program"):
        raise ValueError(f"LOOM_DUAL_SELECT={v!r}: expected launch or program")
    return v


PROGRAM_HEAD = re.compile(r"^command\.def\s+(?:public\s+)?@(\w+)\(\)\s+launch\((.*)\)\s*\{\s*$")


def _program_blocks(lines, tw, alternate):
    """[(start, end)] line ranges (end inclusive: the closing `}` at column 0) of the command programs that launch a twin
    kernel of TW or carry a launch group."""
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].startswith("command.def"):
            j = i + 1
            while lines[j] != "}":
                j += 1
            if lines[i] in alternate or any(GROUP_BEGIN.match(l) or ((m := LAUNCH.match(l)) and m.group(2) in tw) for l in lines[i + 1:j]):
                blocks.append((i, j))
            i = j
        i += 1
    return blocks


def _provider_body(block, tw, target):
    """Lower one raw command body for TARGET, selecting its launch groups and twins."""
    body = block[1:-1]
    if not body or body[-1].strip() != "command.return":
        raise ValueError(f"dual-target select=program: {block[0]} does not end in command.return")
    body = body[:-1]
    out = []
    i = 0
    while i < len(body):
        gm = GROUP_BEGIN.match(body[i])
        if gm:
            ind, alt = gm.groups()
            j = i + 1
            while not GROUP_END.match(body[j]):
                if target == BASE:
                    out.append(body[j])
                j += 1
            if target == TWIN:
                out.append(ind + alt)
            i = j + 1
            continue
        lm = LAUNCH.match(body[i])
        if target == TWIN and lm and lm.group(2) in tw:
            ind, k, grid, args, gty, aty = lm.groups()
            out.append(f"{ind}kernel.launch @{k}{SUFFIX}[{grid}]({args}) : [{gty}]({aty})")
        else:
            out.append(body[i])
        i += 1
    return out


def _program_templates(block, tw, decls, alternate=None):
    m = PROGRAM_HEAD.match(block[0])
    if not m:
        raise ValueError(f"dual-target select=program: cannot read the program header: {block[0][:120]}")
    name, params = m.groups()
    ps = _split_top(params)
    names = ", ".join(p.split(":")[0].strip() for p in ps)
    types = ", ".join(p.split(":", 1)[1].strip() for p in ps)
    tname = f"{name}_body"
    decls.append(f"template.decl @{tname}({params})")
    for tgt in (BASE, TWIN):
        decls.append(f"template.def<@{tname}> target(@{tgt}) @{tname}_{tgt}({params}) {{")
        decls += _provider_body(alternate if alternate and tgt == TWIN else block, tw, tgt)
        decls.append("  template.return\n}")
    return [block[0], f"  template.apply<@{tname}>({names}) : ({types})", "  command.return", "}"]


def twins(text):
    """Symbols K of the kernels that have a `K_rdna4` kernel in TEXT."""
    syms = {m.group("sym") for l in text.split("\n") if (m := KERNEL_DEF.match(l))}
    return sorted(k for k in syms if k + SUFFIX in syms)


GROUP_BEGIN = re.compile(r"^(\s*)// dual-group: " + TWIN + r" (kernel\.launch .*)$")
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
            decls.append(f"template.def<@{name}> target(@{BASE}) @{name}_{BASE}({ps}) {{")
            decls += lines3[:-1]
            decls.append("  template.return\n}")
            decls.append(f"template.def<@{name}> target(@{TWIN}) @{name}_{TWIN}({ps}) {{")
            decls.append(lines3[-1])
            decls.append("  template.return\n}")
        out.append(f"{ind}template.apply<@{sigs[key]}>({', '.join(names)}) : ({', '.join(types)})")
        i = j + 1
    return out


def apply(text, enable=None, select=None, alternate=None):
    """TEXT made dual-target; unchanged when it has no twin kernels (or the pass is off). SELECT: "launch" or "program"
    (default: env LOOM_DUAL_SELECT)."""
    if enable is None:
        enable = enabled()
    if select is None:
        select = select_mode()
    if not enable or "target.decl @device" in text:
        return encodings_first(text)
    tw = set(twins(text))
    if not tw and not alternate:
        return encodings_first(text)
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
            l = l.replace("kernel.def ", f"kernel.def target(@{TWIN if sym != base else BASE}) ", 1)
        out.append(l)
        i += 1
        while lines[i] != "}":
            out.append(lines[i])
            i += 1
        out.append("}")
        i += 1
        if base in tw or sym.endswith(SUFFIX):
            last_def_end = len(out)
    if last_def_end == -1 and alternate:
        last_def_end = next(i for i, line in enumerate(out) if line in alternate)
    decls, sigs, res = [], {}, []
    alternate = alternate or {}
    if select == "program" or alternate:
        stubs, rebuilt, pos = [], [], 0
        for a, b in _program_blocks(out, tw, alternate):
            if select != "program" and out[a] not in alternate:
                continue
            if a < last_def_end:
                raise ValueError(f"program {out[a][:80]} starts before the end of its twin kernels")
            rebuilt += out[pos:a] + [f"\0PROGRAM{len(stubs)}\0"]
            stubs.append(_program_templates(out[a:b + 1], tw, decls, alternate.get(out[a])))
            pos = b + 1
        out = rebuilt + out[pos:]
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
                for tgt, kk in ((BASE, k), (TWIN, k + SUFFIX)):
                    decls.append(f"template.def<@{name}> target(@{tgt}) @{name}_{tgt}({', '.join(ps)}) {{")
                    decls.append(f"  kernel.launch @{kk}[{gv}]({av}) : [{', '.join(gt)}]({', '.join(at)})")
                    decls.append("  template.return\n}")
            res.append(f"{ind}template.apply<@{sigs[key]}>({', '.join(g + a)}) : ({', '.join(gt + at)})")
        else:
            res.append(l)
    # res has one line per line of out: the templates go right after the last twin kernel
    res = res[:last_def_end] + ([""] + decls if decls else []) + res[last_def_end:]
    body = "\n".join(res)
    if select == "program" or alternate:
        for n, stub in enumerate(stubs):
            body = body.replace(f"\0PROGRAM{n}\0", "\n".join(stub))
    body = PROGRAM_DEF.sub(r"\1target(@device) \2", body)
    return HEADER + encodings_first(body)


ENC_LIMIT = 60000   # SSA values defined before an encoding-typed line (counted as _enc_depth does); the parser truncates at 2^16


ENC_BYTES = 4_000_000   # bytes before an encoding-typed line: hrx-loom 0.1.0 fails past 4 MiB (engineer sweep bisected 4,191,941 ok, 4,281,430 fails)

_ENC_LINE = re.compile(r"encoding<layout>|view<[^>]*, %")
# Values a line defines: op results (`%a, %b:2 = ...`, also scf.for induction variables and iter_args `%x = %init`) and
# block / launch arguments (`%x: T`). A use never has `=` or `: ` right after its name (a type annotation is ` : T`).
_VALUE_DEFS = re.compile(r"%[\w.$]+(?::(\d+))?(?=(?:, %[\w.$]+(?::\d+)?)* = )|%[\w.$]+(?=: )")


def _is_enc(part):
    return bool(_ENC_LINE.search(part))


def _enc_depth(text):
    """The largest count of SSA values defined before a line that defines or uses an encoding in a type (or ENC_LIMIT + 1
    when such a line lies past ENC_BYTES of the file). Counts as the parser numbers a module's values (results, packs `%x:N`,
    loop variables, block and launch arguments): within about 0.2 % of the value ids hrx-loom 0.2.0 assigned on
    Qwen3.8-27B's llm.loom (the first truncated reference sat between estimates 65,474 and 65,604)."""
    c = worst = pos = 0
    for l in text.split("\n"):
        pos += len(l.encode()) + 1
        enc = _ENC_LINE.search(l)
        if enc and pos > ENC_BYTES:
            return ENC_LIMIT + 1
        if enc:
            worst = max(worst, c)
        for m in _VALUE_DEFS.finditer(l):
            c += int(m.group(1)) if m.group(1) else 1
    return worst


def encodings_first(text):
    """hrx-loom (0.1.0 and 0.2.0) keeps the value id of an SSA encoding referenced in a type in 16 bits: the text parser
    stores `(uint16_t)` of the module-wide value id (format/text/parser/types.c, loom_parse_type_encoding), so a type
    whose encoding is defined after the module's 65,536th value silently names an unrelated earlier value. The parse
    verifier reports it as ENCODING/004 '... references encoding value %x which is not defined at this point' (Laguna-XS-2.1,
    94k lines); a package registers, and loom-serve fails when a compile request clones such a kernel: `source value %N has
    no target remap` (N = id - 65536; Qwen3.8-27B with both quant/q4_0 choices, whose attention kernels sat 65.6k values
    deep). A file with an encoding-typed line that deep gets its top-level definitions reordered, stably: the funcs that
    lead the file, then every kernel that uses encodings (wherever it was: a weight-converter command program can sit in
    the middle of the kernels), then everything else in its order. Kernel texts are unchanged; smaller files are returned
    as they are. When the encoding kernels alone are too deep, split_library.py (run by build.py after this pass) moves them
    into kernel files; with that split off (LOOM_LIBRARY_SPLIT=off) this raises ValueError."""
    if _enc_depth(text) <= ENC_LIMIT:
        return text
    parts = re.split(r"(?m)^(?=kernel\.def |func\.def |template\.|command\.def )", text)
    head, rest = parts[0], parts[1:]
    first_late = next((i for i, p in enumerate(rest) if p.startswith(("template.", "command.def"))), len(rest))
    funcs = [i for i in range(first_late) if rest[i].startswith("func.def")]
    enc = [i for i, p in enumerate(rest) if p.startswith("kernel.def") and _is_enc(p)]
    moved = set(funcs) | set(enc)
    out = head + "".join([rest[i] for i in funcs + enc] + [p for i, p in enumerate(rest) if i not in moved])
    if _enc_depth(out) > ENC_LIMIT and os.environ.get("LOOM_LIBRARY_SPLIT", "auto").strip().lower() in ("off", "0", "no", "false"):
        raise ValueError(f"encoding-typed lines still lie {_enc_depth(out)} values deep after reordering (limit {ENC_LIMIT}): "
                         "hrx-loom truncates SSA encoding value ids to 16 bits; split the kernel library into more files")
    return out


def apply_file(path):
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read().replace("\r\n", "\n")   # a file written in text mode on Windows has CRLF line ends; the pass works on LF lines
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
            declared = set(re.findall(r"(?m)^kernel\.decl @(\w+)\(", t))   # kernels split_library.py moved to a kernel file
            for m in (LAUNCH.match(l) for l in t.split("\n")):
                if m and m.group(2) in owner and owner[m.group(2)] != n and m.group(2) not in declared:
                    raise ValueError(f"program/{n} launches @{m.group(2)}, whose {SUFFIX} twin is defined in program/{owner[m.group(2)]}: "
                                     "a dual-target kernel must be launched from the file that defines it")
    return [n for n in files if apply_file(os.path.join(program_dir, n))]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        print(p, "rewritten" if apply_file(p) else "unchanged")
