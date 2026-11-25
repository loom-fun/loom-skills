"""Chooses the kernel-unit-test cases a device runs (run_kernel_tests.sh).

usage: ktest_select.py LIB.loom CASES.loom-test OUT.loom-test [ARCH]     (ARCH: gfx1151, gfx1201, ...; default DEVICE_ARCH)

A case launches kernels of the generated library. On a gfx12 device (RDNA4) every launch of a kernel K that has a
`K_rdna4` twin (dual_target.py's convention) goes to the twin; a case launching a gfx11-only kernel there (target(@rdna3)
without a twin: it cannot be compiled for gfx12) is skipped, so one test file covers both architectures. On gfx11
(or LOOM_RDNA4=0) the cases are launched as written. A case whose kernel is not in this build is skipped in both."""
import os
import re
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dual_target  # noqa: E402
import split_library  # noqa: E402

EXPORT = re.compile(r'^kernel\.def\b[^\n]*?\bexport\("([^"]+)"\)', re.M)
SYMBOL = re.compile(r'^kernel\.def\b[^\n]*?\s@([\w.]+)\(', re.M)
RDNA4_ONLY = re.compile(r'^kernel\.def\b[^\n]*?target\(@rdna4\)[^\n]*?\s@([\w.]+)\(', re.M)
RDNA3_ONLY = re.compile(r'^kernel\.def\b[^\n]*?target\(@rdna3\)[^\n]*?\s@([\w.]+)\(', re.M)


def device_arch(environ):
    if environ.get("DEVICE_ARCH"):
        return environ["DEVICE_ARCH"]
    for cmd in (["rocminfo"], ["amd-smi", "static", "--asic"]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            continue
        match = re.search(r"gfx[1-9][0-9a-f]*", result.stdout)
        if match:
            return match.group()
    raise ValueError("set DEVICE_ARCH to generate kernel tests for a family-specific command graph")


def is_gfx12(arch):
    return bool(arch) and arch.startswith("gfx12")


def select(lib, cases, arch=None, rdna4=None):
    """(text of the cases to run, [skip messages])."""
    rdna4 = dual_target.enabled() if rdna4 is None else rdna4
    gfx12 = is_gfx12(arch)
    defined = set(EXPORT.findall(lib)) | set(SYMBOL.findall(lib))
    defined |= set(EXPORT.findall(cases)) | set(SYMBOL.findall(cases))   # test helper kernels
    gfx11_only = set(RDNA3_ONLY.findall(lib)) - {k[:-len(dual_target.SUFFIX)] for k in defined if k.endswith(dual_target.SUFFIX)}
    gfx12_only = {k for k in RDNA4_ONLY.findall(lib) if not k.endswith(dual_target.SUFFIX) or k[:-len(dual_target.SUFFIX)] not in defined}   # K_rdna4 with no K (f16o fusions)
    keep, skipped = [], []
    for b in re.split(r'(?m)^(?=check\.case )', cases):
        if not b.startswith("check.case"):
            keep.append(b)
            continue
        name = re.search(r'@([\w.]+)', b).group(1)
        if gfx12 and rdna4:
            b = re.sub(r'(kernel\.launch @)([\w.]+)(?=\[)',
                       lambda m: m.group(1) + (m.group(2) + dual_target.SUFFIX if m.group(2) + dual_target.SUFFIX in defined else m.group(2)), b)
        used = set(re.findall(r'kernel\.launch @([\w.]+)', b))
        if used - defined:
            skipped.append(f"SKIP {name}: kernel(s) not in this build: {', '.join(sorted(used - defined))}")
        elif not (gfx12 and rdna4) and used & gfx12_only:
            skipped.append(f"SKIP {name}: gfx12-only kernel(s): {', '.join(sorted(used & gfx12_only))}")
        elif gfx12 and used & gfx11_only:
            skipped.append(f"SKIP {name}: gfx11-only kernel(s) without a _rdna4 twin: {', '.join(sorted(used & gfx11_only))}")
        else:
            keep.append(b)
    return "".join(keep), skipped


def _base(k, defined):
    """K for K and its gfx12 twin K_rdna4 (one kernel identity)."""
    return k[:-len(dual_target.SUFFIX)] if k.endswith(dual_target.SUFFIX) and k[:-len(dual_target.SUFFIX)] in defined else k


# Launched kernels hrx-loom 0.2.0's loom-test cannot run as a case root (loom-kb topic sdk, "kernel tests"): reported apart.
UNTESTABLE = {
    r"_listed$": "indirect dispatch (the launch config reads its count from a buffer)",
    r"^g_silu_mul(_lv)?$": "SwiGLU g_silu_mul: the vector bound is proven only for the program's constant width",
}


def untestable(name):
    return next((why for pat, why in UNTESTABLE.items() if re.search(pat, name)), None)


def coverage(lib, selected):
    """(launched, untested): the kernels the library's programs launch (command, template and func bodies; a K / K_rdna4 pair
    counts once) and those of them no selected case launches."""
    defined = set(EXPORT.findall(lib)) | set(SYMBOL.findall(lib))
    launched = set()
    for d in re.split(r'(?m)^(?=\S+\.(?:def|decl) )', lib):
        if d.startswith(("command.def", "template.def", "func.def")):
            launched |= {_base(k, defined) for k in re.findall(r'kernel\.launch @([\w.]+)', d)}
    tested = {_base(k[3:] if k.startswith("kt_") else k, defined) for k in re.findall(r'kernel\.launch @([\w.]+)', selected)}   # kt_K: a copy of K
    return sorted(launched), sorted(launched - tested)


def split_groups(program_dir, name, selected):
    """A split library (split_library.py): the selected cases per module they run against, {file name: case text}.
    A case runs against the kernel file that defines its encoding kernels (a case that launches no moved kernel: against
    the program file). The program-file kernels such a case also launches (rope, split/combine, producers: no encodings,
    so they add no encoding depth) and the helper funcs they call are copied into its group, after the shared preamble."""
    files = [name] + split_library.kernel_files(program_dir)
    texts = {f: open(os.path.join(program_dir, f), encoding="utf-8").read() for f in files}
    owner, parts = {}, {}
    for f in files:
        for p in split_library.TOP.split(texts[f]):
            m = split_library.KDEF.match(p)
            if m:
                owner[m.group(1)] = f
                parts.setdefault(m.group(1), p)
    funcs = {m.group(1): p for p in split_library.TOP.split(texts[name]) for m in [split_library.FUNC.match(p)] if m}
    pre, groups, extra, skipped = [], {f: [] for f in files}, {f: [] for f in files}, []
    for b in re.split(r'(?m)^(?=check\.case )', selected):
        if not b.startswith("check.case"):
            pre.append(b)
            continue
        used = [k for k in dict.fromkeys(re.findall(r'kernel\.launch @([\w.]+)', b)) if k in owner]
        own = {owner[k] for k in used} - {name}
        if len(own) > 1:
            case = re.search(r'@([\w.]+)', b).group(1)
            skipped.append(f"SKIP {case}: launches kernels of several kernel files ({', '.join(sorted(own))})")
            continue
        g = own.pop() if own else name
        groups[g].append(b)
        if g != name:
            extra[g] += [k for k in used if owner[k] == name and k not in extra[g]]
    out = {}
    for f, cs in groups.items():
        if not cs:
            continue
        add = [parts[k] for k in extra[f]]
        need, todo = set(), set(re.findall(r"func\.call @([\w.]+)", "".join(add)))
        while todo:   # the helpers the copied kernels call, transitively, that the kernel file lacks
            fn = todo.pop()
            if fn in funcs and fn not in need and not re.search(rf"(?m)^func\.def\b[^\n@]*@{re.escape(fn)}\(", texts[f]):
                need.add(fn)
                todo |= set(re.findall(r"func\.call @([\w.]+)", funcs[fn]))
        out[f] = "".join(pre + [funcs[n] for n in funcs if n in need] + add + cs)
    return out, skipped


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    arch = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("DEVICE_ARCH", "")
    lib_dir, lib_name = os.path.split(os.path.abspath(sys.argv[1]))
    lib = split_library.library_text(lib_dir, lib_name)   # a split library: the program file and its kernel files
    text, skipped = select(lib, open(sys.argv[2]).read(), arch)
    open(sys.argv[3], "w", newline="\n").write(text)
    import glob
    for old in glob.glob(f"{sys.argv[3][:-len('.loom-test')]}.*.loom-test"):   # a previous run's groups
        os.remove(old)
    if split_library.kernel_files(lib_dir):   # one case file per module: OUT.<library file>.loom-test, run against that file
        groups, more = split_groups(lib_dir, lib_name, text)
        skipped += more
        for f, t in groups.items():
            open(f"{sys.argv[3][:-len('.loom-test')]}.{f[:-len('.loom')]}.loom-test", "w", newline="\n").write(t)
    print(*skipped, sep="\n") if skipped else None
    launched, untested = coverage(lib, text)
    sdk = [k for k in untested if untestable(k)]
    rest = [k for k in untested if not untestable(k)]
    print(f"COVERAGE {len(launched) - len(untested)}/{len(launched)} launched kernels have a case"
          + (f"; untested {', '.join(rest)}" if rest else "") + (f"; not runnable by loom-test (SDK) {', '.join(sdk)}" if sdk else ""))
