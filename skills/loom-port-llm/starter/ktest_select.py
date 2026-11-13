"""Chooses the kernel-unit-test cases a device runs (run_kernel_tests.sh).

usage: ktest_select.py LIB.loom CASES.loom-test OUT.loom-test [ARCH]     (ARCH: gfx1151, gfx1201, ...; default DEVICE_ARCH)

A case launches kernels of the generated library. On a gfx12 device (RDNA4) every launch of a kernel K that has a
`K_rdna4` twin (dual_target.py's convention) goes to the twin; a case launching a gfx11-only kernel there (target(@rdna3)
without a twin: it cannot be compiled for gfx12) is skipped, so one test file covers both architectures. On gfx11
(or LOOM_RDNA4=0) the cases are launched as written. A case whose kernel is not in this build is skipped in both."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dual_target  # noqa: E402

EXPORT = re.compile(r'^kernel\.def\b[^\n]*?\bexport\("([^"]+)"\)', re.M)
SYMBOL = re.compile(r'^kernel\.def\b[^\n]*?\s@([\w.]+)\(', re.M)
RDNA3_ONLY = re.compile(r'^kernel\.def\b[^\n]*?target\(@rdna3\)[^\n]*?\s@([\w.]+)\(', re.M)


def is_gfx12(arch):
    return bool(arch) and arch.startswith("gfx12")


def select(lib, cases, arch=None, rdna4=None):
    """(text of the cases to run, [skip messages])."""
    rdna4 = dual_target.enabled() if rdna4 is None else rdna4
    gfx12 = is_gfx12(arch)
    defined = set(EXPORT.findall(lib)) | set(SYMBOL.findall(lib))
    defined |= set(EXPORT.findall(cases)) | set(SYMBOL.findall(cases))   # test helper kernels
    gfx11_only = set(RDNA3_ONLY.findall(lib)) - {k[:-len(dual_target.SUFFIX)] for k in defined if k.endswith(dual_target.SUFFIX)}
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
        elif gfx12 and used & gfx11_only:
            skipped.append(f"SKIP {name}: gfx11-only kernel(s) without a _rdna4 twin: {', '.join(sorted(used & gfx11_only))}")
        else:
            keep.append(b)
    return "".join(keep), skipped


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    arch = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("DEVICE_ARCH", "")
    text, skipped = select(open(sys.argv[1]).read(), open(sys.argv[2]).read(), arch)
    open(sys.argv[3], "w", newline="\n").write(text)
    print(*skipped, sep="\n") if skipped else None
