#!/bin/bash
# usage: VENV=~/loom/.venv run_kernel_tests.sh PACKAGE TESTDIR   (TESTDIR from make_kernel_tests.py)
pkg=$(cd "$1" && pwd); dir=$2; venv=${VENV:-$HOME/loom/.venv}
bin=$venv/bin; [ -d "$bin" ] || bin=$venv/Scripts   # Windows venvs use Scripts/
cd "$dir" || exit 1
# Keep only cases whose kernels the generator emitted (mixed tensor formats or a changed generator
# can drop a variant a case launches: that was a SYMBOL error for the whole file).
"$bin/python" - "$pkg/program/llm.loom" <<'PY'
import re, sys
lib = open(sys.argv[1]).read()
defined = set(re.findall(r'kernel\.def export\("([^"]+)"\)', lib)) | set(re.findall(r'kernel\.def[^@]*@([\w.]+)', lib))
text = open("cases.loom-test").read()
# A listed kernel (indirect dispatch over a device-written count, which loom-test cannot launch) gets a _slots twin
# for the cases: the same body over a static grid of slots, the first argument the slot count.
twins = []
for name in sorted(set(re.findall(r'kernel\.launch @([\w.]+)_slots\[', text))):
    a = lib.find(f'kernel.def export("{name}_listed")')
    if a < 0:
        continue
    b = lib.index("} launch(", a); end = lib.index("kernel.return\n}\n", b) + len("kernel.return\n}\n")
    grid = lib[a:b].split("%count = index.cast %bounded : i32 to index\n", 1)[1]
    twins.append(f'kernel.def export("{name}_slots") @{name}_slots(%count: index) {{\n' + grid + lib[b:end])
    defined.add(f"{name}_slots")
blocks = re.split(r'(?m)^(?=check\.case )', text)
keep = []
for b in blocks:
    missing = sorted(set(re.findall(r'kernel\.launch @([\w.]+)', b)) - defined)
    if b.startswith("check.case") and missing:
        print(f"SKIP {re.search(r'@([\w.]+)', b).group(1)}: kernel(s) not in this build: {', '.join(missing)}")
        continue
    keep.append(b)
open("cases_used.loom-test", "w").write(keep[0] + "".join(t + "\n" for t in twins) + "".join(keep[1:]))
PY
cat "$pkg/program/llm.loom" cases_used.loom-test > combined.loom
"$bin/loom-test" combined.loom --device=${DEVICE:-amdgpu} > out.json 2> err.txt
grep -v '^{"format' err.txt | head -20
"$bin/python" - <<'PY'
import json, sys
try:
    d = json.load(open("out.json"))
except ValueError:  # loom-test did not get to run the cases: a compile/link error (PARSE, TYPE, SYMBOL...)
    err = [l for l in open("err.txt", errors="replace") if not l.startswith('{"format')]
    out = open("out.json", errors="replace").read()
    print("FAIL compile: loom-test produced no results. Diagnostics (err.txt / out.json):")
    print("".join(err[-30:]) or out[-3000:])
    print("(a case launching a kernel the generator no longer emits shows up as a SYMBOL error: regenerate the tests "
          "with make_kernel_tests.py after changing the generator)")
    sys.exit(1)
bad = 0
for s in d["samples"]:
    fails = [f.get("detail", f.get("message", "")) if isinstance(f, dict) else str(f) for f in s["expectations"].get("failures", [])]
    issues = [i.get("message", "") for i in s.get("issues", [])]
    bad += not s["passed"]
    print(("PASS " if s["passed"] else "FAIL ") + s["case"], *(x[:300] for x in issues + fails[:2]))
print(f"{len(d['samples']) - bad}/{len(d['samples'])} passed")
sys.exit(1 if bad else 0)
PY
