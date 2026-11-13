#!/bin/bash
# usage: VENV=~/loom/.venv run_kernel_tests.sh PACKAGE TESTDIR   (TESTDIR from make_kernel_tests.py)
pkg=$(cd "$1" && pwd); dir=$2; venv=${VENV:-$HOME/loom/.venv}
bin=$venv/bin; [ -d "$bin" ] || bin=$venv/Scripts   # Windows venvs use Scripts/
cd "$dir" || exit 1
# The device's architecture (DEVICE_ARCH, else rocminfo / amd-smi): on gfx12 the cases launch the K_rdna4 twin of every
# kernel that has one and skip cases of gfx11-only kernels. Keep only cases whose kernels the generator emitted (mixed
# tensor formats or a changed generator can drop a variant a case launches: that was a SYMBOL error for the whole file).
arch=${DEVICE_ARCH:-$( (rocminfo 2>/dev/null | grep -o -m1 'gfx[1-9][0-9a-f]*' | head -1) || true)}
[ -n "$arch" ] || arch=$( (amd-smi static --asic 2>/dev/null | grep -o -m1 'gfx[1-9][0-9a-f]*' | head -1) || true)
echo "device arch: ${arch:-unknown}"
"$bin/python" "$(dirname "$0")/ktest_select.py" "$pkg/program/llm.loom" cases.loom-test cases_used.loom-test "$arch"
cat "$pkg/program/llm.loom" cases_used.loom-test > combined.loom
# TEMP debug: dump the flash cases' outputs next to the job's results
"$bin/python" - <<'PY'
import re
t = open("combined.loom").read(); n = [0]
def dump(m):
    n[0] += 1
    return f'  check.file.write.npy value({m.group(2)}) path("dump_{n[0]}.npy") mode(always) : {m.group(4)}\n' + m.group(0)
blocks = re.split(r'(?m)^(?=check\.case )', t)
out = []
for b in blocks:
    if re.match(r'check\.case public @flash_(prefill|batch|cut)', b):
        b = re.sub(r'(?m)^(  check\.expect\.approximate actual\((%\w+)\) expected\((%\w+)\) [^:]*: (tensor<[^>]*>), .*)$', dump, b)
        nm = re.search(r'@(\w+)', b).group(1); open("dumpmap.txt", "a").write(f"{nm} {n[0]}\n")
    out.append(b)
open("combined.loom", "w").write("".join(out))
PY
mkdir -p dump; "$bin/loom-test" combined.loom --device=${DEVICE:-amdgpu} --file-output-root="$PWD/dump" > out.json 2> err.txt
o="$dir/../../out"; [ -d "$o" ] || o="$(dirname "$dir")/out"; mkdir -p "$o/kdump"; cp dump/* dumpmap.txt "$o/kdump/" 2>/dev/null; find . -maxdepth 1 -name 'f*.npy' -size -3M -exec cp {} "$o/kdump/" \; 2>/dev/null; cp cases_used.loom-test "$o/kdump/" 2>/dev/null
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
    neg = "_negcontrol" in s["case"]   # a negative control must FAIL (it feeds the kernel data in the wrong layout)
    ok = (not s["passed"]) if neg else s["passed"]
    bad += not ok
    print(("PASS " if ok else "FAIL ") + s["case"] + (" (negative control failed, as it must)" if neg and ok else " (negative control passed: the test cannot see a layout error)" if neg else ""), *(x[:300] for x in issues + fails[:2]))
print(f"{len(d['samples']) - bad}/{len(d['samples'])} passed")
sys.exit(1 if bad else 0)
PY
