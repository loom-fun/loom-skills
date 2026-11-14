#!/bin/bash
# One suite of the DevLab kernel microbenchmarks (.github/workflows/devlab-kbench.yml): run every listed bench
# variant with loom-benchmark, interleaved (A,B,C,A,B,C,... x repeats) on this runner, and write per-run JSON plus
# a summary table.
#
#   [WORK=~/devlab-kbench] devlab_kbench.sh KBENCH.json SUITE
#
# KBENCH.json: {"sdk": (optional) "ci:ARTIFACT_ID" | version, default = install_loom.sh default, "repeats": N,
#               "suites": [{"suite": name, "dir": "<dir under .github/devlab/kbench>", "files": ["a.loom", ...],
#                           "config": ["k=v", ...] (--config=), "args": ["--measure=...", ...] (loom-benchmark flags),
#                           "repeats": N (optional), "runner": "strix" (default) | "r9700" (workflow runs-on),
#                           "gen": generator script under .github/devlab/kbench (run as `gen CHECKOUT RUNDIR VARIANT...`), "gen_variants": [...]}]}
# Variant name = file name without "bench_" prefix / ".loom" ("bench.loom" is "base"). Benchmarks that read .npy
# weights (check.file.read.npy path("w/x.npy")) that are not in the suite dir get random int8 weights of the
# declared shape (timing-only A/B; they are generated once per runner, never committed).
# Output WORK/<suite>/out: r_<variant>_<i>.json (+.log), summary.json, summary.md.
set -uo pipefail
kb=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); suite=${2:?usage: devlab_kbench.sh KBENCH.json SUITE}
here=$(cd "$(dirname "$0")/../.." && pwd); SV=$here/skills/loom-setup/scripts
WORK=${WORK:-$HOME/devlab-kbench}; mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
q() { python3 -c "import json; k=json.load(open('$kb')); s=next(s for s in k['suites'] if s['suite']=='$suite'); $1"; }
src=$here/.github/devlab/kbench/$(q "print(s['dir'])")
D=$WORK/$suite; out=$D/out; rm -rf "$D"; mkdir -p "$out" "$D/run"
t0=$(date +%s); step() { echo "== $* ($(date +%T), +$(( $(date +%s) - t0 ))s)"; }

root=${LOOM_ROOT:-$WORK/loom-$(python3 -c "import json; print(json.load(open('$kb')).get('sdk','default').replace(':','-'))")}
if [ ! -x "$root/.venv/bin/loom-benchmark" ]; then
  step "install Loom into $root"
  sdk=$(python3 -c "import json; print(json.load(open('$kb')).get('sdk',''))")
  "$SV/install_loom.sh" "$root" ${LOOM_VERSION:-${sdk:+"$sdk"}} > "$out/install.log" 2>&1 || { tail -20 "$out/install.log"; exit 1; }
fi
cat "$root/loom-install.json" > "$out/loom-install.json" 2>/dev/null
LB=$root/.venv/bin/loom-benchmark

# run dir: the suite's sources, plus generated weights
cp -r "$src"/. "$D/run/"; cd "$D/run" || exit 1
# a suite may generate its bench files from this checkout's starter: "gen": script under .github/devlab/kbench, "gen_variants": [...]
gen=$(q "print(s.get('gen', ''))")
if [ -n "$gen" ]; then
  step "generate bench files: $gen $(q "print(' '.join(s.get('gen_variants', [])))")"
  python3 "$here/.github/devlab/kbench/$gen" "$here" "$D/run" $(q "print(' '.join(s.get('gen_variants', [])))") || exit 1
fi
python3 - <<'PY' || exit 1
import glob, os, re
def u8_to_i8(b):  # random bytes mapped into -100..100
    return b.translate(bytes(((x % 201) - 100) & 255 for x in range(256)))
seen = {}
for f in glob.glob("*.loom"):
    for m in re.finditer(r'check\.file\.read\.npy path\("([^"]+)"\) : tensor<(\d+)xi8>', open(f).read()):
        seen[m.group(1)] = int(m.group(2))
for path, n in seen.items():
    if os.path.exists(path): continue
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hdr = "{'descr': '|i1', 'fortran_order': False, 'shape': (%d,), }" % n
    hdr += " " * (63 - (len(hdr) + 10) % 64) + "\n"
    while (10 + len(hdr)) % 64: hdr = hdr[:-1] + " \n"
    import struct
    with open(path, "wb") as fh:
        fh.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode())
        left = n
        while left:
            c = min(left, 1 << 24); fh.write(u8_to_i8(os.urandom(c))); left -= c
    print("generated", path, n)
PY

files=$(q "print(' '.join(s['files']))"); reps=$(q "print(s.get('repeats') or json.load(open('$kb')).get('repeats', 5))")
cfgs=$(q "print(' '.join('--config='+c for c in s.get('config', [])))"); args=$(q "print(' '.join(s.get('args', [])))")
vname() { local v=${1%.loom}; v=${v#bench_}; [ "$v" = bench ] && v=base; echo "$v"; }
echo "suite $suite: files=[$files] repeats=$reps"
rc=0
for i in $(seq 1 "$reps"); do
  for f in $files; do
    v=$(vname "$f"); step "$v run $i/$reps"
    "$LB" "$f" --device=amdgpu $cfgs $args --compile-report=summary --file-output-root="$D/run/fo_${v}_$i" \
      --output="$out/r_${v}_$i.json" > "$out/r_${v}_$i.json.log" 2>&1 || { echo "FAIL $v $i"; tail -5 "$out/r_${v}_$i.json.log"; rc=1; }
    rm -rf "$D/run/fo_${v}_$i"
  done
done

python3 - "$out" "$suite" "$(for f in $files; do vname "$f"; done | tr '\n' ' ')" <<'PY'
import glob, json, os, statistics as S, sys
out, suite, variants = sys.argv[1], sys.argv[2], sys.argv[3].split()
fails = {}
res = {}  # variant -> kernel -> [per-run median ms]
for v in variants:
    for fn in sorted(glob.glob(f"{out}/r_{v}_*.json")):
        try: d = json.load(open(fn))
        except Exception: continue
        by = {}
        for w in d.get("work_items", []):
            if w.get("state") not in (None, "completed", "succeeded", "passed") and not w.get("profile_replay"):
                fails.setdefault(v, {})[w.get("benchmark")] = (w.get("failure") or {}).get("message", w.get("state"))
            for r in w.get("profile_replay", {}).get("summary", {}).get("rows", []):
                n = r.get("function_name", "")
                if n.startswith("k_"): by.setdefault(n, []).append(r["timing"]["mean_ns"] / 1e6)
        for k, x in by.items(): res.setdefault(v, {}).setdefault(k, []).append(S.median(x))
kernels = sorted({k for v in res.values() for k in v})
base = variants[0]
lines = [f"### kbench `{suite}`: median ms of per-run medians (n runs, min-max), vs {base}", "",
         "| kernel | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
for k in kernels:
    cells = []
    for v in variants:
        x = res.get(v, {}).get(k)
        if not x: cells.append("-"); continue
        m = S.median(x); b = res.get(base, {}).get(k)
        rel = f" ({m / S.median(b) - 1:+.1%})" if b and v != base else ""
        cells.append(f"{m:.3f}{rel} n={len(x)} [{min(x):.3f}-{max(x):.3f}]")
    lines.append(f"| {k} | " + " | ".join(cells) + " |")
for v, f in fails.items():
    for b, m in f.items(): lines.append(f"\nFAILED `{v}` {b}: {m[:200]}")
open(f"{out}/summary.md", "w").write("\n".join(lines) + "\n")
json.dump({"suite": suite, "variants": variants, "ms": res, "failures": fails}, open(f"{out}/summary.json", "w"), indent=1)
print("\n".join(lines))
PY
exit $rc
