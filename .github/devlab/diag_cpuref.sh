#!/bin/bash
# Independent reference (devlab-port "post"; W, OUT, SK, LOOM_ROOT): the same llama.cpp build on the CPU
# (-ngl 0 --device none) as a second reference. Records a CPU golden of the short prompts, then reports
#   1. llama.cpp ROCm golden vs llama.cpp CPU golden (how far the two llama.cpp backends agree),
#   2. Loom vs the CPU golden (compare.py),
#   3. with CPU_LIVE=1 (memory for both engines): vision and the thinking-off agent loop with the CPU server as llama.
# A gate failure that llama.cpp's own two backends reproduce is a near-tie misjudgment, not a Loom bug.
set -u
B=$SK/loom-benchmark/scripts; S=$SK/loom-setup/scripts
T=$LOOM_ROOT/.tools/bin/python; V=$LOOM_ROOT/.venv
export LOOM_SERVE_STATE=$W/cpuref-state LLAMACPP_STATE=$W/cpuref-llstate PYTHONDONTWRITEBYTECODE=1
export LLAMACPP_DIR=$($T -c "import json; print(json.load(open('$W/bench/llamacpp_pick.json'))['ranked'][0]['dir'])")
gguf=$($T -c "import json; print(json.load(open('$W/bench/llamacpp_pick.json'))['gguf'])")
mem=$($T -c "import re; print(re.findall(r'--memory_bytes=(\d+)', open('$W/build.log').read())[-1])")
mm=$($T -c "import json; print((json.load(open('$W/cfg.json')).get('vision') or {}).get('mmproj') or '')")
pkill -f loom-serve; pkill -f llama-server; sleep 10
mkdir -p $W/cpuref && cd $W/cpuref
cpu_up() { MMPROJ=$mm LLAMA_SPEC="-ngl 0 --device none" bash $B/llamacpp_server.sh "$gguf" 8131 1 16384 2>&1 | tail -2; }
loom_up() { VENV=$V bash $S/serve.sh cpuref 8162 amdgpu $mem $W/pkg/model.json > /dev/null || echo "loom serve failed"; }
echo "== CPU llama.cpp golden"; cpu_up
t0=$(date +%s); $T $B/golden.py $W/pkg golden_cpu.json --llama http://127.0.0.1:8131 2>&1 | tail -2; echo "recorded in $(( $(date +%s) - t0 )) s"
echo "== 1. llama.cpp ROCm golden vs CPU golden (short prompts)"
PYTHONPATH=$B $T - "$W/pkg" "$W/bench/golden.json" golden_cpu.json <<'PY'
import json, sys
from bench_common import Package, near_tie
pkg = Package(sys.argv[1]); rocm = {g["prompt"]: g for g in json.load(open(sys.argv[2]))}
bad = 0
for c in json.load(open(sys.argv[3])):
    r = rocm.get(c["prompt"])
    if not r: continue
    a, b = r["ids"], c["ids"]; n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
    if n >= min(len(a), len(b)): print(f"  {n:3d}/{len(a):3d} same {c['prompt'][:50]!r}"); continue
    top = dict((int(i), lp) for i, lp in r["top"][n]) if n < len(r["top"]) else {}
    gap = round(top[a[n]] - top[b[n]], 3) if b[n] in top else "not in top-5"
    near = near_tie(top, a[n], b[n]) if top else True
    bad += not near
    print(f"  {n:3d}/{len(a):3d} gap={gap} {'' if near else 'BAD '}rocm {pkg.tok.decode(a[n:n+3])!r} cpu {pkg.tok.decode(b[n:n+3])!r} {c['prompt'][:40]!r}")
print(f"ROCM_VS_CPU bad={bad} (divergences llama.cpp's own two backends make that the gate would call a bug)")
PY
if [ "${CPU_LIVE:-0}" = 1 ]; then
  echo "== 3. live: Loom with the CPU llama.cpp server as reference"; loom_up
  [ -n "$mm" ] && { echo "-- vision"; $T $B/vision_golden.py $W/pkg --url http://127.0.0.1:8162 --llama http://127.0.0.1:8131 2>&1 | grep -E "PASS|FAIL|loom :|llama:" | cut -c1-240; }
  echo "-- agent, thinking off"; $T $B/agent_golden.py $W/pkg http://127.0.0.1:8162 --llama http://127.0.0.1:8131 --thinking off --out agent_off_cpu.json 2>&1 | grep -v '^OK' | tail -6 | cut -c1-400
  bash $B/llamacpp_server.sh --stop 8131
else
  bash $B/llamacpp_server.sh --stop 8131; loom_up
fi
echo "== 2. Loom vs the CPU golden"
$T $B/compare.py $W/pkg golden_cpu.json --url http://127.0.0.1:8162 --out compare_cpu.json 2>&1 | grep -E 'gap=|prompts' | cut -c1-200
bash $S/stop.sh cpuref > /dev/null 2>&1
cp golden_cpu.json compare_cpu.json agent_off_cpu.json $OUT/ 2>/dev/null; true
