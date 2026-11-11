#!/bin/bash
# Gap probe round 2. MODE=sweep: CPU-activity sweep on Qwen3.5-2B prefill. MODE=perf: the card's perf.py --long + 4 streams, base vs +1 spinning core.
set -u
here=$(cd "$(dirname "$0")" && pwd); repo_root=$(cd "$here/../.." && pwd); export SK=$repo_root/skills
W=${WORK:?}; out=$W/out; mkdir -p "$out"
R=${CARD:-loom-fun/unsloth-Qwen3.5-2B-loom}
bash "$here/sysdump.sh" > "$out/sysdump.txt" 2>&1
root=$HOME/devlab-gap/loom
[ -x "$root/.venv/bin/loom-serve" ] || "$SK/loom-setup/scripts/install_loom.sh" "$root" 0.1.0 > "$out/install.log" 2>&1 || { tail "$out/install.log"; exit 1; }
export VENV=$root/.venv; T=$root/.tools/bin/python
pkg=$HOME/devlab-gap/pkg_${R//\//_}
if [ ! -f "$pkg/.ok" ]; then rm -rf "$pkg"; "$VENV/bin/loom-fetch" "https://huggingface.co/$R/resolve/main/model.json" --no-input --materialize=copy --cache "$HOME/devlab-gap/fc" --output="$pkg" > "$out/fetch.log" 2>&1 && touch "$pkg/.ok"; rm -rf "$HOME/devlab-gap/fc"; fi
MID=$(python3 -c "import json; print(json.load(open('$pkg/model.json'))['model']['id'])")
MEM=$(curl -fsSL "https://huggingface.co/$R/resolve/main/info.json" | python3 -c "import json,sys; print(json.load(sys.stdin)['card']['memory_bytes'])")
echo "card $R model $MID mem $MEM"
if [ "$MODE" = sweep ]; then
  WL="2048:8 8192:2"
  for r in 1 2; do
    bash "$here/probe.sh" "$out/base$r" "$pkg" "$MID" "$MEM" - "$WL" > "$out/base$r.log" 2>&1
    SPIN=3 bash "$here/probe.sh" "$out/spin$r" "$pkg" "$MID" "$MEM" - "$WL" > "$out/spin$r.log" 2>&1
    for d in 1/1 1/4 1/9 0.2/4.8 0.05/1; do DUTY=$d bash "$here/probe.sh" "$out/duty${d//\//_}_$r" "$pkg" "$MID" "$MEM" - "$WL" > "$out/duty$r.log" 2>&1; done
    LOOM_EXTRA=--task_worker_spin_us=200000 bash "$here/probe.sh" "$out/twspin$r" "$pkg" "$MID" "$MEM" - "$WL" > "$out/twspin$r.log" 2>&1
  done
else
  export LOOM_SERVE_STATE=$out/state
  for r in 1 2; do for v in base spin; do
    o=$out/perf_${v}$r; mkdir -p "$o"
    python3 "$here/sampler.py" "$o/samples.csv" 0.01 & sp=$!
    "$SK/loom-setup/scripts/serve.sh" s 8461 amdgpu "$MEM" "$pkg/model.json" > "$o/serve.log" 2>&1 || { kill $sp; continue; }
    spin=; [ $v = spin ] && { taskset -c 3 sh -c "while :; do :; done" & spin=$!; }
    "$T" "$SK/loom-benchmark/scripts/perf.py" http://127.0.0.1:8461 $v --model "$MID" --no-campaign --long --log "$o/perf.jsonl" > "$o/perf.out" 2>&1
    "$T" "$SK/loom-benchmark/scripts/c4_steady.py" http://127.0.0.1:8461 4 --model "$MID" --log "$o/c4.jsonl" > "$o/c4.out" 2>&1
    [ -n "$spin" ] && kill $spin
    "$SK/loom-setup/scripts/stop.sh" s; kill $sp; wait $sp
  done; done
fi
for d in "$out"/*/; do [ -f "$d/samples.csv" ] && python3 "$here/clk.py" "$d/samples.csv" "$(basename $d)" | cut -c1-120; done
