#!/bin/bash
# One DevLab job of the local-vs-DevLab gap probe: sysdump, Loom prefill probe (plain + device-profiled), llama-bench.
set -u
here=$(cd "$(dirname "$0")" && pwd); repo_root=$(cd "$here/../.." && pwd); export SK=$repo_root/skills
W=${WORK:?}; out=$W/out; mkdir -p "$out"
R=${CARD:-loom-fun/unsloth-Qwen3.5-2B-loom}; MID=${MODEL_ID:-Qwen/Qwen3.5-2B}; MEM=${MEM:-9000000000}
bash "$here/sysdump.sh" > "$out/sysdump.txt" 2>&1
root=$HOME/devlab-gap/loom
[ -x "$root/.venv/bin/loom-serve" ] || "$SK/loom-setup/scripts/install_loom.sh" "$root" 0.1.0 > "$out/install.log" 2>&1 || { tail "$out/install.log"; exit 1; }
export VENV=$root/.venv
pkg=$HOME/devlab-gap/pkg_${R//\//_}
if [ ! -f "$pkg/.ok" ]; then rm -rf "$pkg"; "$VENV/bin/loom-fetch" "https://huggingface.co/$R/resolve/main/model.json" --no-input --materialize=copy --cache "$HOME/devlab-gap/fc" --output="$pkg" > "$out/fetch.log" 2>&1 && touch "$pkg/.ok"; rm -rf "$HOME/devlab-gap/fc"; fi
bash "$here/probe.sh" "$out/plain1" "$pkg" "$MID" "$MEM" - "${WORKLOAD:-1024:4 2048:6 8192:2}" > "$out/plain1.log" 2>&1
bash "$here/probe.sh" "$out/plain2" "$pkg" "$MID" "$MEM" - "2048:20" > "$out/plain2.log" 2>&1
bash "$here/probe.sh" "$out/prof" "$pkg" "$MID" "$MEM" queue-events,dispatch-events,executable-metadata "2048:4" > "$out/prof.log" 2>&1
python3 -I "$here/parse_prof.py" "$out/prof/p.ireeprof" "$out/prof/p.pkl" > "$out/prof/parse.txt" 2>&1 && python3 -I "$here/analyze.py" "$out/prof/p.pkl" > "$out/prof/analyze.txt" 2>&1
rm -f "$out/prof/p.ireeprof"
if [ -n "${LLAMACPP:-}" ]; then
  lb=$("$SK/loom-benchmark/scripts/fetch_llamacpp.sh" "$HOME/devlab-gap/llamacpp" "$LLAMACPP" 2>"$out/llama_fetch.log" | tail -1)
  g=$(ls "$pkg"/parameter/*.gguf | grep -v mmproj | head -1)
  python3 "$here/sampler.py" "$out/llama_samples.csv" 0.005 & sp=$!
  "$lb/llama-bench" -m "$g" -ngl 99 -fa 1 -p 1024,2048,8192 -n 0 -r 3 -o json > "$out/llama_bench.json" 2> "$out/llama_bench.log"
  kill $sp; wait $sp
fi
grep -h tps "$out"/plain*/pp.jsonl | head -40
cat "$out/prof/analyze.txt" | head -40
