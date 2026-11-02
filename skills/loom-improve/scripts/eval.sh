#!/bin/bash
# Fixed, skill-version-independent evaluation of a finished trial's package (not counted in its
# wall clock). Run it on a QUIET GPU (no other loom-serve / llama-server; check with
# `pgrep -af "loom-serve|llama-server"` and gpu_busy_percent), one package at a time, so perf numbers
# from different skill versions are comparable. Quality uses the gates of a FROZEN skill tree
# (EVAL_SKILLS, default: the skills next to this script), so a skill change can't grade itself.
# usage: [EVAL_SKILLS=/path/to/frozen/skills] [LOOM_ROOT=~/loom] [LLAMACPP_DIR=dir] \
#        eval.sh OUT_DIR PKG GGUF [PORT=8460] [LLAMA_PORT=8430] [MEMORY_BYTES from the package]
#   1. serves PKG alone on PORT
#   2. LEVEL=full gates (FORCE=1: nothing reused from earlier runs) -> OUT_DIR/gates/gates.json
#      (GATES=0: perf only, when the trial's own full gates are trusted)
#   3. perf: llama.cpp (Loom stopped), then Loom: 3 rounds of the short set + 1 long set each
#      (perf.py --no-campaign) -> OUT_DIR/perf.jsonl, medians in OUT_DIR/eval.json
set -u
out=$(mkdir -p "$1" && cd "$1" && pwd); pkg=$(cd "$2" && pwd); gguf=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
port=${4:-8460}; lport=${5:-8430}
here=$(cd "$(dirname "$0")" && pwd); SK=${EVAL_SKILLS:-$here/../..}
B=$SK/loom-benchmark/scripts; SV=$SK/loom-setup/scripts
root=${LOOM_ROOT:-$HOME/loom}; bin=$root/.tools/bin; [ -d "$bin" ] || bin=$root/.tools/Scripts; T=$bin/python   # Windows venvs use Scripts/
export LOOM_SERVE_STATE=$out/state LLAMACPP_STATE=$out/llstate PYTHONDONTWRITEBYTECODE=1
model=$("$T" -c "import json;print(json.load(open('$pkg/llm_config.json'))['model_id'])")
mem=${6:-$("$T" - "$pkg" <<'EOF'
import json, os, sys
p = sys.argv[1]; card = json.load(open(os.path.join(p, "model.json")))
ex = next(iter(card["presets"].values()))["execution"]
cfg = json.load(open(os.path.join(p, "llm_config.json")))
w = os.path.getsize(os.path.realpath(os.path.join(p, "parameter", os.listdir(os.path.join(p, "parameter"))[0])))
print(int(w + ex["pool_capacity"] * 2 * cfg.get("kv_layers", cfg["layers"]) * cfg["kv_heads"] * cfg["head_dim"] * 2 + 6e9))
EOF
)}
busy=$(pgrep -af "loom-serve|llama-server" | grep -v pgrep | wc -l)
[ "$busy" -gt 0 ] && echo "WARNING: $busy other GPU servers are running: perf numbers will not be comparable"
VENV=$root/.venv "$SV/serve.sh" eval "$port" amdgpu "$mem" "$pkg/model.json" || exit 1
trap '"$SV/stop.sh" eval >/dev/null 2>&1' EXIT
# Same reference for every evaluation: the given build, else the latest lemonade release.
LLAMACPP_DIR=${LLAMACPP_DIR:-$("$B/fetch_llamacpp.sh" | tail -1)}; export LLAMACPP_DIR
if [ "${GATES:-1}" = 1 ]; then
  echo "== gates (LEVEL=full, FORCE=1)"
  FORCE=1 LEVEL=full LOOM_ROOT=$root "$B/gates.sh" "$pkg" "http://127.0.0.1:$port" "$gguf" "$out/gates" "$lport" > "$out/gates.out" 2>&1
  tail -1 "$out/gates.out"
fi
echo "== perf (llama.cpp with Loom stopped, then Loom: one model at a time)"
cd "$out"
if [ -n "${LLAMACPP_DIR:-}" ]; then
  "$SV/stop.sh" eval >/dev/null 2>&1
  if "$B/llamacpp_server.sh" "$gguf" "$lport" 4 40960 unified >/dev/null; then
    for r in 1 2 3; do "$T" "$B/perf.py" "http://127.0.0.1:$lport" llama --kind llama --no-campaign --log perf.jsonl >/dev/null; done
    "$T" "$B/perf.py" "http://127.0.0.1:$lport" llama --kind llama --no-campaign --long --log perf.jsonl >/dev/null
    "$B/llamacpp_server.sh" --stop "$lport"
  fi
  VENV=$root/.venv "$SV/serve.sh" eval "$port" amdgpu "$mem" "$pkg/model.json" >/dev/null || exit 1
fi
for r in 1 2 3; do "$T" "$B/perf.py" "http://127.0.0.1:$port" loom --model "$model" --no-campaign --log perf.jsonl >/dev/null; done
"$T" "$B/perf.py" "http://127.0.0.1:$port" loom --model "$model" --no-campaign --long --log perf.jsonl >/dev/null
"$T" - "$B" <<'EOF'
import json, statistics, sys
sys.path.insert(0, sys.argv[1])
from bench_common import perf_metrics
runs = {}
for l in open("perf.jsonl"):
    r = json.loads(l)
    for m, v in perf_metrics(r).items():
        runs.setdefault((r["kind"], m), []).append(v)
med = {f"{k}:{m}": round(statistics.median(v), 1) for (k, m), v in runs.items()}
import os
g = json.load(open("gates/gates.json")) if os.path.exists("gates/gates.json") else {"verdict": "not run", "gates": {}}
res = {"gates": g["verdict"], "per_gate": {n: x["pass"] for n, x in g["gates"].items()}, "perf_median": med,
       "margin": {m.split(":")[1]: round(med[m] / med["llama:" + m.split(":")[1]] - 1, 3)
                  for m in med if m.startswith("loom:") and "llama:" + m.split(":")[1] in med}}
json.dump(res, open("eval.json", "w"), indent=1)
print(json.dumps(res, indent=1))
EOF
