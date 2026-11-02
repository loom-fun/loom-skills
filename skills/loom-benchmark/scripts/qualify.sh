#!/bin/bash
# Everything between "the port serves" and "start optimizing", in one command:
#   1. fetch_llamacpp.sh (latest lemonade-sdk/llamacpp-rocm, pre-approved) + pick_llamacpp.sh: benchmark
#      it and every llama.cpp ROCm build on the machine -> llamacpp_pick.json
#   2. ONE llama-server from the fastest build: 4 slots sharing a unified KV pool big enough for the
#      long goldens and the needle comparison, reused by every step below (no restarts)
#   3. LEVEL=quick gates (goldens, compare, consistency, agent tool loop, sampling, half-context needle)
#      -> a failing gate is reported (exit 1 at the end) but the measurements still run
#   4. llama.cpp baseline: perf.py --kind llama (short and --long) and c4_steady -> perf_log.jsonl
#   5. roofline.py -> roofline.json; the optimization budget is what is left of the end-to-end budget
#      (E2E_BUDGET_MIN, default: loom-roofline/e2e_budget.py, from this model's roofline TG/PP) after the time already spent (WORKDIR/.e2e_start, written by
#      port.sh) and a reserve for the final gates + report (measured from this run's gates)
#   6. Loom baseline: perf.py (short, --long) and c4_steady -> the start rows of targets.jsonl
# usage: [LOOM_ROOT=~/loom] [SEARCH_ROOT=$HOME] [E2E_BUDGET_MIN=60] [BUDGET_H=h] qualify.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT=8130]
#   BUDGET_H overrides the computed optimization budget (the user gave a time). The llama-server is
#   stopped at the end (KEEP_LLAMA=1 keeps it; its URL is printed).
# Prints QUALIFY_RESULT {...}: gates verdict, baseline build, the roofline table, budget minutes.
set -u
pkg=$(cd "$1" && pwd); url=$2; bench=$(mkdir -p "$4" && cd "$4" && pwd); lport=${5:-8130}
gguf=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
S=$(cd "$(dirname "$0")" && pwd); R=$S/../../loom-roofline; root=${LOOM_ROOT:-$HOME/loom}
bin=$root/.tools/bin; [ -d "$bin" ] || bin=$root/.tools/Scripts; T=$bin/python
export LOOM_NO_AUDIT=1 PYTHONDONTWRITEBYTECODE=1 LLAMACPP_STATE=${LLAMACPP_STATE:-$bench/llamacpp-state} LLAMA_CACHE=$bench/llama_cache
model=$("$T" -c "import json;print(json.load(open('$pkg/llm_config.json'))['model_id'])")
ctx=$("$T" -c "import json;print(next(iter(json.load(open('$pkg/model.json'))['presets'].values()))['execution']['context_capacity'])")
cd "$bench"
step() { echo "== $* ($(date +%T))"; }

step "1/6 pick the llama.cpp ROCm baseline build"
if [ ! -s llamacpp_pick.json ]; then
  # The latest lemonade-sdk/llamacpp-rocm release always competes (pre-approved download, ~25 s):
  # new architectures need recent builds (no installed build knew `flint`).
  newest=$("$S/fetch_llamacpp.sh" "${LLAMACPP_HOME:-$HOME/llamacpp-rocm}" | tail -1) || newest=""
  "$S/pick_llamacpp.sh" "$gguf" $((lport + 9)) "${SEARCH_ROOT:-$HOME}${newest:+:$newest}" || exit 1
fi
LLAMACPP_DIR=$("$T" -c "import json;r=json.load(open('llamacpp_pick.json'))['ranked'];print(r[0]['dir'] if r else '')" 2>/dev/null)
if [ -z "$LLAMACPP_DIR" ]; then
  # No installed llama.cpp runs this GGUF (e.g. an architecture newer than every build on the machine).
  # Loom-only qualification: everything that doesn't need a reference, then stop with exit 4.
  echo "NO_LLAMACPP: no installed llama.cpp ROCm build runs this model (reasons above)."
  echo "  Even the latest lemonade-sdk/llamacpp-rocm release doesn't know this architecture: continue"
  echo "  Loom-only; correctness is then NOT verified against a reference: say so in the report."
  NO_LLAMA=1 LEVEL=quick "$S/gates.sh" "$pkg" "$url" "$gguf" "$bench"
  budget=${BUDGET_H:-0.5}
  "$T" "$R/roofline.py" "$gguf" --cfg "$pkg/llm_config.json" --out roofline.json --budget-h "$budget" --reserve-min 0 | tee roofline.txt
  "$T" "$S/perf.py" "$url" loom-baseline --model "$model" --log perf_log.jsonl | grep "^{" | tail -1
  echo "QUALIFY_RESULT {\"gates\": \"LOOM_ONLY\", \"llamacpp\": null}"
  exit 4
fi
export LLAMACPP_DIR
echo "baseline build: $LLAMACPP_DIR"

step "2/6 one shared llama-server (4 slots, unified KV)"
kv=$("$T" -c "print(min($ctx, 65536) + 1024)")
L=http://127.0.0.1:$lport
"$S/llamacpp_server.sh" "$gguf" "$lport" 4 "$kv" unified || exit 1
stop_llama() { [ "${KEEP_LLAMA:-0}" = 1 ] || "$S/llamacpp_server.sh" --stop "$lport"; }
trap stop_llama EXIT

step "3/6 gates (LEVEL=quick)"
gates_rc=0
LEVEL=quick LLAMA_URL=$L "$S/gates.sh" "$pkg" "$url" "$gguf" "$bench" || { gates_rc=1
  echo "WARNING: a gate failed (gates.json, gates_*.log). Measuring anyway so the budget and targets exist;"
  echo "         fix it (or document a proven near-tie) before finalize: a FAIL there fails the audit."; }

step "4/6 llama.cpp baseline"
"$T" "$S/perf.py" "$L" llama-rocm --kind llama --log perf_log.jsonl | tail -1
"$T" "$S/perf.py" "$L" llama-rocm --kind llama --long --log perf_log.jsonl | tail -1
"$T" "$S/c4_steady.py" "$L" 4 --label llama-rocm --log perf_log.jsonl | tail -1
stop_llama

step "5/6 roofline + targets + budget"
[ -z "${E2E_BUDGET_MIN:-}" ] && export MODEL_BUDGET_MIN=$("$T" "$R/e2e_budget.py" "$gguf" --cfg "$pkg/llm_config.json" --context "$ctx" --json | "$T" -c "import json,sys;print(json.load(sys.stdin)['budget_min'])")
"$T" "$R/e2e_budget.py" "$gguf" --cfg "$pkg/llm_config.json" --context "$ctx"
budget=${BUDGET_H:-$("$T" - "$bench" <<'EOF'
import json, os, sys, time
bench = sys.argv[1]
start = next((float(open(p).read().split()[0]) for p in (os.path.join(bench, "..", ".e2e_start"), os.path.join(bench, ".e2e_start"))
              if os.path.exists(p)), time.time())
# The whole job's budget is a function of this model's speed (loom-roofline/e2e_budget.py), unless the
# user gave one (E2E_BUDGET_MIN).
total = float(os.environ.get("E2E_BUDGET_MIN") or os.environ.get("MODEL_BUDGET_MIN") or 60)
g = json.load(open(os.path.join(bench, "gates.json")))
# The final run (LEVEL=full) re-uses the cached llama.cpp references, so it costs about the Loom
# share of this run (~60 %) plus the full-context probe, then a few minutes of perf + report.
reserve = 0.6 * g["seconds"] / 60 + 8
elapsed = (time.time() - start) / 60
opt = max(10.0, total - elapsed - reserve)
print(f"{opt / 60:.2f}")
sys.stderr.write(f"budget: end-to-end {total:.0f} min (this model's roofline budget), elapsed {elapsed:.0f} min, reserve {reserve:.0f} min "
                 f"for final gates + report -> optimization {opt:.0f} min\n")
EOF
)}
longctx=1000,2000,16000; [ "$ctx" -ge 34000 ] && longctx=1000,2000,16000,32000
"$T" "$R/roofline.py" "$gguf" --cfg "$pkg/llm_config.json" --baseline perf_log.jsonl --ctx $longctx \
  --out roofline.json --budget-h "$budget" --reserve-min 0 | tee roofline.txt

step "6/6 Loom baseline"
"$T" "$S/perf.py" "$url" loom-baseline --model "$model" --log perf_log.jsonl | grep -v "^\[campaign\]  " | tail -12
"$T" "$S/perf.py" "$url" loom-baseline --model "$model" --long --log perf_log.jsonl | grep "^{" | tail -1
"$T" "$S/c4_steady.py" "$url" 4 --model "$model" --label loom-baseline --log perf_log.jsonl | grep -v "^\[campaign\]  " | tail -3
"$T" - "$bench" "$budget" <<'EOF'
import json, os, sys
b = sys.argv[1]
g = json.load(open(os.path.join(b, "gates.json")))
roof = json.load(open(os.path.join(b, "roofline.json")))
base = {}
for l in open(os.path.join(b, "perf_log.jsonl")):
    r = json.loads(l)
    if r.get("kind") == "loom":
        base.update({k: v for k, v in r.items() if isinstance(v, (int, float))})
print("QUALIFY_RESULT " + json.dumps({"gates": g["verdict"], "gates_seconds": g["seconds"],
      "llamacpp": json.load(open(os.path.join(b, "llamacpp_pick.json")))["ranked"][0]["dir"],
      "budget_min": round(float(sys.argv[2]) * 60),
      "metrics": {m["metric"]: {k: m.get(k) for k in ("baseline", "target", "stretch", "ceiling")} for m in roof["metrics"]}}))
EOF
exit $gates_rc
