#!/bin/bash
# Everything between "the port serves" and "start optimizing", in one command:
#   1. fetch_llamacpp.sh (latest lemonade-sdk/llamacpp-rocm, pre-approved) + pick_llamacpp.sh: benchmark
#      it and every llama.cpp ROCm build on the machine -> llamacpp_pick.json
#   2. llama.cpp baseline, with Loom stopped: perf.py --kind llama (short and --long) and c4_steady
#      -> perf_log.jsonl. One model at a time: GPU memory is system RAM, so Loom and llama.cpp never
#      run together (Loom is restarted with its own command line afterwards)
#   3. LEVEL=quick gates (goldens, compare, consistency, agent tool loop, sampling, half-context needle)
#      -> a failing gate is reported (exit 1 at the end) but the measurements still run. llama.cpp
#      references are filled with Loom stopped (gates.sh)
#   4. roofline.py -> roofline.json; the optimization budget is what is left of the end-to-end budget
#      (E2E_BUDGET_MIN, default: loom-roofline/e2e_budget.py, from this model's roofline TG/PP) after the time already spent (WORKDIR/.e2e_start, written by
#      port.sh) and a reserve for the final gates + report (measured from this run's gates)
#   5. Loom baseline: perf.py (short, --long) and c4_steady -> the start rows of targets.jsonl
# usage: [LOOM_ROOT=~/loom] [SEARCH_ROOT=$HOME] [E2E_BUDGET_MIN=60] [BUDGET_H=h] qualify.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT=8130]
#   BUDGET_H overrides the computed optimization budget (the user gave a time).
# Prints QUALIFY_RESULT {...}: gates verdict, baseline build, the roofline table, budget minutes.
set -u
. "$(dirname "$0")/_lib.sh"; R=$S/../../loom-roofline
export LOOM_NO_AUDIT=1 LLAMA_CACHE=$bench/llama_cache
cd "$bench"
loom_down   # one model at a time: the picker and the llama.cpp baseline run with Loom stopped
# A package with image input: the shared llama-server also loads the projector (the vision gate uses it).
MMPROJ=$("$T" -c "import json; print((json.load(open('$pkg/llm_config.json')).get('vision') or {}).get('mmproj', ''))" 2>/dev/null); export MMPROJ
trap '"$S/llamacpp_server.sh" --stop "$lport"; loom_up' EXIT   # whatever happens, Loom serves again at the end

# Speculative decoding: which drafter (none, the GGUF's MTP block, a DFlash drafter) and how deep is decided
# by measured decode speed (loom-port-llm/starter/draft/tune.py rebuilds PKG with the winner). Loom is down;
# the candidates are served alone on Loom's port. SPEC=off / SPEC_TUNE=0 skip it.
work=$(dirname "$pkg"); kit=${LOOM_KIT:-$S/../../loom-port-llm/starter}
if [ "${SPEC_TUNE:-1}" = 1 ] && [ -f "$work/cfg.json" ] && [ ! -s draft_tune.json ] && [ ${#loom_argv[@]} -gt 0 ]; then
  step "0/5 speculative decoding: method and depth by measured decode speed"
  mb=$(printf '%s\n' "${loom_argv[@]}" | grep -o -- '--memory_bytes=[0-9]*' | cut -d= -f2)
  VENV=${VENV:-$root/.venv} LOOM_SERVE_STATE=${LOOM_SERVE_STATE:-$work/state} "$T" "$kit/draft/tune.py" "$work" "$loom_port" "$(( (mb + 1073741823) / 1073741824 ))" "$model" 2>&1 | tee draft_tune.log | grep "^tune:\|^TUNE_RESULT"
  grep "^TUNE_RESULT" draft_tune.log | tail -1 | cut -d' ' -f2- > draft_tune.json
  newmb=$("$T" -c "import json; print(json.load(open('draft_tune.json')).get('memory_bytes', ''))" 2>/dev/null)
  if [ -n "$newmb" ]; then   # the restart serves the tuned package with its own memory budget
    for i in "${!loom_argv[@]}"; do case "${loom_argv[$i]}" in --memory_bytes=*) loom_argv[$i]="--memory_bytes=$newmb";; esac; done
  fi
fi

step "1/5 pick the llama.cpp ROCm baseline build"
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
  loom_up || exit 1
  NO_LLAMA=1 LEVEL=quick "$S/gates.sh" "$pkg" "$url" "$gguf" "$bench"
  budget=${BUDGET_H:-0.5}
  "$T" "$R/roofline.py" "$gguf" --cfg "$pkg/llm_config.json" --out roofline.json --budget-h "$budget" --reserve-min 0 | tee roofline.txt
  "$T" "$S/perf.py" "$url" loom-baseline --model "$model" --log perf_log.jsonl | grep "^{" | tail -1
  echo "QUALIFY_RESULT {\"gates\": \"LOOM_ONLY\", \"llamacpp\": null}"
  exit 4
fi
export LLAMACPP_DIR
echo "baseline build: $LLAMACPP_DIR"

step "2/5 llama.cpp baseline (Loom stopped)"
L=http://127.0.0.1:$lport
"$S/llamacpp_server.sh" "$gguf" "$lport" 4 "$(( (ctx < 34000 ? ctx : 34000) + 1024 ))" unified || exit 1
"$T" "$S/perf.py" "$L" llama-rocm --kind llama --log perf_log.jsonl | tail -1
"$T" "$S/perf.py" "$L" llama-rocm --kind llama --long --log perf_log.jsonl | tail -1
"$T" "$S/c4_steady.py" "$L" 4 --label llama-rocm --log perf_log.jsonl | tail -1
"$S/llamacpp_server.sh" --stop "$lport"
loom_up || exit 1

step "3/5 gates (LEVEL=quick)"
gates_rc=0
LEVEL=quick "$S/gates.sh" "$pkg" "$url" "$gguf" "$bench" "$lport" || { gates_rc=1
  echo "WARNING: a gate failed (gates.json, gates_*.log). Measuring anyway so the budget and targets exist;"
  echo "         fix it (or document a proven near-tie) before finalize: a FAIL there fails the audit."; }

step "4/5 roofline + targets + budget"
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

step "5/5 Loom baseline"
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
