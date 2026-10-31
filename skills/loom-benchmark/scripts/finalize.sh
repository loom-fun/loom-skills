#!/bin/bash
# The end of an end-to-end job, in one command, on the final (served) build:
#   1. perf on the final build, llama.cpp then Loom back to back (perf.py short/--long + c4_steady);
#      roofline.json baselines are refreshed from this llama.cpp run (SKIP_LLAMA_PERF=1 skips it)
#   2. LEVEL=full gates (llama.cpp references come from the cache that qualify.sh filled; llama-server
#      only starts on a cache miss) - includes the full-context needle probe
#   3. audit.py (definition of done)
#   4. report.py -> BENCH_DIR/report.html + report.md (every number generated from the logs)
# usage: [LOOM_ROOT=~/loom] finalize.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT=8130]
# Exit 1 if a gate or the audit fails (the report is still written, with the failure in it).
set -u
pkg=$(cd "$1" && pwd); url=$2; bench=$(mkdir -p "$4" && cd "$4" && pwd); lport=${5:-8130}
gguf=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
S=$(cd "$(dirname "$0")" && pwd); root=${LOOM_ROOT:-$HOME/loom}
bin=$root/.tools/bin; [ -d "$bin" ] || bin=$root/.tools/Scripts; T=$bin/python
export PYTHONDONTWRITEBYTECODE=1 LLAMACPP_STATE=${LLAMACPP_STATE:-$bench/llamacpp-state}
model=$("$T" -c "import json;print(json.load(open('$pkg/llm_config.json'))['model_id'])")
[ -s "$bench/llamacpp_pick.json" ] && export LLAMACPP_DIR=${LLAMACPP_DIR:-$("$T" -c "import json;print(json.load(open('$bench/llamacpp_pick.json'))['ranked'][0]['dir'])")}
cd "$bench"
step() { echo "== $* ($(date +%T))"; }
rc=0
step "1/4 final perf: llama.cpp and Loom back to back"
# Load swings between qualify and now (or an A/B run left on the GPU) skew the comparison: measure
# both engines here, one right after the other, and refresh the roofline baselines from it.
if [ -n "${LLAMACPP_DIR:-}" ] && [ "${SKIP_LLAMA_PERF:-0}" != 1 ]; then
  kv=$("$T" -c "import json;print(min(next(iter(json.load(open('$pkg/model.json'))['presets'].values()))['execution']['context_capacity'], 34000) + 1024)")
  if "$S/llamacpp_server.sh" "$gguf" "$lport" 4 "$kv" unified; then
    L=http://127.0.0.1:$lport
    "$T" "$S/perf.py" "$L" llama-final --kind llama --log perf_log.jsonl | grep "^{" | tail -1
    "$T" "$S/perf.py" "$L" llama-final --kind llama --long --log perf_log.jsonl | grep "^{" | tail -1
    "$T" "$S/c4_steady.py" "$L" 4 --label llama-final --log perf_log.jsonl | tail -1
    kill "$(cat "$LLAMACPP_STATE/llamacpp-$lport.pid")" 2>/dev/null; rm -f "$LLAMACPP_STATE/llamacpp-$lport.pid"; sleep 2
    "$T" - <<'PY'
import json
base = {}
for l in open("perf_log.jsonl"):
    r = json.loads(l)
    if r.get("kind") != "llama":
        continue
    for k in ("decode_tps", "decode_tps_2k", "decode_tps_32k", "c4_steady"):
        if k in r:
            base[k] = r[k]
    for p in r.get("prefill", []) + r.get("prefill_long", []):
        n = min((1000, 2000, 8000, 16000, 32000), key=lambda x: abs(x - p["prompt_tokens"]))
        base[f"prefill_{n}"] = p["prefill_tps"]
roof = json.load(open("roofline.json"))
for m in roof["metrics"]:
    b = base.get(m["metric"])
    if b:
        m.update(baseline=b, baseline_eff=round(b / m["ceiling"], 3), margin=round(m["target"] / b - 1, 3),
                 stretch_margin=round(m["stretch"] / b - 1, 3))
json.dump(roof, open("roofline.json", "w"), indent=1)
print("roofline.json baselines refreshed from the back-to-back llama.cpp run")
PY
  else
    echo "WARNING: could not start llama-server; the report uses the qualify baseline"
  fi
fi
"$T" "$S/perf.py" "$url" final --model "$model" --log perf_log.jsonl | grep "^{" | tail -1
"$T" "$S/perf.py" "$url" final --model "$model" --long --log perf_log.jsonl | grep "^{" | tail -1
"$T" "$S/c4_steady.py" "$url" 4 --model "$model" --label final --log perf_log.jsonl | grep -v "^\[campaign\]" | tail -1
step "2/4 gates (LEVEL=full)"
LEVEL=full "$S/gates.sh" "$pkg" "$url" "$gguf" "$bench" "$lport" || rc=1
step "3/4 audit"
"$T" "$S/audit.py" . || rc=1
step "4/4 report"
"$T" "$S/report.py" . --pkg "$pkg" || rc=1
echo "FINALIZE $([ $rc = 0 ] && echo PASS || echo FAIL): $bench/report.html"
exit $rc
