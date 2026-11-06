#!/bin/bash
# The end of an end-to-end job, in one command, on the final (served) build:
#   1. perf on the final build: llama.cpp with Loom stopped, then Loom (short set twice each, long + c4 once);
#      roofline.json baselines are refreshed from these llama.cpp runs (SKIP_LLAMA_PERF=1 skips it)
#   2. LEVEL=full gates (llama.cpp references come from the cache that qualify.sh filled; llama-server
#      only starts on a cache miss) - includes the full-context needle probe
#   3. audit.py (definition of done)
#   4. report.py -> BENCH_DIR/report.html + report.md (every number generated from the logs)
# usage: [LOOM_ROOT=~/loom] finalize.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT=8130]
# Exit 1 if a gate or the audit fails (the report is still written, with the failure in it).
set -u
. "$(dirname "$0")/_lib.sh"
[ -s "$bench/llamacpp_pick.json" ] && export LLAMACPP_DIR=${LLAMACPP_DIR:-$("$T" -c "import json;r=json.load(open('$bench/llamacpp_pick.json'))['ranked'];print(r[0]['dir'] if r else '')")}
[ -z "${LLAMACPP_DIR:-}" ] && { export NO_LLAMA=1; echo "NO_LLAMACPP: Loom-only final gates and perf (no reference build runs this model)"; }
cd "$bench"
rc=0
step "1/4 final perf: llama.cpp (Loom stopped), then Loom"
# One model at a time (GPU memory is system RAM): llama.cpp runs its short set (2 rounds, 1 for slow
# models), the long set and c4 with Loom stopped, then Loom does the same. Each engine's headline is
# the median of its runs (audit: final label; roofline baselines refreshed the same way). Nothing else
# may use the GPU meanwhile. SKIP_LLAMA_PERF=1 measures Loom only.
loom_perf() { "$T" "$S/perf.py" "$url" final --model "$model" --log perf_log.jsonl "$@" | grep "^{" | tail -1; }
# A package with image input: the shared llama-server also loads the projector (the vision gate uses it).
MMPROJ=$("$T" -c "import json; print((json.load(open('$pkg/llm_config.json')).get('vision') or {}).get('mmproj', ''))" 2>/dev/null); export MMPROJ
trap '"$S/llamacpp_server.sh" --stop "$lport"; loom_up' EXIT   # Loom serves again whatever happens
started=0
if [ -n "${LLAMACPP_DIR:-}" ] && [ "${SKIP_LLAMA_PERF:-0}" != 1 ]; then
  loom_down
  kv=$(( (ctx < 34000 ? ctx : 34000) + 1024 ))
  # drafting vs drafting: llama.cpp's speculative setting chosen at qualify (llamacpp_spec.json)
  LLAMA_SPEC=$("$T" -c "import json; print(json.load(open('llamacpp_spec.json'))['args'])" 2>/dev/null) \
    LLAMACPP_ARGS_FILE=llamacpp_args.txt "$S/llamacpp_server.sh" "$gguf" "$lport" 4 "$kv" unified && started=1 || echo "WARNING: could not start llama-server; the report uses the qualify baseline"
fi
L=http://127.0.0.1:$lport
llama_perf() { "$T" "$S/perf.py" "$L" llama-final --kind llama --log perf_log.jsonl "$@" | grep "^{" | tail -1; }
# Two short rounds, one for slow models (< 20 tok/s decode: a round costs minutes there).
rounds=$("$T" -c "
import json
d=[json.loads(l).get('decode_tps') for l in open('perf_log.jsonl') if json.loads(l).get('kind')=='loom'] if __import__('os').path.exists('perf_log.jsonl') else []
d=[x for x in d if x]
print(1 if d and d[-1] < 20 else 2)" 2>/dev/null || echo 2)
if [ $started = 1 ]; then
  for round in $(seq "$rounds"); do llama_perf; done
  llama_perf --long; "$T" "$S/c4_steady.py" "$L" 4 --label llama-final --log perf_log.jsonl | tail -1
  "$S/llamacpp_server.sh" --stop "$lport"
fi
loom_up || exit 1
for round in $(seq "$rounds"); do loom_perf; done
loom_perf --long
"$T" "$S/c4_steady.py" "$url" 4 --model "$model" --label final --log perf_log.jsonl | grep -v "^\[campaign\]" | tail -1
if [ $started = 1 ]; then
  "$T" - "$S" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
from bench_common import perf_metrics
runs = {}
for l in open("perf_log.jsonl"):
    r = json.loads(l)
    if r.get("kind") == "llama" and r.get("label") == "llama-final":
        for k, v in perf_metrics(r).items():
            runs.setdefault(k, []).append(v)
base = {k: sorted(v)[len(v) // 2] for k, v in runs.items()}   # same rule as audit.py's Loom headline
roof = json.load(open("roofline.json"))
for m in roof["metrics"]:
    b = base.get(m["metric"])
    if b:
        m.update(baseline=b, baseline_eff=round(b / m["ceiling"], 3), margin=round(m["target"] / b - 1, 3),
                 stretch_margin=round(m["stretch"] / b - 1, 3))
json.dump(roof, open("roofline.json", "w"), indent=1)
print("roofline.json baselines refreshed from the final llama.cpp runs")
PY
fi
step "2/4 gates (LEVEL=full)"
LEVEL=full "$S/gates.sh" "$pkg" "$url" "$gguf" "$bench" "$lport" || rc=1
step "3/4 audit"
"$T" "$S/audit.py" . || rc=1
step "4/4 report"
"$T" "$S/report.py" . --pkg "$pkg" || rc=1
echo "FINALIZE $([ $rc = 0 ] && echo PASS || echo FAIL): $bench/report.html"
exit $rc
