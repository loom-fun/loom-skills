#!/bin/bash
# The end of an end-to-end job, in one command, on the final (served) build:
#   1. Loom perf on the final build: perf.py (short, --long) + c4_steady, label "final"
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
step "1/4 final Loom perf"
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
