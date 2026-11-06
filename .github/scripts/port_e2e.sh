#!/bin/bash
# The scripted end-to-end pipeline for one GGUF, without an agent: port.sh -> qualify.sh -> finalize.sh, then the
# card evidence (make_info.py perf + gates, on the machine that measured) and the publishable package
# (publish_package.py prepare: origins + model.json.lock). Optimization is what qualify.sh does by itself
# (speculative-decoding method and depth by measured decode speed); no hand-written levers.
# Runs on any Strix Halo (local, or AMD DevLab Dispatch through .github/workflows/devlab-port.yml).
#
#   [SK=skills dir] [REFERENCE=auto|off|loom-fun/REPO] [LOOM_ROOT=DIR/loom] [GPU_LOCK=/tmp/loom-gpu.lock] \
#     port_e2e.sh ORG/REPO[:FILE.gguf] DIR [PORT=8160] [LLAMA_PORT=8130] [MODEL_ID]
#
# DIR/phases.log (`<epoch> <phase>`: loom-improve's kpi.py reads it), DIR/work (port.sh's WORKDIR: pkg, bench, ...),
# DIR/out: evidence (ev_perf.json, ev_gates.json), publish/ (the package as it is uploaded, without weights),
# logs and result.json. GPU_LOCK: every GPU stage holds this flock (one GPU job at a time on a shared machine).
set -uo pipefail
spec=${1:?usage: port_e2e.sh ORG/REPO[:FILE] DIR [PORT] [LLAMA_PORT] [MODEL_ID]}
D=$(mkdir -p "$2" && cd "$2" && pwd); port=${3:-8160}; lport=${4:-8130}; model_id=${5:-}
here=$(cd "$(dirname "$0")/../.." && pwd); SK=${SK:-$here/skills}
B=$SK/loom-benchmark/scripts; SV=$SK/loom-setup/scripts; P=$SK/loom-port-llm
W=$D/work; out=$D/out; mkdir -p "$W" "$out"
export LOOM_SERVE_STATE=$W/state LLAMACPP_STATE=$W/llstate PYTHONDONTWRITEBYTECODE=1
# Long-context gates (needle at half the context, the full-context probe) are deferred to the nightly DevLab run unless
# LONG_GATES=1: finalize.sh with NEEDLE_FULL=0 defers the full-context probe to the nightly DevLab sweep (audit WARN).
[ "${LONG_GATES:-0}" = 1 ] || export NEEDLE_FULL=0
[ -n "${HF_TOKEN:-}" ] || unset HF_TOKEN   # an empty token (a CI secret this repo lacks) fails every hf download
url=http://127.0.0.1:$port
ph() { echo "$(date +%s) $*" >> "$D/phases.log"; echo "== $* ($(date +%T))"; }
lock() { if [ -n "${GPU_LOCK:-}" ]; then flock "$GPU_LOCK" "$@"; else "$@"; fi; }
cleanup() {
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  for f in "$LLAMACPP_STATE"/*.pid "$W"/bench/llamacpp-state/*.pid; do
    [ -f "$f" ] && LLAMACPP_STATE=$(dirname "$f") "$B/llamacpp_server.sh" --stop "$(basename "$f" .pid | sed 's/^llamacpp-//')" >/dev/null 2>&1
  done
  echo "port_e2e.sh: servers stopped"
}
trap cleanup EXIT
result() { printf '{"status": "%s", "stage": "%s", "rc": %s}\n' "$1" "$2" "$3" > "$out/result.json"; cat "$out/result.json"; }

: > "$D/phases.log"
ph install
root=${LOOM_ROOT:-$D/loom}
if [ ! -x "$root/.venv/bin/loom-serve" ] || [ ! -x "$root/.tools/bin/python" ]; then
  "$SV/install_loom.sh" "$root" ${LOOM_VERSION:+"$LOOM_VERSION"} > "$out/install.log" 2>&1 || { tail "$out/install.log"; result failed install 1; exit 1; }
fi
export LOOM_ROOT=$root VENV=$root/.venv; T=$root/.tools/bin/python
# llama.cpp builds: only the ones this run fetched (qualify.sh fetches the latest lemonade release)
export LLAMACPP_HOME=${LLAMACPP_HOME:-$D/llamacpp} SEARCH_ROOT=${SEARCH_ROOT:-${LLAMACPP_HOME:-$D/llamacpp}}
mkdir -p "$LLAMACPP_HOME"

ph port
# port.sh leaves its server running for qualify.sh, so the lock is held from port to finalize as one GPU job.
(
  [ -n "${GPU_LOCK:-}" ] && { exec 9>"$GPU_LOCK"; flock 9; }
  "$P/port.sh" "$spec" "$W" "$port" auto $model_id > "$out/port.log" 2>&1; rc=$?
  tail -4 "$out/port.log"
  [ $rc = 0 ] || { echo "port.sh exit $rc"; exit $rc; }
  gguf=$(ls "$W"/gguf/*.gguf | head -1)
  echo "$(date +%s) qualify" >> "$D/phases.log"
  "$B/qualify.sh" "$W/pkg" "$url" "$gguf" "$W/bench" "$lport" > "$out/qualify.log" 2>&1; echo "qualify exit $?"
  grep "^QUALIFY_RESULT" "$out/qualify.log" | cut -c1-600
  echo "$(date +%s) finalize" >> "$D/phases.log"
  "$B/finalize.sh" "$W/pkg" "$url" "$gguf" "$W/bench" "$lport" > "$out/finalize.log" 2>&1; frc=$?
  echo "finalize exit $frc"; grep "^FINALIZE\|AUDIT" "$out/finalize.log" | tail -3
  exit 0
) ; rc=$?
[ $rc = 0 ] || { ph done; result failed port $rc; exit $rc; }

ph evidence
gguf=$(ls "$W"/gguf/*.gguf | head -1); bench=$W/bench
build=$("$T" -c "import json,re;r=json.load(open('$bench/llamacpp_pick.json'))['ranked'];m=re.search(r'b1[0-9]{3,}', r[0]['dir']) if r else None;print(m.group(0) if m else '')" 2>/dev/null)
src=(--package "$W/pkg")
[ -n "${GITHUB_RUN_ID:-}" ] && src+=(--source-kind devlab --run "$GITHUB_RUN_ID" --url "${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-loom-fun/loom-skills}/actions/runs/$GITHUB_RUN_ID")
if [ -n "$build" ] && [ -s "$bench/perf_log.jsonl" ]; then
  "$T" "$B/card/make_info.py" perf --perf-log "$bench/perf_log.jsonl" --baseline-version "llamacpp-rocm/$build" \
    --baseline-args "$(cat "$bench/llamacpp_args.txt" 2>/dev/null)" "${src[@]}" -o "$out/ev_perf.json" || echo "WARNING: no perf evidence"
fi
"$T" "$B/card/make_info.py" gates "$bench/gates.json" "$W/ktest.log" --time "$(date -u -r "$W/ktest.log" +%FT%T)" "${src[@]}" -o "$out/ev_gates.json" || echo "WARNING: no gate evidence"
source_repo=${spec%%:*}
"$T" "$here/skills/loom-benchmark/scripts/card/publish_package.py" prepare "$W/pkg" "$out/publish" --source "$source_repo" \
  --loom-fetch "$root/.venv/bin/loom-fetch" > "$out/publish.log" 2>&1 || { tail "$out/publish.log"; echo "WARNING: publish prepare failed"; }
cp "$D/phases.log" "$out/" 2>/dev/null
for f in gates.json audit.json perf_log.jsonl draft_tune.json llamacpp_spec.json llamacpp_pick.json roofline.json targets.jsonl \
         llamacpp_args.txt report.md gates_*.log; do cp "$bench"/$f "$out/" 2>/dev/null; done
cp "$W"/reference.json "$W"/cfg.json "$W"/ktest.log "$W"/inspect.log "$W"/build.log "$out/" 2>/dev/null
ph done
v=$("$T" -c "import json; print(json.load(open('$bench/gates.json'))['verdict'])" 2>/dev/null)
a=$(grep -o "AUDIT [A-Z]*" "$out/finalize.log" | tail -1)
echo "gates: $v; audit: $a"
[ "$v" = PASS ] && result ok done 0 || result failed gates 1
