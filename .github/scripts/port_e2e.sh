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
# The full-context probe is deferred to the nightly DevLab sweep (finalize.sh NEEDLE_FULL=0, audit WARN) unless
# LONG_GATES=1. qualify.sh keeps its LEVEL=quick gates (half-context needle included).
[ "${LONG_GATES:-0}" = 1 ] || export NEEDLE_FULL=0
[ -n "${HF_TOKEN:-}" ] || unset HF_TOKEN   # an empty token (a CI secret this repo lacks) fails every hf download
url=http://127.0.0.1:$port
ph() { echo "$(date +%s) $*" >> "$D/phases.log"; echo "== $* ($(date +%T))"; }
lock() { if [ -n "${GPU_LOCK:-}" ]; then flock -o "$GPU_LOCK" "$@"; else "$@"; fi; }   # -o: servers do not inherit the lock
cleanup() {
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  for f in "$LLAMACPP_STATE"/*.pid "$W"/bench/llamacpp-state/*.pid; do
    [ -f "$f" ] && LLAMACPP_STATE=$(dirname "$f") "$B/llamacpp_server.sh" --stop "$(basename "$f" .pid | sed 's/^llamacpp-//')" >/dev/null 2>&1
  done
  echo "port_e2e.sh: servers stopped"
}
trap cleanup EXIT
result() { printf '{"status": "%s", "stage": "%s", "rc": %s}\n' "$1" "$2" "$3" > "$out/result.json"; cat "$out/result.json"; }
# What a failed or finished port leaves in the artifact: the port stage's logs and config (a failure is debugged from
# these), then the bench files.
save_logs() {
  cp "$D/phases.log" "$out/" 2>/dev/null
  for f in reference.json reference.log cfg.json inspect.log chat_verify.log download.log vision.log drafter.log build.log ktest.log; do
    cp "$W/$f" "$out/" 2>/dev/null
  done
  if [ -d "$W/bench" ]; then
    for f in gates.json audit.json perf_log.jsonl draft_tune.json llamacpp_spec.json llamacpp_pick.json roofline.json \
             targets.jsonl llamacpp_args.txt report.md gates_*.log; do cp "$W/bench"/$f "$out/" 2>/dev/null; done
  fi
  return 0
}

: > "$D/phases.log"
ph install
root=${LOOM_ROOT:-$D/loom}
if [ ! -x "$root/.venv/bin/loom-serve" ] || [ ! -x "$root/.tools/bin/python" ]; then
  "$SV/install_loom.sh" "$root" ${LOOM_VERSION:+"$LOOM_VERSION"} > "$out/install.log" 2>&1 || { tail "$out/install.log"; result failed install 1; exit 1; }
fi
export LOOM_ROOT=$root VENV=$root/.venv; T=$root/.tools/bin/python
# build.py stamps the loom-skills commit (kit_source.py): git when SK is a checkout, else LOOM_SKILLS_COMMIT (an
# exported snapshot such as devlab-port's skills_ref)
[ -n "${LOOM_SKILLS_COMMIT:-}" ] || LOOM_SKILLS_COMMIT=$(git -C "$SK" rev-parse HEAD 2>/dev/null) || true
export LOOM_SKILLS_COMMIT
# llama.cpp builds: only the ones this run fetched (qualify.sh fetches the latest lemonade release)
export LLAMACPP_HOME=${LLAMACPP_HOME:-$D/llamacpp} SEARCH_ROOT=${SEARCH_ROOT:-${LLAMACPP_HOME:-$D/llamacpp}}
mkdir -p "$LLAMACPP_HOME"

ph port
# port.sh leaves its server running for qualify.sh, so the lock is held from port to finalize as one GPU job. Each
# stage runs with fd 9 closed (9>&-): a server it leaves running must not inherit the lock (flock -o, as in lock()).
(
  [ -n "${GPU_LOCK:-}" ] && { exec 9>"$GPU_LOCK"; flock 9; }
  "$P/port.sh" "$spec" "$W" "$port" auto $model_id > "$out/port.log" 2>&1 9>&-; rc=$?
  tail -4 "$out/port.log"
  [ $rc = 0 ] || { echo "port.sh exit $rc"; exit $rc; }
  gguf=$(ls "$W"/gguf/*.gguf | head -1)
  echo "$(date +%s) qualify" >> "$D/phases.log"
  "$B/qualify.sh" "$W/pkg" "$url" "$gguf" "$W/bench" "$lport" > "$out/qualify.log" 2>&1 9>&-; echo "qualify exit $?"
  grep "^QUALIFY_RESULT" "$out/qualify.log" | cut -c1-600
  echo "$(date +%s) finalize" >> "$D/phases.log"
  "$B/finalize.sh" "$W/pkg" "$url" "$gguf" "$W/bench" "$lport" > "$out/finalize.log" 2>&1 9>&-; frc=$?
  echo "finalize exit $frc"; grep "^FINALIZE\|AUDIT" "$out/finalize.log" | tail -3
  exit 0
) ; rc=$?
[ $rc = 0 ] || { ph done; save_logs; result failed port $rc; exit $rc; }

ph evidence
gguf=$(ls "$W"/gguf/*.gguf | head -1); bench=$W/bench
build=$("$T" -c "import json,re;r=json.load(open('$bench/llamacpp_pick.json'))['ranked'];m=re.search(r'b1[0-9]{3,}', r[0]['dir']) if r else None;print(m.group(0) if m else '')" 2>/dev/null)
src=(--package "$W/pkg")
# On DevLab (GITHUB_RUN_ID) the evidence says so: the gates count, but a card's performance numbers come from local
# sessions only, so a card published with this perf session is flagged for a local re-measure (card_lib.perf_flags
# "devlab", org_status remeasure: devlab-primary) until a local session replaces it.
[ -n "${GITHUB_RUN_ID:-}" ] && src+=(--source-kind devlab --run "$GITHUB_RUN_ID" --url "${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-loom-fun/loom-skills}/actions/runs/$GITHUB_RUN_ID")
if [ -n "$build" ] && [ -s "$bench/perf_log.jsonl" ]; then
  # 4 streams: llama.cpp's c4_steady rows (perf_log.jsonl) and Loom's (targets.jsonl), the final session's only
  "$T" -c "import json; [print(r['c4_steady']) for r in map(json.loads, open('$bench/perf_log.jsonl')) if r.get('label') == 'llama-final' and r.get('c4_steady')]" > "$out/c4_llama.txt"
  "$T" -c "import json; [print(r['value']) for r in map(json.loads, open('$bench/targets.jsonl')) if r.get('metric') == 'c4_steady' and r.get('label') == 'final']" > "$out/c4_loom.txt" 2>/dev/null
  c4=(); [ -s "$out/c4_loom.txt" ] && c4+=(--c4-loom "$out/c4_loom.txt"); [ -s "$out/c4_llama.txt" ] && c4+=(--c4-llama "$out/c4_llama.txt")
  # no machine paths on a published card: the HF cache as ~/.cache/huggingface, the work dir and home as ~
  hfh=${HF_HOME:-$HOME/.cache/huggingface}
  args=$(sed -e "s|$hfh|~/.cache/huggingface|g" -e "s|$D|~|g" -e "s|$HOME|~|g" "$bench/llamacpp_args.txt" 2>/dev/null)
  "$T" "$B/card/make_info.py" perf --perf-log "$bench/perf_log.jsonl" --baseline-version "llamacpp-rocm/$build" \
    ${args:+--baseline-args "$args"} "${c4[@]}" "${src[@]}" -o "$out/ev_perf.json" || echo "WARNING: no perf evidence"
fi
"$T" "$B/card/make_info.py" gates "$bench/gates.json" "$W/ktest.log" --time "$(date -u -r "$W/ktest.log" +%FT%T)" "${src[@]}" -o "$out/ev_gates.json" || echo "WARNING: no gate evidence"
source_repo=${spec%%:*}
"$T" "$here/skills/loom-benchmark/scripts/card/publish_package.py" prepare "$W/pkg" "$out/publish" --source "$source_repo" \
  --loom-fetch "$root/.venv/bin/loom-fetch" > "$out/publish.log" 2>&1 || { tail "$out/publish.log"; echo "WARNING: publish prepare failed"; }
ph done
save_logs
v=$("$T" -c "import json; print(json.load(open('$bench/gates.json'))['verdict'])" 2>/dev/null)
a=$(grep -o "AUDIT [A-Z]*" "$out/finalize.log" | tail -1)
echo "gates: $v; audit: $a"
if [ "$v" = PASS ]; then result ok done 0; else result failed gates 1; exit 1; fi   # a failed gate fails the job
