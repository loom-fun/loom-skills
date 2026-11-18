#!/usr/bin/env bash
# Run exactly one serial current-Loom frontier comparison on designated DevLab.
# Launch under: timeout --signal=TERM --kill-after=15s 1800s run_frontier.sh OUT_DIR
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: run_frontier.sh OUT_DIR}
test "${GEMMA_E4B_DEVLAB_ACK:-0}" = 1 || { echo 'DevLab acknowledgement missing' >&2; exit 19; }
test -n "${GEMMA_E4B_GGUF:-}" || { echo 'set GEMMA_E4B_GGUF to the exact official-cache checkpoint' >&2; exit 20; }
test -n "${GEMMA_E4B_TOKENIZER:-}" || { echo 'set GEMMA_E4B_TOKENIZER to the pinned tokenizer' >&2; exit 20; }
SDK_ROOT=${LOOM_SDK_ROOT:?run_job.sh must install/verify the pinned SDK first}
VENV="$SDK_ROOT/.venv"
test -x "$VENV/bin/loom-serve" || { echo 'pinned loom-serve is missing; fail before weight hash or GPU start' >&2; exit 21; }
test -x "$VENV/bin/python" || { echo 'pinned Python is missing' >&2; exit 21; }
mkdir -p "$OUT"
ROOT=$(git rev-parse --show-toplevel)
SV="$ROOT/skills/loom-setup/scripts"
test "$(sha256sum "$SV/serve.sh" | cut -d' ' -f1)" = 05ff396c1e5d889ad16226183d7f98a2531e5114293a4eabf783fc1e0223dba3 || { echo 'clean-source serve.sh pin mismatch' >&2; exit 18; }
test "$(sha256sum "$SV/stop.sh" | cut -d' ' -f1)" = 49bccbd184270d2d1106daf845bc3de888b836d6a44dce3f646152eef9047f3e || { echo 'clean-source stop.sh pin mismatch' >&2; exit 18; }
test "$(sha256sum "$SV/memcheck.sh" | cut -d' ' -f1)" = c590ff8b5a05e2190560048ddc7de2ff09cac646517826dbdcaeb650acc4f895 || { echo 'clean-source memcheck.sh pin mismatch' >&2; exit 18; }
printf 'source_checkout_head=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" > "$OUT/run_provenance.txt"

# This hashes the actual checkpoint exactly once, refuses source drift, and builds both arms.
GEMMA_E4B_DEVLAB_ACK=1 GEMMA_E4B_GGUF="$GEMMA_E4B_GGUF" \
GEMMA_E4B_TOKENIZER="$GEMMA_E4B_TOKENIZER" \
GEMMA_E4B_FETCH_BYTES="${GEMMA_E4B_FETCH_BYTES:-unknown}" \
GEMMA_E4B_FETCH_SECONDS="${GEMMA_E4B_FETCH_SECONDS:-unknown}" \
  "$HERE/prepare_packages.sh" "$OUT/builds"

BASELINE="$OUT/builds/baseline"
CANDIDATE="$OUT/builds/candidate"
"$VENV/bin/python" "$HERE/runner/sdk_pin.py" matches "$SDK_ROOT" "$CANDIDATE/llm_config.json" || {
  echo 'installed SDK does not match the frozen package pin; fail before GPU start' >&2; exit 22;
}

ACTIVE_NAME=''; ACTIVE_MONITOR=''; ACTIVE_DIR=''
cleanup() {
  if [ -n "$ACTIVE_NAME" ]; then
    LOOM_SERVE_STATE="$ACTIVE_DIR/state" "$SV/stop.sh" "$ACTIVE_NAME" >/dev/null 2>&1 || true
  fi
  if [ -n "$ACTIVE_MONITOR" ] && [ -n "$ACTIVE_DIR" ]; then
    touch "$ACTIVE_DIR/monitor.stop"
    wait "$ACTIVE_MONITOR" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

run_arm() {
  local arm=$1 package=$2 port=$3
  local name="gemma-e4b-min-ram-$arm" dir="$OUT/results/$arm"
  mkdir -p "$dir/state"
  ACTIVE_NAME="$name"; ACTIVE_DIR="$dir"
  local cgroup_args=()
  if [ -n "${GEMMA_E4B_CGROUP_ROOT:-}" ]; then
    cgroup_args=(--cgroup-root "$GEMMA_E4B_CGROUP_ROOT")
  fi
  "$VENV/bin/python" "$HERE/cold_capacity_targeted.py" monitor \
    --pid-file "$dir/state/$name.pid" --model "$package/model.json" \
    --timeout 1740 --interval 0.1 --out "$dir" "${cgroup_args[@]}" &
  ACTIVE_MONITOR=$!
  for _ in $(seq 1 100); do
    [ -f "$dir/monitor.ready" ] && break
    sleep 0.1
  done
  [ -f "$dir/monitor.ready" ] || { echo "monitor failed to start for $arm" >&2; return 30; }
  printf '%s\n' "serve_$arm" > "$dir/phase"
  LOOM_ACTIVE_WAIT=1 VENV="$VENV" LOOM_SERVE_STATE="$dir/state" \
    timeout --signal=TERM --kill-after=10s 240s \
      "$SV/serve.sh" "$name" "$port" amdgpu none "$package/model.json" -- --pool_backing=fixed \
      > "$dir/serve.log" 2>&1 || { cat "$dir/serve.log" >&2; return 31; }
  printf '%s\n' "probe_$arm" > "$dir/phase"
  timeout --signal=TERM --kill-after=10s 900s "$VENV/bin/python" \
    "$HERE/cold_capacity_targeted.py" probe "$package" "http://127.0.0.1:$port" \
    --preset chat --target-prompt-tokens 65535 --timeout 600 --out "$dir" \
    > "$dir/probe.stdout" 2>&1 || { cat "$dir/probe.stdout" >&2; return 32; }
  LOOM_SERVE_STATE="$dir/state" "$SV/stop.sh" "$name" || return 33
  ACTIVE_NAME=''
  touch "$dir/monitor.stop"
  wait "$ACTIVE_MONITOR" || return 34
  ACTIVE_MONITOR=''
  ACTIVE_DIR=''
  python3 - "$dir/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if not d.get('passed'):
    raise SystemExit(d.get('error','probe failed'))
PY
  python3 "$HERE/cold_capacity_targeted.py" validate-monitor \
    "$dir/memory_summary.json" "$dir/monitor_metadata.json" "$dir/summary.json" \
    > "$dir/memory_coverage.json"
}

# Sample a stable pre-arm idle baseline for the target PCI device before any server starts.
timeout --signal=TERM --kill-after=2s 15s "$VENV/bin/python" "$HERE/cold_capacity_targeted.py" capture-idle \
  --samples 5 --interval 1 --stable-range-bytes 16777216 --out "$OUT/pre_arm1_idle.json" || {
    echo 'pre-arm GPU counters are not stable/attributable; no arm was started' >&2; exit 39;
  }

# Candidate first, then baseline. Servers never overlap. Only HTTP 503 from the same first short
# answer may retry within 120 seconds; no hidden warmup or restart. Then require target VRAM and GTT
# to settle near pre-arm idle for five stable samples, with 60 seconds maximum, before arm2 starts.
run_arm candidate "$CANDIDATE" 18871
if ! timeout --signal=TERM --kill-after=5s 70s "$VENV/bin/python" \
    "$HERE/cold_capacity_targeted.py" settle --idle "$OUT/pre_arm1_idle.json" \
    --timeout 60 --interval 1 --tolerance-bytes 67108864 --stable-range-bytes 16777216 \
    --stable-samples 5 --out "$OUT/inter_arm_settle.json"; then
  python3 - "$OUT/baseline_contamination.json" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); evidence=out.parent/'inter_arm_settle.json'
settle=json.loads(evidence.read_text()) if evidence.exists() else {}
json.dump({'baseline_status':'contaminated_not_run',
           'reason':settle.get('error','inter-arm settle command failed or hit its 70-second process guard'),
           'settle_elapsed_seconds':settle.get('elapsed_seconds',settle.get('settled_after_seconds')),
           'candidate_result_retained':'results/candidate/summary.json',
           'settle_evidence':'inter_arm_settle.json'},open(sys.argv[1],'w'),indent=2)
PY
  echo 'baseline arm refused: target GPU allocations did not settle; candidate is not a RAM win' >&2
  exit 40
fi
run_arm baseline "$BASELINE" 18872
python3 "$HERE/compare_arms.py" "$OUT/results/baseline/summary.json" \
  "$OUT/results/candidate/summary.json" > "$OUT/arm_comparison.json"
python3 - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); cmp=json.loads((out/'arm_comparison.json').read_text())
arms={name:json.loads((out/'results'/name/'summary.json').read_text()) for name in ('baseline','candidate')}
matched=bool(cmp.get('matched_outputs_exactly_equal'))
results={'matched_oracles': 'PASS' if matched else 'FAIL',
         'target_only_frontier': 'PASS',
         'candidate_context_tokens': str(arms['candidate']['execution']['context_capacity']),
         'baseline_context_tokens': str(arms['baseline']['execution']['context_capacity']),
         'candidate_prompt_tokens_per_second': str(round(arms['candidate']['prompt_tokens_per_second'], 3)),
         'baseline_prompt_tokens_per_second': str(round(arms['baseline']['prompt_tokens_per_second'], 3)),
         'candidate_fixed_storage_bytes': str(arms['candidate']['active']['fixed_storage_total_bytes']),
         'baseline_fixed_storage_bytes': str(arms['baseline']['active']['fixed_storage_total_bytes'])}
json.dump({'variant':'gemma-e4b-min-ram','results':results,
           'correctness_match':matched},open(out/'summary.json','w'),indent=2)
PY
echo "bounded two-arm probe passed: $OUT"
