#!/usr/bin/env bash
# Internal entrypoint; run.sh owns the single whole-job deadline.
set -euo pipefail
ROOT=${LOOM_COMPACT_CHECKOUT:?run through run.sh}
HERE="$ROOT/.github/devlab/compact-kv"
OUT=${1:?absolute output directory required}
HEAD=$(git -C "$ROOT" rev-parse HEAD)
test "${GITHUB_SHA:?exact dispatch SHA required}" = "$HEAD"
python3 "$HERE/materialize_runner.py" --out "$OUT/runner" --expected-head "$HEAD"
exec "$OUT/runner/run_job.sh" "$OUT"
