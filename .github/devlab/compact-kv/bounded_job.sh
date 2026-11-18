#!/usr/bin/env bash
# Internal entrypoint; run.sh owns the single whole-job deadline.
set -euo pipefail
ROOT=${LOOM_COMPACT_CHECKOUT:?run through run.sh}
HERE="$ROOT/.github/devlab/compact-kv"
OUT=${1:?absolute output directory required}
HEAD=$(git -C "$ROOT" rev-parse HEAD)
test "${GITHUB_SHA:?exact dispatch SHA required}" = "$HEAD"
RUNNER_PIN=$(python3 -c 'import runpy,sys; print(runpy.run_path(sys.argv[1])["PIN"])' "$HERE/materialize_runner.py")
[[ "$RUNNER_PIN" =~ ^[0-9a-f]{40}$ ]] || { echo 'invalid frozen runner commit' >&2; exit 20; }
mkdir -p "$OUT"
printf 'git fetch --no-tags --depth=1 origin %s (55s TERM +5s KILL)\n' "$RUNNER_PIN" > "$OUT/runner-source-fetch.log"
GIT_TERMINAL_PROMPT=0 timeout --signal=TERM --kill-after=5s 55s \
  git -C "$ROOT" fetch --no-tags --depth=1 origin "$RUNNER_PIN" >> "$OUT/runner-source-fetch.log" 2>&1
git -C "$ROOT" cat-file -e "$RUNNER_PIN^{commit}"
printf 'verified runner commit %s\n' "$RUNNER_PIN" >> "$OUT/runner-source-fetch.log"
python3 "$HERE/materialize_runner.py" --out "$OUT/runner" --expected-head "$HEAD"
exec "$OUT/runner/run_job.sh" "$OUT"
