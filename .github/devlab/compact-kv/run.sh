#!/usr/bin/env bash
# Designated Strix DevLab entrypoint only. 1790s TERM +10s KILL <=1800s total.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
test "${GEMMA_E4B_DEVLAB_ACK:-0}" = 1 || { echo 'reviewed DevLab acknowledgement required' >&2; exit 19; }
OUT=${1:?usage: run.sh ABSOLUTE_OUTPUT_DIR}
[[ "$OUT" = /* ]] || { echo 'absolute output path required' >&2; exit 20; }
export LOOM_COMPACT_CHECKOUT="$ROOT" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
exec timeout --signal=TERM --kill-after=10s 1790s bash "$HERE/bounded_job.sh" "$OUT"
