#!/bin/bash
# usage: VENV=~/loom/.venv run_tests.sh [MMPROJ.gguf H W]   - vision kernel unit tests (vs NumPy), then, given an
# mmproj and a processed image size (multiples of patch*merge), the whole encoder vs ref_vit.py
# (per-row cosine; ref_vit itself is checked against llama-mtmd-debug, see reference/vision.md).
here=$(cd "$(dirname "$0")" && pwd); T=${PY:-${LOOM_ROOT:-$HOME/loom}/.tools/bin/python}; bad=0
export VENV=${VENV:-${LOOM_ROOT:-$HOME/loom}/.venv}
cd "$here"
for t in test_basic test_gemm test_embed test_attn test_resize_hdr; do
  out=$("$T" "$t.py" 2>&1); echo "$out" | grep -E "^(PASS|FAIL|COMPILE)"; echo "$out" | grep -q -E "^(FAIL|COMPILE)" && bad=1
done
if [ -n "${1:-}" ]; then
  out=$("$T" ../test_encoder.py "$1" "$2" "$3" 10 1.0e+00 2>&1); echo "$out" | tail -3
  echo "$out" | grep -q "row-cos min 0.999" || { echo "FAIL encoder: per-row cosine below 0.999"; bad=1; }
fi
exit $bad
