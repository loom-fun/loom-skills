#!/bin/bash
# Vision encoder vs ref_vit.py over image shapes (devlab-port "post"; W, SK, LOOM_ROOT): which shapes fall below
# run_tests.sh's per-row cosine 0.999 (the wide case, 1600x256, did), to localize the op (aspect ratio, width, length).
set -u
K=$SK/loom-port-llm/starter; T=$LOOM_ROOT/.tools/bin/python
pkill -f loom-serve; pkill -f llama-server; sleep 5
mm=$($T -c "import json; print(json.load(open('$W/pkg/llm_config.json'))['vision']['mmproj'])")
cd $K/vision
for hw in "256 1600" "1600 256" "256 1024" "256 512" "512 1600" "256 3200" "480 640" "768 1024" "1024 1024" "288 1216"; do
  set -- $hw
  r=$(VENV=$LOOM_ROOT/.venv VISION_TEST_DIR=$W/vt $T test_encoder.py "$mm" $1 $2 10 1.0e+00 2>&1 | grep -o 'row-cos min [0-9.e-]*\|max abs err [0-9.e-]*' | tr '\n' ' ')
  echo "H=$1 W=$2 T=$(( $1 * $2 / 256 )): $r"
done
