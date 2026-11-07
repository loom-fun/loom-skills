#!/bin/bash
# Diagnostics for the Ornith-1.5-35B-A3B gate failures (devlab-port "post"; W = port_e2e work dir, SK, LOOM_ROOT):
#   1. the vision encoder vs the NumPy reference (checked against llama-mtmd) at the wide image's processed size 1600x256
#   2. the failing requests (vision "wide", agent thinking off) on numerics variants of the same package: as built,
#      f32 activations (ACT_Q8=0), decode-only (NO_GEMM=1: no prefill GEMMs), and drafting off. A token that moves with
#      the numerics is a near-tie of Loom's own; one that every variant keeps, while llama.cpp disagrees, is a bug.
set -u
B=$SK/loom-benchmark/scripts; K=$SK/loom-port-llm/starter; S=$SK/loom-setup/scripts
T=$LOOM_ROOT/.tools/bin/python; V=$LOOM_ROOT/.venv
export LOOM_SERVE_STATE=$W/diag-state LLAMACPP_STATE=$W/diag-llstate LLAMA_CACHE=$W/bench/llama_cache PYTHONDONTWRITEBYTECODE=1
export LLAMACPP_DIR=$($T -c "import json; print(json.load(open('$W/bench/llamacpp_pick.json'))['ranked'][0]['dir'])")
mm=$($T -c "import json; print(json.load(open('$W/pkg/llm_config.json'))['vision']['mmproj'])")
model=$($T -c "import json; print(json.load(open('$W/pkg/llm_config.json'))['model_id'])")
gguf=$(ls $W/gguf/*.gguf | head -1)
echo "== 1. encoder vs ref_vit at 256x1600 ($mm)"
VENV=$V LOOM_ROOT=$LOOM_ROOT $K/vision/tests/run_tests.sh "$mm" 256 1600 2>&1 | tail -4
cd $W
for v in asbuilt act_f32 nogemm nodraft; do
  echo "== 2. variant $v"
  case $v in
    asbuilt) envs=();; act_f32) envs=(ACT_Q8=0);; nogemm) envs=(NO_GEMM=1);; nodraft) envs=();;
  esac
  draft=(); [ $v = nodraft ] && draft=(--draft none)
  rm -rf pkg_$v
  env "${envs[@]}" LOOM_ROOT=$LOOM_ROOT $T $K/build.py cfg.json pkg_$v --tokenizer tok/tokenizer.json --model-id "$model" "${draft[@]}" > build_$v.log 2>&1 || { tail -3 build_$v.log; continue; }
  mem=$(grep -o -- "--memory_bytes=[0-9]*" build_$v.log | cut -d= -f2)
  $S/stop.sh diag > /dev/null 2>&1
  VENV=$V $S/serve.sh diag 8162 amdgpu $mem pkg_$v/model.json > /dev/null || { echo "serve failed"; continue; }
  mkdir -p diag_$v && cd diag_$v
  MMPROJ=$mm $T $B/vision_golden.py ../pkg_$v --url http://127.0.0.1:8162 --cases wide 2>&1 | grep -E "^(PASS|FAIL)"
  $T -c "import json; r = json.load(open('vision_golden.json'))[0]; print('  loom :', repr(r['loom'][:120])); print('  llama:', repr(r['llama'][:120]))"
  $T $B/agent_golden.py ../pkg_$v http://127.0.0.1:8162 --llama http://127.0.0.1:8130 --thinking off --out agent_think_off.json > agent.log 2>&1; grep -v "^OK " agent.log | tail -6
  cd $W
  $S/stop.sh diag > /dev/null 2>&1
done
