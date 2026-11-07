#!/bin/bash
# Diagnostics for the Ornith-1.5-122B long-context miss (devlab-port "post"; W, SK, LOOM_ROOT): the 61k needle and the
# compare goldens on variants of the same package: as built, prefill attention on the split-K path (NO_FLASH2=1:
# no head-grouped flash kernel), and decode-only numerics for linears (ACT_Q8=0). Gate references come from the cache.
set -u
B=$SK/loom-benchmark/scripts; K=$SK/loom-port-llm/starter; S=$SK/loom-setup/scripts
T=$LOOM_ROOT/.tools/bin/python; V=$LOOM_ROOT/.venv
export LOOM_SERVE_STATE=$W/diag-state LLAMACPP_STATE=$W/diag-llstate LLAMA_CACHE=$W/bench/llama_cache PYTHONDONTWRITEBYTECODE=1
export LLAMACPP_DIR=$($T -c "import json; print(json.load(open('$W/bench/llamacpp_pick.json'))['ranked'][0]['dir'])")
model=$($T -c "import json; print(json.load(open('$W/pkg/llm_config.json'))['model_id'])")
mem=$($T -c "import re; print(re.findall(r'--memory_bytes=(\d+)', open('$W/build.log').read())[-1])")
cd $W
for v in asbuilt noflash actf32; do
  echo "== variant $v"
  case $v in asbuilt) envs=(X=1);; noflash) envs=(NO_FLASH2=1);; actf32) envs=(ACT_Q8=0);; esac
  rm -rf pkg_$v
  env "${envs[@]}" LOOM_ROOT=$LOOM_ROOT $T $K/build.py cfg.json pkg_$v --tokenizer tok/tokenizer.json --model-id "$model" --draft none > build_$v.log 2>&1 || { tail -3 build_$v.log; continue; }
  $S/stop.sh diag > /dev/null 2>&1
  VENV=$V $S/serve.sh diag 8162 amdgpu $mem pkg_$v/model.json > /dev/null || { echo "serve failed"; continue; }
  mkdir -p diag_$v && cd diag_$v
  $T $B/needle.py ../pkg_$v http://127.0.0.1:8162 --llama http://127.0.0.1:8130 --llama-max 62000 --out needle.json --sizes 61440 --probes 1 --no-over-capacity 2>&1 | grep -E "^(OK|FAIL)" | cut -c1-260
  $T $B/compare.py ../pkg_$v $W/bench/golden.json --llama http://127.0.0.1:8130 --url http://127.0.0.1:8162 --out compare.json 2>&1 | tail -1
  cd $W
  $S/stop.sh diag > /dev/null 2>&1
done
