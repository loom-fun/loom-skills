# Sourced by gates.sh, qualify.sh and finalize.sh: their shared arguments, paths and helpers.
# Arguments: PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT=8130]
pkg=$(cd "$1" && pwd); url=$2; bench=$(mkdir -p "$4" && cd "$4" && pwd); lport=${5:-8130}
gguf=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")   # absolute: the scripts cd into BENCH_DIR
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); root=${LOOM_ROOT:-$HOME/loom}
bin=$root/.tools/bin; [ -d "$bin" ] || bin=$root/.tools/Scripts; T=$bin/python   # Windows venvs use Scripts/
export PYTHONDONTWRITEBYTECODE=1 LLAMACPP_STATE=${LLAMACPP_STATE:-$bench/llamacpp-state}
model=$("$T" -c "import json;print(json.load(open('$pkg/llm_config.json'))['model_id'])")
ctx=$("$T" -c "import json;print(next(iter(json.load(open('$pkg/model.json'))['presets'].values()))['execution']['context_capacity'])")
step() { echo "== $* ($(date +%T))"; }
