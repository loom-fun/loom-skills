#!/bin/bash
# Start / stop the vLLM ROCm reference (vllm_shim.py: a llama-server look-alike) for a safetensors package, the way
# llamacpp_server.sh does for a GGUF. gates.sh and devlab_perf.sh use it when llm_config.json "source.kind" is "safetensors".
# usage: vllm_server.sh CHECKPOINT_DIR_OR_REPO [PORT=8130] [MAX_MODEL_LEN=36864]
#        vllm_server.sh --stop PORT
#        vllm_server.sh --install       (only the install; prints the venv python as its last line)
#        vllm_server.sh --env           (after --install: shell `export` lines for anything else that runs vLLM from that venv, e.g.
#                                        eval "$(vllm_server.sh --env)" before `vllm serve`: ROCM_PATH/PATH/LD_LIBRARY_PATH/PYTORCH_ROCM_ARCH for the HIP dev
#                                        packages Quark's MXFP4 emulation JIT needs, the userland MPI libs)
#   VLLM_VENV      the environment (default $VLLM_STATE/venv); created with uv (python 3.12) on first use
#   VLLM_VERSION / VLLM_ROCM_VARIANT   the pinned wheel (defaults below: vllm.ai/rocm stable index); the cache key names it
#   VLLM_REVISION  checkpoint revision passed to vLLM (recorded in /props and the card)
#   VLLM_STATE     pid/log/mem dir (default /tmp; devlab_perf.sh sets a job dir)
#   VLLM_EXTRA_ENV  space-separated NAME=VALUE for the engine (e.g. VLLM_QUARK_EMU_MEM_OPT=1)
#   Writes $VLLM_STATE/vllm-PORT.{pid,log,mem,install.log}; MemAvailable / RssAnon every 10 s in .mem (memory findings).
set -e
state=${VLLM_STATE:-/tmp}; mkdir -p "$state"
VLLM_VERSION=${VLLM_VERSION:-0.18.0}; VLLM_ROCM_VARIANT=${VLLM_ROCM_VARIANT:-rocm700}
if [ "${1:-}" = --stop ]; then
  f=$state/vllm-${2:?--stop PORT}.pid; [ -f "$f" ] || exit 0; pid=$(cat "$f")
  kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true   # the shim leads a process group with vLLM's EngineCore child: both go
  for _ in $(seq 120); do pgrep -g "$pid" >/dev/null 2>&1 || kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done; kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true   # GPU memory is free once they are gone
  rm -f "$f"; exit 0
fi
venv=${VLLM_VENV:-$state/venv}
ldextra() { export LD_LIBRARY_PATH=$(ls -d "$venv"/extra/usr/lib/*/ "$venv"/extra/usr/lib/*/openmpi/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}; }
# Quark's MXFP4 emulation (what vLLM runs on a GPU without native MXFP4) JIT-compiles HIP kernels at first use: it needs hipcc + the HIP headers,
# which the runners lack (R9700: none; no passwordless sudo on Strix). The TheRock pip packages bring them: rocm[libraries,devel] from
# repo.amd.com/rocm/whl/<arch index>, into $venv-rocm. VLLM_NO_HIPDEV=1 skips it (checkpoints that never emulate).
rocmenv() {
  local r
  if [ -d "$venv-rocm" ] && [ -x "$venv-rocm/bin/rocm-sdk" ]; then
    r=$("$venv-rocm/bin/rocm-sdk" path --root 2>/dev/null) || r=
    if [ -n "$r" ]; then export ROCM_PATH=$r HIP_PATH=$r ROCM_HOME=$r PATH=$r/bin:$PATH LD_LIBRARY_PATH=$r/lib:${LD_LIBRARY_PATH:-}; fi
  fi
  [ -n "${GFX_ARCH:-}" ] && export PYTORCH_ROCM_ARCH=$GFX_ARCH
  return 0
}
hipdev() {
  [ "${VLLM_NO_HIPDEV:-0}" = 1 ] && return 0
  GFX_ARCH=$("$venv/bin/python" -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])" 2>/dev/null || true)
  rocmenv
  command -v hipcc >/dev/null && [ -f "${ROCM_PATH:-/opt/rocm}/include/hip/hip_runtime.h" ] && return 0
  local idx; case "$GFX_ARCH" in gfx120*) idx=gfx120X-all;; gfx1151) idx=gfx1151;; gfx1150) idx=gfx1150;; gfx110*) idx=gfx110X-all;; *) idx=gfx120X-all;; esac
  echo "HIP dev packages for $GFX_ARCH ($idx)" | tee -a "$state/vllm-install.log"
  uv venv --clear --python 3.12 "$venv-rocm" >> "$state/vllm-install.log" 2>&1 \
    && uv pip install --python "$venv-rocm/bin/python" "rocm[libraries,devel]" --index-url "https://repo.amd.com/rocm/whl/$idx/" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match >> "$state/vllm-install.log" 2>&1 \
    || { echo "WARNING: rocm[devel] install failed (Quark emulation JIT will fail)"; tail -5 "$state/vllm-install.log"; return 0; }
  rocmenv
  echo "ROCM_PATH=${ROCM_PATH:-unset} hipcc=$(command -v hipcc || echo none) hip_runtime.h=$(ls "${ROCM_PATH:-/x}"/include/hip/hip_runtime.h 2>&1 | tail -1)" | tee -a "$state/vllm-install.log"
}
install() {
  ldextra
  [ -x "$venv/bin/python" ] && "$venv/bin/python" -c "import vllm,sys; sys.exit(0 if vllm.__version__.split('+')[0]=='$VLLM_VERSION' else 1)" 2>/dev/null && { GFX_ARCH=$("$venv/bin/python" -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])" 2>/dev/null || true); rocmenv; return 0; }
  local t0=$(date +%s)
  export PATH=$HOME/.local/bin:$PATH
  if ! command -v uv >/dev/null; then   # no uv on the runner: the official installer (to ~/.local/bin), else pip
    { curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh; } >> "$state/vllm-install.log" 2>&1 \
      || python3 -m pip install --user -q --break-system-packages uv >> "$state/vllm-install.log" 2>&1 || true
  fi
  command -v uv >/dev/null || { echo "no uv and it could not be installed:"; tail -10 "$state/vllm-install.log"; return 1; }
  # triton compiles a launcher against Python.h: system python without the -dev headers uses uv's standalone python instead
  local py="--python 3.12"; [ -f /usr/include/python3.12/Python.h ] || sudo -n apt-get install -y python3.12-dev >> "$state/vllm-install.log" 2>&1 || true
  [ -f /usr/include/python3.12/Python.h ] || py="--managed-python --python 3.12"
  sudo -n apt-get install -y libdrm-amdgpu1 >> "$state/vllm-install.log" 2>&1 || true   # libdrm_amdgpu.so (device names); best effort
  uv venv --clear $py "$venv" && uv pip install --python "$venv/bin/python" "vllm==$VLLM_VERSION" pillow \
    --extra-index-url "https://wheels.vllm.ai/rocm/$VLLM_VERSION/$VLLM_ROCM_VARIANT" --index-strategy unsafe-best-match \
    >> "$state/vllm-install.log" 2>&1 || { echo "vLLM install failed:"; tail -20 "$state/vllm-install.log"; return 1; }
  # torch's ROCm wheel links libmpi_cxx.so.40 (Open MPI) and libnuma: not on a bare runner image
  if ! ldconfig -p 2>/dev/null | grep -q 'libmpi_cxx\.so\.40'; then
    { sudo -n apt-get install -y libopenmpi3t64 libnuma1 || { sudo -n apt-get update -qq && sudo -n apt-get install -y libopenmpi3t64 libnuma1; }; } >> "$state/vllm-install.log" 2>&1 \
      || echo "WARNING: could not apt-install libopenmpi3t64 (torch import will fail)"
  fi
  # no passwordless sudo (Strix runner): the missing .debs are downloaded (no root needed) and unpacked into $venv/extra, put on LD_LIBRARY_PATH
  if ! "$venv/bin/python" -c "import torch" >/dev/null 2>&1; then
    local x=$venv/extra dl=$venv/extra-debs; mkdir -p "$x" "$dl"
    ( cd "$dl" && for pkg in $(apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts --no-breaks --no-replaces --no-enhances libopenmpi3t64 libnuma1 2>/dev/null | grep '^[a-z0-9]'); do
        dpkg -s "$pkg" >/dev/null 2>&1 || apt-get download "$pkg" >/dev/null 2>&1; done
      for d in *.deb; do [ -f "$d" ] && dpkg -x "$d" "$x"; done ) >> "$state/vllm-install.log" 2>&1
  fi
  ldextra
  "$venv/bin/python" -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'devices', torch.cuda.device_count())" 2>&1 | tee -a "$state/vllm-install.log" | tail -3
  hipdev
  echo "INSTALLED vllm $VLLM_VERSION $VLLM_ROCM_VARIANT in $(( $(date +%s) - t0 )) s" | tee -a "$state/vllm-install.log"
}
if [ "${1:-}" = --install ]; then install && echo "$venv/bin/python"; exit; fi
if [ "${1:-}" = --env ]; then
  ldextra
  GFX_ARCH=$("$venv/bin/python" -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])" 2>/dev/null || true); rocmenv
  for v in LD_LIBRARY_PATH ROCM_PATH HIP_PATH ROCM_HOME PATH PYTORCH_ROCM_ARCH; do [ -n "${!v:-}" ] && printf 'export %s=%q\n' "$v" "${!v}"; done; exit 0
fi
ckpt=${1:?usage: vllm_server.sh CHECKPOINT [PORT] [MAX_MODEL_LEN]}; port=${2:-8130}; mlen=${3:-36864}
if curl -s -m 2 localhost:$port/health >/dev/null 2>&1; then echo "port $port is busy"; exit 1; fi
install
here=$(cd "$(dirname "$0")" && pwd)
args=("$ckpt" --port "$port" --max-model-len "$mlen"); [ -n "${VLLM_REVISION:-}" ] && args+=(--revision "$VLLM_REVISION")

ldextra
env ${VLLM_EXTRA_ENV:-} VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0} setsid nohup "$venv/bin/python" "$here/vllm_shim.py" "${args[@]}" > "$state/vllm-$port.log" 2>&1 &
echo $! > "$state/vllm-$port.pid"
( p=$(cat "$state/vllm-$port.pid"); while kill -0 $p 2>/dev/null; do
    echo "$(date +%T) MemAvailable_GiB $(awk '/^MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) RssAnon_GiB $(awk '/^RssAnon/{printf "%.1f", $2/1048576}' /proc/$p/status 2>/dev/null) vram_used_GiB $(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | awk '{s=$1>s?$1:s} END{printf "%.1f", s/1073741824}')"; sleep 10; done ) >> "$state/vllm-$port.mem" 2>/dev/null &
t0=$(date +%s)
for i in $(seq 1800); do   # a 20 GB checkpoint load + engine warmup: minutes
  curl -s localhost:$port/health | grep -q ok && { echo "READY vLLM $VLLM_VERSION on :$port (pid $(cat "$state/vllm-$port.pid"), load $(( $(date +%s) - t0 )) s)"; curl -s localhost:$port/props > "$state/vllm-props.json"; exit 0; }
  kill -0 $(cat "$state/vllm-$port.pid") 2>/dev/null || { echo "vLLM died:"; tail -30 "$state/vllm-$port.log"; exit 1; }
  sleep 1
done
echo "timeout"; exit 1
