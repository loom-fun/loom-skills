#!/bin/bash
# Start the llama.cpp ROCm reference server (lemonade-sdk/llamacpp-rocm build) on a GGUF.
# usage: LLAMACPP_DIR=~/llamacpp-rocm llamacpp_server.sh MODEL.gguf [PORT=8130] [PARALLEL=4] [CTX_PER_SLOT=8192] [unified]
#   golden / long-context runs: PARALLEL=1 CTX_PER_SLOT=34000
#   `unified` (5th arg): one KV pool of CTX_PER_SLOT tokens shared by PARALLEL slots (-kvu), so a single
#   request can use all of it and 4 streams still run: one server covers goldens, needle and perf.
#   LLAMA_SPEC="--spec-type draft-mtp --spec-draft-n-max 3" (llama_spec_tune.py's choice) adds speculative
#   decoding: the performance runs use it, so Loom is compared drafting vs drafting; references never do.
#   stop: llamacpp_server.sh --stop PORT   (LLAMACPP_STATE: pid/log dir, default /tmp)
#   LLAMACPP_ARGS_FILE=F writes the exact server arguments to F (finalize.sh: llamacpp_args.txt, the card's baseline)
# Refuses a build that does not see a ROCm device: on Windows, builds missing their HIP/BLAS DLLs
# silently fall back to CPU.
set -e
state=${LLAMACPP_STATE:-/tmp}; mkdir -p "$state"
if [ "${1:-}" = --stop ]; then   # the server this script started on PORT (only if that pid is still a llama-server)
  f=$state/llamacpp-${2:?--stop PORT}.pid; [ -f "$f" ] || exit 0; pid=$(cat "$f")
  if [ -r /proc/$pid/cmdline ]; then tr '\0' ' ' < /proc/$pid/cmdline | grep -q llama-server && kill "$pid" 2>/dev/null || true
  else kill "$pid" 2>/dev/null || true; fi
  # wait until it has exited (its memory is free) before anything loads the next model
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done; kill -9 "$pid" 2>/dev/null
  rm -f "$f"; exit 0
fi
gguf=$1; port=${2:-8130}; np=${3:-4}; per=${4:-8192}; mode=${5:-}
case "$gguf" in /*|?:*) ;; *) gguf=$PWD/$gguf;; esac   # the server runs from LLAMACPP_DIR
dir=${LLAMACPP_DIR:?set LLAMACPP_DIR to the unzipped llamacpp-rocm release}
listeners() { if command -v ss >/dev/null; then ss -ltnp | grep ":$port "; else netstat -ano -p tcp | tr -d '\r' | grep LISTENING | grep ":$port "; fi; }
if curl -s -m 2 localhost:$port/health >/dev/null 2>&1 || listeners >/dev/null; then
  echo "port $port is busy (another llama-server or loom-serve?)"; listeners || true; exit 1
fi
cd "$dir"
exe=./llama-server; [ -f $exe.exe ] && { exe=$exe.exe; command -v cygpath >/dev/null && gguf=$(cygpath -w "$gguf"); }
# Libraries: the build dir, plus any ROCm runtime it links but doesn't ship (lemonade's rocm-stable
# builds use the TheRock runtime from ~/.cache/lemonade/bin/therock-wheels). LLAMACPP_LIBS= adds dirs.
libpath=$PWD${LLAMACPP_LIBS:+:$LLAMACPP_LIBS}
probe=$exe; [ -f libggml-hip.so ] && probe=./libggml-hip.so   # the HIP backend is dlopened: ldd it, not the exe
if command -v ldd >/dev/null && LD_LIBRARY_PATH=$libpath ldd $probe 2>/dev/null | grep -q "not found"; then
  # Add one runtime install's lib dirs, all of them (the HIP runtime dlopens more than ldd lists).
  for sp in $(ls -d "$HOME"/.cache/lemonade/bin/therock-wheels/*/venv/lib/python*/site-packages 2>/dev/null) /opt/rocm; do
    cand=$libpath
    for d in "$sp"/_rocm_sdk_core/lib "$sp"/_rocm_sdk_core/lib/llvm/lib "$sp"/_rocm_sdk_libraries/lib "$sp"/lib; do
      [ -d "$d" ] && cand=$cand:$d
    done
    LD_LIBRARY_PATH=$cand ldd $probe 2>/dev/null | grep -q "not found" || { libpath=$cand; break; }
  done
  LD_LIBRARY_PATH=$libpath ldd $probe 2>/dev/null | grep "not found" && { echo "$dir/llama-server: missing libraries (set LLAMACPP_LIBS)"; exit 1; }
fi
export LD_LIBRARY_PATH=$libpath
$exe --list-devices 2>&1 | grep -q ROCm || { echo "$dir/llama-server sees no ROCm device:"; $exe --list-devices 2>&1 | tail -5; exit 1; }
ctxarg=(-c $((np * per))); [ "$mode" = unified ] && ctxarg=(-c $per -kvu)
# -cram 0: no host-RAM prompt cache. Every reference request is independent (cache_prompt false), and
# an old build (b10820, a sliding-window model) answered the PREVIOUS prompt from that cache.
mm=(); [ -n "${MMPROJ:-}" ] && { mm=(--mmproj "$MMPROJ"); export LLAMA_MEDIA_MARKER='<__media__>'; }   # vision_golden.py's marker
spec=(); [ -n "${LLAMA_SPEC:-}" ] && read -r -a spec <<< "$LLAMA_SPEC"
args=(-ngl 99 -cram 0 "${mm[@]}" "${spec[@]}" "${ctxarg[@]}" -np $np)
# LLAMACPP_ARGS_FILE: record the server's exact arguments (model path, host and port left out), the card's baseline args
if [ -n "${LLAMACPP_ARGS_FILE:-}" ]; then echo "${args[*]}" > "$LLAMACPP_ARGS_FILE"; fi
nohup $exe -m "$gguf" "${args[@]}" --port $port --host 127.0.0.1 > $state/llamacpp-$port.log 2>&1 &
echo $! > $state/llamacpp-$port.pid
for i in $(seq 300); do
  curl -s localhost:$port/health | grep -q ok && { echo "READY llama.cpp $($exe --version 2>&1 | grep -m1 version) on :$port (pid $(cat $state/llamacpp-$port.pid))"; exit 0; }
  kill -0 $(cat $state/llamacpp-$port.pid) 2>/dev/null || { echo "llama-server died:"; tail -20 $state/llamacpp-$port.log; exit 1; }
  sleep 1
done
echo "timeout"; exit 1
