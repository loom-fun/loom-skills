#!/bin/bash
# Start loom-serve in the background and wait until every model is registered.
# usage: VENV=~/loom/.venv serve.sh NAME PORT DEVICE MEMORY_BYTES MODEL_JSON [MODEL_JSON...] [-- extra loom-serve flags]
#   DEVICE: amdgpu | 'xdna://0?columns=8' (NPU; add -- --pool_backing=fixed)
#   MEMORY_BYTES: always set it (or `none` with -- --pool_backing=fixed, which sizes its pools itself) (e.g. 16000000000); it caps weights + KV pool + workspace.
#   amdgpu gets ?completion_spin_ns=4294967295 (active wait): Loom's retire thread spins on the GPU's completion
#   signal while work is in flight (one core, 0% when idle) instead of parking on the KFD interrupt. On Strix Halo
#   firmware that keys the memory-fabric clocks to CPU activity (SMU 0x0a640600, BIOS 1.12: AMD DevLab), a parked
#   CPU leaves FCLK/UCLK ~20% low and Loom 10-45% slower; HIP (llama.cpp) always keeps a core busy. LOOM_ACTIVE_WAIT=0
#   keeps the SDK's default wait. A device URI that already sets completion_spin_ns is passed unchanged.
#   The value is the SDK's maximum (uint32 ns, 4.29 s): the spin window restarts at every completion and parks once
#   a submission runs longer than it; with 1 s, multi-second long-prefill epochs parked (Laguna 32k: 1187 vs 1539 tok/s).
# Logs/PIDs: ~/.cache/loom-serve/NAME.{log,pid,port} (pick unique NAMEs). Stop: stop.sh NAME
# Readiness = /v1/status registration "ready"; compile happens at startup (cached after the first run),
# weights load lazily on the first request. Works on Linux and in Git Bash on Windows.
set -u
name=$1; port=$2; dev=$3; mem=$4; shift 4
models=(); extra=()
while [ $# -gt 0 ]; do [ "$1" = "--" ] && { shift; extra=("$@"); break; }; models+=("--model=$1"); shift; done
case "$dev" in
  amdgpu|amdgpu://*) if [ "${LOOM_ACTIVE_WAIT:-1}" != 0 ] && [ "${dev#*completion_spin_ns=}" = "$dev" ]; then
    case "$dev" in amdgpu) dev="amdgpu://?";; *\?*) dev="$dev&";; *) dev="$dev?";; esac
    dev="${dev}completion_spin_ns=4294967295"; fi;;
esac
venv=${VENV:-$HOME/loom/.venv}
bin=$venv/bin; [ -d "$bin" ] || bin=$venv/Scripts   # Windows venvs use Scripts/
py=$bin/python
state=${LOOM_SERVE_STATE:-$HOME/.cache/loom-serve}; mkdir -p "$state"   # NAME.pid / NAME.log / NAME.port live here
# Listening sockets on $port: `ss` on Linux, `netstat -ano` (last column = Windows PID) elsewhere.
listeners() { if command -v ss >/dev/null; then ss -ltnp | grep ":$port "; else netstat -ano -p tcp | tr -d '\r' | grep LISTENING | grep ":$port "; fi; }
owns_port() { if command -v ss >/dev/null; then listeners | grep -q "pid=$1,"; else listeners | grep -q " $(cat /proc/$1/winpid 2>/dev/null || echo $1)\$"; fi; }
if listeners >/dev/null; then echo "port $port busy:"; listeners; exit 1; fi
# A package may declare how it must be served (llm_config.json "serve": {"args": [...], "committed_bytes": N}): e.g.
# --pool_backing=fixed, which takes no --memory_bytes (it sizes its pools itself); memcheck then checks committed_bytes.
if [ ${#models[@]} -gt 0 ] && [ -f "$(dirname "${models[0]#--model=}")/llm_config.json" ]; then
  IFS='|' read -r pkgargs pkgbytes < <("$py" -c "import json,sys; s=json.load(open(sys.argv[1])).get('serve') or {}; print(' '.join(s.get('args', [])) + '|' + str(s.get('committed_bytes', '')))" "$(dirname "${models[0]#--model=}")/llm_config.json" 2>/dev/null)
  if [ -n "${pkgargs:-}" ]; then
    extra=($pkgargs "${extra[@]}")
    case " $pkgargs " in *" --pool_backing=fixed "*) echo "serve.sh: the package serves with $pkgargs (no --memory_bytes); committed ${pkgbytes:-unknown} bytes" >&2; mem=none; memneed=${pkgbytes:-};; esac
  fi
fi
memflag=(--memory_bytes=$mem); [ "$mem" != none ] || memflag=()   # MEMORY_BYTES=none: no cap (--pool_backing=fixed takes none)
[ "$dev" != amdgpu ] || { [ "$mem" = none ] && [ -z "${memneed:-}" ]; } || "$(dirname "$0")/memcheck.sh" "${memneed:-$mem}" "loom-serve ${extra[*]:-} --memory_bytes=${mem}" || exit 1   # only budgets above the cap are checked
nohup "$bin/loom-serve" --device="$dev" --listen=127.0.0.1:$port "${memflag[@]}" --workers=8 "${models[@]}" "${extra[@]}" > $state/$name.log 2>&1 &
echo $! > $state/$name.pid; echo $port > $state/$name.port; start=$(date +%s)
while kill -0 $(cat $state/$name.pid) 2>/dev/null; do
  st=$(curl -s -m 2 http://127.0.0.1:$port/v1/status | "$py" -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(",".join(x["loom"]["registration"] for x in d) if d else "none")' 2>/dev/null)
  case "$st" in
    *failed*) echo "FAILED $name"; curl -s http://127.0.0.1:$port/v1/status | "$py" -c 'import sys,json; [print(x["id"], json.dumps(x["loom"].get("error") or x["loom"].get("compilation",{}).get("diagnostics"))[:3000]) for x in json.load(sys.stdin)["data"]]'
      # A failed registration keeps the process (and the port); stop it so a rebuild can retry.
      kill $(cat $state/$name.pid) 2>/dev/null; rm -f $state/$name.pid; exit 2;;
  esac
  if [ -n "$st" ] && [ "$st" != none ] && ! echo "$st" | tr ',' '\n' | grep -qv '^ready$'; then
    pid=$(cat $state/$name.pid)
    owns_port $pid || { echo "port $port answered but is not owned by pid $pid (stale server?)"; listeners; exit 3; }
    # Registration is ready once the program compiled (cached kernels make this ~1 s); weights load on the first request.
    echo "READY $name on :$port pid $pid after $(( $(date +%s) - start ))s ($st)"; exit 0
  fi
  sleep 1
done
echo "DIED $name"; grep -v heartbeat $state/$name.log | tail -n 40; exit 1
