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
# The gates render prompts and raw replies with the chat_probe.py the package was built with. port.sh leaves an
# example kit it applied (gemma4, glimmer: chat formats the starter can't model) at WORKDIR/kit next to PKG;
# without LOOM_KIT the starter's renders those prompts differently (agent gate: render_ok false on every turn).
[ -n "${LOOM_KIT:-}" ] || [ ! -f "$(dirname "$pkg")/kit/chat_probe.py" ] || export LOOM_KIT=$(dirname "$pkg")/kit

# One model at a time: GPU memory is system RAM on Strix Halo, so Loom and llama.cpp never run
# together. loom_down stops the loom-serve listening on $url's port and remembers its exact command
# line; loom_up starts that command again (same models, same --memory_bytes), updates serve.sh's
# pid file and waits for registration. No-ops when no loom-serve owns the port (or without /proc).
loom_port=${url##*:}; loom_port=${loom_port%%/*}
loom_down() {
  local pid; pid=$(ss -ltnpH "sport = :$loom_port" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  { [ -n "$pid" ] && [ "$(cat /proc/$pid/comm 2>/dev/null)" = loom-serve ]; } || return 0
  loom_cwd=$(readlink /proc/$pid/cwd); mapfile -d '' -t loom_argv < /proc/$pid/cmdline
  # the serve.sh name whose pid file holds this pid, so loom_up restarts under the same name (a stale
  # NAME.port from an earlier session on the same port would otherwise get it, and stop.sh NAME would miss it)
  local f; loom_name=; for f in "${LOOM_SERVE_STATE:-$HOME/.cache/loom-serve}"/*.pid; do
    [ "$(cat "$f" 2>/dev/null)" = "$pid" ] && { loom_name=$(basename "$f" .pid); break; }; done
  echo "[one model at a time] stopping loom-serve pid $pid on :$loom_port"
  kill "$pid"; for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.5; done; kill -9 "$pid" 2>/dev/null
}
loom_up() {
  [ ${#loom_argv[@]} -gt 0 ] || return 0
  local state=${LOOM_SERVE_STATE:-$HOME/.cache/loom-serve} name=gates-$loom_port pid st
  local f; f=$(grep -lx "$loom_port" "$state"/*.port 2>/dev/null | head -1); [ -n "$f" ] && name=$(basename "$f" .port)
  [ -n "${loom_name:-}" ] && name=$loom_name
  mkdir -p "$state"
  # exec: the background subshell becomes loom-serve, so no shell is left holding the caller's stdout
  # (`qualify.sh ... | tail` otherwise never ends: the subshell kept the pipe open while it waited) and $! is
  # the server's own PID.
  (cd "$loom_cwd" || exit 1; exec nohup "${loom_argv[@]}" >> "$state/$name.log" 2>&1 < /dev/null) &
  echo $! > "$state/$name.pid"
  echo "$loom_port" > "$state/$name.port"; pid=$(cat "$state/$name.pid")
  for _ in $(seq 1800); do
    kill -0 "$pid" 2>/dev/null || { echo "loom-serve died on restart (see $state/$name.log)"; return 1; }
    st=$(curl -s -m 2 "$url/v1/status" | "$T" -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(",".join(x["loom"]["registration"] for x in d) if d else "none")' 2>/dev/null)
    case "$st" in *fail*) echo "loom-serve registration failed on restart: $st"; return 1;; esac
    if [ -n "$st" ] && [ "$st" != none ] && ! echo "$st" | tr ',' '\n' | grep -qv '^ready$'; then
      # the pid that owns the port (not the launching subshell's): stop.sh reads it from the pid file
      pid=$(ss -ltnpH "sport = :$loom_port" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2); echo "$pid" > "$state/$name.pid"
      echo "[one model at a time] loom-serve back on :$loom_port (pid $pid)"; loom_argv=(); return 0
    fi
    sleep 1
  done
  echo "loom-serve did not become ready on restart"; return 1
}
