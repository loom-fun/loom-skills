#!/usr/bin/env bash
set -euo pipefail
exec > >(tee "$WORK/out/downstream-gpu.log") 2>&1
cd "$WORK"
repo=loom-fun/unsloth-Qwen3-0.6B-loom
revision=564d578f4e487c6fbabd84915439f873651b5953
cache="$WORK/hf/hub/models--loom-fun--unsloth-Qwen3-0.6B-loom"
snapshot="$cache/snapshots/$revision"
python3 -m venv "$WORK/gpu-sdk"
source "$WORK/gpu-sdk/bin/activate"
wheel=hrx_loom-0.3.0-py3-none-manylinux_2_39_x86_64.whl
curl -fsSL --retry 3 "https://github.com/ROCm/hrx-system/releases/download/v0.3.0/$wheel" -o "$wheel"
echo "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f  $wheel" | sha256sum -c -
python -m pip install -q "./$wheel"
export HF_HUB_CACHE="$WORK/hf/hub"
mkdir -p "$cache/refs"
timeout 240s loom-fetch "huggingface:$repo:chat@$revision" --materialize=copy --output="$snapshot" --no-input
printf '%s\n' "$revision" > "$cache/refs/main"
printf '{"loom":"v0.2.0"}\n' > "$snapshot/.lemonade-loom.json"
python - "$snapshot" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
card = json.loads((root / "model.json").read_text())
cfg = json.loads((root / "llm_config.json").read_text())
assert cfg["sdk"]["hrx_loom"] == "0.2.0" and card["services"]["chat"]["version"] == 1
print("Testing the exact published SDK 0.2.0 package with retained controller ABI 1")
PY
sha256sum "$snapshot/model.json"
free_port() {
  python - <<'PY'
import socket
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}
stop_child() {
  if [ -n "${pid:-}" ]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}
trap stop_child EXIT
wait_http() {
  local url=$1
  for _ in $(seq 1 100); do
    curl -fsS --max-time 1 "$url" && return
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
chat() {
  local endpoint=$1 model=$2 out=$3
  python - "$endpoint" "$model" "$out" <<'PY'
import json
from pathlib import Path
import sys
import urllib.request

endpoint, model, out = sys.argv[1:]
body = {"model": model, "messages": [{"role": "user", "content": "What is two plus two? Reply with the answer only. /no_think"}],
        "temperature": 0, "max_tokens": 48}
r = urllib.request.Request(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(r, timeout=120) as response:
    data = json.load(response)
Path(out).write_text(json.dumps(data, indent=2) + "\n")
content = data["choices"][0]["message"].get("content") or ""
assert "4" in content or "four" in content.lower(), data
print(model, repr(content))
PY
}

lemonade_smoke() {
  local port
  port=$(free_port)
  mkdir -p "$WORK/lemon-cache" "$WORK/lemon-config"
  "$WORK/lemonade/build/lemond" "$WORK/lemon-cache" "$WORK/lemon-config" --port "$port" --host 127.0.0.1 > "$WORK/out/lemonade-server.log" 2>&1 &
  pid=$!
  if ! wait_http "http://127.0.0.1:$port/api/v1/health"; then
    tail -40 "$WORK/out/lemonade-server.log"
    return 1
  fi
  chat "http://127.0.0.1:$port/api/v1/chat/completions" "Qwen3-0.6B-Loom" "$WORK/out/lemonade-chat.json" || return 1
  find "$WORK/lemon-cache" -name loom-serve -type f -exec {} --version \; | grep 'loom-serve 0.3.0'
}
rc=0
if ! lemonade_smoke; then rc=1; fi
stop_child
pid=

extension_smoke() {
  cc -I "$WORK/extension/loom-embed" "$WORK/extension/validation/devlab/embed_probe.c" \
    -L "$WORK/hrx-build" -Wl,-rpath,"$WORK/hrx-build" -lloom_text_embed -lm -o "$WORK/embed-probe" || return 1
  timeout 180s "$WORK/embed-probe" "$snapshot" 6 2 chat | tee "$WORK/out/extension-logits.json"
}
if ! extension_smoke; then rc=1; fi

gufo_smoke() {
  local port binary model asset
  asset=gufo_0.36.0-1_amd64.deb
  curl -fsSL --retry 3 "https://github.com/gufo-org/gufo/releases/download/v0.36.0/$asset" -o "$asset" || return 1
  echo "c43b9e5e785e6819dd5c45aef48845618453bd9db3bd3808d51198030564016d  $asset" | sha256sum -c - || return 1
  dpkg-deb -x "$asset" gufo-app || return 1
  binary=$(find "$WORK/gufo-app" -path '*/bin/gufo' -type f -print -quit)
  find "$WORK/gufo-app" -name libloom_text_embed.so -print > "$WORK/out/gufo-lib-paths.txt"
  if [ -z "$binary" ] || [ ! -s "$WORK/out/gufo-lib-paths.txt" ]; then
    find "$WORK/gufo-app" -maxdepth 4 -type f | head -60
    return 1
  fi
  while IFS= read -r path; do
    cp --remove-destination "$WORK/libloom_text_embed-gufo.so" "$path" || return 1
  done < "$WORK/out/gufo-lib-paths.txt"
  "$binary" --version || return 1
  port=$(free_port)
  "$binary" serve llm --model "$snapshot" --sessions 2 --port "$port" --host 127.0.0.1 > "$WORK/out/gufo-server.log" 2>&1 &
  pid=$!
  if ! wait_http "http://127.0.0.1:$port/health"; then
    tail -40 "$WORK/out/gufo-server.log"
    return 1
  fi
  model=$(curl -fsS "http://127.0.0.1:$port/v1/models" | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])') || return 1
  chat "http://127.0.0.1:$port/v1/chat/completions" "$model" "$WORK/out/gufo-chat.json"
}
if ! gufo_smoke; then rc=1; fi
stop_child
pid=
exit "$rc"
