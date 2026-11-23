#!/usr/bin/env bash
set -euo pipefail
exec > >(tee "$WORK/out/downstream-gpu.log") 2>&1
cd "$WORK"
repo=loom-fun-staging/unsloth-Qwen3-0.6B-loom
revision=f3ffda86d7bd1fa98d615f4972a7d3f82345294c
# The registered Lemonade model keeps its catalog alias. Seed its isolated cache
# with the exact stage revision instead of mutating the bundled registry.
cache="$WORK/hf/hub/models--loom-fun--unsloth-Qwen3-0.6B-loom"
snapshot="$cache/snapshots/$revision"
python3 -m venv "$WORK/gpu-sdk"
source "$WORK/gpu-sdk/bin/activate"
wheel=hrx_loom-0.3.0-py3-none-manylinux_2_39_x86_64.whl
curl -fsSL --retry 3 "https://github.com/ROCm/hrx-system/releases/download/v0.3.0/$wheel" -o "$wheel"
echo "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f  $wheel" | sha256sum -c -
python -m pip install -q "./$wheel[hf]"
export HF_HUB_CACHE="$WORK/hf/hub"
mkdir -p "$cache/refs" "$cache/snapshots"
materialized="$WORK/materialized-qwen"
timeout 240s loom-fetch "huggingface:$repo:chat@$revision" --materialize=copy --output="$materialized" --no-input
rm -rf -- "$snapshot"
mv "$materialized" "$snapshot"
printf '%s\n' "$revision" > "$cache/refs/main"
auth=()
if [ -n "${HF_TOKEN:-}" ]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); fi
mkdir "$WORK/raw-card"
for f in model.json model.json.lock info.json; do
  curl -fsSL --retry 3 "${auth[@]}" "https://huggingface.co/$repo/resolve/$revision/$f" -o "$WORK/raw-card/$f"
done
cp "$WORK/raw-card/info.json" "$snapshot/info.json"
python - "$snapshot" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
card = json.loads((root / "model.json").read_text())
cfg = json.loads((root / "llm_config.json").read_text())
assert cfg["sdk"]["hrx_loom"] == "0.3.0" and card["services"]["chat"]["version"] == 1
print("Testing the exact staged SDK 0.3.0 card with retained controller ABI 1")
PY
sha256sum "$WORK/raw-card/model.json"
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
        "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}
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
  local mode=$1 port
  port=$(free_port)
  mkdir -p "$WORK/lemon-cache" "$WORK/lemon-config"
  (
    ulimit -S -l 8192
    echo "Lemonade and its backend memlock limit: $(ulimit -l) KiB"
    exec "$WORK/lemonade/build/lemond" "$WORK/lemon-cache" "$WORK/lemon-config" --port "$port" --host 127.0.0.1
  ) > "$WORK/out/lemonade-$mode-server.log" 2>&1 &
  pid=$!
  if ! wait_http "http://127.0.0.1:$port/api/v1/health"; then
    tail -40 "$WORK/out/lemonade-$mode-server.log"
    return 1
  fi
  if ! chat "http://127.0.0.1:$port/api/v1/chat/completions" "Qwen3-0.6B-Loom" "$WORK/out/lemonade-$mode-chat.json"; then
    tail -42 "$WORK/out/lemonade-$mode-server.log"
    return 1
  fi
  find "$WORK/lemon-cache" -name loom-serve -type f -exec {} --version \; | grep 'loom-serve 0.3.0'
}
rc=0
if ! lemonade_smoke materialized; then rc=1; fi
stop_child
pid=

cp "$WORK/raw-card/model.json" "$WORK/raw-card/model.json.lock" "$snapshot/"
if ! lemonade_smoke raw; then rc=1; fi
stop_child
pid=

probe() {
  local name=$1 headers=$2 library=$3
  cc -I "$headers" "$WORK/extension/validation/devlab/embed_probe.c" \
    -L "$library" -Wl,-rpath,"$library" -lloom_text_embed -lm -o "$WORK/$name-embed-probe" || return 1
  timeout 180s "$WORK/$name-embed-probe" "$snapshot" 6 2 chat | tee "$WORK/out/$name-logits.json"
}
if ! probe extension "$WORK/extension/loom-embed" "$WORK/hrx-build"; then rc=1; fi
mkdir "$WORK/gufo-runtime"
cp "$WORK/libloom_text_embed-gufo.so" "$WORK/gufo-runtime/libloom_text_embed.so"
if ! probe gufo "$WORK/gufo/src/models/loom/embed" "$WORK/gufo-runtime"; then rc=1; fi
exit "$rc"
