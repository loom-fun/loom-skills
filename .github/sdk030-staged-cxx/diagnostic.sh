#!/usr/bin/env bash
set -euo pipefail
recipes=$(cd "$(dirname "$0")" && pwd)
cd "$WORK"
for file in abi1-package consumer-revisions runtime-sha256; do cp "tested/$file.txt" "out/$file.txt"; done
cp tested/{gufo,extension}-staged-abi1.json out/
test "$(sed -n '1p' out/consumer-revisions.txt)" = "gufo=ec6ce36a149363ca21521517ace52efe5c506e0c"
test "$(sed -n '2p' out/consumer-revisions.txt)" = "extension=1f155e6a1be7a9d850266fb54f206bb61cf94867"
mapfile -t hashes < <(awk '{ print $1 }' out/runtime-sha256.txt)
echo "${hashes[0]}  tested/embed/gufo/lib/libloom_text_embed.so" | sha256sum -c -
echo "${hashes[1]}  tested/embed/extension/lib/libloom_text_embed.so" | sha256sum -c -

python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
wheel=hrx_loom-0.3.0-py3-none-manylinux_2_39_x86_64.whl
curl -fsSL --retry 3 "https://github.com/ROCm/hrx-system/releases/download/v0.3.0/$wheel" -o "$wheel"
echo "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f  $wheel" | sha256sum -c -
python -m pip install -q "./$wheel[hf]"
repo=loom-fun-staging/unsloth-Qwen3.8-Flash-Next-loom
revision=473e5815cedcf3117fcac905c27ed3434f9fe836
echo "$repo@$revision" > out/package.txt
loom-fetch "huggingface:$repo:vision-mtp@$revision" --materialize=share --output="$WORK/flash" --no-input
python - "$GITHUB_WORKSPACE/skills/loom-benchmark/scripts/card" <<'PY' | tee out/package-check.txt
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from card_lib import programs_sha256
model = json.loads(Path("flash/model.json").read_text())
settings = json.loads(Path("flash/llm_config.json").read_text())
selected = model["presets"]["vision-mtp"]["service"]
abi = model["services"][selected]["version"]
digest = programs_sha256("flash")
assert abi == 2 and settings["sdk"]["hrx_loom"] == "0.3.0"
assert digest == "c0cf0958949af15430bb8791eb167e8108a5a374f3f5aa3828eb55f0d88ac920"
print(f"vision-mtp: controller ABI {abi}, SDK {settings['sdk']['hrx_loom']}, programs {digest}")
PY

ulimit -c 0
ulimit -l 8192
test "$(ulimit -Hl)" = 8192
test "$(ulimit -Sl)" = 8192
printf 'soft=%s KiB hard=%s KiB\n' "$(ulimit -Sl)" "$(ulimit -Hl)" > out/memlock.txt
rc=0
for consumer in gufo extension; do
  case "$consumer" in
    gufo) url=https://raw.githubusercontent.com/loom-fun/gufo/ec6ce36a149363ca21521517ace52efe5c506e0c/src/models/loom/embed/loom_text_embed.h ;;
    extension) url=https://raw.githubusercontent.com/loom-fun/loom-gufo-extension/1f155e6a1be7a9d850266fb54f206bb61cf94867/loom-embed/loom_text_embed.h ;;
  esac
  runtime="$WORK/tested/embed/$consumer"
  curl -fsSL --retry 3 "$url" -o "$runtime/loom_text_embed.h"
  loader="$runtime/lib/ld-linux-x86-64.so.2"
  chmod +x "$loader"
  cc -O0 -g -I "$runtime" "$recipes/controller-abi.c" -L "$runtime/lib" \
    -Wl,--allow-shlib-undefined -lloom_text_embed -o "$runtime/diagnose-abi"
  args=("$loader" --library-path "$runtime/lib" "$runtime/diagnose-abi" "$WORK/flash" vision-mtp flash)
  if timeout 1200s "${args[@]}" 2>&1 | tee "out/$consumer-flash.log"; then continue; fi
  rc=1
  if command -v gdb >/dev/null; then
    timeout 300s gdb -q -batch -ex 'set pagination off' -ex 'set print thread-events off' \
      -ex 'set debuginfod enabled off' -ex run -ex 'bt 28' --args "${args[@]}" \
      > "out/$consumer-backtrace.txt" 2>&1 || true
  fi
done
exit "$rc"
