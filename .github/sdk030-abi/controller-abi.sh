#!/usr/bin/env bash
set -euo pipefail
exec > >(tee "$WORK/out/controller-abi.log") 2>&1
recipes=$(cd "$(dirname "$0")" && pwd)
cd "$WORK"
source_rev=e2c41a720363911c04606ce0f4417e2d55630352
gufo_rev=0dabe4e12f891c4c2ae2c4c0aa1bf997d599000c
extension_rev=7e7dc5eba547f3fbd9793f2829d0266c93717178
checkout() {
  git init -q "$1"
  git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3"
  git -C "$1" checkout -q --detach FETCH_HEAD
  test "$(git -C "$1" rev-parse HEAD)" = "$3"
}
checkout gufo gufo-org/gufo "$gufo_rev"
checkout extension loom-fun/loom-gufo-extension "$extension_rev"
printf 'gufo=%s\nextension=%s\n' "$gufo_rev" "$extension_rev" > "$WORK/out/revisions.txt"
hostname > "$WORK/out/runner.txt"
python3 - <<'PY'
from pathlib import Path

for name, source in (
    ("gufo", "gufo/src/models/loom/embed/loom_text_embed.c"),
    ("extension", "extension/loom-embed/loom_text_embed.c"),
):
    path = Path(source)
    text = path.read_text()
    start = text.index("// Loom streams weights through io_uring")
    end = text.index("static iree_status_t loom_text_embed_open_devices", start)
    text = text[:start] + text[end:]
    for line in (
        "#include <sys/resource.h>\n",
        '#include "iree/base/tooling/flags.h"\n',
        "  IREE_RETURN_IF_ERROR(loom_text_embed_select_parameter_mode());\n",
    ):
        assert text.count(line) == 1, (name, line)
        text = text.replace(line, "")
    path.write_text(text)
    Path(f"out/{name}-loom_text_embed.c").write_text(text)
PY
git -C gufo diff --check
git -C extension diff --check
mkdir hrx
curl -fsSL --retry 3 "https://codeload.github.com/ROCm/hrx-system/tar.gz/$source_rev" -o hrx.tgz
echo '3b02486d07d32d03901f41915605a21840211db4e93b040dfa55283d68d27fd8  hrx.tgz' | sha256sum -c -
tar -C hrx --strip-components=1 -xzf hrx.tgz
python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
wheel=hrx_loom-0.3.0-py3-none-manylinux_2_39_x86_64.whl
curl -fsSL --retry 3 "https://github.com/ROCm/hrx-system/releases/download/v0.3.0/$wheel" -o "$wheel"
echo "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f  $wheel" | sha256sum -c -
python -m pip install -q "./$wheel[hf]"

legacy=loom-fun/unsloth-Qwen3-0.6B-loom
legacy_rev=564d578f4e487c6fbabd84915439f873651b5953
loom-fetch "huggingface:$legacy:chat@$legacy_rev" --materialize=share --output="$WORK/legacy" --no-input
fixture="$WORK/hrx/loom/docs/examples/serving/model-publication/model/tiny-text"
loom-fetch "$fixture/model.json" --materialize=copy --output="$WORK/abi2" --no-input
python - <<'PY'
import json
from pathlib import Path
legacy = json.loads(Path("legacy/model.json").read_text())
settings = json.loads(Path("legacy/llm_config.json").read_text())
fixture = json.loads(Path("abi2/model.json").read_text())
assert settings["sdk"]["hrx_loom"] == "0.2.0" and legacy["services"]["chat"]["version"] == 1
assert fixture["services"]["chat"]["version"] == 2
PY

extra=()
if command -v clang >/dev/null && command -v clang++ >/dev/null; then
  extra+=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)
fi
cmake -S hrx -B hrx-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DIREE_ENABLE_POSITION_INDEPENDENT_CODE=ON -DIREE_BUILD_TESTS=OFF \
  -DIREE_ENABLE_WERROR_FLAG=OFF -DLIBHRX_BUILD=OFF \
  -DIREE_HAL_DRIVER_AMD_XDNA=OFF -DLOOM_TARGET_ALL_AMD_GPU=ON \
  -DIREE_CMAKE_TRY_FILE="$WORK/gufo/src/models/loom/embed/loom_text_embed.cmake" \
  "${extra[@]}"
mkdir -p "$WORK/out/embed"
rc=0
for consumer in gufo extension; do
  if [ "$consumer" = gufo ]; then
    headers="$WORK/gufo/src/models/loom/embed"
  else
    headers="$WORK/extension/loom-embed"
    cmake -S hrx -B hrx-build -DIREE_CMAKE_TRY_FILE="$headers/loom_text_embed.cmake"
  fi
  cmake --build hrx-build --target loom_text_embed
  runtime="$WORK/out/embed/$consumer"
  mkdir "$runtime"
  cp "$WORK/hrx-build/libloom_text_embed.so" "$runtime/"
  cp "$headers/loom_text_embed.h" "$runtime/"
  ldd "$runtime/libloom_text_embed.so" > "$WORK/out/$consumer-ldd.txt"
  cc -I "$headers" "$WORK/extension/validation/devlab/embed_probe.c" \
    -L "$runtime" -Wl,-rpath,"$runtime" -lloom_text_embed -lm -o "$runtime/logits-probe"
  cc -D_GNU_SOURCE -I "$headers" "$recipes/controller-abi.c" \
    -L "$runtime" -Wl,-rpath,'$ORIGIN' -lloom_text_embed -o "$runtime/controller-abi"
  (
    ulimit -l 8192 || exit
    test "$(ulimit -Hl)" = 8192 && test "$(ulimit -Sl)" = 8192 || exit
    timeout 180s "$runtime/logits-probe" "$WORK/legacy" 6 2 chat
  ) | tee "$WORK/out/$consumer-legacy.json" || rc=1
  (
    ulimit -l 8192 || exit
    test "$(ulimit -Hl)" = 8192 && test "$(ulimit -Sl)" = 8192 || exit
    timeout 120s "$runtime/controller-abi" "$WORK/abi2" default tiny
  ) | tee "$WORK/out/$consumer-abi2.txt" || rc=1
done
exit "$rc"
