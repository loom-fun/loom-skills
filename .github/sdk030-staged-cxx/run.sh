#!/usr/bin/env bash
set -euo pipefail
recipes=$(cd "$(dirname "$0")" && pwd)
cd "$WORK"

checkout() {
  git init -q "$1"
  git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3"
  git -C "$1" checkout -q --detach FETCH_HEAD
  test "$(git -C "$1" rev-parse HEAD)" = "$3"
  printf '%s=%s\n' "$1" "$3" >> out/consumer-revisions.txt
}
checkout gufo loom-fun/gufo 6f6550b80e91b4533bc92edeb8abdcbdd44bf80d
checkout extension loom-fun/loom-gufo-extension 1f155e6a1be7a9d850266fb54f206bb61cf94867
test "$(git -C gufo rev-parse HEAD:src/models/loom/runtime.cmake)" = e86a4db65f9f4e112d43c5b8107084e3d127ccea
test "$(git -C gufo rev-parse HEAD:.devops/nix/hrx/loom-runtime.nix)" = 7d11617b6f6f83cb18ddeaa25b75ce8975322e2c
test "$(git -C extension rev-parse HEAD:nix/loom-runtime.nix)" = 59d4427daf0344a1302a6b2a442be54eebb5025c

python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
wheel=hrx_loom-0.3.0-py3-none-manylinux_2_39_x86_64.whl
curl -fsSL --retry 3 "https://github.com/ROCm/hrx-system/releases/download/v0.3.0/$wheel" -o "$wheel"
echo "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f  $wheel" | sha256sum -c -
python -m pip install -q "./$wheel[hf]"

fetch_models() (
  legacy=loom-fun-staging/unsloth-Qwen3-0.6B-loom
  legacy_rev=f3ffda86d7bd1fa98d615f4972a7d3f82345294c
  echo "$legacy@$legacy_rev" > out/abi1-package.txt
  loom-fetch "huggingface:$legacy:chat@$legacy_rev" --materialize=share --output="$WORK/staged-abi1" --no-input
  flash=loom-fun-staging/unsloth-Qwen3.8-Flash-Next-loom
  flash_rev=473e5815cedcf3117fcac905c27ed3434f9fe836
  echo "$flash@$flash_rev" > out/package.txt
  loom-fetch "huggingface:$flash:vision-mtp@$flash_rev" --materialize=share --output="$WORK/flash" --no-input
  python - "$GITHUB_WORKSPACE/skills/loom-benchmark/scripts/card" <<'PY' | tee out/package-check.txt
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from card_lib import programs_sha256

def read(root, name):
    return json.loads(Path(root, name).read_text())

legacy = read("staged-abi1", "model.json")
assert legacy["services"]["chat"]["version"] == 1
assert read("staged-abi1", "llm_config.json")["sdk"]["hrx_loom"] == "0.3.0"
model = read("flash", "model.json")
preset = model["presets"]["vision-mtp"]
abi = model["services"][preset["service"]]["version"]
settings = read("flash", "llm_config.json")
digest = programs_sha256("flash")
assert abi == 2 and settings["sdk"]["hrx_loom"] == "0.3.0"
assert digest == "c0cf0958949af15430bb8791eb167e8108a5a374f3f5aa3828eb55f0d88ac920"
print(f"vision-mtp: controller ABI {abi}, SDK {settings['sdk']['hrx_loom']}, programs {digest}")
PY
)
fetch_models > out/fetch.log 2>&1 &
fetch_pid=$!
trap 'kill "$fetch_pid" 2>/dev/null || true' EXIT

mkdir configure empty-pkgconfig
export GUFO_SOURCE="$WORK/gufo"
cat > configure/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.26)
project(GufoLoomRuntimeValidation LANGUAGES C CXX)
include("$ENV{GUFO_SOURCE}/src/models/loom/runtime.cmake")
EOF
extra=()
if command -v clang >/dev/null && command -v clang++ >/dev/null; then
  extra+=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)
fi
(
  PKG_CONFIG_LIBDIR="$WORK/empty-pkgconfig" PKG_CONFIG_PATH= cmake -S configure -B gufo-build -G Ninja "${extra[@]}"
  cmake --build gufo-build --target gufo_loom_runtime
) > out/gufo-build.log 2>&1 || { tail -90 out/gufo-build.log; exit 1; }
sdk_src="$WORK/gufo-build/loom/s"
sdk_build="$WORK/gufo-build/loom/b"
grep -Fx 'LOOM_IMPORT_CXX:BOOL=ON' "$sdk_build/CMakeCache.txt" | tee out/gufo-cxx-option.txt
mkdir -p out/embed/{gufo,extension}
cp "$sdk_build/libloom_text_embed.so" gufo/src/models/loom/embed/loom_text_embed.h out/embed/gufo/

# Use the standalone extension's pinned importer setting and production CMake
# target with the already built SDK objects. Its Nix wrapper is verified separately.
grep -F '"-DLOOM_IMPORT_CXX=ON"' extension/nix/loom-runtime.nix | tee out/extension-cxx-option.txt
(
  cmake -S "$sdk_src" -B "$sdk_build" \
    -DLOOM_IMPORT_CXX=ON -DIREE_CMAKE_TRY_FILE="$WORK/extension/loom-embed/loom_text_embed.cmake"
  cmake --build "$sdk_build" --target loom_text_embed
) > out/extension-build.log 2>&1 || { tail -90 out/extension-build.log; exit 1; }
cp "$sdk_build/libloom_text_embed.so" extension/loom-embed/loom_text_embed.h out/embed/extension/
wait "$fetch_pid" || { tail -80 out/fetch.log; exit 1; }
trap - EXIT

ulimit -l 8192
test "$(ulimit -Hl)" = 8192
test "$(ulimit -Sl)" = 8192
printf 'soft=%s KiB hard=%s KiB\n' "$(ulimit -Sl)" "$(ulimit -Hl)" > out/memlock.txt
rc=0
for consumer in gufo extension; do
  runtime="$WORK/out/embed/$consumer"
  cc -I "$runtime" "$WORK/extension/validation/devlab/embed_probe.c" \
    -L "$runtime" -Wl,-rpath,"$runtime" -lloom_text_embed -lm -o "$runtime/logits-probe"
  timeout 180s "$runtime/logits-probe" "$WORK/staged-abi1" 6 2 chat | tee "out/$consumer-staged-abi1.json" || rc=1
  cc -I "$runtime" "$recipes/controller-abi.c" \
    -L "$runtime" -Wl,-rpath,"$runtime" -lloom_text_embed -o "$runtime/controller-abi"
  timeout 1500s "$runtime/controller-abi" "$WORK/flash" vision-mtp flash 2>&1 | tee "out/$consumer-flash.log" || rc=1
done
exit "$rc"
