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
checkout gufo loom-fun/gufo bccf173b2664432d03d068413a91639bdd6242f7
checkout extension loom-fun/loom-gufo-extension 1ed03da691cf16d78a35dfee7243185402dd6133
test "$(git -C gufo rev-parse HEAD:src/models/loom/runtime.cmake)" = 34842b69fc538e971665c12fcbfc0bd3db418bdf
test "$(git -C gufo rev-parse HEAD:.devops/nix/hrx/loom-runtime.nix)" = 7d11617b6f6f83cb18ddeaa25b75ce8975322e2c
test "$(git -C extension rev-parse HEAD:nix/loom-runtime.nix)" = 59d4427daf0344a1302a6b2a442be54eebb5025c
test "$(git -C gufo rev-parse HEAD:src/models/loom/embed/loom_text_embed.c)" = 2d6eea9cd4de230c59d434c7c871bf37d97761fe
test "$(git -C extension rev-parse HEAD:loom-embed/loom_text_embed.c)" = 1f38137f47680d390d4245499b1f1fd37311e7d6

timeout 15 docker info --format '{{.ServerVersion}}' > out/docker-version.txt
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
import hashlib
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
for source, expected in {
    "program/shared/program/text/qwen3_8/publish.cxx": "c34431565f494a901676fb45c354fa28009054aefdd790358d9d50f6e3161bb9",
    "program/shared/program/text/publication.h": "b0daed695a786b992e66c2a06b24bc17cada14d342ca217dc368e3cc6b20870f",
}.items():
    actual = hashlib.sha256(Path("flash", source).read_bytes()).hexdigest()
    assert actual == expected, source
    print(f"{source} {actual}")
print(f"vision-mtp: controller ABI {abi}, SDK {settings['sdk']['hrx_loom']}, programs {digest}")
PY
)
fetch_models > out/fetch.log 2>&1 &
fetch_pid=$!
trap 'kill "$fetch_pid" 2>/dev/null || true' EXIT

nix_image=nixos/nix@sha256:9b6a776642088021e9251e7cdb47988747008178ca2f7398cb3e1b365bde3372
echo "$nix_image" > out/nix-image.txt
docker run --rm --name "sdk030-staged-nix-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT" \
  --mount "type=bind,src=$WORK/gufo,dst=/gufo,readonly" \
  --mount "type=bind,src=$WORK/extension,dst=/extension,readonly" \
  --mount "type=bind,src=$recipes,dst=/recipes,readonly" \
  --mount "type=bind,src=$WORK/out,dst=/proof" \
  --env "PROOF_OWNER=$(id -u):$(id -g)" \
  "$nix_image" sh /recipes/nix-bundle.sh > out/nix-build.log 2>&1 ||
  { tail -90 out/nix-build.log; exit 1; }
wait "$fetch_pid" || { tail -80 out/fetch.log; exit 1; }
trap - EXIT

ulimit -l 8192
test "$(ulimit -Hl)" = 8192
test "$(ulimit -Sl)" = 8192
printf 'soft=%s KiB hard=%s KiB\n' "$(ulimit -Sl)" "$(ulimit -Hl)" > out/memlock.txt
rc=0
for consumer in gufo extension; do
  runtime="$WORK/out/embed/$consumer"
  loader="$runtime/lib/ld-linux-x86-64.so.2"
  timeout 180s "$loader" --library-path "$runtime/lib" "$runtime/logits-probe" \
    "$WORK/staged-abi1" 6 2 chat | tee "out/$consumer-staged-abi1.json" || rc=1
  checkpoint="$WORK/$consumer-checkpoint.bin"
  if timeout 1500s "$loader" --library-path "$runtime/lib" "$runtime/controller-abi" \
    "$WORK/flash" vision-mtp flash "$checkpoint" 2>&1 | tee "out/$consumer-flash.log"; then
    timeout 1500s "$loader" --library-path "$runtime/lib" "$runtime/controller-abi" \
      "$WORK/flash" vision-mtp flash-cold "$checkpoint" 2>&1 | tee "out/$consumer-flash-cold.log" || rc=1
  else
    rc=1
  fi
done
exit "$rc"
