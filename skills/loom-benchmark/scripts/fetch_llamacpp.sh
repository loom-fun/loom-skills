#!/bin/bash
# Download the latest lemonade-sdk/llamacpp-rocm release for this GPU (pre-approved source: no need
# to ask the user). Idempotent: a release already unpacked is reused. Prints the build dir last.
# usage: fetch_llamacpp.sh [DEST=~/llamacpp-rocm] [TAG=latest] [GFX=auto]
#   GFX: gfx1151 (Strix Halo), gfx1150, ...; auto reads it from llama.cpp-visible ROCm (rocminfo) or
#   defaults to gfx1151. OS asset: ubuntu (Linux) or windows (Git Bash).
# Run by qualify.sh before pick_llamacpp.sh, so the newest build always competes with the installed
# ones (a new model architecture usually needs a recent build: no installed build knew `flint`).
set -eu
dest=${1:-$HOME/llamacpp-rocm}; tag=${2:-latest}; gfx=${3:-auto}
if [ "$gfx" = auto ]; then
  gfx=$( (rocminfo 2>/dev/null || true) | grep -o "gfx11[0-9][0-9]" | head -1)
  gfx=${gfx:-gfx1151}
fi
os=ubuntu; case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) os=windows;; esac
api=https://api.github.com/repos/lemonade-sdk/llamacpp-rocm/releases/$([ "$tag" = latest ] && echo latest || echo "tags/$tag")
auth=(); [ -n "${GH_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $GH_TOKEN")   # avoids API rate limits on CI
meta=$(curl -fsSL "${auth[@]}" "$api") || { echo "cannot reach the GitHub API for lemonade-sdk/llamacpp-rocm" >&2; exit 1; }
read -r rtag url < <(python3 -c '
import json, sys
d = json.loads(sys.argv[1]); want = f"-{sys.argv[2]}-rocm-{sys.argv[3]}-x64.zip"
a = [x for x in d["assets"] if x["name"].endswith(want) and not x["name"].startswith("qualification")]
print(d["tag_name"], a[0]["browser_download_url"] if a else "")' "$meta" "$os" "$gfx")
[ -n "$url" ] || { echo "release $rtag has no $os $gfx asset" >&2; exit 1; }
dir=$dest/$rtag
if [ ! -x "$dir/llama-server" ] && [ ! -f "$dir/llama-server.exe" ]; then
  mkdir -p "$dir"
  echo "downloading $url" >&2
  curl -fsSL --retry 3 -o "$dir/release.zip" "$url" >&2
  (cd "$dir" && unzip -qo release.zip && rm -f release.zip && chmod +x llama-* 2>/dev/null || true)
fi
LD_LIBRARY_PATH=$dir "$dir/llama-server" --list-devices 2>&1 | grep -q ROCm \
  || echo "WARNING: $dir/llama-server lists no ROCm device (llamacpp_server.sh will refuse it)" >&2
echo "$dir"
