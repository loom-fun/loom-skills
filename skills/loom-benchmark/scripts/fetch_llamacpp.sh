#!/bin/bash
# Download the latest lemonade-sdk/llamacpp-rocm release for this GPU (pre-approved source: no need
# to ask the user). Idempotent: a release already unpacked is reused. Prints the build dir last.
# usage: fetch_llamacpp.sh [DEST=~/llamacpp-rocm] [TAG=latest] [GFX=auto]
#   LLAMACPP_BACKEND=vulkan: the official ggml-org/llama.cpp Vulkan release TAG (bNNNN; GFX unused) instead, e.g.
#   fetch_llamacpp.sh ~/llamacpp-vulkan b12211. The archive's sha256 must equal data/llamacpp-vulkan.sha256's pin
#   (a tag without a pin: GitHub's asset digest); the build dir holds release.sha256. The Vulkan baseline
#   (llamacpp_server.sh with the same LLAMACPP_BACKEND) runs the same workloads as the ROCm one.
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
backend=${LLAMACPP_BACKEND:-rocm}
case $backend in
  rocm) repo=lemonade-sdk/llamacpp-rocm;;
  vulkan) repo=ggml-org/llama.cpp; [ "$tag" != latest ] || echo "note: an unpinned Vulkan release is checked against GitHub's digest only" >&2;;
  *) echo "unknown LLAMACPP_BACKEND $backend (rocm or vulkan)" >&2; exit 1;;
esac
api=https://api.github.com/repos/$repo/releases/$([ "$tag" = latest ] && echo latest || echo "tags/$tag")
auth=(); [ -n "${GH_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $GH_TOKEN")   # avoids API rate limits on CI
meta=$(curl -fsSL --retry 3 "${auth[@]}" "$api") || { echo "cannot reach the GitHub API for $repo" >&2; exit 1; }
py=${PY:-python3}   # PY: an interpreter of your own (weaver passes its bundled one)
read -r rtag url digest < <("$py" -c '
import json, sys
d = json.load(sys.stdin); backend = sys.argv[4]
if backend == "vulkan":   # llama-bNNNN-bin-ubuntu-vulkan-x64.tar.gz / llama-bNNNN-bin-win-vulkan-x64.zip
    kind, ext = ("ubuntu", "tar.gz") if sys.argv[2] == "ubuntu" else ("win", "zip")
    name = "llama-" + d["tag_name"] + "-bin-" + kind + "-vulkan-x64." + ext
    a = [x for x in d["assets"] if x["name"] == name]
else:
    want = f"-{sys.argv[2]}-rocm-{sys.argv[3]}-x64.zip"
    a = [x for x in d["assets"] if x["name"].endswith(want) and not x["name"].startswith("qualification")]
print(d["tag_name"], a[0]["browser_download_url"] if a else "", (a[0].get("digest") or "").removeprefix("sha256:") if a else "")' - "$os" "$gfx" "$backend" <<< "$meta")
[ -n "$url" ] || { echo "release $rtag has no $os $backend $gfx asset" >&2; exit 1; }
dir=$dest/$rtag
if [ ! -x "$dir/llama-server" ] && [ ! -f "$dir/llama-server.exe" ]; then
  mkdir -p "$dir"
  echo "downloading $url" >&2
  curl -fsSL --retry 6 --retry-all-errors --retry-delay 10 -o "$dir/release.zip" "$url" >&2   # GitHub's release CDN answers 504 for minutes at times
  if [ "$backend" = vulkan ]; then   # pinned archive: refuse any other bytes
    got=$(sha256sum "$dir/release.zip" | cut -d' ' -f1); asset=${url##*/}
    pin=$(grep -h " $asset\$" "$(dirname "$0")/../data/llamacpp-vulkan.sha256" | cut -d' ' -f1)
    [ "$got" = "${pin:-$digest}" ] || { echo "$asset sha256 $got, expected ${pin:-$digest}" >&2; rm -rf "$dir"; exit 1; }
    echo "$got" > "$dir/release.sha256"
  fi
  # the Vulkan release for Linux is a tar.gz with one top directory: unpacked flat (release.zip is just the download's name)
  if [ "$backend" = vulkan ] && [ "$os" = ubuntu ]; then
    (cd "$dir" && tar xzf release.zip --strip-components=1 && rm -f release.zip)
  else
  # unzip if present, else Python's zipfile (keeps the executable bits that unzip would restore).
  (cd "$dir" && { if command -v unzip >/dev/null; then unzip -qo release.zip; else "$py" -c '
import os, sys, zipfile
with zipfile.ZipFile("release.zip") as z:
    for i in z.infolist():
        p = z.extract(i)
        mode = i.external_attr >> 16
        if mode:
            os.chmod(p, mode & 0o7777)' ; fi; } && rm -f release.zip && chmod +x llama-* 2>/dev/null || true)
  fi
fi
dev=ROCm; [ "$backend" = vulkan ] && dev=Vulkan
LD_LIBRARY_PATH=$dir "$dir/llama-server" --list-devices 2>&1 | grep -q $dev \
  || echo "WARNING: $dir/llama-server lists no $dev device (llamacpp_server.sh will refuse it)" >&2
echo "$dir"
