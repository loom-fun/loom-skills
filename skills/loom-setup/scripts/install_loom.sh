#!/bin/bash
# Install the Loom SDK (hrx-loom) + the matching model library, from scratch.
# usage: [VENV_NAME=.venv] [PYTHON=python3] install_loom.sh [ROOT=~/loom] [VERSION|release:VERSION|pypi:VERSION|latest|ci-latest|ci:ARTIFACT_ID]
#   0.3.0 (default), 0.2.0, 0.1.0, pypi:V: hrx-loom from PyPI (0.1.0 is what packages without an SDK pin were built with)
#   release:V, or a bare pre-release VERSION (0.2.0rc1, 0.3.0a1): hrx-loom from the ROCm/hrx-system GitHub release vV (a
#                     pre-release, not on PyPI): the wheel asset, its sha256 verified against the release's SHA256SUMS;
#                     no gh needed
#   latest:           newest hrx-loom on PyPI (the starter's templates may need updating for a new ABI)
#   ci-latest:        newest "loom-sdk-wheel-linux-Release" artifact from ROCm/hrx-system main (development builds;
#                     Windows: "loom-sdk-wheel-windows"); needs an authenticated gh
#   ci:ARTIFACT_ID:   that CI artifact
# The library (ROCm/hrx-loom-library) is checked out at the catalog revision the installed SDK names in
# hrx_loom-*.dist-info/loom.json, so catalog cards and the SDK agree.
# Writes ROOT/loom-install.json, the SDK's identity (every build reports version 0.1.0, so the version alone
# doesn't say which SDK compiled or served a package): {"hrx_loom": version, "source": "release" | "pypi" | "ci",
# "hrx_system_sha": full ROCm/hrx-system commit (a release: the tag's commit), "artifact_id": CI artifact id, "branch": its branch or tag,
# "wheel_sha256": sha256 of the wheel file}. For a PyPI wheel the commit is that of the hrx-system release tag
# v<version> whose asset has the same sha256 (null without gh, or when they differ). loom-port-llm's build.py
# stamps it into each package's llm_config.json; the card scripts record it with each measurement.
# Needs: python >= 3.12 with venv, git (and gh + unzip for ci-*). Windows: run from Git Bash (venvs use Scripts/).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${1:-$HOME/loom}; which=${2:-0.3.0}
# Which source a VERSION means: release:V and bare pre-release versions (0.2.0rc1) are GitHub releases; pypi:V and the rest PyPI.
case "$which" in
  release:*) which=release:${which#release:};;
  pypi:*) which=${which#pypi:};;
  ci-latest|ci:*|latest) ;;
  *[0-9]rc[0-9]*|*[0-9]a[0-9]*|*[0-9]b[0-9]*) which=release:$which;;
esac
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) win=1; bin=Scripts; py=${PYTHON:-py -3};; *) win=0; bin=bin; py=${PYTHON:-python3};; esac
mkdir -p "$root"; cd "$root"
venv=${VENV_NAME:-.venv}
$py -m venv $venv
$venv/$bin/python -m pip install -q --upgrade pip
newest() {  # newest unexpired artifact from the upstream main branch
  gh api "repos/ROCm/hrx-system/actions/artifacts?name=$1&per_page=50" \
    -q '[.artifacts[] | select((.expired | not) and .workflow_run.head_branch == "main" and .workflow_run.repository_id == .workflow_run.head_repository_id)] | .[0].id'
}
art=; branch=; sha=; source=pypi
case "$which" in
  ci-latest|ci:*)
    artifact_kind=loom-sdk-wheel-linux-Release
    [ "$win" != 1 ] || artifact_kind=loom-sdk-wheel-windows
    if [ "$which" = ci-latest ]; then
      art=$(newest "$artifact_kind")
      art=${art%$'\r'}
      [[ $art =~ ^[0-9]+$ ]] || { echo "no unexpired main SDK wheel artifact found; choose a reviewed artifact by ID" >&2; exit 1; }
    else art=${which#ci:}; fi
    [[ $art =~ ^[0-9]+$ ]] || { echo "invalid SDK artifact ID" >&2; exit 1; }
    source=ci
    read -r artifact_name artifact_expired branch sha digest < <(gh api "repos/ROCm/hrx-system/actions/artifacts/$art" \
      -q '[.name, .expired, (.workflow_run.head_branch // "unknown"), .workflow_run.head_sha, (.digest // "-")] | @tsv')
    digest=${digest%$'\r'}
    [[ $artifact_name == "$artifact_kind" && $artifact_expired == false && $sha =~ ^[0-9a-f]{40}$ ]] || {
      echo "artifact $art is not an available $artifact_kind SDK wheel" >&2; exit 1;
    }
    echo "artifact $art ($branch ${sha:0:12})"
    mkdir -p "wheel-$art"
    sdk_archive="wheel-$art/sdk.zip"
    found=$(sha256sum "$sdk_archive" 2>/dev/null | cut -d' ' -f1 || true)
    if [[ $digest != sha256:"$found" || -z $found ]]; then
      gh api "repos/ROCm/hrx-system/actions/artifacts/$art/zip" > "$sdk_archive.part"
      mv -f -- "$sdk_archive.part" "$sdk_archive"
    fi
    whl=$("$venv/$bin/python" - "$sdk_archive" "$digest" "$win" <<'PY'
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from zipfile import ZipFile

archive, expected, windows = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
if expected != "-":
    with archive.open("rb") as stream:
        actual = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        sys.exit("SDK artifact SHA-256 does not match GitHub metadata")
platform = r"win_amd64" if windows else r"manylinux_[\w.]+_x86_64"
name = re.compile(rf"hrx_loom-[0-9][A-Za-z0-9_.!]*-py3-none-{platform}\.whl")
with ZipFile(archive) as package:
    wheels = [(item, PurePosixPath(item.filename).name) for item in package.infolist()
              if name.fullmatch(PurePosixPath(item.filename).name) and not item.is_dir()]
    if len(wheels) != 1:
        sys.exit("SDK artifact must contain one plain release wheel for this platform")
    item, wheel_name = wheels[0]
    wheel = archive.with_name(wheel_name)
    temporary = wheel.with_suffix(".whl.part")
    with package.open(item) as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    temporary.replace(wheel)
sys.stdout.write(wheel.as_posix())
PY
    );;
  release:*)
    # A GitHub release of ROCm/hrx-system: the wheel asset, verified against the release's SHA256SUMS. No gh or token.
    source=release; rel=${which#release:}; branch=v$rel
    if [ $win = 1 ]; then asset=hrx_loom-$rel-py3-none-win_amd64.whl; else asset=hrx_loom-$rel-py3-none-manylinux_2_39_x86_64.whl; fi
    url=https://github.com/ROCm/hrx-system/releases/download/v$rel
    mkdir -p wheel-$rel
    if [ ! -f wheel-$rel/$asset ]; then
      curl -fsSL --retry 3 -o wheel-$rel/$asset.part "$url/$asset" && mv -f wheel-$rel/$asset.part wheel-$rel/$asset
    fi
    curl -fsSL --retry 3 -o wheel-$rel/SHA256SUMS "$url/SHA256SUMS"
    want=$(awk -v n="$asset" '{f=$2; sub(/^\*/,"",f); sub(/\r$/,"",f); if (f==n) print $1}' wheel-$rel/SHA256SUMS)
    [ -n "$want" ] || { echo "$asset is not listed in the release's SHA256SUMS"; exit 1; }
    got=$($venv/$bin/python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' wheel-$rel/$asset)
    got=${got%$'\r'}
    [ "$got" = "$want" ] || { rm -f wheel-$rel/$asset; echo "sha256 mismatch for $asset: SHA256SUMS $want, downloaded $got"; exit 1; }
    # The tag's commit (an annotated tag peels with ^{}); plain git, no gh.
    sha=$(git ls-remote https://github.com/ROCm/hrx-system "refs/tags/v$rel^{}" "refs/tags/v$rel" | awk '{print $1}' | head -1)
    whl=wheel-$rel/$asset;;
  *)
    # PyPI: download the wheel first so its sha256 can be recorded, then install that file.
    spec=hrx-loom; [ "$which" = latest ] || spec="hrx-loom==$which"
    rm -rf wheel-pypi; $venv/$bin/pip download -q --no-deps -d wheel-pypi "$spec"
    whl=$(ls wheel-pypi/hrx_loom-*.whl | head -1);;
esac
wheel_sha=$($venv/$bin/python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$whl")
wheel_sha=${wheel_sha%$'\r'}
if [ "$source" != ci ]; then
  verified_sha=$($venv/$bin/python "$here/sdk_pin.py" verify-release-wheel "$(basename "$whl")" "$wheel_sha")
  verified_sha=${verified_sha%$'\r'}
  if [ -n "$verified_sha" ]; then
    sha=$verified_sha; branch=v$(basename "$whl" | cut -d- -f2)
  fi
fi
$venv/$bin/pip install -q --force-reinstall --no-deps "$whl" && $venv/$bin/pip install -q "$whl[hf]"
version=$($venv/$bin/python -c 'import importlib.metadata as m;print(m.version("hrx-loom"))')
version=${version%$'\r'}
if [ $source = pypi ] && [ -z "$sha" ] && command -v gh >/dev/null; then
  # The PyPI wheel is the hrx-system release asset with the same sha256: record the release tag's commit.
  q=".assets[] | select(.name == \"$(basename "$whl")\") | .digest"
  digest=$(gh api "repos/ROCm/hrx-system/releases/tags/v$version" -q "$q" 2>/dev/null || true)
  digest=${digest%$'\r'}
  if [ "$digest" = "sha256:$wheel_sha" ]; then
    branch=v$version; sha=$(gh api "repos/ROCm/hrx-system/commits/v$version" -q .sha 2>/dev/null || true)
  fi
fi
sha=${sha%$'\r'}
$venv/$bin/python -c '
import json, sys
v, src, sha, art, branch, wsha = sys.argv[1:]
print(json.dumps({"hrx_loom": v, "source": src, "hrx_system_sha": sha or None, "artifact_id": int(art) if art else None,
                  "branch": branch or None, "wheel_sha256": wsha}, indent=1, sort_keys=True))' \
  "$version" "$source" "$sha" "$art" "$branch" "$wheel_sha" > loom-install.json
echo "hrx-loom $version from $source${sha:+, hrx-system ${sha:0:12}} (loom-install.json)"
# The SDK names its compatible catalog as "github:ROCm/hrx-loom-library@<revision>".
catalog=$($venv/$bin/python - <<'EOF'
import glob, json, os, site
for d in site.getsitepackages():
    for f in glob.glob(os.path.join(d, "hrx_loom-*.dist-info", "loom.json")):
        print(json.load(open(f)).get("catalog", "").rpartition("@")[2]); raise SystemExit
EOF
)
catalog=${catalog%$'\r'}
[ -d library/.git ] || git clone -q https://github.com/ROCm/hrx-loom-library library
if [ -n "$catalog" ]; then
  if [[ $catalog =~ ^[0-9a-f]{40}$ ]]; then
    git -C library fetch -q origin "$catalog" 2>/dev/null || git -C library fetch -q origin
    git -C library checkout -q "$catalog"
  else   # a tag (release SDKs: loom.json names @v0.2.0)
    git -C library fetch -q --force origin tag "$catalog" && git -C library checkout -q FETCH_HEAD
  fi
  echo "library at catalog revision ${catalog:0:12}"
else
  echo "SDK names no catalog revision (development build); library left at $(git -C library rev-parse --short HEAD)"
fi
# Tools venv for ports/benchmarks/UI (kept separate from the SDK venv).
[ -d .tools ] || $py -m venv .tools
.tools/$bin/pip install -q gguf numpy tokenizers requests jinja2 fastapi uvicorn httpx pillow
$venv/$bin/loom-serve --help >/dev/null && echo "OK: hrx-loom $version in $root/$venv/$bin; tools venv $root/.tools"
