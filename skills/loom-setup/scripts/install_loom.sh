#!/bin/bash
# Install the Loom SDK (hrx-loom) + the matching model library, from scratch.
# usage: [VENV_NAME=.venv] [PYTHON=python3] install_loom.sh [ROOT=~/loom] [VERSION|latest|ci-latest|ci:ARTIFACT_ID]
#   0.1.0 (default):  hrx-loom from PyPI at the version the loom-port-llm starter is validated with
#   latest:           newest hrx-loom on PyPI (the starter's templates may need updating for a new ABI)
#   ci-latest:        newest "loom-sdk-wheel-linux-Release" artifact from ROCm/hrx-system main (development builds;
#                     Windows: "loom-sdk-wheel-windows"); needs an authenticated gh
#   ci:ARTIFACT_ID:   that CI artifact
# The library (ROCm/hrx-loom-library) is checked out at the catalog revision the installed SDK names in
# hrx_loom-*.dist-info/loom.json, so catalog cards and the SDK agree.
# Writes ROOT/loom-install.json, the SDK's identity (every build reports version 0.1.0, so the version alone
# doesn't say which SDK compiled or served a package): {"hrx_loom": version, "source": "pypi" | "ci",
# "hrx_system_sha": full ROCm/hrx-system commit, "artifact_id": CI artifact id, "branch": its branch or tag,
# "wheel_sha256": sha256 of the wheel file}. For a PyPI wheel the commit is that of the hrx-system release tag
# v<version> whose asset has the same sha256 (null without gh, or when they differ). loom-port-llm's build.py
# stamps it into each package's llm_config.json; the card scripts record it with each measurement.
# Needs: python >= 3.12 with venv, git (and gh + unzip for ci-*). Windows: run from Git Bash (venvs use Scripts/).
set -euo pipefail
root=${1:-$HOME/loom}; which=${2:-0.1.0}
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) win=1; bin=Scripts; py=${PYTHON:-py -3};; *) win=0; bin=bin; py=${PYTHON:-python3};; esac
mkdir -p "$root"; cd "$root"
venv=${VENV_NAME:-.venv}
$py -m venv $venv
$venv/$bin/python -m pip install -q --upgrade pip
newest() {  # newest unexpired artifact named $1, preferring main and v* tags
  gh api "repos/ROCm/hrx-system/actions/artifacts?name=$1&per_page=50" \
    -q '[.artifacts[] | select(.expired | not)] | (map(select(.workflow_run.head_branch == "main" or ((.workflow_run.head_branch // "") | startswith("v")))) + .) | .[0].id'
}
art=; branch=; sha=; source=pypi
case "$which" in
  ci-latest|ci:*)
    if [ "$which" = ci-latest ]; then
      if [ $win = 1 ]; then art=$(newest loom-sdk-wheel-windows); else art=$(newest loom-sdk-wheel-linux-Release); fi
    else art=${which#ci:}; fi
    [ -n "$art" ] && [ "$art" != null ] || { echo "no SDK wheel artifact found"; exit 1; }
    source=ci
    read -r branch sha < <(gh api repos/ROCm/hrx-system/actions/artifacts/$art -q '.workflow_run.head_branch + " " + .workflow_run.head_sha')
    echo "artifact $art ($branch ${sha:0:12})"
    mkdir -p wheel-$art
    # `gh run download` trips a path-traversal check on these zips; fetch the zip directly.
    [ -f wheel-$art/sdk.zip ] || gh api repos/ROCm/hrx-system/actions/artifacts/$art/zip > wheel-$art/sdk.zip
    unzip -o -q wheel-$art/sdk.zip -d wheel-$art
    # Some artifacts bundle +asan/+debug/+tracy variants next to the release wheel; take the plain one.
    whl=$(ls wheel-$art/*.whl | grep -v '+' | head -1);;
  *)
    # PyPI: download the wheel first so its sha256 can be recorded, then install that file.
    spec=hrx-loom; [ "$which" = latest ] || spec="hrx-loom==$which"
    rm -rf wheel-pypi; $venv/$bin/pip download -q --no-deps -d wheel-pypi "$spec"
    whl=$(ls wheel-pypi/hrx_loom-*.whl | head -1);;
esac
$venv/$bin/pip install -q --force-reinstall --no-deps "$whl" && $venv/$bin/pip install -q "$whl[hf]"
wheel_sha=$($venv/$bin/python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$whl")
version=$($venv/$bin/python -c 'import importlib.metadata as m;print(m.version("hrx-loom"))')
if [ $source = pypi ] && command -v gh >/dev/null; then
  # The PyPI wheel is the hrx-system release asset with the same sha256: record the release tag's commit.
  q=".assets[] | select(.name == \"$(basename "$whl")\") | .digest"
  digest=$(gh api "repos/ROCm/hrx-system/releases/tags/v$version" -q "$q" 2>/dev/null || true)
  if [ "$digest" = "sha256:$wheel_sha" ]; then
    branch=v$version; sha=$(gh api "repos/ROCm/hrx-system/commits/v$version" -q .sha 2>/dev/null || true)
  fi
fi
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
[ -d library/.git ] || git clone -q https://github.com/ROCm/hrx-loom-library library
if [ -n "$catalog" ]; then
  git -C library fetch -q origin "$catalog" 2>/dev/null || git -C library fetch -q origin
  git -C library checkout -q "$catalog" && echo "library at catalog revision ${catalog:0:12}"
else
  echo "SDK names no catalog revision (development build); library left at $(git -C library rev-parse --short HEAD)"
fi
# Tools venv for ports/benchmarks/UI (kept separate from the SDK venv).
[ -d .tools ] || $py -m venv .tools
.tools/$bin/pip install -q gguf numpy tokenizers requests jinja2 fastapi uvicorn httpx pillow
$venv/$bin/loom-serve --help >/dev/null && echo "OK: hrx-loom $($venv/$bin/python -c 'import importlib.metadata as m;print(m.version("hrx-loom"))') in $root/$venv/$bin; tools venv $root/.tools"
