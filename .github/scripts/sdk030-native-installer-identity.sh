#!/bin/bash
set -euo pipefail

platform=$1
expected_tree=66250a2690d7e4abc7a4d3def5106f8b9c5359dc
git diff --exit-code "$expected_tree" HEAD -- skills

runner_scratch=$RUNNER_TEMP
bin=bin
expected_wheel=6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f
selections=(0.3.0)
if [[ "$platform" == windows ]]; then
  runner_scratch=$(cygpath -u "$runner_scratch")
  native_python_dir=$(cygpath -u 'C:\Program Files\Python312')
  test -x "$native_python_dir/python.exe"
  export PATH="$native_python_dir:$PATH"
  export PYTHON=python
  "$PYTHON" --version
  bin=Scripts
  expected_wheel=06afbc88370fde741037ebd38dba91c2184cb4d2fbd5cee337720cd494a06752
  selections+=(release:0.3.0)
fi
scratch=$(mktemp -d "$runner_scratch/sdk-install-$GITHUB_RUN_ID-XXXXXX")
evidence=$scratch/evidence
mkdir -p "$evidence"
if [[ "$platform" == windows ]]; then
  printf 'SDK_INSTALL_EVIDENCE=%s\n' "$(cygpath -w "$evidence")" >> "$GITHUB_ENV"
else
  printf 'SDK_INSTALL_EVIDENCE=%s\n' "$evidence" >> "$GITHUB_ENV"
fi
printf '%s\n' "$expected_tree" > "$evidence/installer-source.txt"
git rev-parse HEAD > "$evidence/workflow-head.txt"
unset GH_TOKEN GITHUB_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN

for selection in "${selections[@]}"; do
  kind=pypi
  [[ "$selection" != release:* ]] || kind=release
  root=$scratch/$kind
  mkdir "$evidence/$kind"
  bash skills/loom-setup/scripts/install_loom.sh "$root" "$selection" | tee "$evidence/$kind/install.log"
  cp "$root/loom-install.json" "$evidence/$kind/loom-install.json"
  git -C "$root/library" rev-parse HEAD > "$evidence/$kind/catalog-head.txt"
  "$root/.venv/$bin/loom-fetch" --help > "$evidence/$kind/fetch-help.txt"
  "$root/.venv/$bin/loom-serve" --help > "$evidence/$kind/serve-help.txt"
  "$root/.venv/$bin/python" - "$root" "$kind" "$expected_wheel" <<'PY' | tee "$evidence/$kind/verification.json"
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

root, kind, expected = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
identity = json.loads((root / 'loom-install.json').read_text())
assert identity == {
    'artifact_id': None, 'branch': 'v0.3.0', 'hrx_loom': '0.3.0',
    'hrx_system_sha': 'e2c41a720363911c04606ce0f4417e2d55630352',
    'source': kind, 'wheel_sha256': expected,
}, identity
wheel, = root.glob('wheel-*/*.whl')
with wheel.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
distribution = importlib.metadata.distribution('hrx-loom')
assert distribution.version == '0.3.0'
catalog = json.loads(distribution.read_text('loom.json'))['catalog'].rpartition('@')[2]
assert catalog
def revision(ref):
    return subprocess.check_output(['git', '-C', str(root / 'library'), 'rev-parse', ref], text=True).strip()
resolved = revision('HEAD')
assert resolved == revision(f'{catalog}^{{commit}}')
print(json.dumps({'identity': identity, 'wheel': wheel.name, 'catalog_declared': catalog,
                  'catalog_head': resolved, 'normal_launchers': ['loom-fetch --help', 'loom-serve --help']}, indent=2))
PY
done
