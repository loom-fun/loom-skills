#!/usr/bin/env bash
# Single bounded official DevLab job. The caller owns the 1800 s hard timeout.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
CHECKOUT_HEAD=$(git -C "$ROOT" rev-parse HEAD)
if [ -n "${GITHUB_SHA:-}" ] && [ "$CHECKOUT_HEAD" != "$GITHUB_SHA" ]; then
  echo "checkout SHA mismatch: HEAD=$CHECKOUT_HEAD GITHUB_SHA=$GITHUB_SHA" >&2; exit 18
fi
TRACKED_CHANGES=$(git -C "$ROOT" status --porcelain --untracked-files=no)
if [ -n "$TRACKED_CHANGES" ]; then
  echo 'checkout has tracked changes; refusing to run an unreviewed tree' >&2; exit 18
fi
OUT=${1:?usage: run_job.sh OUT_DIR}
JOB_DIR=$(dirname "$OUT")
mkdir -p "$OUT" "$JOB_DIR/assets"
GGUF=''; TOKENIZER="$JOB_DIR/assets/tokenizer.json"; SDK_ROOT=''; PY='python3'
sanitize_output() {
  rm -rf "$OUT/builds/baseline" "$OUT/builds/candidate"
  rm -f "$OUT/builds/target_text_build_config.runtime.json"
  "$PY" - "$OUT" "$OUT" "$GGUF" "$TOKENIZER" "$SDK_ROOT" "$ROOT" "$HERE" "$JOB_DIR" 2>/dev/null <<'PY' || true
import sys
from pathlib import Path
out=Path(sys.argv[1]); secrets=[x for x in sys.argv[2:] if x]
for path in out.rglob('*'):
    if not path.is_file(): continue
    try: text=path.read_text()
    except (UnicodeError,OSError): continue
    for secret in secrets: text=text.replace(secret,'<job-local-path>')
    path.write_text(text)
PY
  rm -rf "$JOB_DIR/assets" "$JOB_DIR"/loom-sdk*
}
trap sanitize_output EXIT
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"

CONFIG="$HERE/target_text_llm_config.json"
SDK_ROOT=$("$ROOT/skills/loom-setup/scripts/ensure_sdk.sh" "$JOB_DIR/loom-sdk" "$CONFIG")
VENV="$SDK_ROOT/.venv"
PY="$SDK_ROOT/.tools/bin/python"
FETCH_PY="$PY"
"$FETCH_PY" -c 'import huggingface_hub' >/dev/null 2>&1 || FETCH_PY="$VENV/bin/python"
"$FETCH_PY" -c 'import huggingface_hub' >/dev/null 2>&1 || { echo 'pinned SDK install lacks huggingface_hub; stop before GPU work' >&2; exit 22; }
test -x "$VENV/bin/loom-serve" && test -x "$PY" || { echo 'pinned SDK tools are incomplete' >&2; exit 20; }
"$PY" "$HERE/runner/sdk_pin.py" matches "$SDK_ROOT" "$CONFIG" || { echo 'installed SDK pin mismatch' >&2; exit 21; }

# Bind the target DRM card to its read-only KFD gfx target before fetching model weights.
"$PY" "$HERE/cold_capacity_targeted.py" identify-gpu --out "$OUT/hardware_identity.json"

# Use the exact immutable public checkpoint revision in the official HF cache. One cache fetch only.
FETCH_JSON="$JOB_DIR/assets/checkpoint_fetch.json"
PYTHONPATH="$HERE/runner${PYTHONPATH:+:$PYTHONPATH}" "$FETCH_PY" - "$FETCH_JSON" <<'PY'
import json, os, sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download
from hf_download import download_with_optional_token
repo='unsloth/gemma-4-E4B-it-GGUF'
revision='bfc15c382204943c3a8fff0c750b94ae2364d7a3'
filename='gemma-4-E4B-it-Q4_K_M.gguf'
cache=Path(os.environ['HF_HUB_CACHE'])
start=time.monotonic(); cache_hit=True
try:
    path=download_with_optional_token(hf_hub_download, repo_id=repo, filename=filename,
                         revision=revision, cache_dir=str(cache), local_files_only=True)
except Exception:
    cache_hit=False
    path=download_with_optional_token(hf_hub_download, repo_id=repo, filename=filename,
                         revision=revision, cache_dir=str(cache))
elapsed=time.monotonic()-start
p=Path(path); size=p.stat().st_size
json.dump({'repo':repo,'revision':revision,'filename':filename,'bytes':size,
           'fetch_bytes':0 if cache_hit else size,'cache_hit':cache_hit,
           'fetch_elapsed_seconds':elapsed,'fetch_bytes_basis':'full file size on cache miss; resumable transfer bytes are not exposed by hf_hub_download',
           'cache_path_kind':'official_huggingface_hub_cache'},
          open(sys.argv[1],'w'),indent=2)
print(json.dumps({'checkpoint_bytes':size,'fetch_elapsed_seconds':elapsed}),flush=True)
PY
GGUF=$(PYTHONPATH="$HERE/runner${PYTHONPATH:+:$PYTHONPATH}" "$FETCH_PY" -c 'import json,os,sys; from hf_download import download_with_optional_token; from huggingface_hub import hf_hub_download; d=json.load(open(sys.argv[1])); print(download_with_optional_token(hf_hub_download, repo_id=d["repo"],filename=d["filename"],revision=d["revision"],cache_dir=os.environ["HF_HUB_CACHE"]))' "$FETCH_JSON")
FETCH_BYTES=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["fetch_bytes"])' "$FETCH_JSON")
FETCH_SECONDS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["fetch_elapsed_seconds"])' "$FETCH_JSON")
FETCH_CACHE_HIT=$("$PY" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["cache_hit"]).lower())' "$FETCH_JSON")

# Exact tokenizer from the maintained E4B card source; one immutable revision and SHA check.
TOKENIZER="$JOB_DIR/assets/tokenizer.json"
"$FETCH_PY" - "$TOKENIZER" <<'PY'
import hashlib,sys,urllib.request
url='https://huggingface.co/loom-fun/unsloth-gemma-4-E4B-it-loom/resolve/1d1f17f3438cdc9c3d0fc83255c9c8f9a68d38f7/tokenizer.json'
data=urllib.request.urlopen(url,timeout=60).read()
sha=hashlib.sha256(data).hexdigest()
if len(data)!=15142283 or sha!='fad8ef1277f412fd6276f1fef8169f1c4b36dc3e1b784888c58ae458470900fd':
    raise SystemExit(f'tokenizer pin mismatch: bytes={len(data)} sha256={sha}')
open(sys.argv[1],'wb').write(data)
print(f'tokenizer_verified bytes={len(data)} sha256={sha}',flush=True)
PY

export GEMMA_E4B_DEVLAB_ACK=1 GEMMA_E4B_GGUF="$GGUF" GEMMA_E4B_TOKENIZER="$TOKENIZER"
export GEMMA_E4B_BUILD_PYTHON="$PY"
export GEMMA_E4B_FETCH_BYTES="$FETCH_BYTES" GEMMA_E4B_FETCH_SECONDS="$FETCH_SECONDS"
export GEMMA_E4B_FETCH_CACHE_HIT="$FETCH_CACHE_HIT" LOOM_SDK_ROOT="$SDK_ROOT"
export LOOM_KB_ROOT="$ROOT/skills/loom-kb"
timeout --signal=TERM --kill-after=10s 1700s "$HERE/run_frontier.sh" "$OUT"

echo "frozen single-job experiment complete: $OUT"
