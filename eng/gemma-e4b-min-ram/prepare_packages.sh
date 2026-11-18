#!/usr/bin/env bash
# Build both exact target-only arms from the same frozen config/checkpoint.
# Run only on the lead-designated DevLab checkout, after GPU-job dispatch review.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: prepare_packages.sh OUT_DIR}
test "${GEMMA_E4B_DEVLAB_ACK:-0}" = 1 || { echo 'run only on the designated DevLab; set GEMMA_E4B_DEVLAB_ACK=1 after confirming isolated runner' >&2; exit 19; }
mkdir -p "$OUT"
GGUF=${GEMMA_E4B_GGUF:?set to the exact pinned Q4_K_M file on DevLab}
TOKENIZER=${GEMMA_E4B_TOKENIZER:?set to the pinned exact card tokenizer.json}
BUILD_PY=${GEMMA_E4B_BUILD_PYTHON:-python3}
EXPECTED_KIT=e40fd8d3d77174bf809860ae91f6db2214a9e207dad341b663dc3af22f0127fd
EXPECTED_BUILDER=faeb722cee49d011cd3422d79832dfb9533710845f7f9e8041b841ff9d363db4
EXPECTED_TOKENIZER=fad8ef1277f412fd6276f1fef8169f1c4b36dc3e1b784888c58ae458470900fd
EXPECTED_CHECKPOINT=36c0282999c77e97bca7354f5a7aa8cd43b0f4167e1f4febfa755073428d1b41
EXPECTED_BUILD_CONFIG=30f607b7e0ed58dbcde90df0518395639c61c6970c1ad2a6e80ddc0390e63345

test "$(sha256sum "$HERE/kit/build.py" | cut -d' ' -f1)" = "$EXPECTED_BUILDER" || { echo 'builder hash mismatch' >&2; exit 21; }
kit_stamp=$("$BUILD_PY" "$HERE/kit/kit_source.py" stamp "$HERE/kit")
kit_sha=$("$BUILD_PY" - "$HERE/kit/KIT_SOURCE" "$kit_stamp" <<'PY'
import json,sys
source=json.load(open(sys.argv[1])); stamp=json.loads(sys.argv[2])
if source['skills_commit'] != 'bacdd3ed241eb62f9578b909ef63eaa0751f3c49' or source['dirty'] is not False:
    raise SystemExit('kit source is not the clean pinned main commit')
if stamp['kit']['modified'] is not False or source.get('kb_root') is not None:
    raise SystemExit('kit is modified or embeds a nonportable KB path')
print(source['kit_sha256'])
PY
)
test "$kit_sha" = "$EXPECTED_KIT" || { echo 'kit source pin mismatch' >&2; exit 21; }
test "$(sha256sum "$TOKENIZER" | cut -d' ' -f1)" = "$EXPECTED_TOKENIZER" || { echo 'tokenizer hash mismatch' >&2; exit 22; }
test "$(stat -c %s "$TOKENIZER")" = 15142283 || { echo 'tokenizer byte size mismatch' >&2; exit 22; }
test "$(sha256sum "$HERE/target_build_config.json" | cut -d' ' -f1)" = "$EXPECTED_BUILD_CONFIG" || { echo 'build config hash mismatch' >&2; exit 22; }
test "$(stat -Lc %s "$GGUF")" = 4977171584 || { echo 'checkpoint byte size mismatch' >&2; exit 23; }
before=$(stat -Lc '%d:%i:%s:%Y' "$GGUF")
"$BUILD_PY" - "$OUT/meminfo_before_hash.json" <<'PY'
import json,sys
keys={'MemTotal','MemFree','MemAvailable','Cached','SReclaimable','Shmem'}
d={}
for line in open('/proc/meminfo'):
 k, v = line.split(':',1)
 if k in keys: d[k]=int(v.split()[0])*1024
json.dump(d,open(sys.argv[1],'w'),indent=2)
PY
hash_start=$(date +%s.%N)
actual_checkpoint=$(sha256sum "$GGUF" | cut -d' ' -f1)
hash_end=$(date +%s.%N)
after=$(stat -Lc '%d:%i:%s:%Y' "$GGUF")
test "$before" = "$after" || { echo 'checkpoint identity changed while hashing' >&2; exit 23; }
test "$actual_checkpoint" = "$EXPECTED_CHECKPOINT" || { echo 'actual checkpoint SHA256 mismatch' >&2; exit 23; }
"$BUILD_PY" - "$OUT/meminfo_after_hash.json" <<'PY'
import json,sys
keys={'MemTotal','MemFree','MemAvailable','Cached','SReclaimable','Shmem'}
d={}
for line in open('/proc/meminfo'):
 k, v = line.split(':',1)
 if k in keys: d[k]=int(v.split()[0])*1024
json.dump(d,open(sys.argv[1],'w'),indent=2)
PY

"$BUILD_PY" - "$HERE/target_build_config.json" "$GGUF" "$OUT/target_text_build_config.runtime.json" <<'PY'
import json, os, sys
src, gguf, dst = sys.argv[1:]
d = json.load(open(src))
if 'vision' in d or 'assistant' in d or 'drafter' in d:
    raise SystemExit('target-only source config unexpectedly carries vision/assistant/drafter components')
if len(d.get('tensors', {})) != 720:
    raise SystemExit('target tensor geometry differs from the frozen 720-tensor E4B build config')
d['gguf'] = os.path.abspath(gguf)
json.dump(d, open(dst, 'w'), indent=1)
PY

builder="$HERE/kit/build.py"
for arm in baseline candidate; do
  if [ "$arm" = baseline ]; then context=131072; rows=4; else context=65536; rows=1; fi
  pkg="$OUT/$arm"
  test ! -e "$pkg" || { echo "refusing to overwrite $pkg" >&2; exit 24; }
  "$BUILD_PY" "$builder" "$OUT/target_text_build_config.runtime.json" "$pkg" \
    --tokenizer "$TOKENIZER" --context "$context" --rows "$rows" \
    --prefill 1024 --memory-gib auto --draft none \
    --model-id google/gemma-4-E4B-it | tee "$OUT/$arm.build.log"
  "$BUILD_PY" - "$pkg/model.json" "$context" "$rows" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); expected_context, expected_rows = map(int, sys.argv[2:])
matches=[]
for name,p in d['presets'].items():
    e=p['execution']
    if e.get('context_capacity') == expected_context and e.get('rows') == expected_rows:
        matches.append((name,e))
if len(matches) != 1:
    raise SystemExit(f'expected one exact context/rows preset; got {matches!r}')
print(json.dumps({'preset':matches[0][0], 'execution':matches[0][1]}, sort_keys=True))
PY
done
printf '%s  %s\n' "$actual_checkpoint" "gemma-4-E4B-it-Q4_K_M.gguf" > "$OUT/checkpoint_pin.sha256"
printf 'devlab_checkpoint_hash_elapsed_start=%s\ndevlab_checkpoint_hash_elapsed_end=%s\ncheckpoint_identity_device_inode_size_mtime=%s\ncheckpoint_fetch_bytes_estimate=%s\ncheckpoint_fetch_cache_hit=%s\ncheckpoint_fetch_seconds=%s\nsource_checkout_head=%s\n' \
  "$hash_start" "$hash_end" "$before" "${GEMMA_E4B_FETCH_BYTES:-unknown}" "${GEMMA_E4B_FETCH_CACHE_HIT:-unknown}" "${GEMMA_E4B_FETCH_SECONDS:-unknown}" \
  "$(git -C "$ROOT" rev-parse HEAD)" > "$OUT/preparation_provenance.txt"
sha256sum "$HERE/target_build_config.json" "$TOKENIZER" \
  "$OUT"/baseline/model.json "$OUT"/candidate/model.json \
  "$OUT"/baseline/program/storage.loom "$OUT"/candidate/program/storage.loom \
  "$OUT"/baseline/program/llm.loom "$OUT"/candidate/program/llm.loom > "$OUT/package_hashes.sha256"
echo "built exact target-only packages: $OUT"
