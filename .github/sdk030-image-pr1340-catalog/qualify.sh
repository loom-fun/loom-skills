#!/usr/bin/env bash
set -euo pipefail
repo=$PWD
exec > >(tee "$WORK/evidence/qualification.log") 2>&1
cd "$WORK"
checkout() {
  local dir=$1 url=$2 ref=$3 expected=$4
  git init -q "$dir"
  git -C "$dir" fetch -q --depth=1 "$url" "$ref"
  git -C "$dir" checkout -q --detach FETCH_HEAD
  test "$(git -C "$dir" rev-parse HEAD)" = "$expected"
}
catalog=https://github.com/ROCm/hrx-loom-library.git
checkout port https://github.com/loom-fun/loom-skills.git "$PR_HEAD" "$PR_HEAD"
checkout release "$catalog" refs/tags/v0.3.0 "$RELEASE_CATALOG_HEAD"
checkout flux "$catalog" "$FLUX_CATALOG_HEAD" "$FLUX_CATALOG_HEAD"
printf 'PR_HEAD=%s\nRELEASE_CATALOG_HEAD=%s\nFLUX_CATALOG_HEAD=%s\n' \
  "$PR_HEAD" "$RELEASE_CATALOG_HEAD" "$FLUX_CATALOG_HEAD" | tee "$WORK/evidence/revisions.txt"
bash "$WORK/port/skills/loom-setup/scripts/install_loom.sh" "$WORK/sdk" 0.3.0
cp "$WORK/sdk/loom-install.json" "$WORK/evidence/sdk.json"
python3 - "$WORK/evidence/sdk.json" <<'PY'
import json
from pathlib import Path
import sys
identity = json.loads(Path(sys.argv[1]).read_text())
assert identity["hrx_loom"] == "0.3.0", identity
assert identity["wheel_sha256"] == "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f", identity
PY
check_card() {
  local family=$1 source=$2 result=$WORK/evidence/$1
  mkdir "$result"
  python3 "$WORK/port/skills/loom-kb/kb.py" route --arch "$family" --device strix-halo --topic image-porting \
    --relative | tee "$result/route.txt"
  grep -Fx "arch/image/diffusion/$family/README.md" "$result/route.txt"
  grep -Fx 'device/rdna3/strix-halo/image-porting.md' "$result/route.txt"
  grep -Fx 'topic/image-porting.md' "$result/route.txt"
  python3 "$WORK/port/skills/loom-port-image/scripts/stage_hf.py" "$source" "$WORK/$family"
  python3 "$repo/.github/sdk030-image-pr1340-catalog/verify.py" "$WORK/$family" "$source" "$result" "$WORK/sdk/.venv/bin"
}
rc=0
(set -e; check_card krea2 "$WORK/release/model/krea/Krea-2-Turbo") & krea=$!
(set -e; check_card flux2 "$WORK/flux/model/black-forest-labs/FLUX.2-klein-4B") & flux=$!
wait "$krea" || rc=1
wait "$flux" || rc=1
exit "$rc"
