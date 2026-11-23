#!/usr/bin/env bash
set -euo pipefail
exec > >(tee "$WORK/evidence/qualification.log") 2>&1
cd "$WORK"
git init -q port
git -C port remote add origin https://github.com/loom-fun/loom-skills.git
git -C port fetch --depth=3 origin "$PR_HEAD"
git -C port checkout -q --detach FETCH_HEAD
test "$(git -C port rev-parse HEAD)" = "$PR_HEAD"
git init -q catalog
git -C catalog remote add origin https://github.com/ROCm/hrx-loom-library.git
git -C catalog fetch --depth=1 origin "$CATALOG_HEAD"
git -C catalog checkout -q --detach FETCH_HEAD
test "$(git -C catalog rev-parse HEAD)" = "$CATALOG_HEAD"
printf 'PR_HEAD=%s\nCATALOG_HEAD=%s\n' "$PR_HEAD" "$CATALOG_HEAD" | tee "$WORK/evidence/revisions.txt"

bash "$WORK/port/skills/loom-setup/scripts/install_loom.sh" "$WORK/sdk" 0.3.0
cp "$WORK/sdk/loom-install.json" "$WORK/evidence/sdk.json"
python3 - "$WORK/evidence/sdk.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    identity = json.load(f)
assert identity["hrx_loom"] == "0.3.0", identity
assert identity["wheel_sha256"] == "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f", identity
PY
tools="$WORK/sdk/.tools/bin"
sdk="$WORK/sdk/.venv/bin"
scripts="$WORK/port/skills/loom-port-image/scripts"
kb="$WORK/port/skills/loom-kb/kb.py"
rc=0
(
  cd "$WORK/port"
  checks=0
  "$tools/python" -m unittest discover -s tests -p 'test_port_image.py' -v || checks=1
  "$tools/python" -m unittest discover -s tests -p 'test_kb.py' -v || checks=1
  "$tools/python" "$kb" lint || checks=1
  "$tools/python" "$kb" dupes || checks=1
  "$tools/python" fleet/scope_lint.py --base HEAD~2 --head HEAD --local --no-impact --scope-block 'kind: architecture' || checks=1
  exit "$checks"
) |& tee "$WORK/evidence/checks.log" || rc=1

export HF_HOME="$WORK/hf"
catalog="$WORK/catalog/model/Tongyi-MAI/Z-Image-Turbo"
"$tools/python" "$scripts/stage_hf.py" "$catalog" "$WORK/staged"
sha256sum "$WORK/staged/model.json" "$WORK/staged/model.json.lock" | tee -a "$WORK/evidence/revisions.txt"
cp "$WORK/staged/model.json" "$WORK/evidence/staged-model.json"
"$sdk/loom-fetch" "$WORK/staged/model.json" --preset=image --materialize=share \
  --cache="$WORK/asset-cache" --output="$WORK/materialized" --no-input |& tee "$WORK/evidence/fetch.log"
cp "$WORK/materialized/model.json" "$WORK/evidence/materialized-model.json"
cp "$catalog/test/reference.json" "$WORK/evidence/reference.json"
"$tools/python" - "$catalog/test/reference.json" "$WORK/prompts.json" <<'PY'
import json
from pathlib import Path
import sys
ref = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(json.dumps({"prompts": [{"id": "fox", "text": ref["short_prompt"]}], "seeds": [0]}))
PY
port=$("$tools/python" - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)
"$tools/python" "$scripts/bench_images.py" --target "z-image=loom:$WORK/materialized/model.json" \
  --alternate --repeats 2 --steps 8 --cfg-scale 0 --loom-guidance z-image=0 --size 384x384 \
  --prompt-file "$WORK/prompts.json" --out "$WORK/evidence/benchmark" --port "$port" \
  --loom-serve "$sdk/loom-serve" --device amdgpu |& tee "$WORK/evidence/generate.log"
"$tools/python" - "$WORK/evidence" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

out = Path(sys.argv[1])
case = out / "benchmark/z-image/fox_s0_384x384"
meta = json.loads((case / "meta.json").read_text())
assert meta["steps"] == 8 and meta["cfg_scale"] == 0
assert meta["deterministic_across_repeats"]
with Image.open(case / "image.png") as image:
    assert image.size == (384, 384)
    rgb = image.convert("RGB")
    assert all(high - low > 20 for low, high in rgb.getextrema())
    actual = hashlib.sha256(rgb.tobytes()).hexdigest()
reference = json.loads((out / "reference.json").read_text())
expected = reference["rgb_sha256"]["bf16"]["384x384"]
result = {"actual_rgb_sha256": actual, "reference_rgb_sha256": expected, "exact": actual == expected,
          "deterministic": meta["deterministic_across_repeats"], "seconds": meta["seconds"]}
(out / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
print(result)
assert actual == expected, result
PY
exit "$rc"
