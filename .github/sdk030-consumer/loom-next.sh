#!/usr/bin/env bash
set -euo pipefail
exec > >(tee "$WORK/out/loom-next.log") 2>&1
workspace=$PWD
cd "$WORK"
revision=ff6e60a1368a77f7593c8969fabc5d4db7aacd16
git init -q loom-next
git -C loom-next fetch -q --depth 1 https://github.com/loom-fun/loom-next.git "$revision"
git -C loom-next -c advice.detachedHead=false checkout -q FETCH_HEAD

cd loom-next
python3 - <<'PY'
from pathlib import Path

def update(path, old, new):
    file = Path(path)
    text = file.read_text()
    assert text.count(old) == 1, (path, old, text.count(old))
    file.write_text(text.replace(old, new))

provenance = "loom_next/generator/provenance.py"
update(provenance, "0.2.0 is on PyPI:", "0.3.0 is on PyPI:")
update(provenance, "# the commit is the v0.2.0 tag", "# the commit is the v0.3.0 tag")
update(provenance, '"hrx_loom": "0.2.0"', '"hrx_loom": "0.3.0"')
update(provenance, "5148f9e3a264af400ec4bae38bd7b06bf551d28cc33b7a61d67f7bcefa590438",
                   "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f")
update(provenance, "0cd370f2baaa0cfef47cc23007f1a109906f8a52",
                   "e2c41a720363911c04606ce0f4417e2d55630352")
update("README.md",
       "**Targets Loom 0.2.0** (`hrx-loom` 0.2.0), the version every loom-fun package is published for.",
       "**Targets Loom 0.3.0** (`hrx-loom` 0.3.0). The generated packages use the version 1 controller ABI, which Loom 0.3.0 still supports for existing Loom 0.2 packages.")
update("README.md", "Packages need Loom 0.2.0.", "The generated package documents its pinned Loom 0.3.0 runtime.")
update("README.md", 'pip install "hrx-loom[hf]==0.2.0"', 'pip install "hrx-loom[hf]==0.3.0"')
PY
git rev-parse HEAD > "$WORK/out/loom-next.base"
git diff --check
git diff --binary > "$WORK/out/loom-next.patch"

python3 -m venv "$WORK/next-venv"
source "$WORK/next-venv/bin/activate"
export VENV=$VIRTUAL_ENV
wheel=hrx_loom-0.3.0-py3-none-manylinux_2_39_x86_64.whl
python -m pip download --no-deps --dest "$WORK" hrx-loom==0.3.0
echo "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f  $WORK/$wheel" | sha256sum -c -
python -m pip install -q "$WORK/$wheel" gguf numpy tokenizers requests jinja2 huggingface_hub
loom-serve --version
slug=unsloth-Qwen3-0.6B-loom
python -m loom_next evaluate --slugs "$slug" --use all --out "$WORK/evaluate" --jobs 1
python - "$WORK" "$slug" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys

from loom_next import corpus, workspace
from loom_next.port import build_args

root, slug = sys.argv[1:]
fixture = corpus.models()[slug]
published = corpus.published_config(slug)
assert published["sdk"]["hrx_loom"] == "0.2.0"
work = Path(root) / "next-package"
work.mkdir()
corpus.write_stubs(fixture, str(work))
shutil.copy(corpus.tokenizer(slug), work)
kit = workspace.make(str(work))
cfg, warnings = workspace.inspect(kit, str(work), os.path.basename(published["gguf"]))
assert cfg is not None, warnings
model = published["model_id"]
cfg["model_id"] = model
status, output = workspace.build(kit, str(work), cfg, build_args(model))
print(output)
assert status == 0
new = json.loads((work / "pkg" / "llm_config.json").read_text())
assert new["sdk"]["hrx_loom"] == "0.3.0"
assert new["generator"]["hrx_loom"]["hrx_loom"] == "0.3.0"
card = json.loads((work / "pkg" / "model.json").read_text())
assert card["services"]["chat"]["version"] == 1
print("Loom-next read the 0.2.0 fixture and generated v0.3.0 metadata with controller ABI 1")
PY
python "$workspace/skills/loom-migrate/scripts/sweep.py" --sdk "$VIRTUAL_ENV" "$WORK/next-package/pkg" --json "$WORK/out/loom-next-link.json"
