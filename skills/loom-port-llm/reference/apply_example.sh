#!/bin/bash
# usage: apply_example.sh EXAMPLE KIT [CFG]     (EXAMPLE: gemma4 | lfm2moe | flint | glimmer, with or without -example;
#                                               starter: the plain starter)
# Makes KIT the current starter plus a reference port: copies the starter to KIT (when KIT doesn't exist yet),
# then the example's own files (everything but README.md, cfg.json and starter.patch), then applies
# starter.patch (the example's changes to starter files). A hunk that doesn't apply stops with the patch
# error: merge that hunk by hand (the example's README says what it does).
# CFG (the port's cfg.json from inspect_gguf.py): every field of the example's reviewed cfg.json except
# gguf is copied into it (the tensor table and the path stay the port's own).
# Writes KIT/KIT_SOURCE (starter/kit_source.py): the loom-skills commit the kit came from, which build.py stamps
# into every package it builds (llm_config.json "generator"), and a hash that tells a later build whether the
# kit was edited since.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=${1%-example}; ex=$here/$name-example; kit=$2
if [ "$name" = starter ]; then
  [ -e "$kit" ] && { echo "$kit exists; apply_example.sh starter only makes a new kit"; exit 2; }
  cp -r "$here/../starter" "$kit"
  python3 "$kit/kit_source.py" write "$kit" --skills-dir "$here/.."
  echo "OK: $kit = starter"; exit 0
fi
[ -d "$ex" ] || { echo "no example '$1' (have: starter $(cd "$here" && ls -d *-example | tr '\n' ' '))"; exit 2; }
[ -e "$kit" ] || cp -r "$here/../starter" "$kit"
(cd "$ex" && find . -type f ! -name README.md ! -name cfg.json ! -name starter.patch | sed 's|^\./||') | while read -r f; do
  mkdir -p "$kit/$(dirname "$f")"; cp "$ex/$f" "$kit/$f"
done
[ ! -s "$ex/starter.patch" ] || patch -p1 --forward --no-backup-if-mismatch -s -d "$kit" < "$ex/starter.patch"
python3 "$kit/kit_source.py" write "$kit" --example "$name" --skills-dir "$here/.."
if [ -n "${3:-}" ]; then
  python3 - "$ex/cfg.json" "$3" <<'PY'
import json, sys
rev, cfg = json.load(open(sys.argv[1], encoding="utf-8")), json.load(open(sys.argv[2], encoding="utf-8"))
cfg.update({k: v for k, v in rev.items() if k not in ("gguf", "tensors")})
json.dump(cfg, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
PY
  echo "OK: $kit = starter + $(basename "$ex"); reviewed fields of $ex/cfg.json merged into $3"
else
  echo "OK: $kit = starter + $(basename "$ex") (merge the reviewed $ex/cfg.json: rerun with CFG as 3rd argument)"
fi
