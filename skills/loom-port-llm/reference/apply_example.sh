#!/bin/bash
# usage: apply_example.sh EXAMPLE KIT     (EXAMPLE: gemma4 | lfm2moe | flint | glimmer, with or without -example)
# Makes KIT the current starter plus a reference port: copies the starter to KIT (when KIT doesn't exist yet),
# then the example's own files (everything but README.md, cfg.json and starter.patch), then applies
# starter.patch (the example's changes to starter files). A hunk that doesn't apply stops with the patch
# error: merge that hunk by hand (the example's README says what it does).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ex=$here/${1%-example}-example; kit=$2
[ -d "$ex" ] || { echo "no example '$1' (have: $(cd "$here" && ls -d *-example | tr '\n' ' '))"; exit 2; }
[ -e "$kit" ] || cp -r "$here/../starter" "$kit"
(cd "$ex" && find . -type f ! -name README.md ! -name cfg.json ! -name starter.patch | sed 's|^\./||') | while read -r f; do
  mkdir -p "$kit/$(dirname "$f")"; cp "$ex/$f" "$kit/$f"
done
[ ! -s "$ex/starter.patch" ] || patch -p1 --forward --no-backup-if-mismatch -s -d "$kit" < "$ex/starter.patch"
echo "OK: $kit = starter + $(basename "$ex") (reviewed config: $ex/cfg.json)"
