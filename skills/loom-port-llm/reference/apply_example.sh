#!/bin/bash
# usage: apply_example.sh EXAMPLE KIT [CFG]     (EXAMPLE: gemma4 | lfm2moe | flint | glimmer, with or without -example;
#                                               starter: the plain starter)
# Makes KIT the current starter plus a reference port: copies the starter to KIT (when KIT doesn't exist yet),
# then the example's own files (everything but README.md, cfg.json and starter.patch), then applies
# starter.patch (the example's changes to starter files). A hunk that doesn't apply stops with the patch
# error: merge that hunk by hand (the example's README says what it does).
# CFG (the port's cfg.json from inspect_gguf.py): every field of the example's reviewed config is copied into it
# except the port's own (gguf, tensors, name, and the keys naming one package's files). The reviewed config is the
# example's cfg.json, or, for an example whose port is published as a CATEGORY A loom-fun package
# (examples.json), that package's llm_config.json at the pinned revision (../find_reference.py example-cfg).
# Writes KIT/KIT_SOURCE (starter/kit_source.py): the loom-skills commit the kit came from, which build.py stamps
# into every package it builds (llm_config.json "generator"), and a hash that tells a later build whether the
# kit was edited since.
set -euo pipefail
# Resolve a working Python once: `python3`/`python` on PATH are the Microsoft Store stubs on some
# Windows installs, an existing Loom venv already holds the right one, and `py -3` knows where the
# real interpreter lives. LOOM_PYTHON overrides the search.
py_find() {  # $1: dir whose .venv may already provide an interpreter (may be empty)
  for p in "${1:-}/.venv/Scripts/python.exe" "${1:-}/.venv/bin/python" "${LOOM_PYTHON:-}"; do
    [ -n "$p" ] && [ -x "$p" ] && { printf '%s\n' "$p"; return 0; }
  done
  for c in python3 python py; do
    p=$(command -v "$c" 2>/dev/null || true); [ -n "$p" ] || continue
    case "$p" in *WindowsApps*) continue ;; esac
    if [ "$c" = py ]; then
      p=$("$p" -3 -c 'import sys;print(sys.executable)' 2>/dev/null || true)
      p=$(printf '%s' "$p" | tr '\\' '/')
      case "$p" in ''|*WindowsApps*) continue ;; esac
    fi
    "$p" -c 'import sys' >/dev/null 2>&1 || continue
    printf '%s\n' "$p"; return 0
  done
  return 1
}
PY=$(py_find "") || { echo "STOP: no working python3/python found (WindowsApps stubs are skipped; set LOOM_PYTHON)" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
name=${1%-example}; ex=$here/$name-example; kit=$2
if [ "$name" = starter ]; then
  [ -e "$kit" ] && { echo "$kit exists; apply_example.sh starter only makes a new kit"; exit 2; }
  cp -r "$here/../starter" "$kit"
  "$PY" "$kit/kit_source.py" write "$kit" --skills-dir "$here/.."
  echo "OK: $kit = starter"; exit 0
fi
[ -d "$ex" ] || { echo "no example '$1' (have: starter $(cd "$here" && ls -d *-example | tr '\n' ' '))"; exit 2; }
[ -e "$kit" ] || cp -r "$here/../starter" "$kit"
(cd "$ex" && find . -type f ! -name README.md ! -name cfg.json ! -name starter.patch | sed 's|^\./||') | while read -r f; do
  mkdir -p "$kit/$(dirname "$f")"; cp "$ex/$f" "$kit/$f"
done
[ ! -s "$ex/starter.patch" ] || patch -p1 --forward --no-backup-if-mismatch -s -d "$kit" < "$ex/starter.patch"
"$PY" "$kit/kit_source.py" write "$kit" --example "$name" --skills-dir "$here/.."
if [ -n "${3:-}" ]; then
  rev=$(mktemp); "$PY" "$here/../find_reference.py" example-cfg "$name" -o "$rev" || { rm -f "$rev"; exit 1; }
  "$PY" - "$rev" "$3" <<'PY'
import json, sys
rev, cfg = json.load(open(sys.argv[1], encoding="utf-8")), json.load(open(sys.argv[2], encoding="utf-8"))
cfg.update(rev)
json.dump(cfg, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
PY
  rm -f "$rev"
  echo "OK: $kit = starter + $(basename "$ex"); the example's reviewed config merged into $3"
else
  echo "OK: $kit = starter + $(basename "$ex") (merge the reviewed config: rerun with CFG as 3rd argument)"
fi
