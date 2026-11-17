#!/bin/bash
# The Loom install a package needs; prints its root on stdout (progress and errors on stderr).
# usage: root=$(ensure_sdk.sh ROOT [CONFIG.json])      CONFIG: the package's llm_config.json (or a catalog.json)
#   No "sdk" pin in CONFIG: ROOT, installed by install_loom.sh (LOOM_VERSION, else its default) when missing.
#   A pin (sdk_pin.py): ROOT when it already is that SDK, else ROOT-<artifact id | version>, installed from the pinned
#   wheel when missing, so pinned and default SDKs live side by side and one model never changes another's install.
#   LOOM_VERSION set explicitly overrides the pin (a warning says so).
# Every script that installs Loom for a package calls this (port.sh, port_e2e.sh, nightly_gates.sh, devlab_*.sh).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$1; cfg=${2:-}
# Resolve a working Python once: `python3`/`python` on PATH are the Microsoft Store stubs on some
# Windows installs, an existing Loom venv already holds the right one, and `py -3` knows where the
# real interpreter lives. LOOM_PYTHON overrides the search.
py_find() {  # $1: dir whose .venv may already provide an interpreter (may be empty)
  if [ -n "${1:-}" ]; then
    for p in "$1/.venv/Scripts/python.exe" "$1/.venv/bin/python"; do
      [ -x "$p" ] && { printf '%s\n' "$p"; return 0; }
    done
  fi
  if [ -n "${LOOM_PYTHON:-}" ] && [ -x "$LOOM_PYTHON" ]; then printf '%s\n' "$LOOM_PYTHON"; return 0; fi
  # Fall back to the interpreter's well-known install roots: a SYSTEM/GPO shell on Windows can have
  # neither `python` nor `py` on PATH (verified on toth, a scheduled task running as SYSTEM).
  for p in /c/Python3*/python.exe /c/Program\ Files/Python3*/python.exe \
           /c/Users/*/AppData/Local/Programs/Python/Python3*/python.exe; do
    [ -x "$p" ] && { printf '%s\n' "$p"; return 0; }
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
PY=$(py_find "$root") || { echo "ensure_sdk.sh: no working python3/python found (WindowsApps stubs are skipped; set LOOM_PYTHON)" >&2; exit 1; }
sdk_pin() { "$PY" "$here/sdk_pin.py" "$@"; }
# The Loom venv is complete when both entry points run.
sdk_ready() {  # $1: root
  [ -x "$1/.venv/bin/loom-serve" ] || [ -x "$1/.venv/Scripts/loom-serve.exe" ] || return 1
  [ -x "$1/.tools/bin/python" ] || [ -x "$1/.tools/Scripts/python.exe" ] || return 1
}
which=; [ -n "$cfg" ] && [ -f "$cfg" ] && which=$(sdk_pin show "$cfg" 2>/dev/null || true)
pinned=0
if [ -n "${LOOM_VERSION:-}" ]; then
  [ -n "$which" ] && echo "WARNING: LOOM_VERSION=$LOOM_VERSION overrides the package's SDK pin ($which)" >&2
  which=$LOOM_VERSION
elif [ -n "$which" ]; then
  pinned=1
  sdk_pin matches "$root" "$cfg" >/dev/null 2>&1 || root=$root-${which##*:}
fi
if ! sdk_ready "$root" || { [ $pinned = 1 ] && ! sdk_pin matches "$root" "$cfg" >/dev/null 2>&1; }; then
  echo "installing Loom ${which:-(default)} into $root" >&2
  "$here/install_loom.sh" "$root" ${which:+"$which"} >&2 || {
    [ $pinned = 1 ] && echo "The package pins SDK $which; a CI artifact expires after ~90 days. Refresh the pin (loom-setup/SKILL.md, 'SDK pins')." >&2
    exit 1; }
  [ $pinned = 0 ] || sdk_pin matches "$root" "$cfg" >/dev/null 2>&1 || { echo "installed SDK is not the pinned one ($cfg)" >&2; exit 1; }
fi
echo "$root"
