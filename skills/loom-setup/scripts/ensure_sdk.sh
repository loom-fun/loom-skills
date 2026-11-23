#!/bin/bash
# The Loom install a package needs; prints its root on stdout (progress and errors on stderr).
# usage: root=$(ensure_sdk.sh ROOT [CONFIG.json])      CONFIG: the package's llm_config.json (or a catalog.json)
#   No CONFIG: ROOT, installed by install_loom.sh (LOOM_VERSION, else its default) when missing.
#   CONFIG without an "sdk" pin means hrx-loom 0.1.0 (sdk_pin.py): the packages published before pins existed.
#   Otherwise the pin (sdk_pin.py): ROOT when it already is that SDK, else ROOT-<artifact id | version>, installed from the
#   pinned wheel when missing, so SDKs of different packages live side by side and one model never changes another's install.
#   LOOM_VERSION set explicitly overrides the pin (a warning says so).
# Every script that installs Loom for a package calls this (port.sh, port_e2e.sh, nightly_gates.sh, devlab_*.sh).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$1; cfg=${2:-}
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) bin=Scripts; suffix=.exe;; *) bin=bin; suffix=;; esac
which=; if [ -n "$cfg" ] && [ -f "$cfg" ]; then which=$(python3 "$here/sdk_pin.py" show "$cfg"); fi
pinned=0
if [ -n "${LOOM_VERSION:-}" ]; then
  [ -n "$which" ] && echo "WARNING: LOOM_VERSION=$LOOM_VERSION overrides the package's SDK pin ($which)" >&2
  which=$LOOM_VERSION
elif [ -n "$which" ]; then
  pinned=1
  python3 "$here/sdk_pin.py" matches "$root" "$cfg" || root=$root-${which##*:}
fi
if [ ! -x "$root/.venv/$bin/loom-serve$suffix" ] || [ ! -x "$root/.tools/$bin/python$suffix" ] || { [ $pinned = 1 ] && ! python3 "$here/sdk_pin.py" matches "$root" "$cfg"; }; then
  echo "installing Loom ${which:-(default)} into $root" >&2
  "$here/install_loom.sh" "$root" ${which:+"$which"} >&2 || {
    if [ $pinned = 1 ]; then
      echo "Installing SDK $which failed." >&2
      case "$which" in ci:*) echo "A CI artifact expires after ~90 days. Refresh the pin (loom-setup/SKILL.md, 'SDK pins')." >&2;; esac
    fi
    exit 1; }
  [ $pinned = 0 ] || python3 "$here/sdk_pin.py" matches "$root" "$cfg" || { echo "installed SDK is not the pinned one ($cfg)" >&2; exit 1; }
fi
echo "$root"
