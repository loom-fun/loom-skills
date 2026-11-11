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
which=; [ -n "$cfg" ] && [ -f "$cfg" ] && which=$(python3 "$here/sdk_pin.py" show "$cfg")
pinned=0
if [ -n "${LOOM_VERSION:-}" ]; then
  [ -n "$which" ] && echo "WARNING: LOOM_VERSION=$LOOM_VERSION overrides the package's SDK pin ($which)" >&2
  which=$LOOM_VERSION
elif [ -n "$which" ]; then
  pinned=1
  python3 "$here/sdk_pin.py" matches "$root" "$cfg" || root=$root-${which##*:}
fi
if [ ! -x "$root/.venv/bin/loom-serve" ] || [ ! -x "$root/.tools/bin/python" ] || { [ $pinned = 1 ] && ! python3 "$here/sdk_pin.py" matches "$root" "$cfg"; }; then
  echo "installing Loom ${which:-(default)} into $root" >&2
  "$here/install_loom.sh" "$root" ${which:+"$which"} >&2 || {
    [ $pinned = 1 ] && echo "The package pins SDK $which; a CI artifact expires after ~90 days. Refresh the pin (loom-setup/SKILL.md, 'SDK pins')." >&2
    exit 1; }
  [ $pinned = 0 ] || python3 "$here/sdk_pin.py" matches "$root" "$cfg" || { echo "installed SDK is not the pinned one ($cfg)" >&2; exit 1; }
fi
echo "$root"
