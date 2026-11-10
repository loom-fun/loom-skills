#!/bin/bash
# Python is already a Loom prerequisite; no Node or symlink privileges needed.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$here/scripts/install_skills.py" "$@"
elif command -v python >/dev/null 2>&1; then
  exec python "$here/scripts/install_skills.py" "$@"
else
  exec py -3 "$here/scripts/install_skills.py" "$@"
fi
