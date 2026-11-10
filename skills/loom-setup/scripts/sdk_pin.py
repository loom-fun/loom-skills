#!/usr/bin/env python3
"""A package's Loom SDK pin: the hrx-loom wheel it needs, when the default (PyPI 0.1.0) will not do.

A package whose programs need a newer SDK (a catalog package using parameters.gather needs 0.2.0.dev0) carries
  "sdk": {"hrx_loom": "0.2.0.dev0", "source": "ci", "artifact_id": 13628412621, "hrx_system_sha": "...",
          "wheel_sha256": "...", "expires": "2026-11-22"}
in its llm_config.json (written by build/catalog.py from the kit's catalog.json). ensure_sdk.sh installs that SDK.

  sdk_pin.py show FILE          the install_loom.sh version argument ("ci:ARTIFACT_ID" or a PyPI version); nothing
                                for a file without a pin
  sdk_pin.py matches ROOT FILE  exit 0 when ROOT/loom-install.json is the pinned SDK (the wheel's sha256, else the
                                artifact id or the version)
  sdk_pin.py stamp ROOT FILE    refresh the pin: FILE["sdk"] = ROOT/loom-install.json (+ the artifact's expiry, when
                                gh can tell). CI artifacts expire after ~90 days: see loom-setup/SKILL.md
"""
import json
import os
import subprocess
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pin(path):
    return load(path).get("sdk") or None


def which(sdk):
    return f"ci:{sdk['artifact_id']}" if sdk.get("source") == "ci" else sdk["hrx_loom"]


def matches(root, sdk):
    try:
        have = load(os.path.join(root, "loom-install.json"))
    except (OSError, ValueError):
        return False
    if sdk.get("wheel_sha256") and have.get("wheel_sha256"):
        return sdk["wheel_sha256"] == have["wheel_sha256"]
    if sdk.get("source") == "ci":
        return have.get("artifact_id") == sdk["artifact_id"]
    return have.get("hrx_loom") == sdk["hrx_loom"] and have.get("source") == "pypi"


def stamp(root, path):
    have = load(os.path.join(root, "loom-install.json"))
    sdk = {k: have.get(k) for k in ("hrx_loom", "source", "artifact_id", "hrx_system_sha", "wheel_sha256")}
    if have.get("artifact_id"):
        r = subprocess.run(["gh", "api", f"repos/ROCm/hrx-system/actions/artifacts/{have['artifact_id']}", "-q", ".expires_at"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            sdk["expires"] = r.stdout.strip()[:10]
    cfg = load(path)
    cfg["sdk"] = {k: v for k, v in sdk.items() if v is not None}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
        f.write("\n")
    print(f"{path}: sdk {json.dumps(cfg['sdk'])}")


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "show" and len(argv) == 2:
        sdk = pin(argv[1])
        print(which(sdk) if sdk else "")
    elif cmd == "matches" and len(argv) == 3:
        sdk = pin(argv[2])
        return 0 if sdk and matches(argv[1], sdk) else 1
    elif cmd == "stamp" and len(argv) == 3:
        stamp(argv[1], argv[2])
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
