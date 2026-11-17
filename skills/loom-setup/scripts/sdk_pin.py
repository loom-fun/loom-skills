#!/usr/bin/env python3
"""A package's Loom SDK pin: the hrx-loom wheel it needs.

A package carries
  "sdk": {"hrx_loom": "0.2.0", "source": "pypi", "hrx_system_sha": "...", "wheel_sha256": "..."}
in its llm_config.json (written by build/catalog.py from the kit's catalog.json, or by `stamp`). source is "pypi" (0.2.0 and 0.1.0), "release" (a
ROCm/hrx-system GitHub release, for pre-releases such as 0.2.0rc1; no expiry), or "ci" (a development artifact; expires after ~90 days, "expires").
A package WITHOUT a pin was built before pins existed and means hrx-loom 0.1.0 (PyPI), explicitly: the published
v0.1.0 cards keep running on 0.1.0 while the installer default is 0.2.0.

  sdk_pin.py show FILE          the install_loom.sh version argument: "release:VERSION", "ci:ARTIFACT_ID" or a PyPI version
                                ("0.1.0" for a file without a pin)
  sdk_pin.py matches ROOT FILE  exit 0 when ROOT/loom-install.json is the pinned SDK (the wheel's sha256, else the
                                artifact id, else the version and source)
  sdk_pin.py stamp ROOT FILE    refresh the pin: FILE["sdk"] = ROOT/loom-install.json (+ a CI artifact's expiry, when
                                gh can tell). CI artifacts expire after ~90 days: see loom-setup/SKILL.md
"""
import json
import os
import subprocess
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


DEFAULT_PIN = {"hrx_loom": "0.1.0", "source": "pypi"}   # a package without a pin


def pin(path):
    return load(path).get("sdk") or dict(DEFAULT_PIN)


def which(sdk):
    if sdk.get("source") == "ci":
        return f"ci:{sdk['artifact_id']}"
    return f"release:{sdk['hrx_loom']}" if sdk.get("source") == "release" else sdk["hrx_loom"]


def matches(root, sdk):
    try:
        have = load(os.path.join(root, "loom-install.json"))
    except (OSError, ValueError):
        return False
    if sdk.get("wheel_sha256") and have.get("wheel_sha256"):
        return sdk["wheel_sha256"] == have["wheel_sha256"]
    if sdk.get("source") == "ci":
        return have.get("artifact_id") == sdk["artifact_id"]
    return have.get("hrx_loom") == sdk["hrx_loom"] and have.get("source") == sdk.get("source", "pypi")


def stamp(root, path):
    have = load(os.path.join(root, "loom-install.json"))
    sdk = {k: have.get(k) for k in ("hrx_loom", "source", "artifact_id", "branch", "hrx_system_sha", "wheel_sha256")}
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
        print(which(pin(argv[1])))
    elif cmd == "matches" and len(argv) == 3:
        sdk = pin(argv[2])
        return 0 if matches(argv[1], sdk) else 1
    elif cmd == "stamp" and len(argv) == 3:
        stamp(argv[1], argv[2])
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
