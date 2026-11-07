#!/usr/bin/env python3
"""Which AMD GPU this is, and the roofline constants that go with it.

Rooflines are only as good as the device behind them: a ceiling off by 3x turns every
target it produces into a wasted optimization campaign. The Strix Halo figures the
skills carried (216 GB/s, 59 TFLOP/s) are 3.1x and 2.5x above a Strix Point's, and the
two parts are close enough in name and ISA (gfx1150 vs gfx1151, same RDNA3.5 CU) to be
assumed alike.

Resolution order:
  1. LOOM_GFX (the override for CI, containers and anything else that knows better)
  2. the PCI device id in sysfs, which needs no ROCm install - the hrx-loom wheel
     bundles its own compiler and runtime, so a machine that only has Loom has no
     rocminfo at all
  3. rocminfo, when it is there
An unresolved device is not an error here: roofline.py imports this module, and its
tests (and any other import) must keep working on a CI box with no AMD GPU. Use
resolved_profile() and take the warning.

usage: device.py [--json|--get FIELD] [--sysfs DIR]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TARGET = "gfx1151"      # the device every skill was calibrated on
FALLBACK_NOTE = "no profile matched; using the calibrated reference device"


def devices(path=None):
    with open(path or os.path.join(HERE, "devices.json")) as f:
        return json.load(f)


def sysfs_pci_ids(sysfs_dir="/sys/class/drm"):
    try:
        cards = sorted(os.listdir(sysfs_dir))
    except OSError:
        return []
    out = []
    for card in cards:
        try:
            with open(os.path.join(sysfs_dir, card, "device", "uevent")) as f:
                for line in f:
                    if line.startswith("PCI_ID="):
                        pid = line.split("=", 1)[1].strip().lower()
                        if pid not in out:
                            out.append(pid)
        except OSError:
            pass
    return out


def rocminfo_gfx():
    try:
        p = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in (p.stdout or "").splitlines():
        for token in line.replace(",", " ").split():
            if token.startswith("gfx") and token[3:].isalnum():
                return token.lower()
    return None


def resolve(dev=None, sysfs_dir="/sys/class/drm"):
    """-> (target or None, how it was decided)"""
    known = devices()
    env = dev or os.environ.get("LOOM_GFX")
    if env:
        env = env.lower()
        return (env, "LOOM_GFX/--gfx") if env in known else (None, f"LOOM_GFX={env} is not in devices.json")
    ids = {i: t for t, v in known.items() for i in v.get("pci_ids", [])}
    for pid in sysfs_pci_ids(sysfs_dir):
        if pid in ids:
            return ids[pid], f"sysfs PCI id {pid}"
    g = rocminfo_gfx()
    if g in known:
        return g, "rocminfo"
    if g:
        return None, f"rocminfo reports {g}, which has no profile"
    return None, f"no profile for PCI ids {sysfs_pci_ids(sysfs_dir) or ['none found']}"


def resolved_profile(dev=None, sysfs_dir="/sys/class/drm", warn=True, strict=False):
    """The profile to build ceilings with, falling back to the reference device with a
    warning on stderr. Never raises: importers include the card tests and any box
    without an AMD GPU. strict=True exits instead, for callers about to act on the
    answer (downloading a build for that ISA, say), where an assumed device costs more
    than a stopped script."""
    target, how = resolve(dev, sysfs_dir)
    known = devices()
    if target is None:
        if strict:
            print(f"device.py: {how}; pass LOOM_GFX=<target> or GFX=<target>", file=sys.stderr)
            sys.exit(1)
        if warn:
            print(f"device.py: {how}; assuming {DEFAULT_TARGET}. Pass LOOM_GFX=<target> to override, "
                  f"or add the device to devices.json (calibrate_device.py measures the constants).",
                  file=sys.stderr)
        target = DEFAULT_TARGET
        how = f"{FALLBACK_NOTE} ({how})"
    prof = dict(known[target])
    prof.update(target=target, detected_via=how)
    return prof


def main():
    argv = sys.argv[1:]
    sysfs = "/sys/class/drm"
    strict = "--strict" in argv
    if strict:
        argv.remove("--strict")
    if "--sysfs" in argv:
        i = argv.index("--sysfs")
        sysfs = argv[i + 1]
        del argv[i:i + 2]
    prof = resolved_profile(sysfs_dir=sysfs, strict=strict)
    if argv and argv[0] == "--get":
        field = argv[1]
        if field not in prof:
            print(f"no field {field} (have: {' '.join(sorted(prof))})", file=sys.stderr)
            return 1
        print(prof[field])
    elif argv and argv[0] == "--json":
        print(json.dumps(prof))
    else:
        for k in sorted(prof):
            print(f"{k}={prof[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
