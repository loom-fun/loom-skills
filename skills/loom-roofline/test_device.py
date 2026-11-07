#!/usr/bin/env python3
"""check.sh-style assertions for device.py. No GPU required: the sysfs directory is a fixture.

usage: python3 test_device.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device  # noqa: E402

fails = []


def check(name, got, want):
    if got != want:
        fails.append(f"{name}: got {got!r}, want {want!r}")
        print(f"FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"OK   {name}")


def sysfs(entries):
    """A fixture /sys/class/drm: {card: PCI_ID}."""
    d = tempfile.mkdtemp()
    for card, pid in entries.items():
        os.makedirs(os.path.join(d, card, "device"))
        with open(os.path.join(d, card, "device", "uevent"), "w") as f:
            f.write(f"DRIVER=amdgpu\nPCI_ID={pid}\n")
    return d


known = device.devices()
check("devices.json parses and holds both mobile parts", sorted(known), ["gfx1150", "gfx1151"])
check("gfx1151 keeps the constants the skills were calibrated on",
      (known["gfx1151"]["bw"], known["gfx1151"]["flops"], known["gfx1151"]["layer_us"]),
      (216000000000, 59000000000000, 35))
check("gfx1150 is the 16-CU part: lower bw and flops than gfx1151",
      (known["gfx1150"]["bw"] < known["gfx1151"]["bw"], known["gfx1150"]["flops"] < known["gfx1151"]["flops"]),
      (True, True))

check("Strix Point PCI id resolves", device.resolve(sysfs_dir=sysfs({"card1": "1002:150E"}))[0], "gfx1150")
check("Strix Halo PCI id resolves (uppercase in sysfs)", device.resolve(sysfs_dir=sysfs({"card0": "1002:1586"}))[0], "gfx1151")
check("the AMD GPU is found even when another device is listed first",
      device.resolve(sysfs_dir=sysfs({"card0": "8086:a000", "card1": "1002:150e"}))[0], "gfx1150")
check("no AMD GPU: unresolved", device.resolve(sysfs_dir=sysfs({"card0": "8086:a000"}))[0], None)
check("empty sysfs: unresolved", device.resolve(sysfs_dir="/nonexistent")[0], None)

os.environ["LOOM_GFX"] = "gfx1151"
check("LOOM_GFX overrides a different GPU in sysfs",
      device.resolve(sysfs_dir=sysfs({"card1": "1002:150e"}))[0], "gfx1151")
os.environ["LOOM_GFX"] = "gfx9999"
check("LOOM_GFX naming an unprofiled part is not resolved to a guess",
      device.resolve(sysfs_dir=sysfs({"card1": "1002:150e"}))[0], None)
del os.environ["LOOM_GFX"]

prof = device.resolved_profile(sysfs_dir=sysfs({"card1": "1002:150e"}), warn=False)
check("resolved_profile carries the name and the resolution reason",
      (prof["target"], "sysfs" in prof["detected_via"]), ("gfx1150", True))
fallback = device.resolved_profile(sysfs_dir="/nonexistent", warn=False)
check("unresolved falls back to the calibrated reference device", fallback["target"], "gfx1151")
check("and says so, instead of pretending", "no profile" in fallback["detected_via"], True)

# The failure mode this whole file exists to prevent: the two parts share an ISA family and
# 0x150e is one hex digit from 0x1586, yet their ceilings differ by 3x.
check("the profiles are far enough apart that guessing is not harmless",
      known["gfx1151"]["bw"] / known["gfx1150"]["bw"] > 2.5, True)

print("\n" + (f"{len(fails)} FAILURES" if fails else "all device.py checks pass"))
sys.exit(1 if fails else 0)
