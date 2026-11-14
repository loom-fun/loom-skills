#!/usr/bin/env python3
"""The device of a DevLab runner key, for devlab_perf.sh (and a check of the workflows' runs-on tables).

usage: runner_device.py KEY            print "DEVICE POOL_BYTES PRESET_SUFFIX" (POOL_BYTES 0 / PRESET_SUFFIX - when the device has none); exit 1 if no
                                       device has runner KEY (stderr names the known keys)
       runner_device.py --labels       print {runner key: runs-on labels} as JSON (devices with a runner)
Everything comes from skills/loom-kb/device/<family>/<device>/device.json (runner.key, runner.labels, memory_bytes, preset_suffix).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "skills", "loom-kb"))
import registry  # noqa: E402

if sys.argv[1:] == ["--labels"]:
    print(json.dumps(registry.runner_labels()))
    sys.exit(0)
key = sys.argv[1]
for name, d in registry.devices().items():
    if d["runner"]["key"] == key and d["runner"]["labels"]:
        print(name, d.get("memory_bytes") or 0, d.get("preset_suffix") or "-")
        sys.exit(0)
sys.exit(f"unknown runner {key} ({' | '.join(registry.runner_labels())})")
