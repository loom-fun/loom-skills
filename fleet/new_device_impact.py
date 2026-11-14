#!/usr/bin/env python3
"""The impact report of a NEW device: every model of the fleet, all rows, on the device's target family (no kernel changed, the whole
fleet is unmeasured there). Feed it to sample_plan.py to get the first measurement wave:

    new_device_impact.py DEVICE > impact_new_device.json
    sample_plan.py impact_new_device.json --gpus DEVICE [--emit-jobs DIR]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_lib as L  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tgt = L.GPUS[L.registry.device(sys.argv[1])["device"]]
    slugs = sorted(json.load(open(os.path.join(L.ROOT, "fleet", "manifest.json")))["models"])
    print(json.dumps({"models": {s: {"kernels": [{"name": "new-device", "target": tgt, "family": "config", "rows": list(L.ROWS)}]} for s in slugs}}, indent=1))
