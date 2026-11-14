#!/usr/bin/env python3
"""Which of the fleet fits a device's memory, from the fit table's sizing columns (weights+scratch `w`, KV bytes per position `kvpp`,
native `ctx`): usage: fit_device.py DEVICE [--fit .github/devlab/r9700_fit.json] [--json]. Verdicts as r9700_fit.py's.

APPROXIMATION: the table was sized for the R9700 (its `<first>-r9700` prefill chunk where a package has one, so `w` can be a little
larger than the default preset needs). Good enough to plan a first wave; .github/scripts/r9700_fit.py rebuilds the real table and still
names radeon-r9700 (a finding of the device-row dry run). A device without `memory_bytes` fits everything (Strix Halo: package's own).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-kb"))
import registry  # noqa: E402

HEADROOM = 2_000_000_000   # card_lib.SERVE_HEADROOM


def verdicts(device, rows, headroom=HEADROOM):
    pool = registry.device(device).get("memory_bytes")
    out = []
    for r in rows:
        fixed = r["w"]
        kvpp = r.get("kvpp") or (r["kv"] / r["ctx"])   # some rows carry only kv at ctx
        kv = kvpp * r["ctx"]
        if pool is None:
            v, served = "fits (native ctx)", r["ctx"]
        elif fixed + headroom >= pool:
            v, served = "does not fit", 0
        elif fixed + kv + headroom <= pool:
            v, served = "fits (native ctx)", r["ctx"]
        else:
            v, served = "fits (reduced ctx)", int((pool - fixed - headroom) // kvpp)
        out.append({"slug": r["slug"], "w": fixed, "need": fixed + kv + headroom, "ctx": r["ctx"], "served_context": served, "verdict": v})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("device")
    ap.add_argument("--fit", default=os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = verdicts(a.device, json.load(open(a.fit)))
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for r in res:
            print(f"{r['slug']:55s} w {r['w'] / 1e9:5.1f} GB need {r['need'] / 1e9:6.1f} GB  {r['verdict']}" +
                  (f" (served ctx {r['served_context']})" if r["verdict"].startswith("fits (red") else ""))
        n = {}
        for r in res:
            n[r["verdict"]] = n.get(r["verdict"], 0) + 1
        print(a.device, "memory", registry.device(a.device).get("memory_bytes"), n)
