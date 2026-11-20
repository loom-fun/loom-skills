#!/usr/bin/env python3
"""Per-step power and GPU clock of a devlab_perf.sh job run with POWER_LOG=1 (a local machine's evidence of its power mode).

usage: power_steps.py OUT_DIR [--busy-w 40] [--band 120 125]

Reads OUT_DIR/power.tsv (epoch, power1_input in microwatts, freq1_input in Hz; one sample every 0.2 s) and OUT_DIR/step_times.tsv
(step, start, end). A sample is busy when the package draws at least --busy-w watts (server loads, idle waits between requests and
the client's tokenization fall below it). Per step and per engine (loom, llamacpp-rocm, llamacpp-vulkan, gufo): the number of busy
samples, their median W and median GHz, and the 95th percentile W. A step whose busy median lies outside --band is flagged
(prints FLAG); memory-bound decode can draw less than the package limit, so a flag is a fact to state, not an error.
Writes OUT_DIR/power_steps.json; make_info.py perf --power-evidence reads it.
"""
import argparse
import json
import os
import statistics


def engine(step):
    kind = step.split(":")[0]
    if kind.startswith("gufo"):
        return "gufo"
    if kind.startswith("llama"):
        return "llamacpp-vulkan" if "@vulkan" in kind else "llamacpp-rocm"
    if kind in ("perf", "c4", "profile", "ab"):
        return "loom"
    return None


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def stats(samples, busy_w):
    busy = [(w, f) for w, f in samples if w >= busy_w]
    if not busy:
        return {"samples": len(samples), "busy": 0}
    ws, fs = [w for w, _ in busy], [f for _, f in busy]
    return {"samples": len(samples), "busy": len(busy), "median_w": round(statistics.median(ws), 1), "p95_w": round(pct(ws, 0.95), 1),
            "median_ghz": round(statistics.median(fs), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--busy-w", type=float, default=40.0)
    ap.add_argument("--band", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    a = ap.parse_args()
    samples = []
    for line in open(os.path.join(a.out, "power.tsv")):
        p = line.split()
        if len(p) == 3:
            samples.append((float(p[0]), int(p[1]) / 1e6, int(p[2]) / 1e9))
    steps = [line.rstrip("\n").split("\t") for line in open(os.path.join(a.out, "step_times.tsv")) if line.strip()]
    rows, per_engine = [], {}
    for i, (name, t0, t1) in enumerate(steps):
        t0, t1 = float(t0), float(t1)
        mine = [(w, f) for t, w, f in samples if t0 <= t < t1]
        r = {"step": name, "index": i, "engine": engine(name), **stats(mine, a.busy_w)}
        if a.band and r.get("busy") and not a.band[0] <= r["median_w"] <= a.band[1]:
            r["flag"] = f"median {r['median_w']} W outside {a.band[0]:g}-{a.band[1]:g} W"
        rows.append(r)
        if r["engine"]:
            per_engine.setdefault(r["engine"], []).extend(mine)
    eng = {k: stats(v, a.busy_w) for k, v in per_engine.items()}
    doc = {"source": "amdgpu hwmon power1_input / freq1_input, 0.2 s samples", "busy_w": a.busy_w, "band": a.band,
           "steps": rows, "engines": eng}
    json.dump(doc, open(os.path.join(a.out, "power_steps.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['step']:40s} {r.get('median_w', '-'):>6} W {r.get('median_ghz', '-'):>6} GHz  busy {r.get('busy', 0)}/{r['samples']}"
              + (f"  FLAG {r['flag']}" if r.get("flag") else ""))
    for k, v in eng.items():
        print(f"ENGINE {k:16s} {v.get('median_w', '-')} W {v.get('median_ghz', '-')} GHz (busy {v.get('busy', 0)})")


if __name__ == "__main__":
    main()
