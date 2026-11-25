#!/usr/bin/env python3
"""Per-kernel table of a loom-serve session started with --print_device_statistics (devlab_perf.sh step profile:V).

usage: profile_stats.py SERVE_LOG REQUESTS_JSON [--tokens-per-request N | --steps N] [--md OUT.md] [--json OUT.json]
The HAL prints at shutdown, per dispatched function: count, total and average device time ("IREE HAL device statistics:").
Rows are grouped by kernel family (FAMILIES: first matching regex of the function name); ms/token = total / tokens the
scenario processed (prefill: prompt tokens of the measured requests; decode: generated tokens), % = share of the device-busy total."""
import argparse
import json
import re
import sys

UNITS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
ROW = re.compile(r"session=\d+ dispatch (\S+)\s+count=(\d+) total=([\d.]+) (ns|us|ms|s) avg=([\d.]+) (ns|us|ms|s)")
# (family, regex over the function name); first match wins; extended once the real names are known (see profile_families.json next to this script)
FAMILIES = []


def parse(text):
    rows = {}
    for m in ROW.finditer(text):
        n, c, t, u = m.group(1), int(m.group(2)), float(m.group(3)), m.group(4)
        r = rows.setdefault(n, [0, 0.0]); r[0] += c; r[1] += t * UNITS[u]
    return rows   # name -> [count, total ms]


def family(name, fams):
    for f, rx in fams:
        if re.search(rx, name):
            return f
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("requests")
    ap.add_argument("--steps", type=float, default=0, help="decode: the verify steps of the request (ms/step column)")
    ap.add_argument("--md"); ap.add_argument("--json")
    ap.add_argument("--families", help="JSON [[family, regex], ...]")
    a = ap.parse_args()
    fams = json.load(open(a.families)) if a.families else FAMILIES
    rows = parse(open(a.log, errors="replace").read())
    req = json.loads(open(a.requests).read().strip().splitlines()[-1])
    reqs = req.get("requests") or []
    includes_warmup = req.get("warmup_mode") == "perf-essay"
    if includes_warmup:
        reqs = [req["warmup"], *reqs]
    if req["scenario"] in ("pf1k", "pf2k", "pf8k", "pf32k"):
        toks = sum(r["prompt_tokens"] for r in reqs)   # the warm-up's few tokens are in the totals too (negligible vs thousands)
    else:
        toks = sum(r["completion_tokens"] for r in reqs)
    total = sum(v[1] for v in rows.values())
    by = {}
    for n, (c, t) in rows.items():
        f = by.setdefault(family(n, fams), {"count": 0, "ms": 0.0, "kernels": {}})
        f["count"] += c; f["ms"] += t; f["kernels"][n] = [c, t]
    wall = sum((r["ttft_s"] if req["scenario"] not in ("dec", "dec2k", "dec32k") else r["total_s"]) for r in reqs) * 1e3
    if req["scenario"] == "c4":   # concurrent: the wall is the longest stream
        wall = max(r["total_s"] for r in reqs) * 1e3
    res = {"scenario": req["scenario"], "includes_warmup": includes_warmup, "tokens": toks, "busy_ms": round(total, 2), "wall_ms": round(wall, 1), "steps": a.steps or None,
           "ms_per_token_busy": round(total / toks, 4) if toks else None, "families": {}, "kernels": {}}
    L = [f"### {req['scenario']}: {toks} tokens, device busy {total:.1f} ms ({total / toks:.4f} ms/token), request wall {wall:.1f} ms"
         + (f", {a.steps:.0f} verify steps ({total / a.steps:.2f} ms/step)" if a.steps else ""), "",
         "| family | dispatches | ms | ms/token | % of busy |" + (" ms/step |" if a.steps else ""), "|---|---|---|---|---|" + ("---|" if a.steps else "")]
    for f, d in sorted(by.items(), key=lambda kv: -kv[1]["ms"]):
        res["families"][f] = {"count": d["count"], "ms": round(d["ms"], 2), "ms_per_token": round(d["ms"] / toks, 5), "pct": round(100 * d["ms"] / total, 1)}
        L.append(f"| {f} | {d['count']} | {d['ms']:.2f} | {d['ms'] / toks:.5f} | {100 * d['ms'] / total:.1f} |" + (f" {d['ms'] / a.steps:.3f} |" if a.steps else ""))
    L += ["", "| kernel | family | dispatches | ms | ms/token | % | avg us |", "|---|---|---|---|---|---|---|"]
    for n, (c, t) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
        res["kernels"][n] = {"count": c, "ms": round(t, 3)}
        L.append(f"| {n} | {family(n, fams)} | {c} | {t:.2f} | {t / toks:.5f} | {100 * t / total:.1f} | {1e3 * t / c:.1f} |")
    print("\n".join(L))
    if a.md: open(a.md, "w").write("\n".join(L) + "\n")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
