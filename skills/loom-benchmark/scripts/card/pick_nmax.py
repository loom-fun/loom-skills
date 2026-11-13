#!/usr/bin/env python3
"""DIAGNOSTIC: tabulate a llama.cpp draft n-max sweep (not a card input: reference/methodology.md rule 2 uses the documented n-max or the default).

  pick_nmax.py JOB_OUT_DIR [--candidates 1 2 3 4 0]

JOB_OUT_DIR is a devlab-perf job's out/ with perf_llamacpp-<backend>.np1_n<N>.jsonl files (N = 0: llama.cpp's default n-max), one per
candidate and backend (steps llama@rocm.np1:3:-:N and llama@rocm.np1:3:long:N). Per backend it picks the n-max with the best
geometric mean over the 1-stream rows (decode, decode_2k, decode_32k, prefill_*), prints the table and writes nmax.json:
  {"rocm": {"best": N, "geomean": {N: G}, "rows": {N: {metric: mean}}}, "vulkan": {...}}
One value for all rows of a backend; no per-row choice.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_info import metric_samples, perf_rows  # noqa: E402

ROWS = ("decode", "decode_2k", "decode_32k", "prefill_1k", "prefill_2k", "prefill_8k", "prefill_16k", "prefill_32k")


def means(path):
    return {k: sum(v) / len(v) for k, v in metric_samples(perf_rows(path)).items() if k in ROWS and v}


def pick(out_dir, candidates=(1, 2, 3, 4, 0)):
    res = {}
    for bk in ("rocm", "vulkan"):
        rows = {}
        for n in candidates:
            f = os.path.join(out_dir, f"perf_llamacpp-{bk}.np1_n{n}.jsonl")
            if os.path.exists(f):
                rows[n] = means(f)
        if not rows:
            continue
        common = [r for r in ROWS if all(r in m for m in rows.values())]   # the same rows for every candidate
        geo = {n: math.exp(sum(math.log(m[r]) for r in common) / len(common)) for n, m in rows.items()} if common else {}
        if geo:
            res[bk] = {"best": max(geo, key=geo.get), "geomean": {str(n): round(g, 2) for n, g in geo.items()},
                       "rows": {str(n): {k: round(v, 1) for k, v in m.items()} for n, m in rows.items()}, "over": common}
    return res


if __name__ == "__main__":
    d = sys.argv[1]
    cand = [int(x) for x in sys.argv[sys.argv.index("--candidates") + 1:]] if "--candidates" in sys.argv else (1, 2, 3, 4, 0)
    r = pick(d, cand)
    json.dump(r, open(os.path.join(d, "nmax.json"), "w"), indent=1)
    for bk, v in r.items():
        print(bk, "best n-max", v["best"] or "default", "geomean", v["geomean"])
