#!/usr/bin/env python3
"""Drafted decode rows of a same-job ABBA, normalised per verification pass: tok/s, tokens per pass and ms per pass of every session of two
variants (devlab_perf.sh perf_<V>.jsonl), and the new/old ratio of each. A drafted row's tok/s = tokens per pass / time per pass; perf.py's
prompts start with a nonce, so the generated text, and with it the drafted-token acceptance, differs between sessions. ms per pass is the
cost the SDK and the programs control: a row whose tok/s is out of its band while ms per pass is within it moved with the acceptance.

usage: pass_time.py JOB_DIR [--old fun] [--new staging] [--json OUT.json]
Passes per row: decode tokens - accepted drafted tokens (perf.py draft_accepted, hrx-loom 0.3.0), else tokens / (1 + acceptance x depth)
(hrx-loom 0.2.0: acceptance = accepted / proposed, proposed = passes x the configured depth from model_<V>.json; exact to about one pass)."""
import argparse
import json
import os
import re
import statistics
import sys

ROWS = ("decode_tps", "decode_tps_2k", "decode_tps_32k")


def depth(job, v):
    try:
        d = re.findall(r'"draft_depth":\s*(\d+)', open(os.path.join(job, f"model_{v}.json")).read())
        return int(d[0]) if d else None
    except OSError:
        return None


def sessions(job, v):
    """{row: [{tps, tokens, passes, ms_per_pass}]} of perf_<v>.jsonl"""
    dp, out = depth(job, v), {}
    for line in open(os.path.join(job, f"perf_{v}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if "label" not in r:
            continue
        for k in ROWS:
            if not r.get(k):
                continue
            n = ((r.get("completion_tokens") or {}).get(k) or [None])[-1]
            passes = None
            if n and k in (r.get("draft_accepted") or {}):
                passes = n - r["draft_accepted"][k]
            elif n and k in (r.get("acceptance") or {}) and dp:
                passes = n / (1 + r["acceptance"][k] * dp)
            ms = 1000 * n / r[k] / passes if passes else None
            out.setdefault(k, []).append({"tps": r[k], "tokens": n, "passes": passes and round(passes, 1),
                                          "tokens_per_pass": round(n / passes, 3) if passes else None, "ms_per_pass": ms and round(ms, 3)})
    return out


def compare(job, old="fun", new="staging"):
    o, n = sessions(job, old), sessions(job, new)
    res = {}
    for k in ROWS:
        if not o.get(k) or not n.get(k):
            continue
        med = lambda s, f: statistics.median([x[f] for x in s if x[f]]) if any(x[f] for x in s) else None
        r = {"old": o[k], "new": n[k], "tps_ratio": round(med(n[k], "tps") / med(o[k], "tps"), 4)}
        if med(o[k], "ms_per_pass") and med(n[k], "ms_per_pass"):
            r["ms_per_pass_ratio"] = round(med(n[k], "ms_per_pass") / med(o[k], "ms_per_pass"), 4)
            r["tokens_per_pass_ratio"] = round(med(n[k], "tokens_per_pass") / med(o[k], "tokens_per_pass"), 4)
        res[k] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("--old", default="fun"); ap.add_argument("--new", default="staging"); ap.add_argument("--json")
    a = ap.parse_args()
    res = compare(a.job, a.old, a.new)
    for k, r in res.items():
        print(k)
        for side in ("old", "new"):
            s = r[side]
            print(f"  {side}: tok/s {[x['tps'] for x in s]} | tokens/pass {[x['tokens_per_pass'] for x in s]} | ms/pass {[x['ms_per_pass'] for x in s]}")
        print(f"  new/old: tok/s {r['tps_ratio'] - 1:+.1%}" + (f", ms/pass {r['ms_per_pass_ratio'] - 1:+.1%}, tokens/pass {r['tokens_per_pass_ratio'] - 1:+.1%}"
                                                          if "ms_per_pass_ratio" in r else " (undrafted or no counters)"))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
