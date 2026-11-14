#!/usr/bin/env python3
"""Replay strix_ab_compare.py results (the --json files of past A/B jobs) under the per-row bands of methodology.json and table what changes
against the flat band they were decided with.

usage: replay_bands.py FILE.json [FILE.json ...] [--flat 0.03] [--md]

Per file: every metric's status flat (|ratio - 1| <= flat) vs per-row (its row's band), the file's verdict flat vs per-row (pass when no
metric is out, else rerun), and, for a confirming re-run (first_run in the file, the first run found among the inputs by job name), the
out_of_margin verdict (a metric out in both runs in the same direction). Gates and text are not replayed (unchanged by bands).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import registry  # noqa: E402


def statuses(res, band_of):
    """{metric: (status, direction)} of one compare result; acceptance rows as 'acceptance:<row>' (their band is the acceptance row's)."""
    out = {}
    for k, e in res["metrics"].items():
        r = e.get("ratio")
        out[k] = ("MISSING", 0) if r is None else (("PASS" if abs(r - 1) <= band_of(k) else "OUT"), 1 if r > 1 else -1)
    for k, e in (res.get("acceptance") or {}).items():
        r = e.get("ratio")
        ok = (r is not None and abs(r - 1) <= band_of("spec_acceptance")) or (e.get("old") == 0 and e.get("new") == 0)
        out[f"acceptance:{k}"] = ("PASS" if ok else ("MISSING" if r is None else "OUT"), 0 if r is None else (1 if r > 1 else -1))
    return out


def verdict(st, prev=None):
    outs = {k for k, (s, _) in st.items() if s != "PASS"}
    if prev is None:
        return "pass" if not outs else "rerun"
    margin = {k for k in outs if prev.get(k, ("PASS", 0))[0] != "PASS" and prev[k][1] == st[k][1]}
    return "pass" if not margin else "out_of_margin"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--flat", type=float, default=0.03)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)
    runs, allruns = {}, []
    for f in a.files:
        d = json.load(open(f))
        allruns.append((f, d))
        runs.setdefault(d["job"], (f, d))   # the first file of a job: what a confirming re-run's first_run names
    flat, per = (lambda row: a.flat), registry.band
    rows, changed = [], []
    for f, d in sorted(allruns, key=lambda fd: fd[0]):
        sf, sp = statuses(d, flat), statuses(d, per)
        prev = d.get("first_run", {}).get("job")
        pf = pp = None
        if prev in runs and runs[prev][1] is not d:
            pf, pp = statuses(runs[prev][1], flat), statuses(runs[prev][1], per)
        vf, vp = verdict(sf, pf), verdict(sp, pp)
        flips = {k: (sf[k][0], sp[k][0]) for k in sf if sf[k][0] != sp[k][0]}
        rows.append((os.path.basename(f), vf, vp, flips))
        if flips or vf != vp:
            changed.append(rows[-1])
    n = len(rows)
    flips = [x for r in rows for x in r[3].values()]
    print(f"{n} results; verdict changes {sum(1 for r in rows if r[1] != r[2])}; files with a metric status change {len(changed)}; "
          f"metric statuses PASS->OUT {flips.count(('PASS', 'OUT'))}, OUT->PASS {flips.count(('OUT', 'PASS'))}")
    if a.md:
        print("\n| result | flat verdict | per-row verdict | metric status changes (flat -> per-row) |\n|---|---|---|---|")
    for name, vf, vp, flips in changed:
        txt = ", ".join(f"{k} {x}->{y}" for k, (x, y) in flips.items()) or "-"
        print((f"| {name} | {vf} | {vp} | {txt} |" if a.md else f"{name:60} {vf:14} -> {vp:14} {txt}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
