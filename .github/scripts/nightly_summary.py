"""Markdown table of the nightly gates: one row per card, one column per gate (nightly-gates.yml's summary job).

usage: python3 nightly_summary.py ARTIFACTS_DIR [--models .github/nightly-models.json] >> "$GITHUB_STEP_SUMMARY"
Reads every ARTIFACTS_DIR/**/summary.json (nightly_gates.sh) and perf.md; a card in the model list with no
summary is shown as MISSING (its job failed before the gates, e.g. fetch or serve). Exit 1 unless every
listed card has a PASS verdict.
"""
import argparse
import glob
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("artifacts")
ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nightly-models.json"))
ap.add_argument("--only", default="", help="comma-separated repos this run covered (default: every listed card)")
a = ap.parse_args()

runs = {}
for f in glob.glob(os.path.join(a.artifacts, "**", "summary.json"), recursive=True):
    s = json.load(open(f))
    s["_dir"] = os.path.dirname(f)
    runs[(s["repo"], s["level"])] = s
listed = [m["repo"] for m in json.load(open(a.models))["models"]]
want = [r for r in listed if not a.only or r in a.only.split(",")] + sorted({r for r, _ in runs} - set(listed))
order = {"dev": 0, "fast": 1, "quick": 2, "full": 3}
gates = []
for s in runs.values():
    gates += [g for g in s["gates"] if g not in gates]

print("## Nightly gates on the published loom-fun cards\n")
print("| card | level | verdict | job time | gates time | " + " | ".join(gates) + " |")
print("|---|---|---|---|---|" + "---|" * len(gates))
ok = True
for r in want:
    mine = sorted((k for k in runs if k[0] == r), key=lambda k: order.get(k[1], 9))
    if not mine:
        ok = False
        print(f"| {r} | | **MISSING** | | | " + " | ".join("" for _ in gates) + " |")
        continue
    for k in mine:
        s = runs[k]
        ok &= s["verdict"] == "PASS"
        cells = []
        for g in gates:
            x = s["gates"].get(g)
            cells.append("" if x is None else f"{'PASS' if x['pass'] else '**FAIL**'} ({x['seconds']} s)")
        v = s["verdict"] if s["verdict"] == "PASS" else f"**{s['verdict']}**"
        print(f"| {r} | {s['level']} | {v} | {s['seconds'] / 60:.1f} min | {s.get('gates_seconds') or 0} s | " + " | ".join(cells) + " |")
for k in sorted(runs, key=lambda k: (want.index(k[0]) if k[0] in want else 99, order.get(k[1], 9))):
    p = os.path.join(runs[k]["_dir"], "perf.md")
    if os.path.exists(p):
        print(f"\n### {k[0]} ({k[1]}): performance (report only)\n")
        print(open(p).read())
print("\nLogs: the `nightly-<card>` artifacts (gates.out, gates_*.log, serve logs).")
sys.exit(0 if ok else 1)
