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
    runs[s["repo"]] = s
listed = [m["repo"] for m in json.load(open(a.models))["models"]]
want = [r for r in listed if not a.only or r in a.only.split(",")] + [r for r in runs if r not in listed]
gates = []
for s in runs.values():
    gates += [g for g in s["gates"] if g not in gates]

print("## Nightly gates on the published loom-fun cards\n")
print("| card | level | verdict | time | " + " | ".join(gates) + " |")
print("|---|---|---|---|" + "---|" * len(gates))
ok = True
for r in want:
    s = runs.get(r)
    if s is None:
        ok = False
        print(f"| {r} | | **MISSING** | | " + " | ".join("" for _ in gates) + " |")
        continue
    ok &= s["verdict"] == "PASS"
    cells = []
    for g in gates:
        x = s["gates"].get(g)
        cells.append("" if x is None else f"{'PASS' if x['pass'] else '**FAIL**'} ({x['seconds']} s)")
    v = s["verdict"] if s["verdict"] == "PASS" else f"**{s['verdict']}**"
    print(f"| {r} | {s['level']} | {v} | {s['seconds'] // 60} min | " + " | ".join(cells) + " |")
for r in want:
    p = os.path.join(runs[r]["_dir"], "perf.md") if r in runs else ""
    if p and os.path.exists(p):
        print(f"\n### {r}: performance (report only)\n")
        print(open(p).read())
print("\nLogs: the `nightly-<card>` artifacts (gates.out, gates_*.log, serve logs).")
sys.exit(0 if ok else 1)
