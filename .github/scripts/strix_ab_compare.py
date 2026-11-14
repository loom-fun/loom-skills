#!/usr/bin/env python3
"""Compare the two packages of a Strix A/B job (strix_ab_jobs.py): new (staging) / old (fun) per card metric.

usage: strix_ab_compare.py JOB_OUT_DIR [--band 0.03] [--fun-info loom-fun info.json] [--json out.json]

Per metric (card_lib.METRIC_KEYS) and per drafted metric's acceptance, decided on the MEDIAN of the serve sessions per side, a
session's value being the median of its samples (user rule, 2026-11-14 10:38): old, new, new/old, PASS inside +-band, else OUT (a re-run decides). Plus: greedy text (text_fun.json vs
text_staging.json, per prompt), func / ktest / gates of the staging package (summary.json), and new vs the published loom-fun card value.
Exit 0 always; the verdict is in the JSON ("verdict": pass | rerun | gate_fail; with --prev FIRST.json: pass | out_of_margin | gate_fail).
A missing metric (a step failed) counts as out of band: re-run.
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import card_lib  # noqa: E402
from make_info import loom_sessions, read_rows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--band", type=float, default=0.03)
ap.add_argument("--fun-info", help="the published loom-fun info.json (card values)")
ap.add_argument("--json")
ap.add_argument("--prev", help="the first run's --json (this run is the confirming re-run): verdict out_of_margin when a metric is "
                "outside the band in both runs in the same direction, else pass")
a = ap.parse_args()
o = a.out


def side(v, nd):
    """{metric: [per-session median]} and {metric: [per-session median acceptance]} of one package (user rule 2026-11-14 10:38:
    each metric is decided on the median of the sessions; a session's value is the median of its samples). The same function
    builds the cards' values (make_info.py perf)."""
    p, c4 = os.path.join(o, f"perf_{v}.jsonl"), os.path.join(o, f"c4_{nd}.txt")
    out, _, acc = loom_sessions(read_rows(p) if os.path.exists(p) else [], c4 if os.path.exists(c4) else None)
    return out, acc


old, oacc = side("fun", "funnd")
new, nacc = side("staging", "stagingnd")
card = {}
if a.fun_info:
    info = card_lib.load_info(a.fun_info)
    card = {k: m.get("loom") for k, (m, _) in card_lib.card_metrics(card_lib.gpu_view(info, "strix-halo")).items()}

metrics = {}
for k in card_lib.METRIC_KEYS:
    if k == "spec_acceptance":
        continue
    med = lambda x: round(statistics.median(x), 1) if x else None
    ov, nv = med(old.get(k)), med(new.get(k))
    if ov is None and nv is None:
        continue
    r = round(nv / ov, 4) if ov and nv else None
    st = "PASS" if r is not None and abs(r - 1) <= a.band else ("MISSING" if r is None else "OUT")
    e = {"old": ov, "new": nv, "ratio": r, "status": st, "old_sessions": old.get(k), "new_sessions": new.get(k)}
    if card.get(k):
        e["card"] = card[k]
        e["new_vs_card"] = round(nv / card[k], 4) if nv else None
    metrics[k] = e
acc = {}
for k in sorted(set(oacc) | set(nacc)):
    ov = round(statistics.median(oacc[k]), 4) if oacc.get(k) else None
    nv = round(statistics.median(nacc[k]), 4) if nacc.get(k) else None
    r = round(nv / ov, 4) if ov and nv else None
    acc[k] = {"old": ov, "new": nv, "ratio": r,
              "status": "PASS" if (r is not None and abs(r - 1) <= a.band) or (ov == 0 and nv == 0) else ("MISSING" if r is None else "OUT")}

# greedy text: the Loom texts of both packages, prompt by prompt
text = {"status": "MISSING"}
tf, ts = os.path.join(o, "text_fun.json"), os.path.join(o, "text_staging.json")
if os.path.exists(tf) and os.path.exists(ts):
    A, B = json.load(open(tf)), json.load(open(ts))
    ta, tb = A.get("texts") or [], B.get("texts") or []
    same = [x == y for x, y in zip(ta, tb)]
    def lead(x, y):
        x, y = json.dumps(x), json.dumps(y)
        n = 0
        while n < min(len(x), len(y)) and x[n] == y[n]:
            n += 1
        return n
    text = {"status": "MATCH" if same and all(same) and len(ta) == len(tb) else "DIFF", "prompts": len(ta),
            "identical": sum(same), "common_prefix_chars": [lead(x, y) for x, y in zip(ta, tb)]}

summ = json.load(open(os.path.join(o, "summary.json")))["results"] if os.path.exists(os.path.join(o, "summary.json")) else {}
gates = {k: v for k, v in summ.items() if k.split(":")[0] in ("func", "ktest", "gates")}
def gate_ok(k, v):
    t = k.split(":")[0]
    if t == "func":
        return v.strip().lower().startswith("pass")
    if t == "ktest":
        return v.startswith("PASS") or v.startswith("SKIP")
    return v.startswith("GATES PASS")
gate_status = {k: ("PASS" if gate_ok(k, v) else "FAIL") for k, v in gates.items()}

outs = [k for k, e in metrics.items() if e["status"] != "PASS"] + [f"acceptance:{k}" for k, e in acc.items() if e["status"] != "PASS"]
fails = [k for k, s in gate_status.items() if s != "PASS"]
worst = min(((abs(e["ratio"] - 1), k, e["ratio"]) for k, e in metrics.items() if e["ratio"]), default=None, key=lambda x: -x[0])
res = {"job": summ and json.load(open(os.path.join(o, "summary.json")))["job"], "runner": open(os.path.join(o, "runner.txt")).read().strip() if os.path.exists(os.path.join(o, "runner.txt")) else None,
       "band": a.band, "metrics": metrics, "acceptance": acc, "text": text, "gates": gates, "gate_status": gate_status,
       "out_of_band": outs, "gate_failures": fails,
       "worst_metric": worst[1] if worst else None, "worst_ratio": worst[2] if worst else None,
       "verdict": "gate_fail" if fails else ("rerun" if outs else "pass")}
if a.prev:   # the stop rule: re-run once; still outside the band in the same direction = OUT OF MARGIN
    p = json.load(open(a.prev))
    def sides(x):
        out = {}
        for k, e in x["metrics"].items():
            if e["status"] == "OUT":
                out[k] = 1 if e["ratio"] > 1 else -1
        for k, e in x["acceptance"].items():
            if e["status"] == "OUT":
                out[f"acceptance:{k}"] = 1 if e["ratio"] > 1 else -1
        return out
    sp, sn = sides(p), sides(res)
    res["first_run"] = {"job": p["job"], "runner": p["runner"], "out_of_band": p["out_of_band"]}
    res["out_of_margin"] = sorted(k for k in sn if sp.get(k) == sn[k])
    res["verdict"] = "gate_fail" if fails else ("out_of_margin" if res["out_of_margin"] else "pass")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
print(f"{res['job']} on {res['runner']}: verdict {res['verdict']}  worst {res['worst_metric']} {res['worst_ratio']}")
print(f"{'metric':14} {'old':>9} {'new':>9} {'new/old':>8} {'status':7} {'card':>9} {'new/card':>8}")
for k, e in metrics.items():
    print(f"{k:14} {e['old'] or '-':>9} {e['new'] or '-':>9} {e['ratio'] or '-':>8} {e['status']:7} {e.get('card') or '-':>9} {e.get('new_vs_card') or '-':>8}")
for k, e in acc.items():
    print(f"{'acc ' + k:14} {e['old']!s:>9} {e['new']!s:>9} {e['ratio'] or '-':>8} {e['status']}")
print("text:", text)
for k, v in gates.items():
    print(f"{gate_status[k]:5} {k}: {v[:160]}")
