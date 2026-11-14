#!/usr/bin/env python3
"""The promotion ratchet (plan Appendix A section 4): may the HEAD build of (model, GPU) replace the published package?

usage: promote_check.py --model SLUG --gpu strix-halo|radeon-r9700 COMPARE.json [COMPARE2.json ...]
                       [--baselines fleet/baselines.json] [--methodology M.json] [--near-tie] [--json OUT]

COMPARE*.json: strix_ab_compare.py outputs of the same A/B (old = the published package, new = the HEAD build), oldest first;
a second file is the confirming re-run. Decision, per methodology.json's ratchet rule:
  promote           every row >= published - band, the gates pass, the greedy text matches (or is a near-tie: --near-tie), and no
                    improvement is waiting for confirmation
  improved-pending  nothing regressed, but a row improved by more than its band in fewer than `confirming_runs` (2) runs: the
                    improvement is recorded, not trusted; it reaches the card after a second run on the same side
  rerun             a row is out of band low in a single run (or cannot be decided: a bimodal flag): a re-run decides
  hold              a row is below its T1 reject threshold, or below its band in the last two runs in the same direction, or a gate
                    fails, or the text differs: the published package keeps serving and a regression ticket is filed against HEAD
Rows are judged one at a time; a gain never pays for a loss. `baseline_drift` lists rows whose new value is below the stored
baseline value by more than twice the band (different runner, so informational): the slow cumulative drift check.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_lib as L  # noqa: E402
import verdict as V  # noqa: E402

LOW = ("regressed", "reject")


def row_decision(per_run, confirming):
    """per_run: this row's verdict dicts, oldest first. -> (state, reason)"""
    last = per_run[-1]
    v = last["verdict"]
    if len(per_run) >= 2:
        a, b = per_run[-2]["verdict"], v
        if a in LOW and b in LOW:
            return "hold", f"below band in both runs ({per_run[-2]['ratio']}, {last['ratio']})"
        if a == b == "improved":
            return "ok", "improved, confirmed in 2 runs"
        if "flag" in (a, b):
            return "rerun", "undecidable (bimodal mode check) in a run"
        return "ok", "within band or noise (the runs disagree)" if (a != b) else "within band"
    if v == "reject":
        return "hold", f"below the T1 reject threshold ({last['ratio']} < {last.get('reject')})"
    if v == "regressed":
        return "rerun", f"below its band in one run ({last['ratio']}): a re-run decides"
    if v in ("flag", "missing"):
        return "rerun", last.get("why") or "no value"
    if v == "improved" and confirming > 1:
        return "pending", f"improved ({last['ratio']}) in one run: needs {confirming} runs"
    return "ok", "within band"


def check(slug, gpu, compares, baselines=None, meth=None, near_tie=False, rows_only=None):
    meth = meth or L.load_methodology()
    confirming = int(meth["ratchet"].get("confirming_runs", 2))
    runs = [V.model_verdict(c, slug, rows_only, meth) for c in compares]
    rows, reasons = {}, []
    for r in L.ROWS:
        per = [x["rows"][r] for x in runs if r in x["rows"]]
        if not per:
            continue
        st, why = row_decision(per, confirming)
        rows[r] = {"ratios": [p["ratio"] for p in per], "verdicts": [p["verdict"] for p in per], "state": st, "reason": why}
    last = runs[-1]
    gate_fail = last["gate_failures"]
    text = last["text"]
    states = [x["state"] for x in rows.values()]
    if gate_fail:
        decision, reasons = "hold", ["gate failed: " + ", ".join(gate_fail)]
    elif text == "DIFF" and not near_tie:
        decision, reasons = "hold", ["greedy text differs (review as a near-tie with --near-tie)"]
    elif "hold" in states:
        decision, reasons = "hold", [f"{r}: {x['reason']}" for r, x in rows.items() if x["state"] == "hold"]
    elif "rerun" in states:
        decision, reasons = "rerun", [f"{r}: {x['reason']}" for r, x in rows.items() if x["state"] == "rerun"]
    elif "pending" in states:
        decision, reasons = "improved-pending", [f"{r}: {x['reason']}" for r, x in rows.items() if x["state"] == "pending"]
    else:
        decision, reasons = "promote", []
    drift = []
    b = ((baselines or {}).get("models", {}).get(slug) or {}).get(gpu) or {}
    for r, e in ((compares[-1].get("metrics") or {}).items()):
        if r in b and e.get("new") and b[r].get("value"):
            ratio = e["new"] / b[r]["value"]
            if ratio < 1 - 2 * meth["bands"][r][0]:
                drift.append({"row": r, "new": e["new"], "baseline": b[r]["value"], "ratio": round(ratio, 3), "run": b[r].get("run")})
    return {"model": slug, "gpu": gpu, "decision": decision, "reasons": reasons, "rows": rows, "baseline_drift": drift,
            "bands": meth["source"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("compares", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", required=True, choices=sorted(L.GPUS))
    ap.add_argument("--baselines", default=os.path.join(L.ROOT, "fleet", "baselines.json"))
    ap.add_argument("--methodology")
    ap.add_argument("--near-tie", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    base = json.load(open(a.baselines)) if os.path.exists(a.baselines) else None
    res = check(a.model, a.gpu, [json.load(open(p)) for p in a.compares], base, L.load_methodology(a.methodology), a.near_tie)
    print(f"{a.model} on {a.gpu}: {res['decision']}")
    for r in res["reasons"]:
        print("  ", r)
    for d in res["baseline_drift"]:
        print(f"   baseline drift {d['row']}: {d['new']} vs published {d['baseline']} ({d['ratio']})")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
