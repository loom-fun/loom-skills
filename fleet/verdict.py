#!/usr/bin/env python3
"""Per-row verdicts of an A/B compare JSON (strix_ab_compare.py --json) against the methodology bands (plan Appendix A sections 4-5).

usage: verdict.py COMPARE.json [--model SLUG] [--rows decode,prefill_8k] [--methodology methodology.json] [--json OUT]

Row verdicts (rows are judged one at a time, no geomean trading):
  pass       |ratio - 1| <= the row's band
  improved   ratio > 1 + band (recorded, not trusted: it reaches a card only after confirming runs, promote_check.py)
  regressed  ratio < 1 - band
  reject     ratio < the row's T1 reject threshold (blocks a PR); `extend`: within 0.02 of it, rerun with 6+6 sessions
  flag       cannot decide: a bimodal model whose modes cannot be compared like with like (a mode with < 2 sessions on a side, a
             different number of modes per side, or an out-of-band ratio between unimodal sides that a mode shift would explain);
             a change in the frequency of the slow mode is flagged, never a silent pass
  missing    the row has no value on a side
Model verdict = the worst row, also gate failures (gate_fail) and a changed greedy text (text_diff, near-tie review needed).
The sampled verdict of a stratum is the worst member verdict (worst()).
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_lib as L  # noqa: E402

# worst first
SEVERITY = ["gate_fail", "reject", "regressed", "text_diff", "flag", "missing", "pass", "improved"]
MODE_GAP = 0.06      # two modes of a bimodal model are ~10 % apart (plan section 5); sessions further apart than 6 % split


def worst(verdicts):
    vs = [v for v in verdicts if v]
    return min(vs, key=SEVERITY.index) if vs else "pass"


def sessions(entry, side):
    s = entry.get(f"{side}_sessions") or entry.get(f"{side}_runs")
    return [x for x in (s or []) if x]


def modes(values):
    """Split sessions into at most two modes (gap larger than MODE_GAP of the median); [low, high] lists."""
    v = sorted(values)
    if len(v) < 2:
        return [v]
    med = statistics.median(v)
    gaps = [(v[i + 1] - v[i], i) for i in range(len(v) - 1)]
    g, i = max(gaps)
    return [v[:i + 1], v[i + 1:]] if g > MODE_GAP * med else [v]


def band_verdict(ratio, band, reject):
    if ratio is None:
        return "missing", False
    near = abs(ratio - reject) <= 0.02
    if ratio < reject:
        return "reject", near
    if ratio < 1 - band:
        return "regressed", near
    if ratio > 1 + band:
        return "improved", False
    return "pass", near


def row_verdict(entry, row, bimodal, meth):
    band, reject = meth["bands"][row]
    ratio = entry.get("ratio")
    old_s, new_s = sessions(entry, "old"), sessions(entry, "new")
    if ratio is None:
        return {"verdict": "missing", "ratio": None, "band": band}
    if not bimodal:
        v, near = band_verdict(ratio, band, reject)
        return {"verdict": v, "ratio": ratio, "band": band, "reject": reject, "extend": near}
    mo, mn = modes(old_s), modes(new_s)
    if len(old_s) >= 2 and len(new_s) >= 2 and len(mo) == len(mn) == 1:
        # both sides unimodal: an out-of-band ratio ~ the mode gap could be a mode shift, not a change
        v, near = band_verdict(ratio, band, reject)
        if v in ("regressed", "reject", "improved") and abs(ratio - 1) <= 0.13:
            return {"verdict": "flag", "ratio": ratio, "band": band, "why": "unimodal sides, ratio within a mode gap: a mode shift cannot be excluded"}
        return {"verdict": v, "ratio": ratio, "band": band, "reject": reject, "extend": near}
    if len(mo) != len(mn) or any(len(m) < 2 for m in mo + mn):
        return {"verdict": "flag", "ratio": ratio, "band": band, "why": f"modes per side {len(mo)} vs {len(mn)}: slow-mode frequency changed or a mode has < 2 sessions"}
    rs = [round(statistics.median(b) / statistics.median(a), 4) for a, b in zip(mo, mn)]
    res = [band_verdict(r, band, reject) for r in rs]
    return {"verdict": worst([r[0] for r in res]), "ratio": min(rs), "mode_ratios": rs, "band": band, "reject": reject,
            "extend": any(r[1] for r in res)}


def model_verdict(compare, slug="", rows=None, meth=None):
    meth = meth or L.load_methodology()
    bimodal = L.is_bimodal(slug, meth)
    out = {}
    for row in L.ROWS:
        if rows is not None and row not in rows:
            continue
        e = (compare.get("metrics") or {}).get(row)
        if e is None:
            continue
        out[row] = row_verdict(e, row, bimodal, meth)
    gates = [k for k, s in (compare.get("gate_status") or {}).items() if s != "PASS"]
    text = (compare.get("text") or {}).get("status")
    vs = [r["verdict"] for r in out.values()]
    if gates:
        vs.append("gate_fail")
    if text == "DIFF":
        vs.append("text_diff")
    return {"model": slug, "verdict": worst(vs), "rows": out, "gate_failures": gates, "text": text, "bimodal": bimodal}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("compare")
    ap.add_argument("--model", default="")
    ap.add_argument("--rows")
    ap.add_argument("--methodology")
    ap.add_argument("--json")
    a = ap.parse_args()
    meth = L.load_methodology(a.methodology)
    c = json.load(open(a.compare))
    res = model_verdict(c, a.model, a.rows.split(",") if a.rows else None, meth)
    print(f"{a.model or os.path.basename(a.compare)}: {res['verdict']}  (bands: {meth['source']})")
    for r, v in res["rows"].items():
        print(f"  {r:12s} {v['verdict']:10s} ratio {v['ratio']}  band +-{v['band']}" + ("  EXTEND to 6+6 sessions" if v.get("extend") else "")
              + (f"  [{v['why']}]" if v.get("why") else ""))
    if res["gate_failures"]:
        print("  gate failures:", ", ".join(res["gate_failures"]))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
