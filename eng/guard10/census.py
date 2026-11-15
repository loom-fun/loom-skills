#!/usr/bin/env python3
"""Metric census (measure 1): per model x GPU, every card row of the currently published loom-fun-staging card against its reference,
counted better / within band / worse. Re-runnable; reads the live HF repos (cached by huggingface_hub), no GPU.

  Strix Halo reference: the official loom-fun card (loom-fun/<slug> main).
  R9700 reference:      the pre-campaign loom-fun-staging revision pinned in fleet/manifest.json (phase-2 values).
  Current:              loom-fun-staging/<slug> main (or --rev-dir for a local snapshot is not supported; use --staging-rev SLUG=REV).
Bands: methodology.json per row (sample_lib.load_methodology): better > 1+band, worse < 1-band, else within. Rows: decode, prefill, streams4
(spec_acceptance is informational and not counted). A row the reference has but the current card lacks counts as worse (missing); a row
only the current card has, or neither, is not counted. A GPU the reference has no results for is not counted.

usage: census.py [--out eng/guard10/census.md] [--json census.json] [--models slug,slug]
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "fleet"))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import card_lib  # noqa: E402
import sample_lib as L  # noqa: E402

GPU_LIST = ["strix-halo", "radeon-r9700"]


def fetch(repo, rev=None):
    from huggingface_hub import hf_hub_download
    try:
        return card_lib.load_info(hf_hub_download(repo, "info.json", revision=rev, force_download=rev is None))
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"warn: {repo}@{rev}: {type(e).__name__}\n")
        return None


def card_rows(info, gpu):
    """{row: (value, programs_sha256 of its session)} of one GPU's card; {} when the GPU has no results."""
    if not info:
        return {}
    dev = (info.get("devices") or {}).get(gpu)
    if not card_lib.has_results(dev):
        return {}
    v = card_lib.gpu_view(info, gpu)
    sessions = (v.get("performance") or {}).get("sessions", [])
    return {k: (m.get("loom"), (sessions[i].get("programs_sha256") if i < len(sessions) else None))
            for k, (m, i) in card_lib.card_metrics(v).items() if k in L.ROWS and m.get("loom")}


def state_of(info, gpu):
    return ((info or {}).get("devices") or {}).get(gpu, {}).get("state")


def classify(ref, cur, band):
    if cur is None:
        return "worse", None
    r = cur / ref
    return ("better" if r > 1 + band else "worse" if r < 1 - band else "within"), round(r, 4)


def census(slugs, manifest, meth):
    res = {g: {"better": 0, "within": 0, "worse": 0, "models": {}, "worse_rows": []} for g in GPU_LIST}
    for s in slugs:
        cur = fetch(f"loom-fun-staging/{s}")
        refs = {"strix-halo": fetch(f"loom-fun/{s}"),
                "radeon-r9700": fetch(f"loom-fun-staging/{s}", manifest[s]["staging"]["revision"])}
        for g in GPU_LIST:
            ref_rows, cur_rows = card_rows(refs[g], g), card_rows(cur, g)
            if not ref_rows:
                res[g]["models"][s] = None
                continue
            c = {"better": 0, "within": 0, "worse": 0}
            for row, (rv, rsha) in ref_rows.items():
                band = meth["bands"][row][0]
                cv, csha = cur_rows.get(row, (None, None))
                k, ratio = classify(rv, cv, band)
                c[k] += 1
                if k == "worse":
                    if cv is None:
                        dev = ((cur or {}).get("devices") or {}).get(g) or {}
                        why = ("current card has no results for this GPU (" + (state_of(cur, g) or "no performance session; not yet measured on staging") + ")"
                               if not cur_rows else "row not on the current card (not measured, or does not fit this GPU)")
                    elif rsha and csha and rsha == csha:
                        why = "same programs as the reference: runner/session noise"
                    elif rsha and csha:
                        why = "programs differ from the reference (a refactor/campaign change or a retune); cause not recorded in the card"
                    else:
                        why = "session without programs_sha256; cause not recorded"
                    res[g]["worse_rows"].append({"model": s, "row": row, "ref": rv, "cur": cv, "ratio": ratio, "band": band, "reason": why})
            for k, n in c.items():
                res[g][k] += n
            res[g]["models"][s] = c
    return res


def render(res, meth, manifest, n):
    out = [f"# Metric census (measure 1)", "",
           f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by `eng/guard10/census.py` over {n} models; re-run it for fresh numbers.",
           "Current = loom-fun-staging main. Reference: Strix = official loom-fun card; R9700 = the pre-campaign staging revision in fleet/manifest.json.",
           "Rows: decode x3, prefill x5, streams4 (acceptance not counted). Bands per row from methodology.json "
           "(" + ", ".join(f"{r} {b[0]:.1%}" for r, b in meth["bands"].items()) + "). better > 1+band, worse < 1-band.", ""]
    out += ["## Totals per GPU", "", "| GPU | rows | better | within | worse | not worse |", "|---|---|---|---|---|---|"]
    for g, r in res.items():
        t = r["better"] + r["within"] + r["worse"]
        out.append(f"| {g} | {t} | {r['better']} | {r['within']} | {r['worse']} | {(t - r['worse']) / t:.1%} |" if t else f"| {g} | 0 | | | | |")
    for g, r in res.items():
        out += ["", f"## {g}: per model (better / within / worse)", ""]
        for s, c in sorted(r["models"].items()):
            out.append(f"- {s}: " + (f"{c['better']} / {c['within']} / {c['worse']}" if c else "no reference results (not counted)"))
        out += ["", f"### {g}: worse rows ({len(r['worse_rows'])})", ""]
        if r["worse_rows"]:
            out += ["| model | row | reference | current | ratio | band | reason |", "|---|---|---|---|---|---|---|"]
            for w in sorted(r["worse_rows"], key=lambda w: (w["ratio"] if w["ratio"] is not None else 0)):
                out.append(f"| {w['model']} | {w['row']} | {w['ref']} | {w['cur'] if w['cur'] is not None else '-'} | {w['ratio'] if w['ratio'] is not None else '-'} | "
                           f"{w['band']:.1%} | {w['reason']} |")
        else:
            out.append("none")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "census.md"))
    ap.add_argument("--json")
    ap.add_argument("--models")
    a = ap.parse_args()
    manifest = json.load(open(os.path.join(ROOT, "fleet", "manifest.json")))["models"]
    slugs = a.models.split(",") if a.models else sorted(manifest)
    meth = L.load_methodology()
    res = census(slugs, manifest, meth)
    open(a.out, "w").write(render(res, meth, manifest, len(slugs)))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print("census done: " + "; ".join(f"{g} {r['better']} better / {r['within']} within / {r['worse']} worse" for g, r in res.items()))


if __name__ == "__main__":
    main()
