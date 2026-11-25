#!/usr/bin/env python3
"""Per-model, per-GPU SDK delta report (loom-migrate SKILL.md step 7, version entry section 5b), generated from sdk-evidence/1.

usage: sdk_delta_report.py EVIDENCE.json [-o REPORT.md] [--band 3.0] [--strict]

EVIDENCE.json is the document `fleet/regen.py evidence STATE.json -o EVIDENCE.json` writes from a `--mode sdk-loom --programs production`
sweep: for every model and claimed device the old card's Loom rows, the rows the new SDK measured on the SAME programs, the deltas, the
regressions over the band with their explanation, and the job that measured them. Nothing here is typed by hand: re-run it after every collect.

Verdict per model and device: a row is "equal or better" when its delta is >= -band percent (the ratchet's noise band, default 3); the
pair passes when at least two thirds of its measured rows are. Every regression over the band must carry an explanation (an upstream issue
or a root cause with its same-job ABBA link); --strict exits 1 while any is missing, or while a qualified device has no measured row.
"""
import argparse
import json
import sys

ROWS = ["decode", "decode_2k", "decode_32k", "prefill_1k", "prefill_2k", "prefill_8k", "prefill_16k", "prefill_32k", "streams4"]
SHORT = {"decode": "decode", "decode_2k": "dec 2k", "decode_32k": "dec 32k", "prefill_1k": "pf 1k", "prefill_2k": "pf 2k",
         "prefill_8k": "pf 8k", "prefill_16k": "pf 16k", "prefill_32k": "pf 32k", "streams4": "4 streams"}
SUPERMAJORITY = 2 / 3


def _pct(v):
    return "-" if v is None else f"{v:+.1f}"


def _link(text, url):
    return f"[{text}]({url})" if url else str(text)


def _explanation(reg):
    ex = reg.get("explanation")
    if not ex:
        return None
    if isinstance(ex, dict):
        return _link(ex.get("text") or "explained", ex.get("url"))
    return str(ex)


def pair_summary(dev, band):
    """(measured rows, equal-or-better rows, rows worse than the band, regressions list) of one device entry"""
    delta = {k: v for k, v in (dev.get("delta_pct") or {}).items() if k in ROWS}
    worse = sorted(k for k, v in delta.items() if v < -band)
    return delta, len(delta) - len(worse), worse


def collect(doc, band=3.0):
    """rows of the report: one dict per (slug, device) with its status, deltas, verdict and the regressions to explain"""
    out = []
    for slug, mod in sorted(doc.get("models", {}).items()):
        for dev_name, dev in sorted(mod.get("devices", {}).items()):
            row = {"slug": slug, "device": dev_name, "status": dev.get("status"), "job_url": dev.get("job_url"), "reason": dev.get("reason"),
                   "kind": mod.get("kind", "text"), "delta": {}, "ok": 0, "n": 0, "verdict": None, "regressions": [], "image": dev.get("image"),
                   "acceptance": None}
            if row["kind"] == "image":
                img = dev.get("image") or {}
                row["verdict"] = "pixels identical" if img.get("pixels_match") else ("pixels differ" if "pixels_match" in img else None)
            else:
                delta, ok, worse = pair_summary(dev, band)
                row.update(delta=delta, ok=ok, n=len(delta))
                regs = {r["row"]: r for r in dev.get("regressions") or []}
                for k in worse:
                    r = regs.get(k) or {"row": k, "pct": delta[k], "confirmed_abba_pct": (dev.get("confirmed") or {}).get(k), "explanation": None}
                    row["regressions"].append(r)
                if delta:
                    row["verdict"] = "pass" if ok / len(delta) >= SUPERMAJORITY else "FAIL"
                acc = dev.get("spec_acceptance") or {}
                if acc.get("old") is not None or acc.get("new") is not None:
                    row["acceptance"] = (acc.get("old"), acc.get("new"))
            out.append(row)
    return out


def problems(rows):
    """what --strict refuses: a regression with no explanation, a qualified text pair with nothing measured, a failed verdict"""
    bad = []
    for r in rows:
        for g in r["regressions"]:
            if not _explanation(g):
                bad.append(f"{r['slug']} {r['device']} {g['row']} {g['pct']:+.1f} %: no explanation")
        if r["status"] == "qualified" and r["kind"] != "image" and not r["n"]:
            bad.append(f"{r['slug']} {r['device']}: qualified but no measured row")
        if r["verdict"] == "FAIL":
            bad.append(f"{r['slug']} {r['device']}: fewer than two thirds of the rows are equal or better ({r['ok']}/{r['n']})")
    return bad


def render(doc, band=3.0):
    rows = collect(doc, band)
    sdk = doc.get("sdk") or {}
    models = {r["slug"] for r in rows}
    ln = [f"SDK delta, hrx-loom {sdk.get('old')} -> {sdk.get('new')}: the same programs, the previous card's Loom rows vs the new SDK's rows. "
          f"{len(models)} models, {len(rows)} model-device pairs; evidence {doc.get('schema')} generated {doc.get('generated')}, "
          f"harness {((doc.get('harness') or {}).get('commit') or '-')[:8]}. Delta = new / old - 1 in percent (positive is faster); "
          f"equal or better means >= -{band:g} %.", ""]
    ln += ["### Verdict per model and GPU", "", "| Model | GPU | Status | Rows equal or better | Over the band | Job |", "|---|---|---|---:|---|---|"]
    for r in rows:
        if r["kind"] == "image":
            ln.append(f"| {r['slug']} | {r['device']} | {r['status']} | {r['verdict'] or '-'} | - | {_link('job', r['job_url']) if r['job_url'] else '-'} |")
            continue
        worse = ", ".join(f"{SHORT[g['row']]} {g['pct']:+.1f} %" for g in r["regressions"]) or "-"
        score = f"{r['ok']}/{r['n']} ({r['verdict']})" if r["n"] else (r["reason"] or r["status"] or "-")
        ln.append(f"| {r['slug']} | {r['device']} | {r['status']} | {score} | {worse} | {_link('job', r['job_url']) if r['job_url'] else '-'} |")
    ln += ["", "### Delta per row (percent)", "", "| Model | GPU | " + " | ".join(SHORT[k] for k in ROWS) + " | accepted per pass old -> new |",
           "|---|---|" + "---:|" * len(ROWS) + "---|"]
    for r in rows:
        if r["kind"] == "image" or not r["n"]:
            continue
        acc = "-" if not r["acceptance"] else " -> ".join("-" if x is None else f"{x:.3g}" for x in r["acceptance"])
        ln.append(f"| {r['slug']} | {r['device']} | " + " | ".join(_pct(r["delta"].get(k)) for k in ROWS) + f" | {acc} |")
    regs = [(r, g) for r in rows for g in r["regressions"]]
    ln += ["", "### Regressions over the band", ""]
    if not regs:
        ln.append("None.")
    else:
        ln += ["| Model | GPU | Row | Delta | Same-job ABBA | Explanation |", "|---|---|---|---:|---:|---|"]
        for r, g in regs:
            ln.append(f"| {r['slug']} | {r['device']} | {SHORT[g['row']]} | {g['pct']:+.1f} % | {_pct(g.get('confirmed_abba_pct'))} | "
                      f"{_explanation(g) or '**UNEXPLAINED**'} |")
    other = [r for r in rows if r["status"] != "qualified"]
    ln += ["", "### Pairs without a qualified 0.3.0 measurement", ""]
    if not other:
        ln.append("None.")
    else:
        ln += ["| Model | GPU | Status | Reason |", "|---|---|---|---|"]
        ln += [f"| {r['slug']} | {r['device']} | {r['status']} | {(r['reason'] or '-')[:200]} |" for r in other]
    held = doc.get("held") or {}
    if held:
        ln += ["", "### Held out of the release", "", "| Model | Reason |", "|---|---|"]
        ln += [f"| {s} | {(h.get('reason') or '-')[:200]} |" for s, h in sorted(held.items())]
    bad = problems(rows)
    ln += ["", f"Checks: {'all regressions explained, every qualified pair measured' if not bad else str(len(bad)) + ' open'}."]
    ln += [f"- {b}" for b in bad]
    return "\n".join(ln) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("evidence")
    ap.add_argument("-o", "--out")
    ap.add_argument("--band", type=float, default=3.0)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    with open(a.evidence) as f:
        doc = json.load(f)
    text = render(doc, a.band)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 1 if a.strict and problems(collect(doc, a.band)) else 0


if __name__ == "__main__":
    sys.exit(main())
