#!/usr/bin/env python3
"""Generate the end-to-end report from the bench/ logs, so no number is typed by hand.

usage: report.py [BENCH_DIR=.] [--pkg PKG] [--title TITLE] [--notes notes.md]
  -> BENCH_DIR/report.html (single page, light/dark) and BENCH_DIR/report.md

Reads: audit.json (headline table; run audit.py first), gates.json + gates_runs.jsonl,
compare.json, agent_golden.json, needle.json, gates_sampling.log, gates_consistency.log,
llamacpp_pick.json, roofline.json, targets.jsonl, decisions.jsonl, ../.e2e_start and the
package's llm_config.json/model.json. --notes adds your own markdown-free text (one paragraph
per blank-line-separated block) under "Notes": what was optimized, what was skipped and why.
"""
import argparse
import html
import json
import os
import re
import time

ap = argparse.ArgumentParser()
ap.add_argument("bench", nargs="?", default=".")
ap.add_argument("--pkg")
ap.add_argument("--title")
ap.add_argument("--notes")
a = ap.parse_args()
B = lambda f: os.path.join(a.bench, f)


def load(f, default=None):
    try:
        return json.load(open(B(f)))
    except (OSError, ValueError):
        return default


def jl(f):
    try:
        return [json.loads(l) for l in open(B(f)) if l.strip()]
    except OSError:
        return []


def tail(f, n=6):
    try:
        return [l.rstrip() for l in open(B(f), encoding="utf-8", errors="replace")][-n:]
    except OSError:
        return []


audit = load("audit.json", {})
gates = load("gates.json", {})
runs = jl("gates_runs.jsonl")
pick = load("llamacpp_pick.json", {})
roof = load("roofline.json", {})
cmp_rows = load("compare.json", [])
agent = load("agent_golden.json", {})
needle = load("needle.json", {})
decisions = jl("decisions.jsonl")
targets = jl("targets.jsonl")
cfg, preset = {}, {}
if a.pkg:
    cfg = json.load(open(os.path.join(a.pkg, "llm_config.json")))
    card = json.load(open(os.path.join(a.pkg, "model.json")))
    preset = next(iter(card["presets"].values())).get("execution", {})
model = cfg.get("model_id") or (roof.get("gguf") or "model").split("/")[-1]
title = a.title or f"{model} on Loom vs llama.cpp ROCm"

# timeline
start = None
for p in (B("../.e2e_start"), B(".e2e_start")):
    if os.path.exists(p):
        start = float(open(p).read().split()[0])
        break
now = time.time()
elapsed_min = round((now - start) / 60) if start else None

# gates summary
cmp_ok = [r for r in cmp_rows if not r.get("error")]
gaps = [r["divergence_gap"] for r in cmp_ok if isinstance(r.get("divergence_gap"), float)]
turns = agent.get("turns", [])
nres = needle.get("results", [])
samp = [l for l in tail("gates_sampling.log", 10) if l.startswith(("OK", "FAIL"))]
cons = [l for l in tail("gates_consistency.log", 10) if l.startswith(("PASS", "FAIL", "prompt tokens"))]

lines = []
lines.append(f"# {title}\n")
bl = []
ORDER = ["decode_tps", "prefill_2000", "prefill_1000", "c4_steady", "decode_tps_32k", "prefill_32000"]
for r in sorted(audit.get("headline", []), key=lambda r: ORDER.index(r["metric"]) if r["metric"] in ORDER else 99):
    if r.get("margin") is not None:
        bl.append(f"{r['metric']} {r['loom']:.0f} vs {r['llama_cpp']:.0f} tok/s ({r['margin']:+.0%})")
lines.append("**Bottom line:** " + ("; ".join(bl[:4]) or "no measurements") + ". "
             + f"Gates: {gates.get('verdict', 'not run')} (level {gates.get('level', '-')})."
             + (f" End-to-end time: {elapsed_min} min." if elapsed_min else "") + "\n")
lines.append("## Performance (final build vs the fastest llama.cpp ROCm build)\n")
lines.append("| metric | llama.cpp | Loom | margin | % of ceiling | target | met |\n|---|---|---|---|---|---|---|")
for r in audit.get("headline", []):
    lc = f"{r['llama_cpp']:.1f}" if r.get("llama_cpp") else "-"
    mg = f"{r['margin']:+.0%}" if r.get("margin") is not None else "-"
    lines.append(f"| {r['metric']} | {lc} | {r['loom']:.1f} | {mg} | {r['pct_of_ceiling']:.0%} | {r['target']:.1f} | {'yes' if r['met'] else 'no'} |")
lines.append("")
lines.append("## Correctness gates\n")
for n, g in gates.get("gates", {}).items():
    lines.append(f"- **{n}**: {'PASS' if g['pass'] else 'FAIL'} ({g.get('seconds', '?')} s)")
if cmp_ok:
    lines.append(f"- golden compare: {len(cmp_ok)} prompts up to {max(r.get('prompt_tokens') or 0 for r in cmp_ok):,} tokens, "
                 f"{sum(r['full_match'] for r in cmp_ok)} exact matches, every divergence a near-tie: "
                 f"{'yes' if not any(r['bad'] for r in cmp_rows) else 'NO'} (max log-prob gap {max(gaps) if gaps else 0:.2f})")
if turns:
    lines.append(f"- agent tool loop: {len(turns)} turns; rendering exact {sum(t['render_ok'] for t in turns)}/{len(turns)}, "
                 f"history reuse {sum(t['reuse_ok'] for t in turns)}/{len(turns)}, valid tool calls {sum(t['calls_ok'] for t in turns)}/{len(turns)}")
if nres:
    lines.append(f"- needle: sizes {sorted({r['n'] for r in nres})} of a {needle.get('context', 0):,}-token context; "
                 f"all probes served: {'yes' if all(r.get('status') == 200 for r in nres) else 'NO'}; "
                 f"over-capacity prompt -> HTTP {needle.get('over_capacity', {}).get('status')}")
for l in samp:
    lines.append(f"- sampling: {l}")
for l in cons:
    lines.append(f"- consistency: {l}")
lines.append("")
lines.append("## Setup\n")
top = (pick.get("ranked") or [{}])[0]
lines.append(f"- GGUF: `{roof.get('gguf', cfg.get('gguf', '?'))}`")
lines.append(f"- llama.cpp ROCm baseline (fastest of {len(pick.get('ranked', []))} builds on the machine): `{top.get('dir', '?')}`")
if preset:
    lines.append(f"- Loom preset: context {preset.get('context_capacity')}, rows {preset.get('rows')}, "
                 f"prefill chunk {preset.get('prefill_capacity')}")
lines.append("- Loom numbers are client-side (HTTP streaming); llama.cpp numbers are its server-side timings, "
             "which slightly favours llama.cpp.")
lines.append("")
if decisions:
    lines.append("## Optimization decisions\n")
    for d in decisions:
        rej = "; ".join(f"{r.get('option')}: {r.get('measured', '')} ({r.get('why', '')})" for r in d.get("rejected", []))
        lines.append(f"- {d.get('decision')}: **{d.get('chosen')}**" + (f" — rejected: {rej}" if rej else ""))
    lines.append("")
ctrl = [c for c in audit.get("checks", []) if c["result"] != "ok"]
lines.append("## Audit\n")
lines.append("AUDIT " + ("PASS" if not [c for c in ctrl if c["result"] == "FAIL"] else "FAIL"))
for c in ctrl:
    lines.append(f"- [{c['result']}] {c['check']}: {c['detail']}")
lines.append("")
if runs:
    lines.append("## Timeline\n")
    for r in runs:
        lines.append(f"- {r['ts']} gates level={r['level']} {r['verdict']} in {r['seconds']} s")
    if start:
        lines.append(f"- end to end: {elapsed_min} min since {time.strftime('%H:%M', time.localtime(start))}")
    lines.append("")
if a.notes and os.path.exists(a.notes):
    lines.append("## Notes\n")
    lines.append(open(a.notes, encoding="utf-8").read().strip() + "\n")
md = "\n".join(lines)
open(B("report.md"), "w", encoding="utf-8").write(md)


def md_to_html(md):
    out, in_list, in_table = [], False, False
    inline = lambda s: re.sub(r"`([^`]+)`", r"<code>\1</code>", re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", html.escape(s)))
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            if not in_table:
                out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</tbody></table>"); in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</tbody></table>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(model.split('/')[-1])} Loom report</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--accent:#0f6b5c;--card:#ffffff}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151614;--fg:#ececea;--muted:#a3a39d;--line:#2d2e2b;--accent:#5cc2ad;--card:#1c1d1b}}}}
:root[data-theme="dark"]{{--bg:#151614;--fg:#ececea;--muted:#a3a39d;--line:#2d2e2b;--accent:#5cc2ad;--card:#1c1d1b}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:920px;margin:0 auto}} h1{{font-size:26px;margin:0 0 12px}} h2{{font-size:18px;margin:28px 0 8px;color:var(--accent)}}
table{{border-collapse:collapse;width:100%;background:var(--card);font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;white-space:nowrap}} th{{color:var(--muted);font-weight:600}}
code{{font-size:13px;word-break:break-all}} li{{margin:3px 0}} p{{margin:8px 0}}
</style></head><body><main>
{md_to_html(md)}
</main></body></html>"""
open(B("report.html"), "w", encoding="utf-8").write(page)
print("wrote", B("report.html"), "and", B("report.md"))
