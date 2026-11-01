#!/usr/bin/env python3
"""KPIs of one end-to-end trial of the loom skills, and the comparison of two trials.

usage (tools venv python, for gguf): kpi.py summarize TRIAL_DIR [--label NAME] [--skill-rev REV] [--log kpis.jsonl]
       kpi.py compare BASE.json NEW.json        (or two labels from --log: kpi.py compare --log kpis.jsonl BASE NEW)

TRIAL_DIR is what the trial prompt (trial_prompt.md) creates: phases.log (`<epoch> <phase>` lines and
`# friction` comments) and work/ with the bench/ directory that qualify.sh / finalize.sh fill.
summarize prints one JSON object (and appends it to --log):
  wall_min        install -> done, minutes (the headline KPI)
  budget_min      this model's roofline budget (loom-roofline/e2e_budget.py); within_budget = the KPI target
  phase_min       minutes per phase (port, port_*, qualify, optimize, finalize, report, ...)
  manual_min      time in hand-work phases (port_* other than port itself, *_debug, *_chat)
  friction        number of `# ...` friction notes the agent logged (each one is a skill gap)
  gates           final gates.json: verdict, level, per-gate pass; audit verdict
  quality         compare max_gap / bad prompts, agent turns ok/total, consistency, needle sizes served
  perf            audit.json headline: per metric Loom, llama.cpp, margin (measurement quality noted)
compare prints a KPI table with REGRESSION / improved / same per KPI and exits 1 on any regression:
  wall_min +10 % or more, any gate PASS -> FAIL, compare max_gap up by > 0.3, a gate that ran before
  missing now, a perf margin down by more than the noise band (10 points by default; contended GPU
  numbers are flagged, not judged).
"""
import argparse
import glob
import json
import os
import sys
import time


def load(p, default=None):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return default


def phases(trial):
    rows, friction = [], 0
    for line in open(os.path.join(trial, "phases.log"), errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            friction += 1
            continue
        t, _, name = line.partition(" ")
        try:
            rows.append((float(t), name.split()[0] if name else "?"))
        except ValueError:
            friction += line.count("#") > 0
    rows.sort()
    dur = {}
    for (t0, n), (t1, _) in zip(rows, rows[1:]):
        dur[n] = round(dur.get(n, 0) + (t1 - t0) / 60, 1)
    done = [t for t, n in rows if n == "done"]
    # Unfinished: up to the last logged phase (a stalled agent must not look like a 20-hour run).
    wall = round(((done[-1] if done else rows[-1][0]) - rows[0][0]) / 60, 1) if rows else None
    manual = round(sum(v for k, v in dur.items() if (k.startswith("port_") or k.endswith(("_debug", "_chat", "_merge")))
                       and k != "port"), 1)
    return {"wall_min": wall, "finished": bool(done), "phase_min": dur, "manual_min": manual, "friction": friction}


def bench_dir(trial):
    cands = [os.path.dirname(p) for p in glob.glob(os.path.join(trial, "work", "**", "gates.json"), recursive=True)]
    cands = [c for c in cands if "llama_cache" not in c]
    return max(cands, key=lambda d: os.path.getmtime(os.path.join(d, "gates.json"))) if cands else None


def summarize(trial, label, rev):
    out = {"label": label or os.path.basename(os.path.abspath(trial)), "skill_rev": rev, "trial": os.path.abspath(trial),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.update(phases(trial))
    b = bench_dir(trial)
    out["bench"] = b
    # The wall-clock target is this model's roofline budget (loom-roofline/e2e_budget.py), not a constant.
    try:
        pkgs = [os.path.dirname(p) for p in glob.glob(os.path.join(trial, "work", "**", "llm_config.json"), recursive=True)
                if os.path.exists(os.path.join(os.path.dirname(p), "model.json"))]
        if pkgs:
            # the final build serves the native context; A/B variants are often built at 8k
            ctx_of = lambda d: next(iter(json.load(open(os.path.join(d, "model.json")))["presets"].values()))["execution"]["context_capacity"]
            pkg = max(pkgs, key=lambda d: (ctx_of(d), os.path.getmtime(d)))
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "loom-roofline"))
            import e2e_budget as E
            cfg = json.load(open(os.path.join(pkg, "llm_config.json")))
            card = json.load(open(os.path.join(pkg, "model.json")))
            ctx = next(iter(card["presets"].values()))["execution"]["context_capacity"]
            gguf = os.path.realpath(os.path.join(pkg, "parameter", os.listdir(os.path.join(pkg, "parameter"))[0]))
            dec, pre = E.speeds(gguf, cfg, [ctx])
            out["budget_min"] = round(E.FIXED_MIN + E.OPT_MIN + sum(E.workload(dec, pre, ctx).values()) / 60)
            out["within_budget"] = out["wall_min"] is not None and out["wall_min"] <= out["budget_min"]
    except Exception as e:  # the budget is informative; never fail the summary on it
        out["budget_error"] = str(e)[:200]
    if not b:
        return out
    g = load(os.path.join(b, "gates.json"), {})
    audit = load(os.path.join(b, "audit.json"), {})
    out["gates"] = {"verdict": g.get("verdict"), "level": g.get("level"), "seconds": g.get("seconds"),
                    "pass": {k: v["pass"] for k, v in g.get("gates", {}).items()},
                    "audit": None if not audit else ("FAIL" if any(c["result"] == "FAIL" for c in audit.get("checks", [])) else "PASS")}
    q = {}
    cmp_rows = load(os.path.join(b, "compare.json"), [])
    if cmp_rows:
        gaps = [r["divergence_gap"] for r in cmp_rows if isinstance(r.get("divergence_gap"), float)]
        q.update(compare_prompts=len(cmp_rows), compare_bad=sum(bool(r.get("bad")) for r in cmp_rows),
                 compare_max_gap=round(max(gaps), 3) if gaps else 0.0,
                 compare_max_tokens=max((r.get("prompt_tokens") or 0) for r in cmp_rows))
    ag = load(os.path.join(b, "agent_golden.json"), {})
    if ag.get("turns"):
        t = ag["turns"]
        q["agent_ok"] = f"{sum(x['render_ok'] and x['reuse_ok'] and x['calls_ok'] for x in t)}/{len(t)}"
    needles = (load(os.path.join(b, "needle.json"), {}) or {}).get("results", []) + \
              (load(os.path.join(b, "needle_full.json"), {}) or {}).get("results", [])
    if needles:
        q["needle_max_served"] = max(r["n"] for r in needles if r.get("status") == 200) if any(r.get("status") == 200 for r in needles) else 0
    out["quality"] = q
    out["perf"] = {r["metric"]: {"loom": r["loom"], "llama": r.get("llama_cpp"), "margin": r.get("margin")}
                   for r in audit.get("headline", [])}
    out["perf_contended"] = any(c["check"].startswith(("plausible: llama.cpp", "baseline: fastest"))
                                for c in audit.get("checks", []) if c["result"] != "ok")
    return out


def compare(a, b, noise=0.10, judge_perf=False):
    rows, reg = [], 0

    def row(k, va, vb, verdict):
        nonlocal reg
        reg += verdict == "REGRESSION"
        rows.append((k, va, vb, verdict))
    wa, wb = a.get("wall_min"), b.get("wall_min")
    if b.get("budget_min"):
        row("wall vs model budget", f"{wb}/{b['budget_min']}", "", "within" if b.get("within_budget") else "OVER BUDGET")
    if wa and wb:
        row("wall_min", wa, wb, "REGRESSION" if wb > wa * 1.10 else "improved" if wb < wa * 0.95 else "same")
    row("manual_min", a.get("manual_min"), b.get("manual_min"), "info")
    row("friction notes", a.get("friction"), b.get("friction"), "info")
    alias = {"golden1": "golden", "golden2": "golden", "needle_full": "needle"}
    norm = lambda d: {alias.get(k, k): (v and norm_acc.get(alias.get(k, k), True)) for k, v in d.items()} if d else {}
    norm_acc = {}
    ga, gb = norm((a.get("gates") or {}).get("pass", {})), norm((b.get("gates") or {}).get("pass", {}))
    for k in sorted(set(ga) | set(gb)):
        va, vb = ga.get(k), gb.get(k)
        v = "REGRESSION" if (va and vb is False) or (va is not None and vb is None) else \
            "improved" if (va is False and vb) else "same"
        row(f"gate {k}", va, vb, v)
    qa, qb = a.get("quality") or {}, b.get("quality") or {}
    if "compare_max_gap" in qa and "compare_max_gap" in qb:
        d = qb["compare_max_gap"] - qa["compare_max_gap"]
        row("compare max_gap", qa["compare_max_gap"], qb["compare_max_gap"], "REGRESSION" if d > 0.3 else "same")
    # In-trial perf is measured while other jobs may share the GPU: informational unless both runs
    # come from eval.sh on a quiet GPU (--perf). Perf regressions are decided on eval.sh numbers.
    pa, pb = a.get("perf") or {}, b.get("perf") or {}
    contended = not judge_perf or a.get("perf_contended") or b.get("perf_contended")
    for m in sorted(set(pa) & set(pb)):
        ma, mb = pa[m].get("margin"), pb[m].get("margin")
        if ma is None or mb is None:
            continue
        v = "info (judge with eval.sh)" if contended else ("REGRESSION" if mb < ma - noise else "improved" if mb > ma + noise else "same")
        row(f"perf {m} margin", f"{ma:+.0%}", f"{mb:+.0%}", v)
    w = max(len(r[0]) for r in rows)
    print(f"{'KPI':{w}}  {'base':>12}  {'new':>12}  verdict   ({a.get('label')} -> {b.get('label')})")
    for k, va, vb, v in rows:
        print(f"{k:{w}}  {str(va):>12}  {str(vb):>12}  {v}")
    print(f"\n{reg} regression(s)")
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summarize", "compare"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--label")
    ap.add_argument("--skill-rev", default="")
    ap.add_argument("--log")
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--perf", action="store_true", help="judge perf margins (only for eval.sh numbers on a quiet GPU)")
    a = ap.parse_args()
    if a.cmd == "summarize":
        s = summarize(a.args[0], a.label, a.skill_rev)
        print(json.dumps(s, indent=1))
        if a.log:
            with open(a.log, "a") as f:
                f.write(json.dumps(s) + "\n")
        return
    if a.log:
        rows = {json.loads(l)["label"]: json.loads(l) for l in open(a.log) if l.strip()}
        base, new = rows[a.args[0]], rows[a.args[1]]
    else:
        base, new = load(a.args[0]), load(a.args[1])
    sys.exit(1 if compare(base, new, a.noise, a.perf) else 0)


if __name__ == "__main__":
    main()
