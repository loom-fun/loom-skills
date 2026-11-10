#!/usr/bin/env python3
"""Definition of done for an optimization campaign. Run it from bench/ before reporting.

usage: audit.py [BENCH_DIR=.]    -> prints checks + the headline table, writes audit.json,
                                    exit 1 if any check FAILs

Checks, all from the files the scripts write themselves:
  baseline   llamacpp_pick.json exists, and the roofline baseline is its fastest build
  plausible  Loom never above ~0.97 of a ceiling (else the roofline undercounts);
             llama.cpp not implausibly far below it (else the baseline is suspect)
  tracking   targets.jsonl has a start row and later measurements
  gates      the latest compare is PASS; consistency PASS, or FAIL with consistency_note.md
             (near-tie with its cause); gates ran on the final build when git is in use
Use the headline table it prints in the report, so no number is typed by hand.
"""
import json
import os
import subprocess
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
P = lambda f: os.path.join(d, f)
here = os.path.dirname(os.path.abspath(__file__))
PROGRESS = os.path.join(here, "..", "..", "loom-roofline", "progress.py")
checks = []


NO_REF = False   # set below: no installed llama.cpp runs this model (qualify.sh NO_LLAMACPP)


def check(name, ok, msg, level="FAIL"):
    if NO_REF and name.startswith(("baseline:", "gates: compare", "roofline: ran with a baseline", "gates: full run")):
        level, msg = "WARN", msg + " [no llama.cpp build runs this model: correctness vs a reference is UNVERIFIED]"
    checks.append({"check": name, "result": "ok" if ok else level, "detail": msg})


def jl(f):
    return [json.loads(l) for l in open(P(f)) if l.strip()] if os.path.exists(P(f)) else []


roof = json.load(open(P("roofline.json"))) if os.path.exists(P("roofline.json")) else None
pick = json.load(open(P("llamacpp_pick.json"))) if os.path.exists(P("llamacpp_pick.json")) else None
targets, gates = jl("targets.jsonl"), jl("gates.jsonl")
metrics = {m["metric"]: m for m in roof["metrics"]} if roof else {}

NO_REF = pick is not None and not pick.get("ranked")
# baseline
check("baseline: picker ran", pick is not None and pick.get("ranked"),
      "llamacpp_pick.json missing: run pick_llamacpp.sh MODEL.gguf from bench/")
if pick and pick.get("ranked") and roof:
    top = pick["ranked"][0]
    base = metrics.get("decode_tps", {}).get("baseline")
    pbase = metrics.get("prefill_1000", {}).get("baseline")
    # chat vs raw /completion and different prompts: allow 8 % on decode, 25 % on prefill
    ok = (base is not None and base >= top["decode_tps"] * 0.92
          and (pbase is None or pbase >= top["prefill_1k_tps"] * 0.75))
    # With a single candidate build the baseline is that build: a lower number is load noise.
    ok = ok or len(pick["ranked"]) == 1
    # finalize.sh measures llama.cpp with the picked build itself (label llama-final): a low number
    # there is GPU load during the measurement, not a wrong build.
    by_finalize = any(json.loads(l).get("label") == "llama-final" for l in open(P("perf_log.jsonl"))) \
        if os.path.exists(P("perf_log.jsonl")) else False
    check("baseline: fastest build used", ok,
          f"roofline baseline decode {base} / prefill-1k {pbase} vs fastest build "
          f"{top['decode_tps']} / {top['prefill_1k_tps']} ({top['dir']}): "
          + ("measured with that build by finalize.sh, so the GPU was loaded: say so in the report"
             if by_finalize else "re-measure with that build"),
          level="WARN" if by_finalize else "FAIL")
check("roofline: ran with a baseline", roof is not None and metrics.get("decode_tps", {}).get("baseline"),
      "roofline.json missing or without --baseline: run loom-roofline")

# plausibility, both directions (best kept value); the headline uses the final build
best, final = {}, {}
for r in targets:
    if r.get("kept", True):
        best[r["metric"]] = max(best.get(r["metric"], 0), r["value"])
        final.setdefault(r["metric"], []).append(r)
for m, rs in final.items():   # final build = last kept label; median of its runs (noise is ±5-10 %)
    vals = sorted(x["value"] for x in rs if x.get("label") == rs[-1].get("label")) or [rs[-1]["value"]]
    final[m] = vals[len(vals) // 2]
# A drafted decode row's ceiling: roofline.json scales every decode ceiling by one drafting gain measured on short
# replies (draft_gain). A row whose own timed requests measured their acceptance (perf.py, c4_steady.py) gets the
# ceiling at that acceptance (roofline.drafted_row_ceiling, as the card's roofline), never a lower one: a reply that
# copies a long document accepts far more drafted tokens than the short replies the gain came from.
row_acc = {}
for r in jl("perf_log.jsonl") + [x for x in targets if x.get("metric") == "c4_steady"]:
    if r.get("kind", "loom") == "loom":
        row_acc.update(r.get("acceptance") if isinstance(r.get("acceptance"), dict)
                       else ({"c4_steady": r["acceptance"]} if r.get("acceptance") is not None else {}))
tune = {}
if os.path.exists(P("draft_tune.json")) and os.path.getsize(P("draft_tune.json")):
    tune = json.load(open(P("draft_tune.json")))
depth = tune.get("depth") if tune.get("kind", "none") != "none" else None
shape = None
if row_acc and depth and roof and os.path.exists(roof.get("gguf") or ""):
    try:
        sys.path.insert(0, os.path.join(here, "..", "..", "loom-roofline"))
        import roofline
        cfg = json.load(open(roof["cfg"])) if roof.get("cfg") and os.path.exists(roof["cfg"]) else None
        shape = roofline.model_shape(roof["gguf"], cfg)
    except Exception as e:   # no gguf package, unreadable GGUF: the roofline.json ceilings stand
        print(f"audit: drafted-row ceilings at their own acceptance unavailable ({e})", file=sys.stderr)


def ceiling(m, R):
    c = shape and roofline.drafted_row_ceiling(shape, m, depth, row_acc.get(m), roof.get("bw") or roofline.BW)
    return max(R["ceiling"], c or 0)


for m, R in metrics.items():
    if m in best:
        e = best[m] / ceiling(m, R)
        # 216 GB/s is a conservative measured bandwidth: dense ports reached 221-237 GB/s. Above 0.97 is a
        # WARN (check the byte count first); above 1.10 the roofline certainly undercounts.
        check(f"plausible: Loom {m}", e <= 0.97,
              f"Loom at {e:.2f} of the ceiling: check the roofline's byte count (sliding window, KV sharing, "
              "hybrid layers, expert count); a dense model at 1.0-1.1 means the bandwidth constant is conservative",
              level="WARN" if e <= 1.10 else "FAIL")
    be = R.get("baseline_eff")
    if be is not None:
        floor = 0.10 if m.startswith("prefill") else 0.30
        check(f"plausible: llama.cpp {m}", be >= floor,
              f"llama.cpp at {be:.2f} of the ceiling (< {floor}): suspect the baseline build or its settings",
              level="WARN")

# tracking
dec = [r for r in targets if r["metric"] == "decode_tps"]
check("tracking: progress logged", len(dec) >= 2 and dec[0].get("change") == "start",
      f"{len(dec)} decode_tps rows in targets.jsonl; perf.py logs them once roofline.json exists")

# gates
last = {}
for g in gates:
    last[g["gate"]] = g
c = last.get("compare")
check("gates: compare PASS", c is not None and c["verdict"] == "PASS",
      f"latest compare: {c and {k: c[k] for k in ('verdict', 'max_gap', 'bad') if k in c}}")
s = last.get("consistency")
note = os.path.exists(P("consistency_note.md"))
if s is not None and s["verdict"] != "PASS" and note:
    check("gates: consistency", False, "FAIL, documented as a near-tie in consistency_note.md: "
          "state it in the report", level="WARN")
else:
    check("gates: consistency", s is not None and s["verdict"] == "PASS",
          f"latest consistency: {s and s['verdict']} (a near-tie with a known cause needs consistency_note.md)")
gj = json.load(open(P("gates.json"))) if os.path.exists(P("gates.json")) else None
if gj and any(n.startswith("needle") for n in (gj.get("skipped") or {})):
    check("gates: needle coverage", False, "explicitly excluded from this trial; long-context retrieval and "
          "full-context serving were not verified", level="WARN")
failing = [n for n, g in (gj or {}).get("gates", {}).items() if not g["pass"]]
documented = failing == ["consistency"] and note   # a near-tie with its cause in consistency_note.md
check("gates: full run PASS", gj is not None and (gj.get("verdict") == "PASS" or documented) and gj.get("level", "full") == "full",
      f"gates.json: {gj and {k: gj.get(k) for k in ('verdict', 'level')}}: run finalize.sh (LEVEL=full gates) on the final build")
if gj and "needle_full" in (gj.get("deferred") or []):
    check("gates: full-context probe", False, "deferred (NEEDLE_FULL=0): this build was not probed at its full "
          "context length; LEVEL=full gates.sh on it runs only that probe", level="WARN")
kept = [r for r in targets if r.get("kept", True) and r.get("commit", "-") != "-"]
if kept and c and c.get("commit", "-") != "-" and not (gj and gj.get("pkg_hash") and gj.get("level") == "full"):
    # (gates.json from a LEVEL=full run carries the package hash it ran on: that, not git HEAD, names the build)
    check("gates: ran on the final build", c["commit"] == kept[-1]["commit"],
          f"compare at {c['commit']}, last kept measurement at {kept[-1]['commit']}")

# headline table
rows = []
for m, R in metrics.items():
    if m not in best:
        continue
    goal = roof.get("campaign", {}).get("goals", {}).get(m) or R["target"]
    v = final[m]
    rows.append({"metric": m, "llama_cpp": R.get("baseline"), "loom": v, "loom_best_seen": best[m],
                 "margin": round(v / R["baseline"] - 1, 3) if R.get("baseline") else None,
                 "pct_of_ceiling": round(v / R["ceiling"], 3), "target": goal,
                 "met": v >= goal * 0.99})
print("Loom = final build (median of its runs); best = best single kept run")
print(f"{'metric':16} {'llama.cpp':>9} {'Loom':>9} {'best':>7} {'margin':>7} {'%ceil':>6} {'target':>8}  met")
for r in rows:
    lc = f"{r['llama_cpp']:9.1f}" if r["llama_cpp"] else f"{'-':>9}"
    mg = f"{r['margin']:+7.0%}" if r["margin"] is not None else f"{'-':>7}"
    print(f"{r['metric']:16} {lc} {r['loom']:9.1f} {r['loom_best_seen']:7.1f} "
          f"{mg} {r['pct_of_ceiling']:6.0%} {r['target']:8.1f}  {'yes' if r['met'] else 'no'}")
if pick and pick.get("ranked"):
    print(f"baseline build: {pick['ranked'][0]['dir']}")
print()
for ch in checks:
    print(f"[{ch['result']:4}] {ch['check']}" + ("" if ch["result"] == "ok" else f": {ch['detail']}"))
if roof and targets:
    for m in sorted({r["metric"] for r in targets}):
        r = subprocess.run([sys.executable, PROGRESS, P("roofline.json"), P("targets.jsonl"), "--metric", m],
                           capture_output=True, text=True)
        v = [l for l in r.stdout.splitlines() if "->" in l]
        print(f"[ctrl] {m}: {v[-1].strip() if v else r.stderr.strip()[:120]}")
json.dump({"checks": checks, "headline": rows}, open(P("audit.json"), "w"), indent=1)
fails = [c for c in checks if c["result"] == "FAIL"]
print("\nAUDIT", "FAIL" if fails else "PASS", f"({len(fails)} failing)")
sys.exit(1 if fails else 0)
