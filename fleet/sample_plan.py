#!/usr/bin/env python3
"""Sampled-testing plan of a change: strata of the touched models -> representatives -> DevLab jobs (plan Appendix A, T1).

usage: sample_plan.py IMPACT.json [--gpus strix-halo,radeon-r9700] [--history coverage.json [--record]]
                      [--configs DIR] [--baselines fleet/baselines.json] [--fit .github/devlab/r9700_fit.json]
                      [--json plan.json] [--emit-jobs OUTDIR] [--sessions 2]

IMPACT.json: the impact report of fleet/diff_fleet.py (per touched model: changed kernels with their target gfx11 | gfx12 | both,
optionally rows and a stratum signature; see sample_lib.normalize_impact for the accepted shapes).

Rules (plan Appendix A section 2):
  * a model is touched on a GPU when a changed kernel targets that GPU's family (gfx11: Strix Halo, gfx12: Radeon R9700);
    a model that does not fit the GPU (the fit table) is skipped on it; rows = the union of the touched kernels' rows
    (only rows the model has a published value for, when --baselines is given);
  * strata = touched models grouped by (changed kernel families, arch, head_dim, G, KV bucket, draft kind, weight formats of the
    changed matmuls, dense/MoE) per GPU; a change that only touches chat/tool text has no perf rows: functional-only stratum;
  * >= 1 representative per stratum per GPU, 2 when the stratum spans a wide shape range (hidden size or KV heads >= 2x apart);
    sub-1B models are not decode canaries when a larger member exists; the preferred member is where the changed kernel weighs most
    (largest hidden size for GEMM/matvec/MoE/GDN/program, longest context for attention); rotation: the least recently measured
    member first (--history), ties by weight;
  * runner time: estimated T1 canary = a third of the model's full A/B (plan section 6), against the full A/B of every touched model.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_lib as L  # noqa: E402
import fit_device  # noqa: E402
registry = L.registry

SUB_1B_BYTES = 0.9e9        # Q4_K_M weights below this: a sub-1B model (decode rows are noisy there, plan section 5)
WIDE_RATIO = 2.0            # a stratum spans a wide shape range when hidden size or KV heads differ by this factor
DECODE_SET = set(L.DECODE_ROWS) | {"streams4"}
ATTN = {"attn_prefill", "attn_decode", "spec"}


def full_ab_minutes(w_bytes):
    """Runner minutes of a full A/B job (3 sessions per side, 9 rows, dev gates): 7 min for a 0.5B model to 103 min for a 32B one
    (plan section 6, runs 43603561113, 43605627702, 43603562298), linear in the weights."""
    g = (w_bytes or 4e9) / 1e9
    return round(min(103.0, max(7.0, 7 + (g - 0.4) * (96.0 / 19.5))), 1)


def load_json(path, default=None):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return default


class Fleet:
    """What the plan knows about each model: llm_config signature, weights, fit per GPU, rows with a published value."""

    def __init__(self, configs, fit_path, baselines_path):
        self.configs = configs
        self.fit = {r["slug"]: r for r in (load_json(fit_path, []) or [])}
        self.baselines = (load_json(baselines_path, {}) or {}).get("models", {})

    def config(self, slug):
        for d in ([self.configs] if self.configs else []) + [os.path.join(L.ROOT, "fleet")]:
            p = os.path.join(d, slug, "llm_config.json")
            if os.path.exists(p):
                with open(p) as fh:
                    return json.load(fh)
        return None

    def weights(self, slug):
        return (self.fit.get(slug) or {}).get("w")

    def fits(self, slug, gpu):
        row = self.fit.get(slug)
        if gpu == "radeon-r9700":   # the fit table is sized for the R9700: its own verdict
            v = (row or {}).get("verdict")
            return v is None or v.startswith("fits")
        if row and "w" in row and registry.device(gpu).get("memory_bytes"):   # another GPU with a memory size: the same sizing against it
            return fit_device.verdicts(gpu, [row])[0]["verdict"].startswith("fits")
        return True

    def rows_with_baseline(self, slug, gpu):
        b = (self.baselines.get(slug) or {}).get(gpu)
        if b is None:
            return None                    # unknown model/GPU: do not filter
        return {r for r in b if r in L.ROWS}   # a '_state' entry (does not fit, unsupported) has no rows


def plan(impact, gpus, fleet, history=None, today=None, granularity="fine", prefer="heaviest"):
    history = history or {}
    today = today or datetime.date.today().isoformat()
    mod = L.normalize_impact(impact)
    strata, untouched = {}, []
    for slug, m in sorted(mod.items()):
        cfg = fleet.config(slug)
        sig = dict(L.signature_from_config(cfg)) if cfg else None
        if m["signature"]:                    # the impact report's own signature wins; the config adds hidden size, context, layers
            sig = {**(sig or {}), **{k: v for k, v in m["signature"].items() if v is not None}}
        for gpu in gpus:
            tgt = L.GPUS[gpu]
            ks = [k for k in m["kernels"] if tgt in k["targets"]]
            if not ks:
                continue
            if sig is None:
                sig = {"arch": "?", "head_dim": None, "G": None, "kv_bucket": None, "draft": "none", "moe": "?", "hidden": 0,
                       "kv_heads": 0, "context": 0}
            if not fleet.fits(slug, gpu):
                strata.setdefault(("__skipped__", gpu), []).append({"slug": slug, "reason": "does not fit " + gpu})
                continue
            fams = sorted({k["family"] for k in ks if k["family"] != "text"}) or ["text"]
            fmts = sorted({f for k in ks for f in k["formats"]})
            rows = sorted({r for k in ks for r in k["rows"]}, key=L.ROWS.index)
            have = fleet.rows_with_baseline(slug, gpu)
            if have is not None:
                rows = [r for r in rows if r in have]
            ksig = {**sig, "arch": "*"} if granularity == "shape" else sig    # shape: the kernel instantiation signature without the arch family
            if granularity == "arch":   # arch: what serving exercises (programs of the arch, draft kind, MoE, projector), not the shapes
                ksig = {**sig, "head_dim": "*", "G": "*", "kv_bucket": "*",
                        "moe": (sig.get("moe") or "dense") + ("+vision" if sig.get("vision") else "")}
            flat = granularity in ("coarse", "arch")
            key = L.stratum_key(ksig, [] if flat else fams, [] if flat else fmts, gpu)
            st = strata.setdefault((key, gpu), [])
            st.append({"slug": slug, "sig": sig, "rows": rows, "families": fams, "kernels": sorted(k["name"] for k in ks),
                       "w": fleet.weights(slug)})
    out = []
    for (key, gpu), members in sorted(strata.items(), key=lambda kv: kv[0]):
        if key == "__skipped__":
            continue
        out.append(select(key, gpu, members, history, today, prefer))
    skipped = [{"gpu": g, **x} for (k, g), v in strata.items() if k == "__skipped__" for x in v]
    return out, skipped


def weight_score(member):
    """Where the changed kernel weighs most: the longest context for attention-family changes, else the largest hidden size."""
    s = member["sig"]
    hid = s.get("hidden") or 0
    if set(member["families"]) & ATTN:
        return (s.get("context") or 0, hid)
    return (hid, s.get("context") or 0)


def select(key, gpu, members, history, today, prefer="heaviest"):
    union_rows = sorted({r for m in members for r in m["rows"]}, key=L.ROWS.index)
    cands = list(members)
    notes = []
    if set(union_rows) & DECODE_SET:
        big = [m for m in cands if (m["w"] or 1e12) >= SUB_1B_BYTES]
        if big and len(big) < len(cands):
            notes.append("sub-1B members not sampled for decode rows: " + ", ".join(m["slug"] for m in cands if m not in big))
            cands = big
    hist = history.get("models", {})
    # least recently measured first (never = ''), then the heaviest member of the changed kernel (prefer="lightest": the smallest
    # weights, for functional checks such as the release smoke, where the stratum's programs matter and runner time is the cost)
    if prefer == "lightest":
        cands.sort(key=lambda m: (hist.get(f"{m['slug']}|{gpu}", ""), bool(m["w"]) and m["w"] < SUB_1B_BYTES, m["w"] or 1e15,
                                  weight_score(m), m["slug"]))   # a sub-1B member only when it is the only kind (weak at tool calls)
    else:
        cands.sort(key=lambda m: (hist.get(f"{m['slug']}|{gpu}", ""), tuple(-x for x in weight_score(m)), m["slug"]))
    reps = [cands[0]]
    why = {cands[0]["slug"]: ["least recently measured" if hist.get(f"{cands[0]['slug']}|{gpu}", "")
                              else ("smallest weights" if prefer == "lightest" else "heaviest for the changed kernel")]}
    if len(cands) > 1:
        hs = [m["sig"].get("hidden") or 0 for m in cands]
        kvs = [m["sig"].get("kv_heads") or 0 for m in cands]
        rh = (max(hs) / min(hs)) if min(hs) else 1.0
        rk = (max(kvs) / min(kvs)) if min(kvs) else 1.0
        if rh >= WIDE_RATIO or rk >= WIDE_RATIO:
            first = reps[0]
            if rh >= rk:
                far = max(cands, key=lambda m: abs((m["sig"].get("hidden") or 0) - (first["sig"].get("hidden") or 0)))
                reason = f"opposite shape extreme: hidden {far['sig'].get('hidden')} vs {first['sig'].get('hidden')}"
            else:
                far = max(cands, key=lambda m: abs((m["sig"].get("kv_heads") or 0) - (first["sig"].get("kv_heads") or 0)))
                reason = f"opposite shape extreme: kv_heads {far['sig'].get('kv_heads')} vs {first['sig'].get('kv_heads')}"
            if far is not first:
                reps.append(far)
                why[far["slug"]] = [reason]
    minutes = {m["slug"]: full_ab_minutes(m["w"]) for m in members}
    return {"key": key, "gpu": gpu, "families": sorted({f for m in members for f in m["families"]}), "members": [m["slug"] for m in members],
            "rows": union_rows, "functional_only": not union_rows,
            "representatives": [{"slug": r["slug"], "why": why[r["slug"]], "rows": union_rows,
                                 "canary_minutes": round(minutes[r["slug"]] / 3, 1), "full_minutes": minutes[r["slug"]]} for r in reps],
            "notes": notes, "full_minutes": round(sum(minutes.values()), 1)}


def totals(strata):
    return {"strata": len(strata), "touched_models": sum(len(s["members"]) for s in strata),
            "sampled_models": sum(len(s["representatives"]) for s in strata),
            "canary_runner_hours": round(sum(r["canary_minutes"] for s in strata for r in s["representatives"]) / 60, 2),
            "full_runner_hours": round(sum(s["full_minutes"] for s in strata) / 60, 2)}


def table(strata, skipped):
    lines = ["| GPU | stratum | members | rows | representatives |", "|---|---|---|---|---|"]
    for s in strata:
        sig = s["key"].split("|")
        short = " ".join(p for p in sig if not p.startswith("gpu="))
        rows = "(functional only)" if s["functional_only"] else ",".join(s["rows"])
        reps = "; ".join(f"{r['slug']} ({'; '.join(r['why'])})" for r in s["representatives"])
        lines.append(f"| {s['gpu']} | {short} | {len(s['members'])} | {rows} | {reps} |")
    for x in skipped:
        lines.append(f"| {x['gpu']} | skipped | {x['slug']} | | {x['reason']} |")
    t = totals(strata)
    lines.append("")
    lines.append(f"strata {t['strata']}; touched model-GPU pairs {t['touched_models']}; sampled {t['sampled_models']}; "
                 f"T1 canary ~{t['canary_runner_hours']} runner-h vs full A/B of every touched model ~{t['full_runner_hours']} runner-h")
    return "\n".join(lines)


# ---- jobs --------------------------------------------------------------------------------------------------------------------
def filter_job(job, rows, sessions):
    """Narrow a generated A/B job (strix_ab_jobs / ab_jobs perf.json entry) to the rows a canary needs: drop the --long perf steps
    when no long row is needed, the 4-stream steps when streams4 is not needed, the gates and llama.cpp steps (a canary compares
    two Loom builds), and cut the session count of the perf and c4 steps to `sessions` per side (ABBA kept)."""
    need_long = bool(set(rows) & set(L.LONG_ROWS))
    need_short = bool(set(rows) & set(L.SHORT_ROWS))
    need_c4 = "streams4" in rows
    steps = []
    seen = {}
    for s in job["steps"]:
        p = s.split(":")
        if p[0] == "perf":
            if p[-1] == "long" and not need_long:
                continue
            if p[-1] != "long" and not need_short:
                continue
            key = "perf-long" if p[-1] == "long" else "perf"
        elif p[0] == "c4":
            if not need_c4:
                continue
            key = "c4"
        elif p[0] == "gates" or p[0].startswith("llama"):
            continue
        else:
            steps.append(s)          # textmatch, func, ktest stay: the canary checks text and the changed kernels
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= 2 * sessions:
            steps.append(s)
    return {**job, "steps": steps, "name": ("t1-" + job["name"].split("-", 1)[1])[:40]}


def emit_jobs(plan_strata, outdir, sessions, refs=None):
    """perf.json per GPU for the representatives, from ab_jobs.py when it exists, else strix_ab_jobs.py (Strix only)."""
    gen = os.path.join(L.ROOT, ".github", "scripts")
    os.makedirs(outdir, exist_ok=True)
    written = []
    for gpu in sorted({s["gpu"] for s in plan_strata}):
        reps = {}
        for s in plan_strata:
            if s["gpu"] == gpu:
                for r in s["representatives"]:
                    reps.setdefault(r["slug"], set()).update(r["rows"])
        slugs = sorted(reps)
        ab = os.path.join(gen, "ab_jobs.py")
        if os.path.exists(ab):
            cmd = [sys.executable, ab, "--device", gpu, "--reps", str(sessions), "--gates", ""] + (["--refs", refs] if refs else []) + slugs
        elif gpu == "strix-halo":
            cmd = [sys.executable, os.path.join(gen, "strix_ab_jobs.py"), "--reps", str(sessions), "--gates", ""] + slugs
        else:
            print(f"no A/B job generator for {gpu} yet (ab_jobs.py): {slugs}", file=sys.stderr)
            continue
        try:
            exp = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
        except subprocess.CalledProcessError as e:   # ab_jobs.py / strix_ab_jobs.py know only the devices with a job generator
            print(f"no job file for {gpu}: {os.path.basename(cmd[1])} failed ({(e.stderr or '').strip().splitlines()[-1:] or e.returncode}); the plan above is the wave", file=sys.stderr)
            continue
        exp["jobs"] = [filter_job(j, sorted(reps[j["card"].split("/")[-1]], key=L.ROWS.index), sessions) for j in exp["jobs"]]
        path = os.path.join(outdir, f"perf_{gpu}.json")
        json.dump(exp, open(path, "w"), indent=1)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("impact")
    ap.add_argument("--gpus", default="strix-halo,radeon-r9700")
    ap.add_argument("--history")
    ap.add_argument("--record", action="store_true", help="write today's date for the chosen representatives into --history")
    ap.add_argument("--configs", help="directory of <slug>/llm_config.json (default fleet/)")
    ap.add_argument("--baselines", default=os.path.join(L.ROOT, "fleet", "baselines.json"))
    ap.add_argument("--fit", default=os.path.join(L.ROOT, ".github", "devlab", "r9700_fit.json"))
    ap.add_argument("--json")
    ap.add_argument("--emit-jobs", help="directory for perf_<gpu>.json (devlab-perf experiment files) of the representatives")
    ap.add_argument("--refs", help="ab_jobs.py --refs MAP.json: {slug: {old: ORG/REPO@REV, new: ORG/REPO@REV}}: old = the published package (baseline), "
                    "new = the candidate uploaded for the head build")
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--granularity", choices=["fine", "coarse", "shape", "arch"], default="fine",
                    help="fine (default, the plan): arch, shape, draft kind, dense/MoE, changed kernel families and formats; "
                         "coarse: without the changed kernel families; shape: without the arch family (the kernel instantiation signature); "
                         "arch: arch, draft kind, dense/MoE and projector only (functional checks: fleet/smoke_plan.py)")
    ap.add_argument("--prefer", choices=["heaviest", "lightest"], default="heaviest",
                    help="representative: heaviest for the changed kernel (default, perf canaries) or smallest weights (functional checks)")
    a = ap.parse_args()
    gpus = [g for g in a.gpus.split(",") if g]
    fleet = Fleet(a.configs, a.fit, a.baselines)
    hist = load_json(a.history, {}) if a.history else {}
    strata, skipped = plan(json.load(open(a.impact)), gpus, fleet, hist, granularity=a.granularity, prefer=a.prefer)
    res = {"impact": a.impact, "gpus": gpus, "strata": strata, "skipped": skipped, "totals": totals(strata)}
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print(table(strata, skipped))
    if a.emit_jobs:
        for p in emit_jobs(strata, a.emit_jobs, a.sessions, a.refs):
            print("wrote", p)
    if a.record and a.history:
        h = hist or {}
        today = datetime.date.today().isoformat()
        for s in strata:
            for r in s["representatives"]:
                h.setdefault("models", {})[f"{r['slug']}|{s['gpu']}"] = today
                h.setdefault("cells", {}).setdefault(s["key"], {})[r["slug"]] = today
        json.dump(h, open(a.history, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
