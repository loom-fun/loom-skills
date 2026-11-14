#!/usr/bin/env python3
"""Measure 5 (does sampling agree with testing every model?): replay of the phase-3 Strix A/B.

usage: measure5_replay.py --base DIR --head DIR --ab-glob 'GLOB/*.json' [--jobs-json ab_minutes.json] [--out measure5_phase3.md]

The "change" of each model is the drift between two program trees (BASE_DIR/<slug>/program vs HEAD_DIR/<slug>/program; here the
loom-fun package vs the loom-fun-staging package). The impact of the change on gfx11 is computed per model (replay_impact), the
touched models are put into strata and the representatives are chosen by sample_plan.py's rule, with the baselines of the loom-fun
cards for the rows. The sampled verdict of a stratum is the worst verdict of its representatives (their own A/B result on the
stratum's rows, the rows the impact analysis predicts); the full verdict of a model is its A/B result on every row, decided by the
promotion ratchet (promote_check) on all its runs. Agreement = the stratum's sampled verdict equals the member's full verdict.
"""
import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import promote_check as P  # noqa: E402
import replay_impact  # noqa: E402
import sample_lib as L  # noqa: E402
import sample_plan as SP  # noqa: E402

GPU = "strix-halo"


def cls(decision):
    """perf class of a ratchet decision: bad = hold (a confirmed regression, a reject or a gate failure), suspect = rerun (undecided),
    ok = promote | improved-pending"""
    return {"hold": "bad", "rerun": "suspect"}.get(decision, "ok")


ORDER = ["bad", "suspect", "ok"]


def worst_cls(cs):
    return min(cs, key=ORDER.index) if cs else "ok"


def load_ab(pattern):
    files = collections.defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        m = re.match(r"(.*)\.(r\w+)\.json$", os.path.basename(f))
        if m:
            files[m.group(1)].append((m.group(2), f))
    return {s: [json.load(open(f)) for _, f in sorted(fs)] for s, fs in files.items()}


def evaluate(impact, ab, fleet, meth, granularity):
    strata, _ = SP.plan(impact, [GPU], fleet, granularity=granularity)
    touched = {m: [k["name"] for k in v["kernels"]] for m, v in impact["models"].items() if v["kernels"]}
    full = {}
    for s, runs in ab.items():
        r = P.check(s, GPU, runs, None, meth, near_tie=True)
        full[s] = {"class": cls(r["decision"]), "decision": r["decision"], "rows": {k: v for k, v in r["rows"].items() if v["state"] in ("hold", "rerun")},
                   "text": runs[-1].get("text", {}).get("status")}
    out = []
    for st in strata:
        rows = st["rows"]
        reps = [r["slug"] for r in st["representatives"]]
        rep_res = {}
        for r in reps:
            if r in ab:
                res = P.check(r, GPU, ab[r], None, meth, near_tie=True, rows_only=rows or None)
                rep_res[r] = {"class": cls(res["decision"]), "decision": res["decision"], "text": ab[r][-1].get("text", {}).get("status")}
        members = [m for m in st["members"] if m in ab]
        # the same members measured one at a time (what a random rotation would pick): single-representative classes
        solo = {}
        for m in members:
            res = P.check(m, GPU, ab[m], None, meth, near_tie=True, rows_only=rows or None)
            solo[m] = {"class": cls(res["decision"]), "text": ab[m][-1].get("text", {}).get("status")}
        out.append({"key": st["key"], "rows": rows, "members": st["members"], "with_data": members, "reps": reps, "rep_res": rep_res,
                    "sampled": worst_cls([v["class"] for v in rep_res.values()]),
                    "sampled_text": "DIFF" if any(v["text"] == "DIFF" for v in rep_res.values()) else "MATCH",
                    "solo": solo, "full_minutes": st["full_minutes"],
                    "rep_minutes": [x["canary_minutes"] for x in st["representatives"]]})
    return out, touched, full


def confusion(strata, full, truth="all"):
    """per member of a stratum: (sampled class vs the member's full class). Counts over members with A/B data.
    truth "all": the member's verdict on every row (the full-fleet test); "rows": on the stratum's rows only."""
    c = collections.Counter()
    detail = []
    for s in strata:
        for m in s["with_data"]:
            f = full[m]["class"] if truth == "all" else s["solo"][m]["class"]
            smp = s["sampled"]
            is_rep = m in s["reps"]
            if smp == f:
                k = "agree"
            elif smp == "ok" and f in ("bad", "suspect"):
                k = "false_pass" if f == "bad" else "agree_suspect_missed"
            elif smp in ("bad", "suspect") and f == "ok":
                k = "false_alarm"
            else:
                k = "partial"       # bad vs suspect
            c[(k, "rep" if is_rep else "non-rep")] += 1
            detail.append((m, s["key"], smp, f, is_rep, k))
    return c, detail


def rotation_robustness(strata, full, truth="all"):
    """for every stratum with >= 2 members with data: if each member in turn were the single representative (its own result on
    the stratum rows), how often would the sampled class equal each other member's full class?"""
    pairs = agree = fp = fa = 0
    for s in strata:
        ms = s["with_data"]
        if len(ms) < 2:
            continue
        for r in ms:
            for m in ms:
                if m == r:
                    continue
                pairs += 1
                a, b = s["solo"][r]["class"], (full[m]["class"] if truth == "all" else s["solo"][m]["class"])
                agree += a == b
                fp += (a == "ok" and b == "bad")
                fa += (a == "bad" and b == "ok")
    return pairs, agree, fp, fa


def hours(minutes_json, slugs):
    """actual runner hours of the phase-3 A/B jobs of these models (last successful first run + the confirming run when there is one)"""
    d = json.load(open(minutes_json)) if minutes_json and os.path.exists(minutes_json) else {}
    tot = 0.0
    for s in slugs:
        e = d.get(s) or {}
        for run in ("r1", "r2"):
            ok = [m for m, c, _ in (e.get(run) or []) if c == "success"]
            if ok:
                tot += ok[-1]
    return tot / 60


def family_subset(impact, fam):
    """the impact of a change that only touches kernels of one family (a single-PR scenario)"""
    models = {}
    for m, v in impact["models"].items():
        ks = [k for k in v["kernels"] if k["family"] == fam]
        if ks:
            models[m] = {"kernels": ks}
    return {**impact, "models": models}


def summarize(strata, touched, full, truth):
    c, detail = confusion(strata, full, truth)
    reps = {r for s in strata for r in s["reps"]}
    members = {m for s in strata for m in s["with_data"]}
    nonrep = members - reps
    t = collections.Counter()
    for (k, who), n in c.items():
        t[(k, who)] += n
    pairs, agree, fp, fa = rotation_robustness(strata, full, truth)
    strata_bad = collections.Counter()
    for s in strata:
        ms = s["with_data"]
        if not ms:
            continue
        truth_cls = worst_cls([(full[m]["class"] if truth == "all" else s["solo"][m]["class"]) for m in ms])
        strata_bad[(s["sampled"], truth_cls)] += 1
    tx = collections.Counter()
    for s_ in strata:
        for m in s_["with_data"]:
            if m not in s_["reps"]:
                tx[f"sampled={s_['sampled_text']}|member={s_['solo'][m]['text']}"] += 1
    return {"text_non_rep": dict(sorted(tx.items())), "strata": len(strata), "touched_models": len(touched), "models_with_data": len(members), "reps": len(reps & members),
            "non_rep_members": len(nonrep), "confusion": {f"{k}|{w}": n for (k, w), n in sorted(t.items())},
            "stratum_pairs": {f"sampled={a}|worst_member={b}": n for (a, b), n in sorted(strata_bad.items())},
            "rotation": {"pairs": pairs, "agree": agree, "false_pass": fp, "false_alarm": fa}}


def impact_of(base, head, workdir, diff_fleet_dir=None):
    """The impact of the drift between two package trees, by fleet/diff_fleet.py (engineer "fleet") on the gfx11 views of both trees;
    replay_impact's own comparison when diff_fleet is not available. Either way: a normalized report for sample_plan."""
    for d in (HERE, diff_fleet_dir):
        if d and os.path.exists(os.path.join(d, "diff_fleet.py")):
            sys.path.insert(0, d)
            import diff_fleet
            tmp = tempfile.mkdtemp(prefix="views_", dir=workdir)
            for side, src in (("base", base), ("head", head)):
                replay_impact.write_view_trees(src, os.path.join(tmp, side), "gfx11")
            rep = diff_fleet.report(os.path.join(tmp, "base"), os.path.join(tmp, "head"))
            for s_ in rep["untouched"]:
                rep["models"][s_] = {"blocks": []}
            shutil.rmtree(tmp)
            return rep, "diff_fleet.py"
    return replay_impact.impact(base, head, "gfx11"), "replay_impact.py (diff_fleet.py not found)"


def run(base, head, ab_glob, configs=None, baselines=None, jobs_json=None, workdir=None, diff_fleet_dir=None):
    meth = L.load_methodology()
    impact, tool = impact_of(base, head, workdir or tempfile.gettempdir(), diff_fleet_dir)
    impact = {"models": {m: {"kernels": [{"name": k["name"], "targets": k["targets"], "family": k["family"], "rows": k["rows"], "formats": k["formats"]}
                                          for k in v["kernels"]], "signature": v["signature"]} for m, v in L.normalize_impact(impact).items()}}
    impact = {"models": {m: {**v, "kernels": [{**k, "target": k["targets"]} for k in v["kernels"]]} for m, v in impact["models"].items()}}
    ab = load_ab(ab_glob)
    fleet = SP.Fleet(configs or head, os.path.join(L.ROOT, ".github", "devlab", "r9700_fit.json"), baselines or os.path.join(L.ROOT, "fleet", "baselines.json"))
    res = {"impact_tool": tool, "meth": meth["source"], "models": len(impact["models"]), "with_data": len(ab), "no_ab_data": sorted(set(impact["models"]) - set(ab)),
           "whole": {}, "family": {}, "hours": {}}
    for gran in ("fine", "coarse", "shape"):
        strata, touched, full = evaluate(impact, ab, fleet, meth, gran)
        res["whole"][gran] = {"summary": summarize(strata, touched, full, "all"), "strata": strata}
        if gran == "fine":
            res["full"], res["touched"] = full, touched
    pred = {}
    for m, v in impact["models"].items():
        if m in res["full"]:
            rows = {r for k in v["kernels"] for r in L.kernel_family_rows(k["name"])[1]}
            bad = set(res["full"][m]["rows"])
            pred[m] = {"bad_rows": sorted(bad), "predicted_rows": sorted(rows), "missed": sorted(bad - rows)}
    res["row_prediction"] = pred
    res["text_diff"] = sorted(m for m, f in res["full"].items() if f["text"] == "DIFF")
    res["untouched_not_ok"] = {m: f["decision"] for m, f in res["full"].items() if m not in res["touched"] and f["class"] != "ok"}
    res["touched_not_ok"] = {m: (f["decision"], sorted(f["rows"])) for m, f in res["full"].items() if m in res["touched"] and f["class"] != "ok"}
    fams = sorted({k["family"] for v in impact["models"].values() for k in v["kernels"]} - {"text"})
    for fam in fams:
        sub = family_subset(impact, fam)
        res["family"][fam] = {}
        for gran in ("fine", "shape"):
            strata, touched, full = evaluate(sub, ab, fleet, meth, gran)
            res["family"][fam][gran] = {"summary": summarize(strata, touched, full, "rows"), "strata": strata}
    h = res["hours"]
    h["full_ab_all_models"] = hours(jobs_json, ab)
    h["full_ab_touched"] = hours(jobs_json, [m for m in ab if m in res["touched"]])
    for gran, v in res["whole"].items():
        reps = {r for s in v["strata"] for r in s["reps"]}
        h[f"canary_{gran}"] = hours(jobs_json, reps) / 3.0
        h[f"reps_full_ab_{gran}"] = hours(jobs_json, reps)
    return res


def render_md(res):
    W, F, H = res["whole"], res["family"], res["hours"]
    n_touched = len(res["touched"])
    cm = lambda sm, k: sum(v for kk, v in sm["confusion"].items() if kk.startswith(k))
    L_ = []
    a = L_.append
    a("# Measure 5, phase-3 replay: does the sampled verdict equal the full-fleet verdict?")
    a("")
    a("Offline replay of the Strix Halo phase-3 A/B (loom-fun package vs loom-fun-staging package, 63 A/B files, 46 models). "
      f"Tools: `{res['impact_tool']}` on the gfx11 views of both trees (impact of the program drift per model), `fleet/sample_plan.py` (strata and representatives), "
      f"`fleet/verdict.py` + `fleet/promote_check.py` (per-row verdicts, bands `{res['meth']}`), `fleet/measure5_replay.py` (this report). "
      "No GPU time was used; every number below is recomputed from the existing A/B data.")
    a("")
    a("## What was replayed")
    a(f"- {res['models']} models. The change of a model = the drift between the two program trees on gfx11 (kernels compared one by one, "
      f"gfx12 twins and target declarations normalized away). {n_touched} models have a non-empty impact; {res['models'] - n_touched} are untouched "
      f"(byte-identical gfx11 programs).")
    a(f"- {res['with_data']} models have A/B data ({len(res['no_ab_data'])} have none: {', '.join(res['no_ab_data'])}); "
      f"{sum(1 for m in res['touched'] if m in res['full'])} of the touched models have data and form the test set.")
    a("- Full verdict of a model = the promotion ratchet on all its A/B runs, every row, per-row bands: `bad` = a row below its band in both of the "
      "last two runs, or below the T1 reject threshold in a single run, or a gate failure; `suspect` = undecided (one run out of band, or a "
      "bimodal model whose modes cannot be compared); `ok` = everything else (improvements are `ok`: recorded, not trusted). Greedy-text "
      "differences are reported separately (phase 3 reviewed them as near-ties).")
    a("- Sampled verdict of a stratum = the worst verdict of its representatives, each judged only on the stratum's predicted rows and only "
      "on the runs of that A/B (a rep that was re-run in phase 3 counts with its confirming run: the T1 sequential extension to 6+6 sessions).")
    a("- Agreement of a model = its stratum's sampled class equals the model's full class. A representative agrees with itself trivially "
      "(up to the row restriction), so the real test is the non-representative members.")
    a("")
    a("## Result: the whole drift as one change")
    a("")
    a("| strata rule | strata | touched (with data) | representatives | non-rep members | non-rep agree | false-pass | false-alarm | strata agree |")
    a("|---|---|---|---|---|---|---|---|---|")
    for g, label in (("fine", "plan Appendix A (arch, shape, draft, dense/MoE, kernel families, formats)"), ("coarse", "without the kernel families"),
                     ("shape", "without the arch family (kernel instantiation signature)")):
        sm = W[g]["summary"]
        sp = sm["stratum_pairs"]
        agree = sum(v for k, v in sp.items() if k.split("|")[0].split("=")[1] == k.split("|")[1].split("=")[1])
        a(f"| {g}: {label} | {sm['strata']} | {sm['touched_models']} ({sm['models_with_data']}) | {sm['reps']} | {sm['non_rep_members']} | "
          f"{cm(sm, 'agree|non-rep')} | {cm(sm, 'false_pass|')} | {cm(sm, 'false_alarm|')} | {agree}/{sum(sp.values())} |")
    a("")
    sm = W["fine"]["summary"]
    a(f"Rotation robustness (plan rule): if every member of a multi-member stratum were the single representative in turn, the sampled class "
      f"equals each other member's full class in {sm['rotation']['agree']}/{sm['rotation']['pairs']} pairs "
      f"(false-pass {sm['rotation']['false_pass']}, false-alarm {sm['rotation']['false_alarm']}). Under the shape rule: "
      f"{W['shape']['summary']['rotation']['agree']}/{W['shape']['summary']['rotation']['pairs']} "
      f"(false-pass {W['shape']['summary']['rotation']['false_pass']}, false-alarm {W['shape']['summary']['rotation']['false_alarm']}).")
    a("")
    a("Models with a confirmed or rejected row in the full test, and what the sampler does with them:")
    a("")
    a("| model | full verdict | bad rows | in the predicted rows | stratum representative |")
    a("|---|---|---|---|---|")
    rep_of = {}
    for s_ in W["fine"]["strata"]:
        for m in s_["members"]:
            rep_of[m] = ", ".join(s_["reps"])
    for m, (d, rows) in sorted(res["touched_not_ok"].items()):
        pr = res["row_prediction"][m]
        a(f"| {m} | {d} | {', '.join(rows)} | {'yes' if not pr['missed'] else 'NO: ' + ', '.join(pr['missed'])} | {rep_of.get(m, '-')} |")
    for m, d in sorted(res["untouched_not_ok"].items()):
        pr = res["row_prediction"][m]
        a(f"| {m} | {d} (untouched: A/A noise of a bimodal model) | {', '.join(pr['bad_rows'])} | no impact, no GPU time | none |")
    a("")
    a("## Result: one change family at a time (what a single PR would have looked like)")
    a("")
    a("Each row restricts the impact to one kernel family (a PR that touches only that family), forms the strata, picks the representatives "
      "and compares them with the other members, both judged on the stratum's rows only. This is the closer analogue of a per-PR sample, but "
      "the A/B result of a model mixes all the families that drifted in it, so a member's verdict on those rows can come from another family "
      "(a confound that makes disagreement more likely, not less).")
    a("")
    a("| family (rows) | strata rule | strata | touched (data) | reps | non-rep members | non-rep agree | false-pass | false-alarm | rotation agree |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for fam, d in sorted(F.items()):
        rows = ",".join(sorted({r for s_ in d["fine"]["strata"] for r in s_["rows"]}, key=L.ROWS.index)) or "-"
        for g in ("fine", "shape"):
            sm = d[g]["summary"]
            a(f"| {fam} ({rows if len(rows) < 40 else str(len(rows.split(','))) + ' rows'}) | {g} | {sm['strata']} | {sm['touched_models']} ({sm['models_with_data']}) | {sm['reps']} | "
              f"{sm['non_rep_members']} | {cm(sm, 'agree|non-rep')} | {cm(sm, 'false_pass|')} | {cm(sm, 'false_alarm|')} | "
              f"{sm['rotation']['agree']}/{sm['rotation']['pairs']} |")
    a("")
    a("## Runner hours")
    a("")
    a(f"- Full A/B of every model with data (actual phase-3 job times, last successful job per run): {H['full_ab_all_models']:.1f} runner-h; of the "
      f"touched models only (what impact analysis alone saves): {H['full_ab_touched']:.1f} runner-h.")
    a(f"- Representatives (plan rule): {H['reps_full_ab_fine']:.1f} runner-h at full-A/B cost; a T1 canary is about a third of that (affected rows only, "
      f"2+2 sessions, no gates; plan section 6 estimate, not measured): {H['canary_fine']:.1f} runner-h. Under the shape rule {H['canary_shape']:.1f} runner-h.")
    a(f"- So for this drift: impact analysis alone removes {H['full_ab_all_models'] - H['full_ab_touched']:.1f} of {H['full_ab_all_models']:.1f} runner-h "
      f"({100 * (H['full_ab_all_models'] - H['full_ab_touched']) / H['full_ab_all_models']:.0f} %); stratified sampling removes only the few non-representative "
      f"members ({H['full_ab_touched'] - H['reps_full_ab_fine']:.1f} runner-h); the rest of the saving ({H['reps_full_ab_fine'] - H['canary_fine']:.1f} runner-h) "
      "comes from the cheaper canary protocol (rows and sessions), which this replay cannot verify.")
    a("")
    a("## Honest limits")
    a("")
    a("- The bands are the plan's calibrated defaults (`skills/loom-kb/method/methodology.json` was not on `refactor` yet); dropping the kernel families from the key (coarse) changes nothing here because models of one arch/shape drifted in the same families.")
    a("- This is a very different change from a PR. The drift between two packages is ~20 independent kernel changes across 28 models "
      "(flash2 hot loop in 27, decode attention in 12, MoE and GDN in a few, command-program changes). Strata keyed on arch family stay small "
      f"(median 1 member: {sum(1 for s_ in W['fine']['strata'] if len(s_['members']) == 1)} of {len(W['fine']['strata'])} strata are singletons), "
      "so sampling saves little here and agreement is mostly trivial. The informative cases are the "
      f"{W['fine']['summary']['non_rep_members']} non-representative members (fine) and the rotation pairs.")
    a("- The four decode_32k regressions of qwen35moe (drafted hd256 kv2 G8, three DFlash models plus one MTP model) are all found: the DFlash "
      "stratum has three members, all `bad`, and any one of them as representative flags the stratum; the MTP model is its own stratum "
      "(draft kind is part of the signature) and is its own representative. The two undrafted qwen35moe models are `ok` and form a different stratum, as plan Appendix A says.")
    a("- Where the sampler misses or over-alarms: members of one stratum that differ in size. With the shape rule (arch dropped), "
      "in the attention-prefill-only scenario Mellum-12B (ok) is merged with Qwen3-30B-A3B (a small prefill_16k regression, 0.968, below the +-2 % band): "
      "with Qwen3-30B as representative Mellum gets a false alarm, with Mellum as representative the regression would pass "
      "(rotation pairs: false-pass 1, false-alarm 1; with the arch family in the key the two are separate strata). Small regressions just "
      "below a band are exactly what a representative of a heterogeneous stratum can miss (plan section 8); this is why the plan keeps the arch family "
      "in the signature and takes 2 representatives for wide strata.")
    a("- Bimodal small models (Qwen2.5-0.5B, Qwen3-0.6B) have byte-identical programs here (no impact), yet the full test leaves them `suspect` "
      "(mode check cannot decide): impact analysis removes this A/A noise from the PR gate; the full test cannot tell noise from change.")
    a("- Impact prediction: every confirmed bad row is inside the rows predicted from the changed kernels; the only rows outside a prediction belong to the "
      "untouched bimodal models (noise).")
    td = res["text_diff"]
    nr_diff = [m for m in td if any(m in s_["with_data"] and m not in s_["reps"] for s_ in W["fine"]["strata"])]
    a(f"- Text: {len(td)} models have a greedy-text difference in phase 3 ({', '.join(td)}); "
      + ("none of them is a non-representative member, so the text check of a representative is not tested against a sibling here."
         if not nr_diff else f"non-representative members among them: {', '.join(nr_diff)}.")
      + " (Whole-drift replay.)")
    tp = F.get("attn_prefill", {}).get("shape", {}).get("summary", {}).get("text_non_rep", {})
    if any(k.split("|")[0].split("=")[1] != k.split("|")[1].split("=")[1] for k in tp):
        a("- Greedy text does not transfer between siblings the way perf does: in the attention-prefill scenario with the shape rule the non-representative "
          f"members' text results against their representative's are {', '.join(f'{k} x{v}' for k, v in sorted(tp.items()))}. A near-tie flip is a property of one "
          "model, so every model needs its own text check in the T3 rotation; the representative's text check only covers the representative.")
    a("- n is small: 4 to 7 non-representative members and 14 to 22 rotation pairs. This supports 'no contradiction found', not a rate. A stronger test "
      "needs the live per-PR runs the tooling now schedules (T1 representative vs T2/T3 full fleet).")
    a("- Runner hours of the canary protocol are an estimate (a third of the full A/B); the replay measured only job times of the full A/B.")
    a("")
    return "\n".join(L_)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--ab-glob", required=True)
    ap.add_argument("--configs", help="<slug>/llm_config.json directory (default: --head)")
    ap.add_argument("--baselines")
    ap.add_argument("--jobs-json")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--workdir", help="where the normalized view trees go (disk, not /tmp)")
    ap.add_argument("--diff-fleet-dir", help="a directory holding diff_fleet.py when this checkout does not")
    a = ap.parse_args()
    res = run(a.base, a.head, a.ab_glob, a.configs, a.baselines, a.jobs_json, a.workdir, a.diff_fleet_dir)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=list)
    if a.md:
        open(a.md, "w").write(render_md(res))
    for g, v in res["whole"].items():
        print("whole", g, json.dumps(v["summary"]))
    for f, d in res["family"].items():
        for g, v in d.items():
            print("family", f, g, json.dumps(v["summary"]))
    print("hours", res["hours"])
    print("untouched not ok", res["untouched_not_ok"], "\ntouched not ok", res["touched_not_ok"])


if __name__ == "__main__":
    main()
