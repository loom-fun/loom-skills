#!/usr/bin/env python3
"""Same-runner A/B jobs for the devlab-perf workflow, on any registered device: write .github/devlab/perf.json for some models.

usage: ab_jobs.py --device strix-halo|radeon-r9700 [--old-org ORG] [--new-org ORG] [--old-rev REV] [--new-rev REV] [--refs MAP.json]
                  [--baselines] [--gates dev] [--reps N] [--suffix -r2] SLUG... > .github/devlab/perf.json

One job per model on ONE runner of the device: the old package (variant "fun") and the new package (variant "staging") served one
after the other with the same prompts and settings, in an ABBA order so a runner's drift cancels:
  perf:fun:1 perf:staging:1 perf:staging:1 perf:fun:1 ... (N serve sessions each, ABBA), the same with :long
                                                                     decode, decode_2k, decode_32k, prefill 1k/2k/8k/16k/32k (+ acceptance)
  c4:funnd c4:stagingnd c4:stagingnd c4:funnd                        4 streams, undrafted (the card's streams4 rule)
  textmatch:fun, textmatch:staging                                   greedy text of both (text_fun.json vs text_staging.json)
  func:staging, ktest:staging, gates:staging:LEVEL                  the new package's functional checks, kernel tests and gates
  (--baselines) llama@rocm.np1 (+long), llamac4@rocm.nodraft, @vulkan  the card's llama.cpp baselines on the same runner, matched drafting
Variant names stay fun (old) and staging (new), so strix_ab_compare.py reads every job unchanged.

Which package is "old" / "new": ORG/SLUG at main, or pinned to a card revision ORG/SLUG@REV (devlab_perf.sh fetches that commit and
writes card_revision.txt / card_revision_fun.txt). --old-rev / --new-rev pin every slug; --refs MAP.json pins per slug:
  {"SLUG": {"old": "org/repo@rev", "new": "org/repo@rev"}}   (either key may be missing: the --old-org / --new-org default applies)
An A/A run (a package against itself, e.g. old = staging@REV, new = staging at main) measures the runner's own noise.

--device strix-halo: the Strix Halo runner (strix_ab_jobs.py is a wrapper with these defaults), the old card's --memory_bytes.
--device radeon-r9700: the R9700 runner, both variants served as the NEW card's R9700 Quick start serves it (r9700_jobs.py): the fit row
  (--fit, default .github/devlab/r9700_fit.json) gives the <first>-r9700 preset (job "require_preset_r9700"), the served context of a
  reduced-context model and the llama.cpp ROCm KV size; the old package, which may carry no such preset, is served with the same
  execution settings (variant "execution" + "cap_pool" = what the fixed pool backing can hold); --pool_backing=fixed and the memory clamp are
  devlab_perf.sh's. A model whose weights do not fit gets no job.
Ingest the NEW package's card rows from one job: skills/loom-benchmark/reference/ab_ingest.md (notes/REGEN_RECIPE.md "measure + ingest").
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import card_lib  # noqa: E402

# device -> the devlab-perf runner key, the job-name prefix, the llama.cpp ROCm build when nightly-models.json names none
DEVICES = {"strix-halo": {"runner": "strix", "prefix": "sab-", "llamacpp": "b1374"},
           "radeon-r9700": {"runner": "r9700", "prefix": "rab-", "llamacpp": "b1375"}}


def abba(k, start=0):
    """fun, staging, staging, fun, fun, staging ... (k sessions each), from the start-th pair's order"""
    out = []
    for i in range(k):
        out += ["fun", "staging"] if (i + start) % 2 == 0 else ["staging", "fun"]
    return out


def ref(org, slug, rev=None):
    return f"{org}/{slug}" + (f"@{rev}" if rev else "")


def r9700_fields(slug, f, m, pool):
    """What r9700_jobs.py derives from the fit row f of a model: (job fields, new-variant execution, old-variant fields, memory) or None
    when the weights do not fit. The new variants are exactly r9700_jobs' card / nodraft."""
    if f["w"] + card_lib.SERVE_HEADROOM >= pool or not f["verdict"].startswith("fits"):
        return None
    mem = min(m["memory_bytes"], pool)
    ctx = card_lib.fit_served_context(f, pool)   # the native context's KV cache does not fit (or the row caps it): serve what does
    if ctx:
        mem = pool
    preset = card_lib.gpu_preset_execution(f, ctx)
    ex_ctx = {"context_capacity": ctx, "pool_capacity": ctx} if ctx and not preset else {}
    # llama.cpp ROCm keeps one 65536-token KV pool; shrink it when that plus ~2.5 GB of buffers does not fit beside the weights
    lkv = None
    kvt = f["kv"] / f["ctx"]
    if f["w"] + 65536 * kvt + 2_500_000_000 > pool:
        lkv = max(36864, min(65536, int((pool - f["w"] - 2_500_000_000) // kvt) // 4096 * 4096))
    job = {**({"llama_kv_rocm": lkv} if lkv else {}), **({"require_preset_r9700": True} if preset else {}),
           "memory_bytes_r9700": mem, **({"served_context": ctx} if ctx else {})}
    # the old package carries no <first>-r9700 preset: the same settings as an execution override (its pool shrunk to what fixed backing holds)
    old = {**({"context": ctx} if ctx else {}), "execution": preset or ex_ctx, "cap_pool": card_lib.fixed_pool_capacity(f)}
    return job, ex_ctx, old, ctx


def build(a):
    dev = DEVICES[a.device]
    refs = json.load(open(a.refs)) if a.refs else {}
    models = {m["repo"].split("/")[-1]: m for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]}
    r9700 = a.device == "radeon-r9700"
    if r9700:
        pool = card_lib.GPUS["radeon-r9700"]["memory_bytes"]
        fit = {r["slug"]: r for r in json.load(open(a.fit))}
    jobs, skipped = [], []
    for slug in a.slugs:
        m = models[slug]
        old = (refs.get(slug) or {}).get("old") or ref(a.old_org, slug, a.old_rev)
        new = (refs.get(slug) or {}).get("new") or ref(a.new_org, slug, a.new_rev)
        # a fast model's decode rate is bimodal per serve session (Qwen3-0.6B: ~297 or ~330 tok/s, both packages): more sessions per side
        dec = card_lib.published_decode(slug, old.split("/")[0])
        n, rr = card_lib.sampling_plan(dec, a.fast_tps)   # the sampling rule (methodology rule 5): fast: 6 sessions, 3 samples per session
        n = a.reps or n
        nc = 4 if rr > 1 else 2
        name = (dev["prefix"] + slug.lower().replace(".", "-"))[:40 - len(a.suffix)] + a.suffix
        lr = a.llama_reps or (n if r9700 else 2)
        steps = ([f"perf:{v}:1" for v in abba(n)] + [f"perf:{v}:1:long" for v in abba(n, start=1)]
                 + [f"c4:{v}nd" for v in abba(nc)] + ["textmatch:fun", "textmatch:staging",
                  "func:staging", "ktest:staging", *([f"gates:staging:{a.gates}"] if a.gates else []),
                  *([] if not a.baselines else [f"llama@rocm.np1:{lr}", f"llama@rocm.np1:{lr}:long", "llamac4@rocm.nodraft",
                                               f"llama@vulkan.np1:{lr}", f"llama@vulkan.np1:{lr}:long", "llamac4@vulkan.perslot.nodraft"])])
        if not r9700:
            jobs.append({
                "name": name, "runner": dev["runner"], **({"repeat_rows": rr} if rr > 1 else {}), "card": new, "memory_bytes": m["memory_bytes"],
                "kit": m["kit"], "llamacpp": m.get("llamacpp") or dev["llamacpp"], "llamacpp_vulkan": "b12211",
                "variants": [{"name": "fun", "card": old}, {"name": "staging"},
                             {"name": "funnd", "card": old, "execution": {"draft_depth": 0}},
                             {"name": "stagingnd", "execution": {"draft_depth": 0}}],
                # every run in its own serve session, interleaved (ABBAAB...): a small model's decode rate is a property of the serve
                # session (Qwen3-0.6B: 297 or 345 tok/s for the same package in two sessions on one runner), so each side's median
                # spans n sessions and neither side gets all the slow ones
                # fast models: "repeat_rows" 3 (perf.py --repeat-rows: decode_2k, decode_32k, prefill_8k measured 3 times per session; a
                # session's value is the median of its samples), 6 sessions of each kind
                "steps": steps})
            continue
        got = r9700_fields(slug, fit[slug], m, pool)
        if not got:
            skipped.append(slug)   # the weights do not fit: no run (the card says "doesn't fit")
            continue
        jf, ex_ctx, o, ctx = got
        nv = lambda vname, ex: {"name": vname, **({"context": ctx} if ctx else {}), **({"execution": ex} if ex else {})}
        ov = lambda vname, extra: {"name": vname, "card": old, **{k: v for k, v in o.items() if k != "execution"},
                                   "execution": {**o["execution"], **extra}}
        jobs.append({
            **{k: v for k, v in jf.items() if k == "llama_kv_rocm"},
            "name": name, "runner": dev["runner"], **({"repeat_rows": rr} if rr > 1 else {}), "card": new,
            **{k: v for k, v in jf.items() if k != "llama_kv_rocm"}, "kit": m["kit"], "llamacpp": dev["llamacpp"], "llamacpp_vulkan": "b12211",
            "variants": [ov("fun", {}), nv("staging", ex_ctx), ov("funnd", {"draft_depth": 0}), nv("stagingnd", {**ex_ctx, "draft_depth": 0})],
            "steps": steps})
    if not jobs:
        sys.exit("no job: " + (", ".join(skipped) + " do not fit the R9700" if skipped else "no model given"))
    if r9700:
        return {"_comment": f"R9700 A/B (ab_jobs.py): per model one job on one R9700 runner, old package (fun) vs new package (staging), served as the new "
                            f"card's R9700 Quick start, same prompts and settings; compare with strix_ab_compare.py. Does not fit: {', '.join(skipped) or 'none'}",
                "runner": "r9700", "memory_bytes": pool, "llama_spec": "matched", "jobs": jobs,
                "card": jobs[0]["card"], "kit": None, "llamacpp": dev["llamacpp"], "llamacpp_vulkan": "b12211"}
    return {"_comment": "Strix Halo A/B of the staging cards (strix_ab_jobs.py): per model one job on one Strix runner, "
                        f"{a.old_org} package (fun) vs {a.new_org} package (staging), same prompts and settings; compare with strix_ab_compare.py",
            "runner": "strix", "memory_bytes": jobs[0]["memory_bytes"], "llama_spec": "matched", "jobs": jobs,
            "card": jobs[0]["card"], "kit": None, "llamacpp": jobs[0]["llamacpp"]}


def parser(device=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slugs", nargs="+")
    if device is None:
        ap.add_argument("--device", choices=sorted(DEVICES), required=True)
    else:
        ap.set_defaults(device=device)
    ap.add_argument("--gates", default="dev")
    ap.add_argument("--reps", type=int, default=None, help="serve sessions per side for the 1-stream rows (default: 3; 6 for a fast model, which also gets --repeat-rows 3)")
    ap.add_argument("--fast-tps", type=float, default=100, help="a model whose published Strix decode is above this gets 6 short-row sessions and 4 c4 sessions per side")
    ap.add_argument("--old-org", default="loom-fun")
    ap.add_argument("--new-org", default="loom-fun-staging")
    ap.add_argument("--old-rev", help="pin the old card of every slug to this revision (ORG/SLUG@REV)")
    ap.add_argument("--new-rev", help="pin the new card of every slug to this revision")
    ap.add_argument("--refs", help='JSON {"SLUG": {"old": "org/repo@rev", "new": "org/repo@rev"}}: per-slug card references (win over the org/rev options)')
    ap.add_argument("--fit", default=os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"), help="radeon-r9700: the fit table")
    ap.add_argument("--baselines", action="store_true", help="also run the llama.cpp ROCm + Vulkan rows (off: the A/B compares Loom with Loom; no llama.cpp re-measure)")
    ap.add_argument("--llama-reps", type=int, default=None, help="llama.cpp runs per 1-stream row (default: 2 on Strix Halo; on the R9700 the Loom session count, as r9700_jobs.py)")
    ap.add_argument("--suffix", default="", help="appended to the job name (a confirming re-run: -r2)")
    return ap


def main(argv=None, device=None):
    a = parser(device).parse_args(argv)
    json.dump(build(a), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
