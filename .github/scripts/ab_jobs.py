#!/usr/bin/env python3
"""Same-runner A/B jobs for the devlab-perf workflow, on any registered device: write .github/devlab/perf.json for some models.

usage: ab_jobs.py --device DEVICE (any registry device with a runner: strix-halo, radeon-r9700, ...) [--old-org ORG] [--new-org ORG] [--old-rev REV] [--new-rev REV] [--refs MAP.json]
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
--device with a memory pool (radeon-r9700): the device's runner, both variants served as the NEW card's Quick start for it serves it
  (device_jobs.py, r9700_jobs.py): the fit row (--fit, default the device's fit table, device.json "fit_table") gives the <first>-r9700 preset (job "require_preset_r9700"), the served context of a
  reduced-context model and the llama.cpp ROCm KV size; the old package, which may carry no such preset, is served with the same
  execution settings (variant "execution" + "cap_pool" = what the fixed pool backing can hold); --pool_backing=fixed and the memory clamp are
  devlab_perf.sh's. A model whose weights do not fit gets no job.

Ingest the NEW package's card rows from one job (JOB = the downloaded artifact dir, OUT = the package dir holding the card to update, RUN/URL = the
run and the job page; the new variant's files are perf_staging.jsonl / c4_stagingnd.txt, the baselines' the same as r9700_jobs.py's):
  S=skills/loom-benchmark/scripts/card
  pull_devlab.py --identity $JOB --gpu GPU --device-out dev.json --hrx-out hrx.json
  make_info.py perf --gpu GPU --loom-jsonl $JOB/perf_staging.jsonl --llama-jsonl $JOB/perf_llamacpp-rocm.np1_n*.jsonl --c4-loom $JOB/c4_stagingnd.txt \
      --c4-llama $JOB/c4_llamacpp-rocm.nodraft_n0.txt --package $OUT --source-kind devlab --run RUN --url URL --device dev.json --hrx-loom hrx.json \
      --baseline-version llamacpp-rocm/b1375 --baseline-args "$(cat $JOB/llamacpp_args-rocm.np1.txt)" --fetched-programs "$(cat $JOB/programs_sha256.txt)" -o ev_perf.json
  make_info.py perf --gpu GPU --backend vulkan --llama-jsonl $JOB/perf_llamacpp-vulkan.np1_n*.jsonl --c4-llama $JOB/c4_llamacpp-vulkan.perslot.nodraft_n0.txt \
      --baseline-version b12211 --baseline-args "$(cat $JOB/llamacpp_args-vulkan.np1.txt)" --baseline-sha256 "$(cat $JOB/llamacpp_release-vulkan.np1.sha256)" \
      --fetched-programs "$(cat $JOB/programs_sha256.txt)" --package $OUT --source-kind devlab --run RUN --url URL --device none --hrx-loom none -o ev_vulkan.json
  make_info.py gates --gpu GPU --time JOB_END_UTC $JOB/gates_staging/gates.json $JOB/ktest_staging.log --package $OUT --source-kind devlab --run RUN --url URL \
      --device dev.json --hrx-loom hrx.json -o ev_gates.json
  make_info.py roofline --gpu GPU --device ROOFLINE_PRESET --package $OUT --evidence ev_perf.json -o ev_roof.json
  make_info.py build $OUT --org ORG --gpus strix-halo,radeon-r9700 --fit .github/devlab/r9700_fit.json --memory-bytes N --from-info info_prev.json \
      --evidence ev_perf.json ev_vulkan.json ev_gates.json ev_roof.json -o info_new.json      (then cp to $OUT/info.json; render_card.py $OUT; --check)
  The job's card_revision.txt names the new package's commit the run measured (the --package dir must hold that revision: --fetched-programs refuses a mismatch).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import card_lib  # noqa: E402
sys.path.insert(0, HERE)
import device_jobs  # noqa: E402

# the devices an A/B can run on: the device registry's devices with a DevLab runner (skills/loom-kb: device.json runner, methodology.json
# baselines per device); a device without a runner or baselines cannot be measured
DEVICES = {}
for _k, _dv in card_lib.registry.devices().items():
    _b = card_lib.registry.methodology()["baselines"].get(_k)
    if _dv["runner"]["labels"] and _b and "rocm" in _b:
        DEVICES[_k] = {"runner": _dv["runner"]["key"], "baselines": _b,
                       "prefix": _dv["runner"].get("ab_prefix") or _dv["runner"]["key"] + "ab-",   # job names: sab-<slug> (strix), rab-<slug> (r9700)
                       "llamacpp": _b["rocm"].get("fallback") or _b["rocm"]["build"],   # the ROCm build when nightly-models.json names none
                       "vulkan": _b["vulkan"]["build"]}


def abba(k, start=0):
    """fun, staging, staging, fun, fun, staging ... (k sessions each), from the start-th pair's order"""
    out = []
    for i in range(k):
        out += ["fun", "staging"] if (i + start) % 2 == 0 else ["staging", "fun"]
    return out


def ref(org, slug, rev=None):
    return f"{org}/{slug}" + (f"@{rev}" if rev else "")


def r9700_fields(slug, f, m, pool, gpu="radeon-r9700"):
    """What device_jobs.py derives from the fit row f of a model on a device with the memory POOL: (job fields, new-variant execution,
    old-variant fields, served context) or None when the weights do not fit. The new variants are exactly device_jobs' card / nodraft."""
    got = device_jobs.device_fields(gpu, f, m, pool)
    return got and got[:4]


def build(a):
    dev = DEVICES[a.device]
    refs = json.load(open(a.refs)) if a.refs else {}
    models = {m["repo"].split("/")[-1]: m for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]}
    pool = card_lib.GPUS[a.device]["memory_bytes"]
    r9700 = pool is not None   # a device with its own memory pool (the R9700's path: fit table, device preset); Strix Halo takes the package's
    NAME = card_lib.GPUS[a.device]["runner"]
    if r9700:
        fit = {r["slug"]: r for r in json.load(open(a.fit or os.path.join(ROOT, card_lib.registry.fit_table(a.device))))}
    jobs, skipped = [], []
    for slug in a.slugs:
        m = models[slug]
        old = (refs.get(slug) or {}).get("old") or ref(a.old_org, slug, a.old_rev)
        new = (refs.get(slug) or {}).get("new") or ref(a.new_org, slug, a.new_rev)
        # a fast model's decode rate is bimodal per serve session (Qwen3-0.6B: ~297 or ~330 tok/s, both packages): more sessions per side
        dec = card_lib.published_decode(slug, a.old_org)   # the published Strix decode (--old-org, never a pinned ref's org: a staging card may carry no Strix number)
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
                "kit": m["kit"], "llamacpp": m.get("llamacpp") or dev["llamacpp"], "llamacpp_vulkan": dev["vulkan"],
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
        got = r9700_fields(slug, fit[slug], m, pool, a.device)
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
            **{k: v for k, v in jf.items() if k != "llama_kv_rocm"}, "kit": m["kit"], "llamacpp": dev["llamacpp"], "llamacpp_vulkan": dev["vulkan"],
            "variants": [ov("fun", {}), nv("staging", ex_ctx), ov("funnd", {"draft_depth": 0}), nv("stagingnd", {**ex_ctx, "draft_depth": 0})],
            "steps": steps})
    if not jobs:
        sys.exit("no job: " + (", ".join(skipped) + f" do not fit the {NAME}" if skipped else "no model given"))
    if r9700:
        return {"_comment": f"{NAME} A/B (ab_jobs.py): per model one job on one {NAME} runner, old package (fun) vs new package (staging), served as the new "
                            f"card's {NAME} Quick start, same prompts and settings; compare with strix_ab_compare.py. Does not fit: {', '.join(skipped) or 'none'}",
                "runner": dev["runner"], "memory_bytes": pool, "llama_spec": "matched", "jobs": jobs,
                "card": jobs[0]["card"], "kit": None, "llamacpp": dev["llamacpp"], "llamacpp_vulkan": dev["vulkan"]}
    return {"_comment": "Strix Halo A/B of the staging cards (strix_ab_jobs.py): per model one job on one Strix runner, "
                        f"{a.old_org} package (fun) vs {a.new_org} package (staging), same prompts and settings; compare with strix_ab_compare.py",
            "runner": dev["runner"], "memory_bytes": jobs[0]["memory_bytes"], "llama_spec": "matched", "jobs": jobs,
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
    ap.add_argument("--fit", default=None, help="a device with a memory pool (radeon-r9700): its fit table (default: device.json fit_table)")
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
