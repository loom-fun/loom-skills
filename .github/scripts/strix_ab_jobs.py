#!/usr/bin/env python3
"""The Strix Halo A/B of the staging cards (devlab-perf, runner strix): write .github/devlab/perf.json for some models.

usage: strix_ab_jobs.py [--gates dev] [--reps 3] [--old-org loom-fun] [--new-org loom-fun-staging] SLUG... > .github/devlab/perf.json

One job per model on ONE Strix Halo runner: the published loom-fun package (variant "fun", old) and the loom-fun-staging package
(variant "staging", new) served one after the other with the same prompts and settings (each card's own default preset, the
loom-fun card's --memory_bytes), in an ABBA order so a runner's drift cancels:
  perf:fun:1 perf:staging:1 perf:staging:1 perf:fun:1 ... (N serve sessions each, ABBA), the same with :long
                                                                     decode, decode_2k, decode_32k, prefill 1k/2k/8k/16k/32k (+ acceptance)
  c4:funnd c4:stagingnd c4:stagingnd c4:funnd                        4 streams, undrafted (the card's streams4 rule), 2 sessions x 3 runs each
  textmatch:fun, textmatch:staging                                   greedy text of both (text_fun.json vs text_staging.json)
  func:staging, ktest:staging, gates:staging:LEVEL                  the staging package's functional checks, kernel tests and gates
  (--baselines) llama@rocm.np1 (+long), llamac4@rocm.nodraft, @vulkan  the card's llama.cpp baselines (ROCm card build, Vulkan b12211, matched drafting)
The staging card's Strix Loom numbers are ingested from this job (variant staging / stagingnd, as r9700_jobs.py's card / nodraft).
Compare with .github/scripts/strix_ab_compare.py (new/old per metric, the +-3 % same-runner tie band).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import card_lib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("slugs", nargs="+")
ap.add_argument("--gates", default="dev")
ap.add_argument("--reps", type=int, default=None, help="serve sessions per side for the 1-stream rows (default: 3; 6 for a fast model, which also gets --repeat-rows 3)")
ap.add_argument("--fast-tps", type=float, default=100, help="a model whose published Strix decode is above this gets 6 short-row sessions and 4 c4 sessions per side")
ap.add_argument("--old-org", default="loom-fun")
ap.add_argument("--new-org", default="loom-fun-staging")
ap.add_argument("--baselines", action="store_true", help="also run the llama.cpp ROCm + Vulkan rows (off: the A/B compares Loom with Loom; no llama.cpp re-measure)")
ap.add_argument("--llama-reps", type=int, default=2)
ap.add_argument("--suffix", default="", help="appended to the job name (a confirming re-run: -r2)")
a = ap.parse_args()

models = {m["repo"].split("/")[-1]: m for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]}
jobs = []


def abba(k, start=0):
    """fun, staging, staging, fun, fun, staging ... (k sessions each), from the start-th pair's order"""
    out = []
    for i in range(k):
        out += ["fun", "staging"] if (i + start) % 2 == 0 else ["staging", "fun"]
    return out


for slug in a.slugs:
    m = models[slug]
    old = f"{a.old_org}/{slug}"
    # a fast model's decode rate is bimodal per serve session (Qwen3-0.6B: ~297 or ~330 tok/s, both packages): more sessions per side
    dec = card_lib.published_decode(slug, a.old_org)
    n, rr = card_lib.sampling_plan(dec, a.fast_tps)   # the sampling rule (methodology rule 5): fast: 6 sessions, 3 samples per session
    n = a.reps or n
    nc = 4 if rr > 1 else 2
    name = ("sab-" + slug.lower().replace(".", "-"))[:40 - len(a.suffix)] + a.suffix
    jobs.append({
        "name": name, "runner": "strix", **({"repeat_rows": rr} if rr > 1 else {}), "card": f"{a.new_org}/{slug}", "memory_bytes": m["memory_bytes"],
        "kit": m["kit"], "llamacpp": m.get("llamacpp") or "b1374", "llamacpp_vulkan": "b12211",
        "variants": [{"name": "fun", "card": old}, {"name": "staging"},
                     {"name": "funnd", "card": old, "execution": {"draft_depth": 0}},
                     {"name": "stagingnd", "execution": {"draft_depth": 0}}],
        # every run in its own serve session, interleaved (ABBAAB...): a small model's decode rate is a property of the serve
        # session (Qwen3-0.6B: 297 or 345 tok/s for the same package in two sessions on one runner), so each side's mean
        # spans n sessions and neither side gets all the slow ones
        # fast models: "repeat_rows" 3 (perf.py --repeat-rows: decode_2k, decode_32k, prefill_8k measured 3 times per session; a
        # session's value is the median of its samples), 6 sessions of each kind
        "steps": [f"perf:{v}:1" for v in abba(n)] + [f"perf:{v}:1:long" for v in abba(n, start=1)]
                 + [f"c4:{v}nd" for v in abba(nc)] + ["textmatch:fun", "textmatch:staging",
                  "func:staging", "ktest:staging", *([f"gates:staging:{a.gates}"] if a.gates else []),
                  *([] if not a.baselines else [f"llama@rocm.np1:{a.llama_reps}", f"llama@rocm.np1:{a.llama_reps}:long", "llamac4@rocm.nodraft",
                                               f"llama@vulkan.np1:{a.llama_reps}", f"llama@vulkan.np1:{a.llama_reps}:long", "llamac4@vulkan.perslot.nodraft"])]})
exp = {"_comment": "Strix Halo A/B of the staging cards (strix_ab_jobs.py): per model one job on one Strix runner, "
                   f"{a.old_org} package (fun) vs {a.new_org} package (staging), same prompts and settings; compare with strix_ab_compare.py",
       "runner": "strix", "memory_bytes": jobs[0]["memory_bytes"], "llama_spec": "matched", "jobs": jobs,
       "card": jobs[0]["card"], "kit": None, "llamacpp": jobs[0]["llamacpp"]}
json.dump(exp, sys.stdout, indent=1)
print()
