#!/usr/bin/env python3
"""The Strix Halo A/B of the staging cards (devlab-perf, runner strix): write .github/devlab/perf.json for some models.

usage: strix_ab_jobs.py [--gates dev] [--reps 3] [--old-org loom-fun] [--new-org loom-fun-staging] SLUG... > .github/devlab/perf.json

One job per model on ONE Strix Halo runner: the published loom-fun package (variant "fun", old) and the loom-fun-staging package
(variant "staging", new) served one after the other with the same prompts and settings (each card's own default preset, the
loom-fun card's --memory_bytes), in an ABBA order so a runner's drift cancels:
  perf:fun:N, perf:staging:N, perf:staging:N:long, perf:fun:N:long   decode, decode_2k, decode_32k, prefill 1k/2k/8k/16k/32k (+ acceptance)
  c4:funnd, c4:stagingnd                                             4 streams, undrafted (the card's streams4 rule)
  textmatch:fun, textmatch:staging                                   greedy text of both (text_fun.json vs text_staging.json)
  func:staging, ktest:staging, gates:staging:LEVEL                  the staging package's functional checks, kernel tests and gates
  llama@rocm.np1 (+long), llamac4@rocm.nodraft, the same @vulkan     the card's llama.cpp baselines (ROCm card build, Vulkan b12211, matched drafting)
The staging card's Strix section is ingested from this job (variant staging / stagingnd, as r9700_jobs.py's card / nodraft).
Compare with .github/scripts/strix_ab_compare.py (new/old per metric, the +-3 % same-runner tie band).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

ap = argparse.ArgumentParser()
ap.add_argument("slugs", nargs="+")
ap.add_argument("--gates", default="dev")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--old-org", default="loom-fun")
ap.add_argument("--new-org", default="loom-fun-staging")
ap.add_argument("--no-baselines", action="store_true", help="skip the llama.cpp ROCm + Vulkan rows (the card's baselines)")
ap.add_argument("--llama-reps", type=int, default=2)
ap.add_argument("--suffix", default="", help="appended to the job name (a confirming re-run: -r2)")
a = ap.parse_args()

models = {m["repo"].split("/")[-1]: m for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]}
n = a.reps
jobs = []
for slug in a.slugs:
    m = models[slug]
    old = f"{a.old_org}/{slug}"
    name = ("sab-" + slug.lower().replace(".", "-"))[:40 - len(a.suffix)] + a.suffix
    jobs.append({
        "name": name, "runner": "strix", "card": f"{a.new_org}/{slug}", "memory_bytes": m["memory_bytes"],
        "kit": m["kit"], "llamacpp": m.get("llamacpp") or "b1374", "llamacpp_vulkan": "b12211",
        "variants": [{"name": "fun", "card": old}, {"name": "staging"},
                     {"name": "funnd", "card": old, "execution": {"draft_depth": 0}},
                     {"name": "stagingnd", "execution": {"draft_depth": 0}}],
        "steps": [f"perf:fun:{n}", f"perf:staging:{n}", f"perf:staging:{n}:long", f"perf:fun:{n}:long",
                  "c4:funnd", "c4:stagingnd", "textmatch:fun", "textmatch:staging",
                  "func:staging", "ktest:staging", *([f"gates:staging:{a.gates}"] if a.gates else []),
                  *([] if a.no_baselines else [f"llama@rocm.np1:{a.llama_reps}", f"llama@rocm.np1:{a.llama_reps}:long", "llamac4@rocm.nodraft",
                                               f"llama@vulkan.np1:{a.llama_reps}", f"llama@vulkan.np1:{a.llama_reps}:long", "llamac4@vulkan.perslot.nodraft"])]})
exp = {"_comment": "Strix Halo A/B of the staging cards (strix_ab_jobs.py): per model one job on one Strix runner, "
                   f"{a.old_org} package (fun) vs {a.new_org} package (staging), same prompts and settings; compare with strix_ab_compare.py",
       "runner": "strix", "memory_bytes": jobs[0]["memory_bytes"], "llama_spec": "matched", "jobs": jobs,
       "card": jobs[0]["card"], "kit": None, "llamacpp": jobs[0]["llamacpp"]}
json.dump(exp, sys.stdout, indent=1)
print()
