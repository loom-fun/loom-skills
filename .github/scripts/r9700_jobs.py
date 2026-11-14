#!/usr/bin/env python3
"""The R9700 per-model measurement job (devlab-perf, runner r9700): write .github/devlab/perf.json for some models.

usage: r9700_jobs.py [--fit VERDICT.json (default .github/devlab/r9700_fit.json)] [--card-org loom-fun-staging|loom-fun] [--build] SLUG... > .github/devlab/perf.json
       r9700_jobs.py --all [...]                every model of .github/nightly-models.json that fits the R9700

One job per model, on ONE R9700 runner, exactly as skills/loom-benchmark/reference/methodology.md requires:
  * Loom: variant "card" (the package as published, drafting as shipped) and variant "nodraft" (draft_depth 0, the 4-stream
    rows); --build rebuilds both from this branch (the dual-target package: use it before the package is on staging).
  * steps: func:card (the functional checks the card claims), ktest:card (the kernel tests, the K_rdna4 twins on gfx12),
    optionally gates:card:LEVEL (--gates dev: golden/consistency against llama.cpp), perf:card:3 + perf:card:3:long (decode, decode_2k, decode_32k,
    prefill 1k/2k/8k/16k/32k), c4:nodraft (4 streams), then llama.cpp ROCm (lemonade gfx120X) and Vulkan (b12211) with the
    package's matched drafting on the same runner: llama@rocm.np1:3 / :3:long, llamac4@rocm.nodraft, llama@vulkan.np1:3 /
    :3:long, llamac4@vulkan.perslot.nodraft.
  * a package whose fit row has "r9700_execution" is served from its `<first>-r9700` preset (model.json, written by make_info.py build --fit);
    the job carries "require_preset_r9700": devlab_perf.sh fails unless the fetched card has that preset, so func, gates, perf and c4
    measure exactly what the card's R9700 Quick start serves.
  * memory budget: what the card's rendered R9700 Quick start line passes as --memory_bytes (card_lib.gpu_memory: the
    package's memory_bytes, at most card_lib.GPUS["radeon-r9700"]["memory_bytes"]; the GPU's whole pool for a reduced context). A model whose native context needs more KV cache than the pool holds after the weights is
    served at card_lib.served_context(...) tokens (variant "context"): the rows beyond it are refused by the server and
    skipped by perf.py (over_context), which the card shows as "doesn't fit". A model whose weights do not fit gets no job.
Ingest with make_info.py perf ... --gpu radeon-r9700 --source-kind devlab (skills/loom-org/SKILL.md step 3).
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
ap.add_argument("slugs", nargs="*", help="card repo names (without the org), e.g. unsloth-Qwen3-0.6B-loom")
ap.add_argument("--all", action="store_true")
ap.add_argument("--fit", default=os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"))
ap.add_argument("--card-org", default="loom-fun-staging")
ap.add_argument("--build", action="store_true", help="rebuild the variants from this branch instead of fetching the card's programs")
ap.add_argument("--gates", default="", help="add the gates step at this LEVEL (dev|fast|quick|full) on the card variant")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

GPU = "radeon-r9700"
POOL = card_lib.GPUS[GPU]["memory_bytes"]
models = {m["repo"].split("/")[-1]: m for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]}
fit = {r["slug"]: r for r in json.load(open(a.fit))}
slugs = sorted(m for m in models if fit.get(m, {}).get("verdict", "").startswith("fits")) if a.all else a.slugs
n = a.reps
jobs = []
skipped = sorted(m for m in models if not fit.get(m, {}).get("verdict", "").startswith("fits")) if a.all else []
for slug in slugs:
    m, f = models[slug], fit[slug]
    if f["w"] + card_lib.SERVE_HEADROOM >= POOL or not f["verdict"].startswith("fits"):
        skipped.append(slug)   # the weights do not fit: no run (the card says "doesn't fit")
        continue
    mem = min(m["memory_bytes"], POOL)
    ctx = None
    ctx = card_lib.fit_served_context(f, POOL)   # the native context's KV cache does not fit (or the row caps it): serve what does
    if ctx:
        mem = POOL
    # a package whose fit row has "r9700_execution" carries the `<first>-r9700` preset (card_lib.gpu_preset_execution): devlab_perf.sh serves it
    # on the R9700 exactly as the card's Quick start does, and its execution already holds the served context: no override is added here
    preset = card_lib.gpu_preset_execution(f, ctx)
    v = lambda name, ex: {"name": name, **({"build": {}} if a.build else {}), **({"context": ctx} if ctx else {}),
                          **({"execution": ex} if ex else {})}
    ex_ctx = {"context_capacity": ctx, "pool_capacity": ctx} if ctx and not preset else {}
    card = {**v("card", ex_ctx)}
    if a.build:
        card.pop("execution", None)
        if preset:
            sys.exit(f"{slug}: --build rebuilds from the default preset's settings, not the R9700 preset's: publish the package and measure the card")
    nodraft = v("nodraft", {**ex_ctx, "draft_depth": 0})
    # llama.cpp ROCm keeps one 65536-token KV pool (devlab_perf.sh); when that plus its ~2.5 GB of buffers and context does not fit the pool
    # beside the weights, shrink it to what fits (at least the 36864 tokens Vulkan uses per slot): "llama_kv_rocm"
    kvt = f["kv"] / f["ctx"]
    lkv = None
    if f["w"] + 65536 * kvt + 2_500_000_000 > POOL:
        lkv = max(36864, min(65536, int((POOL - f["w"] - 2_500_000_000) // kvt) // 4096 * 4096))
    jobs.append({
        **({"llama_kv_rocm": lkv} if lkv else {}),
        "name": slug.lower().replace(".", "-")[:40],
        "card": f"{a.card_org}/{slug}", **({"require_preset_r9700": True} if preset else {}), "kit": m["kit"], "llamacpp": "b1375", "llamacpp_vulkan": "b12211",
        "memory_bytes_r9700": mem, **({"served_context": ctx} if ctx else {}),
        "variants": [card, nodraft],
        "steps": ["func:card", "ktest:card", *([f"gates:card:{a.gates}"] if a.gates else []), f"perf:card:{n}", f"perf:card:{n}:long", "c4:nodraft",
                  f"llama@rocm.np1:{n}", f"llama@rocm.np1:{n}:long", "llamac4@rocm.nodraft",
                  f"llama@vulkan.np1:{n}", f"llama@vulkan.np1:{n}:long", "llamac4@vulkan.perslot.nodraft"]})
if not jobs:
    sys.exit("no job: " + (", ".join(skipped) + " do not fit the R9700" if skipped else "no model given"))
exp = {"_comment": "R9700 per-model measurement (r9700_jobs.py): one job per model on one R9700 runner; "
                   "loom-fun-staging card, llama.cpp ROCm b1375 + Vulkan b12211 with the matched drafting, "
                   "reduced-context models served at served_context. Does not fit: " + (", ".join(skipped) or "none"),
       "runner": "r9700", "memory_bytes": POOL, "llama_spec": "matched", "jobs": jobs,
       "card": jobs[0]["card"], "kit": None, "llamacpp": "b1375", "llamacpp_vulkan": "b12211"}
json.dump(exp, sys.stdout, indent=1)
print()
