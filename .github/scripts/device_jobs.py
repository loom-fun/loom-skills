#!/usr/bin/env python3
"""The per-device per-model measurement job (devlab-perf, the device's runner): write .github/devlab/perf.json for some models.

usage: device_jobs.py --device DEVICE [--fit TABLE.json (default: the device's fit table, skills/loom-kb device.json "fit_table")]
                      [--card-org loom-fun-staging|loom-fun] [--build] SLUG... > .github/devlab/perf.json
       device_jobs.py --device DEVICE --all [...]        every model of .github/nightly-models.json that fits the device
       r9700_jobs.py [...]                                the same with --device radeon-r9700 (the original entry point, same bytes)

Any registry device (skills/loom-kb/device/<family>/<device>/device.json) with a DevLab runner, a memory pool (memory_bytes), a fit table
(.github/scripts/fit.py --device D) and llama.cpp baselines (method/methodology.json "baselines") takes this generator; the device
supplies the runner key, the memory pool, the serve flags (pool backing), the preset suffix and the llama.cpp builds. Below, "the device" is
the R9700 in the original wording; the R9700's jobs are byte-identical to what r9700_jobs.py always wrote.

One job per model, on ONE runner of the device, exactly as skills/loom-benchmark/reference/methodology.md requires:
  * Loom: variant "card" (the package as published, drafting as shipped) and variant "nodraft" (draft_depth 0, the 4-stream
    rows); --build rebuilds both from this branch (the dual-target package: use it before the package is on staging).
  * steps: func:card (the functional checks the card claims), ktest:card (the kernel tests, the K_rdna4 twins on gfx12),
    optionally gates:card:LEVEL (--gates dev: golden/consistency against llama.cpp), perf:card:3 + perf:card:3:long (decode, decode_2k, decode_32k,
    prefill 1k/2k/8k/16k/32k), c4:nodraft (4 streams), then llama.cpp ROCm and Vulkan (the device's pinned builds, methodology.json "baselines")
    with the package's matched drafting on the same runner: llama@rocm.np1:3 / :3:long, llamac4@rocm.nodraft, llama@vulkan.np1:3 /
    :3:long, llamac4@vulkan.perslot.nodraft.
  * a package whose fit row has "<runner key>_execution" (r9700: "r9700_execution") is served from its `<first><preset suffix>` preset
    (model.json, written by make_info.py build --fit); the job carries "require_preset_<runner key>": devlab_perf.sh fails unless the fetched
    card has that preset, so func, gates, perf and c4 measure exactly what the card's Quick start for the device serves.
  * serving: the device is served as the card's Quick start serves it (card_lib.GPUS serve_args: --pool_backing=fixed on the R9700 and Strix Halo, which
    devlab_perf.sh adds), whose KV pool is the preset's pool_capacity (make_info build --fit caps it to the GPU). Below, the --memory_bytes
    budget is the llama.cpp sizing and the elastic variant's: what the card's rendered Quick start line once passed as --memory_bytes
    (card_lib.gpu_memory: the package's memory_bytes, at most the device's memory_bytes; the GPU's whole pool for a reduced context). A model whose
    native context needs more KV cache than the pool holds after the weights is served at card_lib.served_context(...) tokens (variant
    "context"): the rows beyond it are refused by the server and skipped by perf.py (over_context), which the card shows as "doesn't fit". A
    model whose weights do not fit gets no job.
  * --preset full (default) | iterate | final: .github/scripts/job_presets.py (iterate: no gates, smoke instead of func, no baseline engines; final: func + ktest +
    gates once on the card variant, no perf).
Ingest with make_info.py perf ... --gpu DEVICE --source-kind devlab (skills/loom-fleet/reference/publish.md step 3).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import card_lib  # noqa: E402
import matched_spec  # noqa: E402
import job_presets  # noqa: E402

registry = card_lib.registry


VLLM_MAX_LEN = 40960   # vLLM's --max-model-len for the baseline rows: the longest row is a 32k document + its answer; the model's native 262144 does not fit a 32 GB GPU's KV cache


def key_of(gpu):
    """The runner key of a device (device.json runner.key): the suffix of the job fields memory_bytes_<key> and require_preset_<key>."""
    return registry.device(gpu)["runner"]["key"]


def job_devices():
    """The devices a measurement job can run on: a runner, a memory pool and baselines (a device without one is not measurable)."""
    base = registry.methodology()["baselines"]
    return [k for k, d in registry.devices().items() if d["runner"]["labels"] and d.get("memory_bytes") and k in base and "rocm" in base[k]]


def device_fields(gpu, f, m, pool):
    """What a job derives from the fit row f of model m on a device with the memory POOL, or None when the weights do not fit:
    (job fields, execution of the new variants, old-package variant fields, served context). The new variants are the "card" / "nodraft"
    variants of device_jobs; ab_jobs.py uses the same plan."""
    if f["w"] + card_lib.SERVE_HEADROOM >= pool or not f["verdict"].startswith("fits"):
        return None
    k = key_of(gpu)
    mem = min(m["memory_bytes"], pool)
    ctx = card_lib.fit_served_context(f, pool)   # the native context's KV cache does not fit (or the row caps it): serve what does
    if ctx:
        mem = pool
    # a package whose fit row has "<key>_execution" carries the device's preset (card_lib.gpu_preset_execution): devlab_perf.sh serves it
    # on the device exactly as the card's Quick start does, and its execution already holds the served context: no override is added here
    preset = card_lib.gpu_preset_execution(f, ctx, gpu)
    ex_ctx = {"context_capacity": ctx, "pool_capacity": ctx} if ctx and not preset else {}
    # llama.cpp ROCm keeps one 65536-token KV pool (devlab_perf.sh); when that plus its ~2.5 GB of buffers and context does not fit the pool
    # beside the weights, shrink it to what fits (at least the 36864 tokens Vulkan uses per slot): "llama_kv_rocm"
    kvt = f["kv"] / f["ctx"]
    lkv = None
    if f["w"] + 65536 * kvt + 2_500_000_000 > pool:
        lkv = max(36864, min(65536, int((pool - f["w"] - 2_500_000_000) // kvt) // 4096 * 4096))
    job = {**({"llama_kv_rocm": lkv} if lkv else {}), **({f"require_preset_{k}": True} if preset else {}),
           f"memory_bytes_{k}": mem, **({"served_context": ctx} if ctx else {})}
    # the old package of an A/B carries no device preset: the same settings as an execution override (its pool shrunk to what fixed backing holds)
    old = {**({"context": ctx} if ctx else {}), "execution": preset or ex_ctx, "cap_pool": card_lib.fixed_pool_capacity(f, gpu=gpu)}
    return job, ex_ctx, old, ctx, preset


def reference_only(slug):
    """A safetensors package (fleet/<slug>/llm_config.json source.kind): llama.cpp cannot read the checkpoint, so its job has no llama.cpp
    baseline steps (vLLM ROCm is its baseline engine: the vllm: steps once vLLM is the reference) and the gates step runs only where the vLLM
    reference engine exists (vllm_shim.py). None when the slug is a GGUF card, else (has the vLLM reference)."""
    try:
        cfg = json.load(open(os.path.join(ROOT, "fleet", slug, "llm_config.json")))
    except OSError:
        return None
    if (cfg.get("source") or {}).get("kind") != "safetensors":
        return None
    return os.path.exists(os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "vllm_shim.py"))


def not_a_target(slug, gpu):
    """Why the package is not measured on GPU (a weight format without a fast path on its family: card_lib.non_targets), or None."""
    try:
        cfg = json.load(open(os.path.join(ROOT, "fleet", slug, "llm_config.json")))
    except OSError:
        return None
    return card_lib.non_targets(cfg).get(gpu)


def main(argv=None, device=None, tool="device_jobs.py"):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slugs", nargs="*", help="card repo names (without the org), e.g. unsloth-Qwen3-0.6B-loom")
    if device is None:
        ap.add_argument("--device", required=True, choices=job_devices())
    else:
        ap.set_defaults(device=device)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fit", default=None, help="the device's fit table (default: device.json fit_table)")
    ap.add_argument("--card-org", default="loom-fun-staging")
    ap.add_argument("--build", action="store_true", help="rebuild the variants from this branch instead of fetching the card's programs")
    ap.add_argument("--gates", default="", help="add the gates step at this LEVEL (dev|fast|quick|full) on the card variant")
    ap.add_argument("--preset", choices=job_presets.PRESETS, default="full",
                    help="step list: full (default), iterate (no gates, smoke instead of func, no baseline engines: the loop of an optimization), final (func + ktest + gates once on the card variant)")
    ap.add_argument("--reps", type=int, default=None, help="serve sessions (= llama.cpp runs) per 1-stream row: default 3, 6 for a fast model (methodology rule 5)")
    ap.add_argument("--fast-tps", type=float, default=card_lib.FAST_DECODE_TPS, help="a model whose published loom-fun (Strix) decode is above this is fast: 6 sessions, --repeat-rows 3")
    ap.add_argument("--old-org", default="loom-fun", help="the org whose published cards give each model's decode rate")
    a = ap.parse_args(argv)

    GPU = a.device
    if GPU not in job_devices():
        sys.exit(f"{GPU}: no measurement job (needs a DevLab runner, a memory pool, baselines in methodology.json): " + ", ".join(job_devices()))
    POOL = card_lib.GPUS[GPU]["memory_bytes"]
    DEV = registry.device(GPU)   # runner key and llama.cpp baselines: skills/loom-kb (device.json, methodology.json)
    K = key_of(GPU)
    NAME = card_lib.GPUS[GPU]["runner"]   # "R9700": the card's short name of the device
    BASE = registry.methodology()["baselines"][GPU]
    fit_path = a.fit or os.path.join(ROOT, registry.fit_table(GPU))
    if not os.path.exists(fit_path):
        sys.exit(f"{fit_path}: no fit table for {GPU} (fit.py --device {GPU} writes it)")
    models = {m["repo"].split("/")[-1]: m for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]}
    fit = card_lib.fit_rows(fit_path, GPU, ROOT)   # sizing + verdict from the table, the serving settings from the scope rules (exec.*)
    slugs = sorted(m for m in models if fit.get(m, {}).get("verdict", "").startswith("fits")) if a.all else a.slugs
    jobs = []
    skipped = sorted(m for m in models if not fit.get(m, {}).get("verdict", "").startswith("fits")) if a.all else []
    for slug in slugs:
        m, f = models[slug], fit[slug]
        why = not_a_target(slug, GPU)
        if why:
            skipped.append(slug)   # a format without a fast path on this device: the card says "not a target"
            print(f"{slug}: not a target on {NAME}: {why}", file=sys.stderr)
            continue
        got = device_fields(GPU, f, m, POOL)
        if not got:
            skipped.append(slug)   # the weights do not fit: no run (the card says "doesn't fit")
            continue
        jf, ex_ctx, _, ctx, preset = got
        n, rr = card_lib.sampling_plan(card_lib.published_decode(slug, a.old_org), a.fast_tps)   # the one sampling rule, Loom and llama.cpp alike
        n = a.reps or n
        v = lambda name, ex: {"name": name, **({"build": {}} if a.build else {}), **({"context": ctx} if ctx else {}),
                              **({"execution": ex} if ex else {})}
        card = {**v("card", ex_ctx)}
        if a.build:
            card.pop("execution", None)
            if preset:
                sys.exit(f"{slug}: --build rebuilds from the default preset's settings, not the {NAME} preset's: publish the package and measure the card")
        nodraft = v("nodraft", {**ex_ctx, "draft_depth": 0})
        st = reference_only(slug)   # None: a GGUF card (llama.cpp baselines); else a safetensors card: Loom rows only, gates where vLLM is the reference
        base = {} if st is not None else {"llamacpp": BASE["rocm"]["build"], "llamacpp_vulkan": BASE["vulkan"]["build"]}
        llama = [] if st is not None else [f"llama@rocm.np1:{n}", f"llama@rocm.np1:{n}:long", "llamac4@rocm.nodraft",
                                           f"llama@vulkan.np1:{n}", f"llama@vulkan.np1:{n}:long", "llamac4@vulkan.perslot.nodraft"]
        gates = [f"gates:card:{a.gates}"] if a.gates and (st is None or st) else []
        if st:   # vLLM ROCm is the baseline engine of a safetensors card, where the reference engine's install helper exists: same runner, out of the box
            llama = [f"vllm:{n}", f"vllm:{n}:long", f"vllmc4:{n}"]
        jobs.append({
            **({"llama_kv_rocm": jf["llama_kv_rocm"]} if "llama_kv_rocm" in jf and st is None else {}),
            **({"vllm_spec": spec} if st and (spec := matched_spec.vllm_spec(f"loom-fun/{slug}")) else {}),
            **({"vllm_max_len": VLLM_MAX_LEN} if st else {}),
            "name": slug.lower().replace(".", "-")[:40], **({"repeat_rows": rr} if rr > 1 else {}),
            "card": f"{a.card_org}/{slug}", **({f"require_preset_{K}": True} if preset else {}), "kit": m["kit"], **base,
            f"memory_bytes_{K}": jf[f"memory_bytes_{K}"], **({"served_context": ctx} if ctx else {}),
            "variants": [card, nodraft],
            "steps": job_presets.apply(a.preset, ["func:card", "ktest:card", *gates, *["perf:card:1"] * n, *["perf:card:1:long"] * n, "c4:nodraft", *llama], "card",
                                       a.gates or "dev")})
        if a.preset == "final":
            job_presets.trim_variants(jobs[-1])
    if not jobs:
        sys.exit("no job: " + (", ".join(skipped) + f" do not fit the {NAME}" if skipped else "no model given"))
    exp = {"_comment": f"{NAME} per-model measurement ({tool}): one job per model on one {NAME} runner; "
                       f"loom-fun-staging card, llama.cpp ROCm {BASE['rocm']['build']} + Vulkan {BASE['vulkan']['build']} with the matched drafting, "
                       "reduced-context models served at served_context. Does not fit: " + (", ".join(skipped) or "none"),
           "runner": DEV["runner"]["key"], "memory_bytes": POOL, "llama_spec": "matched", "jobs": jobs,
           "card": jobs[0]["card"], "kit": None, "llamacpp": BASE["rocm"]["build"], "llamacpp_vulkan": BASE["vulkan"]["build"]}
    if all("llamacpp" not in j for j in jobs):   # only safetensors cards: no llama.cpp anywhere in the experiment
        for k in ("llama_spec", "llamacpp", "llamacpp_vulkan"):
            exp.pop(k)
    json.dump(exp, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
