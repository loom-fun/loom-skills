# Radeon RX 7800 XT (gfx1101, 16 GB)

Navi 32, 60 CUs, GDDR6 on a 256-bit bus: **624 GB/s peak**, **37.3 TFLOP/s** dense f16 WMMA (AMD spec). Family `rdna3` (gfx11-generic): the programs are the base target's, byte-identical to the Strix Halo ones. This row adds **no new kernels** — only the device entry, roofline preset, runner and (pending) fit table.

## Measured (this session, 2026-11-18)

First measured discrete RDNA3 desktop in the fleet: no DevLab runner is known for this device, so `runner.labels` is empty and the numbers below come from a named machine (soulafein83, CachyOS, Linux 7.3.4, ROCm 10.1 AUR, hrx-loom 0.2.0 PyPI).

- **Activation:** `amdf-info --device=rdna --activate` → `Activation: OK` (PM4, AQL, SDMA queues).
- **Gemma-4-E4B-it** (`chat-16gb` preset, 6.0 GB weights, ctx 73728, rows 4, draft_depth 3): `READY` in 5 s, smoke `finish_reason: stop` („Stolicą Francji jest Paryż."), VRAM 6.01 GB peak, no OOM, no swap.
- **Golden vs llama.cpp:** `PASS` — 23 prompts, 16 full match, median 39, max gap 0.205, `bad=0`.
- **Spec:** 16/16 completions identical, 0 near-tie, drafter acceptance 42% (5106 proposals).
- **Sampling:** temperature 0 == greedy, top_k 1 == greedy, same seed reproduces.
- **Context admission:** 73 653 / 73 728 tokens (99.9%), 31 s.
- **Decode:** 170.5 tok/s (82.6% of the 206.4 tok/s bandwidth ceiling); prefill 1k/2k/8k/16k/32k = 3934/4482/4227/3947/3270 tok/s; 4 streams 329.1 tok/s aggregate; acceptance 29–42%.
- **llama.cpp ROCm b1380 baseline:** decode 102.5 tok/s, prefill 1k/2k = 2684/2825 tok/s (no speculative decoding — MTP context creation fails in that build for this model).
- **VRAM:** 6.01 GB peak with Gemma-4-E4B; 10.36 GB with Nemotron-3.5-Bolt-E4B; 7.92 GB with Qwen3.5-9B; 8.22 GB with Gemma-4-12B; 12.02 GB with GPT-OSS-20B. All under the 15 GB budget.
- **Four additional models** served without changes: Nemotron-3.5-Bolt-E4B (4B), Qwen3.5-9B, Gemma-4-12B, GPT-OSS-20B (MoE) — all `READY`, smoke `stop`, math `391`.

## Caveats

- **No DevLab runner:** `runner.labels` is empty. Results come from a single named machine and are not part of the automated fleet measurements. A card shows this GPU only for a model whose entry has data.
- **`amdf-info` reports only a `system` memory pool** (16.5 GiB RAM) for this device; `Locality: unavailable`. There is no enumerated device-local/VRAM pool. The budget is an explicit `--memory_bytes` argument, not derived from an enumerated pool. Serving still places weights in VRAM (verified with `amd-smi`: 6.01 GB used).
- **`memcheck.sh` refuses budgets above 60% of host RAM** (9.2 GiB on a 16 GiB machine) — a Strix-Halo assumption. On a discrete GPU the budget is VRAM, so `LOOM_MEMORY_CAP_GIB=17` (or equivalent) is required. Documented in the port plan.
- **Roofline bandwidth/flops are SPEC** (624 GB/s, 37.3 TFLOP/s), not measured. A DevLab measurement would replace them (Strix: 216 of 256 peak).
- **`fit_table` is null** and **`prefill_flag` is null**: no per-model fit table yet. The fit tooling is R9700-specific (see `loom-fleet/reference/campaign.md`).

## Preset

`<first>-rx7800xt` (`preset_suffix`), only created when a package needs serving settings that differ from the default. The `chat-16gb` preset from Gemma-4-E4B (ctx 73728, rows 4, draft_depth 3) runs unchanged.

## Roofline

`roofline.py --device radeon-rx7900xtx --bw 624000000000 --flops 37000000000000`: decode ceiling 206.4 tok/s, prefill 73728 ceiling 2976.6 tok/s, c4 ceiling 770.6 tok/s. Loom reaches 82.6% of the decode ceiling.

## Notes for the PR

- Family: `rdna3` (gfx11-generic): the programs are the base target's, byte-identical to the Strix Halo ones. This row adds no kernels.
- The `fleet/devices/` entry and card integration are left to the maintainer's tooling (`make_info.py`); raw evidence (gates, compare, perf, amdf-info, status, logs) is attached to this PR.
- `runner.labels` is empty: results come from a named machine, not a DevLab runner. Data: `device.json`.