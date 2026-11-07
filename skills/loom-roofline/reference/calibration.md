# Calibration: achieved efficiency vs roofline

One section per device: a ceiling, an achieved rate and an efficiency do not travel between
GPUs. `device.py` says which device a run used; `calibrate_device.py` measures a new one.

## Strix Halo (gfx1151)

Ceilings are from `roofline.py` (216 GB/s, 59 TFLOP/s). Achieved numbers are
end-to-end and client-side, from tuned Loom ports. `EFF` in `roofline.py` is the
best (or a representative) value per class. Add one row per finished optimization
campaign, including ones that missed their target.

| Model (arch, quant) | Metric | Ceiling | Loom achieved | Eff | llama.cpp ROCm | Note |
|---|---|---|---|---|---|---|
| Nemotron-3.5-Bolt-E4B (gemma4 dense, Q4_K_M) | decode | 68.1 | 61.1 | 0.90 | 52.7 | adaptive matvec rows, fused QKV, paired-row geglu |
| 〃 | decode @1k | 66.8 | 59.2 | 0.89 | 50.6 | |
| 〃 | decode @2k | 66.2 | 57.9 | 0.87 | 48.7 | |
| 〃 | prefill 1k | 7146 | 1830 | 0.26 | 1593 | head_dim 256/512: no flash2, simple split attention |
| 〃 | prefill 2k | 7094 | 1801 | 0.25 | 1678 | |
| 〃 | c4 steady | 255 | 165 | 0.65 | 157 | |
| Bolt-E4B, run 2 (fresh agent, 1 turn, 1 h budget) | decode | 68.1 | 60.0 | 0.88 | 52.7 (b1355) | adaptive rows only; fused QKV / paired-row not tried |
| 〃 | prefill 1k / 2k | 7146 / 7094 | 2201 / 2339 | 0.31 / 0.33 | 1593 / 1678 | new flash prefill kernel for head_dim 256/512 (~30 min to build) |
| 〃 | c4 steady | 255 | 169 | 0.66 | 157 | |
| LFM2.5-24B-A2B (lfm2moe hybrid: 30 short-conv + 10 attention layers, MoE 64×4, Q4_K_M; fresh agent, 1 turn, 74 min) | decode | 140 | 100.4 | 0.72 | 97.3 (b1355) | MoE decode below 0.78; the roofline step was skipped, so no targets steered the run |
| 〃 | prefill 1k / 2k | 13208 / 12970 | 2677 / 2744 | 0.20 / 0.21 | 2140 / 2213 | prefill chunk 4096: every chunk reads all 64 experts |
| 〃 | c4 steady | 204 | 185.7 | 0.91 | 158.9 | |
| Qwen2.5-0.5B-Instruct (qwen2 dense, Q4_K_M/Q5_0; fresh agent, 1 turn, 35 min, campaign tools used) | decode | 551 (target 377 with per-layer floor) | 289 | 0.52 | 288 (b1355) | small model: per-layer launch cost dominates |
| 〃 | prefill 1k / 2k / 16k | 77769 / 73596 / – | 17571 / 18571 / 11920 | 0.23 / 0.25 / 0.28 | 13851 / 13890 / 8202 | |
| Flint (MoE 256×6, int4 q4_1) | decode | 130 | 99–104 | 0.78 | 62 | expert gather bound |
| 〃 | decode @32k | 82.7 | 63 | 0.76 | 48 | |
| 〃 | prefill 2k | 11946 | ~2650 | 0.22 | 1509 | |
| 〃 | prefill 32k | 7041 | 1976 | 0.28 | 1047 | |
| 〃 | c4 steady | 244 | 181 | 0.74 | 152 | |
| Muse-Glimmer-30B (dense, Q5_K_M) | decode | – | 13.1 | – | 12.4 | |
| 〃 | prefill 2k / 32k | – | ~430 / 420 | – | 355 / 359 | |
| Qwen3-0.6B (dense, Q8_0; starter, no tuning) | decode / decode @2k | – | 252 / 179 | – | 253 / 193 | per-layer launch and attention cost dominate at 2k |
| 〃 | prefill 2k | – | 13.0k | – | 10.5k | |
| 〃 | c4 steady | – | 690 | – | 578 | |

Observations:
- Dense decode reaches about 0.9 of the bandwidth roofline. The last 10 % is the
  per-layer small kernels plus attention.
- Prefill end to end is ~0.25–0.33 of WMMA peak, even though the GEMMs alone reach ~0.5.
  Attention, norms and dispatch make up the rest. A target above ~0.3 needs new
  GEMM work (LDS double-buffering, dequant into fragments).
- These efficiencies were fitted on the same two models, so targets for a new
  architecture are an out-of-sample prediction. Record the result here.

## Speculative decoding (`EFF_DRAFTED`)

Against `drafted_ceiling()`: the bandwidth ceiling of one verify pass of depth + 1 positions (MoE: their distinct
experts) x (1 + acceptance x depth) tokens per pass, the drafter's own passes not counted. Published loom-fun
cards, DFlash depth 3, local session of 2026-11-06:

| Model | Metric | Ceiling | Loom | Eff | Acceptance |
|---|---|---|---|---|---|
| Qwen3.8-27B (dense, Q4_K_M) | decode | 42.4 | 29.2 | 0.69 | 0.76 |
| 〃 | c4 steady | 168.0 | 61.7 | 0.37 | |
| Muse-Glimmer-30B (dense, UD-Q5_K_M) | decode | 38.1 | 25.8 | 0.68 | 0.73 |
| 〃 | c4 steady | 151.4 | 68.8 | 0.45 | |
| Bolt-E4B (dense, Q4_K_M) | decode | 205.1 | 97.6 | 0.48 | 0.67 |
| 〃 | c4 steady | 768.2 | 219.1 | 0.29 | |
| Flint (MoE, Q4_K_M) | decode | 177.8 | 92.4 | 0.52 | 0.68 |
| 〃 | c4 steady | 301.8 | 134.2 | 0.44 | |

Small models sit lower: the drafter and the verify pass's fixed costs are a larger share of a short pass.

## Strix Point (gfx1150)

Radeon 890M (16 CU), 2x DDR5-5600, 61 GiB RAM, hrx-loom 0.1.0 from PyPI, llama.cpp ROCm
b1369 gfx1150 asset. Ceilings are `roofline.py` with the fitted profile (70 GB/s,
23.6 TFLOP/s): `bw` and the per-layer floor come from `calibrate_device.py` over
Qwen3-0.6B-Q4_K_M and Qwen3.5-0.8B-Q4_K_M (12 points, RMS 4%), not from a datasheet.
Achieved rates are `perf.py` rows from **untuned starter ports** - no optimization campaign
has run on this device, so these are the floor to beat, not the best seen.

| Model (arch, quant) | Metric | Ceiling | Loom achieved | Eff | llama.cpp ROCm | Note |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (qwen3 dense, Q4_K_M; starter, no tuning) | decode | 179.1 | 141.3 | 0.79 | 134.4 | the gap is the launch floor, not bandwidth |
| 〃 | decode @2k | 112.9 | 85.0 | 0.75 | 92.0 | below llama.cpp: the KV read is where the starter loses here |
| 〃 | prefill 1k | 23707 | 5978 | 0.25 | 3909 | |
| 〃 | prefill 2k | 21258 | 5480 | 0.26 | 3545 | |
| Qwen3.5-0.8B (gated-delta-net hybrid + vision, Q4_K_M; starter) | decode | 134.2 | 115.6 | 0.86 | 98.5 * | |
| 〃 | decode @2k | 128.2 | 104.1 | 0.81 | - | |
| 〃 | decode @32k | 76.5 | 70.4 | 0.92 | - | long context is bandwidth-clean; the floor is amortized |

`*` the hybrid's llama.cpp figures are `llama-bench` pp256/tg64, not `perf.py`'s workloads, so
that row is indicative only; the Qwen3-0.6B rows are same-workload on both engines.

Observations:
- Dense decode reaches 0.75-0.79 of the ceiling where Halo reaches ~0.9, and the shortfall is
  accounted for: the per-layer launch floor measures 58 us/layer here against Halo's 35-40, so
  a 390 MB step is ~5.6 ms of weights plus ~1.6 ms of launches and sync - 23 % of the step,
  which is what the efficiency column is missing. On the gated-delta-net hybrid the floor is
  ~126 us/layer (3.0 ms of a 9.3 ms step), because its recurrent layers launch more per step.
  Fusing launches outranks byte-cutting on this part more often than the Halo catalog implies.
- Prefill lands at 0.25-0.26 of the 23.6 TFLOP/s peak, inside Halo's reported 0.23-0.33. That
  agreement is what justifies scaling the peak by CU count (16/40); prefill on its own only
  proves >= 6.2 TFLOP/s measured.
- Bandwidth fell 3.1x from Halo while compute fell 2.5x, so the ridge point moves from
  273 to 492 FLOP/byte: work that was compute-bound there is memory-bound here. Treat the
  catalog's tile and chunk sizes as choices to re-derive.
- Ceilings taken from Halo constants are 3.1x too high on this GGUF (552.8 vs 179.1 tok/s) and
  shrink the end-to-end budget for a 0.6B from 55 min to 44, which would stop the pipeline
  before it finished its own gates.
