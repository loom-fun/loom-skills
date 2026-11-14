# Radeon AI PRO R9700 (gfx1201, 32 GB)

Navi 48, 64 CUs, GDDR6 on a 256-bit bus: 640 GB/s peak, 613 GB/s measured; 191 TFLOP/s dense f16 by spec. Two names: the card's GPU key
is `radeon-r9700`; every measurement runs on the DevLab Dispatch **R9700S** runner (same silicon, 2.35 GHz rocminfo clock, 214 TFLOP/s
measured) whose roofline preset is `radeon-r9700s` (the card uses it: `roofline.default`). Both presets are in `device.json`.

- Pool: the board reports 31.86 GiB; loom-serve gets 34.0 GB (`memory_bytes`). A package that does not fit at its native context is
  served at a reduced `served_context`; one whose weights do not fit is that GPU's `unsupported` state.
- Serving: `--pool_backing=fixed` (`card.serve_args`): the elastic backing's trim + compaction after each session stalls the next long
  prefill; fixed backing commits the preset's `pool_capacity` eagerly.
- Preset: `<first>-r9700` (`preset_suffix`; `build.py --prefill-r9700` = `--prefill-device radeon-r9700=N`, a larger prefill chunk, same programs); fit table
  `.github/devlab/r9700_fit.json`. Runners `[self-hosted, devlab-dispatch, radeon-r9700]`, about 3 concurrent.

Data: `device.json`.

## Roofline calibration (DevLab `radeon-r9700`), 2026-11-13

No Loom port runs on gfx12 yet: hrx-loom 0.1.0 rejects the starter kit's gfx11-shaped WMMA kernels (GEMM, flash
prefill) for `amdgpu-rdna4`. The rows below are llama.cpp on the same runners: they set the `radeon-r9700s` preset's
bandwidth and show what the hardware gives. Ceilings: `--device radeon-r9700s` (613 GB/s, 214 TFLOP/s, both measured).

| Model | Metric | Ceiling | llama.cpp ROCm b1375 (gfx120X) | llama.cpp Vulkan b12211 (RADV) | Eff (best) |
|---|---|---|---|---|---|
| Qwen2.5-32B-Instruct Q4_K_M | decode (tg128) | 30.9 | 28.4 | 29.3 | 0.95 |
| 〃 | prefill 512 | ~1200 | 1091 | 776 | 0.91 (int8 MMQ: not bound by the f16 peak) |
| Qwen3-14B Q4_K_M | decode (tg128) | 70.0 | 59.4 | 63.4 | 0.91 |
| Qwen3-0.6B Q4_K_M | decode (tg128) | ~1500 | 358 | 506 | 0.34 (per-layer launch cost dominates) |
