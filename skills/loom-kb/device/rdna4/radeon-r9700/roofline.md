# Radeon AI PRO R9700: roofline calibration

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
