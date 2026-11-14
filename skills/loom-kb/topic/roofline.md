# Topic: roofline (calibration of achieved efficiency; where the remaining gap usually is)

Ceilings are from `roofline.py` (`--device`, presets in the device registry). `EFF` in `roofline.py` is the best (or a representative) value per class.
The measured calibration rows live with the device they were measured on: `device/strix-halo` (the table the efficiencies were fitted on) and
`device/radeon-r9700`. Add one row per finished optimization campaign there, including ones that missed their target.

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

## Where the remaining gap usually is

- **Prefill GEMM efficiency.** Q4_K GEMMs run ~28 TFLOP/s in the model (~30 on real data in isolation), ~70 %
  of the real-data WMMA ceiling (~40–43 TOPS, see above); gufo (an engine tuned for this chip) reaches ~80 %
  with int8 (W8A8 / K-quant×Q8_1) wave64 GEMMs, which halve LDS and activation traffic, and fused
  norm/SwiGLU→Q8_1 epilogues. int8 WMMA is not faster per instruction here (same ~40 TOPS on real data), so
  the gain is energy: fewer bytes moved per MAC. Tried here and slower on real data: a wave64 port of the f16
  GEMM through transposed fragment loads (27.6 vs 30.5 TFLOP/s) and llama.cpp's mmq formulation
  (`gen_gemm.gemm_q8`: Q8_1 activations x Q4_K codes on int8 WMMA, 20.7; 26.2 even with the per-block scaling
  removed): the f16 kernel's staging pipeline is the better one; an int8 kernel would need gufo's deeper
  pipelining to pay off.
- **Gated delta-net prefill** (hybrids): the register recurrence steps through the chunk token by token
  (Qwen3.8-27B: 0.14 ms/token of 2.3, all GDN kernels 0.27); a chunked (WY / parallel-scan) form is the lever.
- **Small models** (≤ 1B), for example Qwen3-0.6B: decode at 2k context ran 179
  vs 193 tok/s for llama.cpp. There, the attention and launch overhead per layer
  dominates rather than weight bandwidth. Fewer, fused launches per layer, and
  fewer splits when context is short, are the levers.
- **MoE decode** is bound by the expert gather: batch the active experts' matvecs
  per token group.
