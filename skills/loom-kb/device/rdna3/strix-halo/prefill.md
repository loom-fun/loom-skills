# Strix Halo: prefill

- **...and through constant data.** Strix Halo's GPU is power-limited: WMMA on constant operands runs
  52–57 TOPS, on random operands ~40 (wave32) / ~43 (wave64), for f16 and int8 alike. A prefill GEMM fed
  `check.generate.fill` constants read 36 TFLOP/s; with the GGUF's real weights and random activations
  30 (the model measured 28). Feed compute-bound benchmarks real weights (`check.file.read.npy`) and
  random activations before trusting a kernel A/B.

## WMMA peak re-measured (kbench `wmma_peak`, 2026-11-19)

Register-only 16x16x16 MMA loop (2560 workgroups x 8 waves x 256 steps x 16 MMAs, no memory traffic in the loop;
`.github/devlab/kbench/wmma_peak/gen_peak.py`), DevLab run 44642983266 and the local Strix Halo, same results within 3 %:

| operands | TFLOP/s (DevLab / local) |
|---|---|
| f16 random (-1..1) | 43.8 / 42.6 |
| f16 all-constant, all-zero; i8 all-constant | 54.5-54.7 / 53.5-54.6 |
| i8 random (-128..127) | 43.9 / 42.9 |
| f16 B = integers 0..15 (raw Q4 codes); f16 B = Q4_K dequant values | +2.6 % / +0.4 % vs f16 random |
| i8 B = 0..15 or -8..7 (W4A8); A -64..63 | +0.3..+1.9 % |
| f16 A and B with 4 mantissa bits | +5..+6 % |

- **Why: a sustained 85 W socket power limit.** Clock and power sampled every 0.1 s on the local Strix Halo, 25 s per
  pattern: after a ~5 s boost at ~100 W, socket power sits at 85.0 W in every case; the GPU clock then holds 2.29 GHz
  on random f16 and 2.76-2.80 GHz on constant/zero data (the 2.9 GHz preset x 40 CUs x 512 = 59.4 TFLOP/s is never
  sustained). The DevLab runner shows the same (power 86-112 W in the boost window, clock 1.8-2.8 GHz).
- Not levers: int8 instead of f16 (same rate), 4-bit-valued weights (W4A8 or raw Q4 codes in f16: <= 3 %), the
  parked vs active completion wait (42.2 vs 42.1 TFLOP/s, same 85 W; the spinning core does not take the GPU's budget
  in this measurement). The only lever on the ceiling itself is the platform power limit (BIOS / SMU, outside Loom).
- Consequence for prefill targets: at 85 W the practical WMMA ceiling on real data is ~43 TFLOP/s = 73 % of the
  roofline's 59 TFLOP/s, so a dense model's end-to-end prefill cannot reach 70 % of `roofline.py`'s prefill ceiling
  (Qwen3.8-27B: 49.7 GFLOP/token -> 881 tok/s at 43.8 TFLOP/s with zero non-GEMM time, 76 % of the 8k ceiling).
