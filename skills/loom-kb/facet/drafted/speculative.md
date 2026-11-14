# Facet: drafted: speculative (verify-step kernels)

## int8 activations for multi-token and verify matvecs (`ACT_Q8`)

- **Fixed: the multi-token target step was the drafting bottleneck** (issue #21). Decode-shaped linears and the
  head kept f32 activations, so each lane of the 4-token matvec held 4 x 32 f32 values (192+ VGPRs, low occupancy)
  and decoded every weight group to f32: Nemotron-3.5-Bolt-E4B, no drafting, 1/2/4 streams took 21.5 / 26.2 / 35.0 ms
  per step (llama.cpp verifies 3-4 tokens for ~+5 ms), Qwen3.8-27B 78.6 / 90.8 / 131.9 ms. The starter now does what
  llama.cpp's `mul_mat_vec_q` does (`ACT_Q8=1`, the default): `g_quant_q8` quantizes each activation once per
  32-value group (`d = amax/127` rounded to F16, `q = round(x/d)`, llama.cpp's `quantize_q8_1` numerics), launched in
  the decode branch beside the GEMM's `g_row_scale_f16`; the `_q8` matvecs keep the starter's layout (one lane per
  group, R rows, TG tokens) but contract the raw weight codes with `vector.dot4i` (Q8_0, Q4_K, Q5_K, Q6_K; other
  formats keep the f32 kernel) and come in 1-, 4-, 8- and 16-token variants; the head (`g_head_mv*_q8`) in 1-, 4-, 8- and
  16-span variants (each 128-row chunk in sub-blocks of 512/TG rows: see below). A token's result is an exact integer dot followed by the same F32
  scaling, and its wave sum is an explicit xor-16/8/4/2/1 butterfly that the multi-token variants run by recursive
  halving (TG + 5 shuffles instead of 5 TG; Qwen3.8-27B with 4 drafting streams: 254 -> 230 ms per epoch) on the same
  pairing tree, so verification of up to 16 tokens rounds exactly like 1-token decode and verify epochs take the WMMA
  GEMM only from 17 tokens (`SPEC_MM_FROM`, was 5). Each token's integer dot is eight accumulating `dot4` steps
  (no vector reduce), and the 8/16-token variants take 16 rows per workgroup (a row's arithmetic does not depend on R:
  explicit ops and butterfly), which cuts the activation reloads: Qwen3.8-27B with 4 drafting streams at depth 3 went
  254 -> 230 (butterfly) -> 177 ms per epoch (chained dots, R = 16), steady c4 37 -> 51 tok/s, above llama.cpp's own
  drafting (48). Running a 16-token variant as two 8-token chunks in one workgroup was no faster. Nemotron 1/2/4
  streams: 21.4 / 22.5 / 24.6 ms per step; drafted results in the tables above. Kernel microbenchmark (`loom-benchmark` dispatch time, Q6_K 2560 x 40960): f32 1/4 tokens 438 / 643 us,
  int8 427 / 521 us. Quantizing inside the matvec instead (no separate launch) was slower: with the adaptive rows
  per workgroup (R = 1-4) every workgroup re-quantizes the whole activation (Q8_0 4 tokens at R=1: 1343 vs 735 us).
  `ACT_Q8=0` restores the f32 path (numerics of earlier packages). The quantize launches cost a fast model a little:
  LFM2.5-24B-A2B (10 ms steps, no drafting) 10.15 vs 9.75 ms per 1-stream step, but 20.3 vs 21.9 ms with 4 streams;
  big models do not notice (Qwen3.8-27B 1-stream decode 12.9 either way). A loom-optimize lever for small models:
  fuse the quantization into the producing norm / attention-combine kernels.
