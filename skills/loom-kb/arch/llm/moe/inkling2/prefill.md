# Family inkling2: prefill

Strix Halo gfx1151, hrx-loom 0.2.0, FP16 KV, chunk (`prefill_capacity`) 4096. Prefill numerics may change (f16
relative bias, reordered sums); every kept change passed the golden/compare/spec gates.

## Contents

- Where prefill goes
- Kept: attention
- Kept: MoE grouped GEMMs
- Kept: dense, controller and elementwise kernels
- Did not help

## Where prefill goes

- 8k at chunk 4096 (round 2, ~5.9 s busy): MoE grouped GEMMs ~24 TFLOP/s (dequant/barrier bound), dense Q8_0 m256 GEMMs
  ~28, full-attention flash2 ~12.5, sliding flash2 ~10, elementwise kernels at DRAM bandwidth.
- 1k (1,243 tokens, 864 ms TTFT, round 6): MoE GEMMs 49 % (m128 padding alone +98 ms at ~78 tokens per expert), non-GEMM
  kernels 18 %. Token 1 is an invisible `<|thought|>`, so a streaming client's TTFT includes the first decode step (+56 ms).
- The 48-layer compute ceiling is ~4,380 tok/s at 43.8 TFLOP/s (~10 GFLOP per token); shipped 1k-32k reach 1,590-1,860.

## Kept: attention

- Transposed-WMMA flash2 (G = 3, D = 128, window 1024) with the relative bias on the fly, far tiles skipped, per-query
  sinks and the sigmoid gate in the kernel.
- Relative bias as a Toeplitz WMMA (3 WMMAs + a wave-private LDS transpose, stride 17; f16, rel err 3e-4, max|bias|
  1.83) and two 16-token tiles per staged KV tile on sliding layers: sliding flash 697 -> ~520 ms at 8k.
- Whole-tile `mask_v = 0` (masked path only where needed): 4,684 -> 2,668 instructions, sliding 525 -> 404 ms, full 356 ->
  314 ms.
- Full layers past 1024 keys split into FAR (distance >= 1025: no bias, mask or sink; TS = 8, 128 queries, 24 waves, exp2
  domain, lazy rescale) and NEAR (last <= 1,152 keys) with an exact f32 merge: 64k full attention 37.1 -> 10.0 s
  (21.3 TFLOP/s); TTFT 64k 84.8 -> 43.4 s, 200k 792 -> 223 s; 32k prefill +29 %. The FAR kernel's later spill fix is in
  long-context.md.

## Kept: MoE grouped GEMMs

- 128-row expert tiles with an f16-output epilogue (drops a row-scale pass): ggu + gdn 2.5 -> 2.2 s at 8k.
- Split tiles (`MOE_SPLIT_TILE`): >= 65 rows on m128, 1-64 rows on a 64-row variant (27,648 B LDS, 96 VGPR, higher
  occupancy tier): MoE -4.7 %.
- Compact indirect tiles: launch only the needed m128 / s64 tiles with an indirect count from a `view<3xi32>`, inside
  `command.concurrent` (`MOE_SPLIT_128I`, `MOE_CONCURRENT`): with the concurrent tiers, 1k prefill +3.7 %.
- NVFP4 dequant inline in the K loop (README), as a Low helper (`MOE_NVFP4_LOW`) with tighter range assumes
  (`MOE_ACT_A32`: scalar-base addressing, -92 VALU in m128): MoE -2.3 % clock-normalised.
- WMMA bf16 router (`g_gemm_bf16_store_m256` + top-k): 163 -> ~22 ms. The router stays exact bf16 (top-k safety).
- `g_moe_gdn` with a direct f16 output, the down scale in the epilogue and a vec4 f16 combine: combine 183 -> 98 ms, gdn
  758 -> 713 ms, -216 MiB workspace.

## Kept: dense, controller and elementwise kernels

- Loop controller apply as a 64x64 WMMA (channel-fastest, dead tiles skipped), then a branchless vector tanh: 205 -> 115 ms
  at 8k. bf16 projections as 64x32 WMMA on the f16 activations: 149 -> ~15 ms.
- Vec4 shortconv fused with the residual add: 137 -> 39 ms at 8k. Tiled shortconv (3 launches -> 1 on sliding K/V): 1.85x.
- Mix + norm + f16 in one pass; the RMS row scale of a unit-RMS row is exactly 1.0 (|y| <= sqrt(2304) = 48), so the
  concat-RMS output skips the row-scale pass exactly. Live-row skip in the mixes.
- Chunk 4096 over 2048: +6 % (needed the workspace cut in porting.md to keep 1M admission).

## Did not help

- MoE tile shapes: m256 (per-K-step dequant/barrier bound; ggu 10.9 -> 12.1 ms per call; m256i code is 16,180 B, over the
  I-cache), m64 everywhere (+6.5 %), m256 everywhere (+19 %), a 32-row split (31 % occupancy), no LDS padding (2.9x slower,
  16-way bank conflicts), pad 0 with 32 KB LDS.
- Dense GEMM variants: cross-group staging (+16-30 %), early dequant (+3-5 %), double buffering (+8 %), `schedule(phased)`
  m256 (15 of 16 `lgkmcnt(0)` became partial waits, 236 VGPR, +1.3 % slower), m128 x n64 bf16 (32 workgroups < 40 CUs).
- Attention: sliding TS = 4 (+12 %, 62.7 KB LDS) and TS = 1, f16 B in LDS (tie), `vpack`, exp2 and fast variants of the
  sliding kernels (244 / 224 B spills). FAR TS = 4 doubles KV traffic (2x slower). A 768-thread workgroup must fit 6
  waves per SIMD, so it is capped at ~216 VGPR (`AMDGPU/052`). `flash2_swa` and `flash2_ts2` are ~17 KB of code, over the
  16 KiB I-cache.
- Fusing the controller norm into its apply (64 workgroups under-fill 40 CUs and spill: 290 -> 610 ms). Chunk 8192: tie at
  8k, +1.5-3.2 GiB workspace, slower at 64k+.
