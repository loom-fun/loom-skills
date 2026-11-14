# RDNA4 family (gfx12-generic, `@rdna4`)

Wave 32, WMMA 16x16 fragments with 8-wide operand vectors (`vector<8xf16>`: RDNA4 halves the fragment registers and doubles the dense
f16 rate per CU). A kernel with a gfx12 version `K` -> `K_rdna4` selects it per device (`dual_target.py`). The twins are text
rewrites of the RDNA3 kernels (`gen_gemm_rdna4.twin()`, `flash2_rdna4.to_rdna4()`, MoE inline in `gen_moe.py`); tuning that only
exists here: `flash2_b2`, `trim`, `vpack`, `prefetch`, `f16o_fuse`. Devices: `radeon-r9700` (gfx1201).

Data: `family.json`.

## GEMM twins: what bounds them (`gemm_r9700` kbench suite)

The gfx12 twins (`gen_gemm_rdna4.py`) run the q4_k m256 kernels at 95-125 TFLOP/s on a 214 TFLOP/s register-only WMMA loop
(`k_peak`); q6_k m256 is within 5 % of q4_k, bf16 (no dequant) is *slower* (0.96 ms vs 0.84 ms at 1024x12288x4096: 3.5x the weight bytes).
Everything below is `devlab-kbench` (suite `gemm_r9700`, runner r9700, kernels exactly as the models build them: `GEMM_SKIP_PADDING`
and `GEMM_N_FASTEST` on) unless marked e2e (`devlab-perf`, same-job llama.cpp ROCm).
- **WMMA and VALU do not overlap on RDNA4.** `k_mix<N>`: 8 independent WMMAs per iteration plus N extra `v_pk_fma_f16` chains
  of 4 instructions: 5.24 ms (N=0), 9.47 (8), 12.5 (16), 18.5 (32), 30.6 ms (64): every additional VALU instruction costs
  the same wherever it sits (about 0.14 of a WMMA each), nothing hides under the matrix unit. A GEMM with ~230 non-WMMA instructions per 64
  WMMAs per K step is therefore capped near 60 % of peak, which is what the kernels reach.
- **Ablations (m256 store, 1024x12288x4096, 0.86 ms = 119 TFLOP/s):** no dequant (weights stage zeros) -12..-18 % (an upper bound: zero operands
  also draw less power), no weight staging stores -11 %, no barriers -2..-5 %, no activation staging stores/loads 0 (hidden), no WMMA and
  no fragment reads -55 %. The non-WMMA half of the kernel is dequant plus LDS traffic and is spread thin; no single piece is worth 15 %.
- **Tried, kernel level (same run, 3 repeats): all within +-3 %.**
  - `dq2`: a thread stages the 16-value half of *both* nibble groups of its page (one 16-byte load instead of two 32-byte ones, one scale/min
    decode per group instead of two): -9 % instructions per step; store/swiglu +2..-2 %; e2e (E2B at its R9700 preset, Qwen3-8B) +-1 %.
  - `sb`: dq2 with the K loop unrolled by one superblock (constant groups, header once per four steps, prefetched a superblock ahead):
    -31 % non-WMMA instructions per step (330 -> 227 besides the 64 WMMAs), store kernel -1..-7 % (three runs: -2.5, -6.6, -7, -1.5), swiglu kernel 0..-2 %. **In a model the swiglu kernel is 3x slower** (E2B 2k prefill -42 %, Qwen3-8B -47 %; the store kernel -3 %); the same kernel text
    is at parity in kbench with real Qwen3-0.6B weights, weights beyond the 64 MB MALL, 1k/2k/4k rows and the models' skip-padding wrapper, so
    the cause was not found (a fully unrolled short K loop after launch-argument specialization is the open suspect). Not for models.
  - packed f16 scale/min decode (`sc`), branch-free activation prefetch (`nb`): instruction counts -6 %, time +-5 %; with `nb`/`sc` the swiglu
    kernel spills (892 B scratch, 2.5x slower).
  - Fewer instructions are not the whole story: -31 % instructions gave ~-3 % time. The mix numbers above say VALU is serial with WMMA, but the
    instructions that disappear are the cheap ones (address math); the conversions (`v_perm`, `v_pk_add`, `v_pk_fma`: 3 per 2 weights),
    `ds_load_b128`/`ds_store_b128` and the barriers remain.
- **Why int8 WMMA (llama.cpp's MMQ) is not the answer here.** The 32-value-block fixup (convert the i32 accumulators, apply
  activation scale x weight scale, accumulate) costs three VALU instructions per accumulator element per 32 K: 384 per wave per 32 K for the
  64 x 64 wave tile, more than the f16 path's whole dequant (~90 per 64 K), and the WMMA does not overlap it (see above). Same conclusion as the
  gfx11 `gemm_q8` (20.7 vs 30 TFLOP/s). Pre-dequantized f16 weights do not pay either (bf16 row above).
- **Tile quantization at E2B sizes.** m256 store at N=1536 (o_proj, down) has 4 x 12 = 48 workgroups at 1k rows for 32 WGPs x 2 slots:
  96 TFLOP/s vs 115-122 for the N >= 6144 shapes. A 256 x 192 tile (12 waves, 64.5 KB LDS, N=1536 = 8 tiles) would fill the slots at 2k
  rows; ~2-3 % of E2B prefill, not built (N not divisible by 192 elsewhere needs predicated edges).
- **Where E2B's gap to llama.cpp ROCm is.** Whole-prefill matmul rate (2 x params x tokens / time): Qwen3-8B Loom 85 vs llama.cpp 62
  TFLOP/s-equivalent (Loom +37 %: GEMM-bound, our GEMMs are faster than theirs); Gemma-4-E2B (2.3 B matmul params, 35 layers) Loom 65 vs llama.cpp 72.
  E2B spends 28 % of its time outside q4_k/q6_k GEMMs (flash 12 %, norm/quant/row-scale 10 %, f32 GEMMs 4 %, PLE 2 %; optgemm's doubling profile
  at 2k), 8B ~8 %. Closing E2B 2k/8k (-10/-13 %) needs those kernels (fusion of norm + row scale + quant, f32 PLE GEMMs, SWA flash), not the q4_k GEMM.
