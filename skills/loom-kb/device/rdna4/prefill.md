# Device family rdna4: prefill (GEMM twins)

## Contents
- GEMM twins: what bounds them (`gemm_r9700`)
- RDNA4 (gfx12) GEMM twins
- MXFP4 split planes on fp8 WMMA (`gemm_mxfp4s`; v1, the sweeps, the second-generation designs and `lvs`)
- Qwen3.8-27B on the R9700 (GEMM shapes, prefill outside the GEMMs)
- MXFP4 card on the R9700: prefill outside the GEMMs

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

## RDNA4 (gfx12) GEMM twins

The gfx12 twins' kbench findings (what bounds them, what was tried, why int8 WMMA is not the answer) are in `device/rdna4`.

## MXFP4 split planes on fp8 WMMA (R9700; choice `prefill.gemm.mxfp4s_f8`; kbench suite `gemm_mxfp4s`, runs 43769100540 .. 43772576318)
- **Ceiling.** Register-only fp8 WMMA chains (`k_peak8`): 337-347 TFLOP/s with zero operands, 312 with random operands (power-capped, `k_peak8r`; ninfer measured the same effect).
- **The kernel.** `gemm_mxfp4s_f8_m256`: 256 x 128 tiles (swiglu 256 x 64 per weight), 8 waves of 64 rows x 64 columns (swiglu: 32 rows x 64 columns of two weights), K step 128 (four
  blocks), the bf8 decode in LDS (one `v_perm` table lookup per four weights, 2^8 held in the epilogue row factor: the same arithmetic as the ggml-layout kernel of
  #871), the activation fragments loaded from global memory straight into registers (`a_direct`). Quark's nibble order (element 2i low) is `vector.interleave` of the low and
  high nibble vectors; `order="split16"` is ggml's. Measured at 2048 rows (random operands, warm L2, one runner):

  | GEMM (rows 2048) | staged activations (LDS) | a_direct (default) | TFLOP/s a_direct |
  |---|---|---|---|
  | store K 17408 N 5120 (ffn down) | 1.70 ms | 1.53 | 238 |
  | store K 5120 N 10240 (gdn qkv) | 1.04 | 0.93 | 232 |
  | store K 5120 N 12288 (attn q + gate) | 1.26 | 1.10 | 233 |
  | store K 5120 N 6144 (gdn z / ffn alike) | 0.63 | 0.56 | 230 |
  | swiglu K 5120 N 17408 (gate + up) | 3.58 | 3.22 (rf 2) | 227 |
- **What bounds it (ablations on a_direct, 17408 x 5120 store).** No decode ALU (stores of zeros): +0.5 %, the decode is free; no barriers: slower (+10 %: the waves of a workgroup
  desynchronise); activation fragments of K substeps 1..7 reusing substep 0's (global loads vanish): -6 %; weight fragments from LDS reusing substep 0: -5 %; no WMMA at all: 0.91 ms of the
  1.57 (the MMA-only time at the random-data peak is ~1.1-1.2 ms, so the WMMA overlaps the rest by about two thirds).
- **Levers that lost or tied.** Double-buffered weight stage with one barrier per step (`dbuf`): -1..-4 % only (a_direct alone -11 %). K step 256 (`kmult` 2): ties a_direct on store, -4 % on
  swiglu. 8 rows of waves x 32 rows (`rf` 8): +100 % (spills the fragment loads); `rf` 2: ties on store, -9 % on swiglu (so swiglu uses it). 128-thread workgroups (`nt` 128): 4-7 % slower.
  Next-step activation fragments carried in registers (`a_pf`): no gain at rf 2, spills at rf 4.

- **Second sweep (pfgemm, kbench runs 43788662357, 43790112436, 43790618518, 43791938638; rows 1024 = the card's prefill chunk, and 2048; all vs the default kernel, same job).** None beats the default by more than noise (+-2 %):
  - Tile shape: 128 x 256 (8 waves of 64 x 64, 240 VGPR, 36 KB LDS) +9..11 % slower on store, +17 % on swiglu (128 x 128 per weight pair); 256 x 256 (16 waves, 512 threads) +20..26 % store, +16 % swiglu; 128 x 128 with 32-row wave tiles +7 %. Larger tiles lose because fewer workgroups fit a WGP (3 x 8 waves at 218 VGPR), not because of global traffic.
  - Software-pipelined fragment loads (`swp`, the next K substep's A and B fragments loaded before this substep's WMMAs): store +0.2..+2.8 %, swiglu +1.2 % (no gain); with 32-row wave tiles (`rf` 2 + `swp`, 190 VGPR store) -1..-2 % at 2048 rows then -0.9..+1.3 % in a second job: noise.
  - Grid order swapped (output tile fastest, `swz`): +0.2..+2 % store, +4.7 % swiglu at 2048 rows. L2 prefetch of the activation lines two steps ahead (`pfa`, `view.prefetch` read/l2, one scalar prefetch per row of the wave's share): tie (-0.2..+0.5 %).
  - f16 output (`yf16`, halving the output write): store +3 % slower at 2048 rows, ties at 1024: the f32 write is not a bound (the swiglu consumer saving is therefore not worth a producer change).
  - LDS pitch of both stages (`pad` bytes after each 128-value row): 16 (default) and 32 / 80 / 112 tie; 0 is 2.3-4x slower, 8 +25 %, 48 +7 %, 64 +36 %: the pitch sets the bank mapping of the fragment reads, 144 B is already on the good side.
  - Occupancy and wave quantization: at 1024 rows the 17408 x 5120 store has 160 tiles (1.7 x 96 resident workgroups) and still runs at the same TFLOP/s as at 2048 rows (234 vs 238), so the tail is not a loss. Remaining gap to the 312 TFLOP/s random-data peak (76 %): WMMA issue overlaps the staging and fragment traffic only partly (ablations above); every restructuring tried moves the time by under 3 %.
- **Second-generation designs (ogemm, kbench runs 43864345377 .. 43871481153; choice `prefill.gemm.mxfp4s_f8_kernel`).** Read the ISA first (recipe: topic `optimize-loop`, "Local pre-flight").
  The v1 loop per 128-K step: 128 WMMA, ~409 SALU
  (a 64-bit address chain per activation fragment load), 238 waits, every A load waited on right after the barrier.
  - Zero operands make the GEMM 14-17 % faster (register-only WMMA: 19 %); no WMMA at all: 58 % of the time. The kernel is mostly bound by WMMA toggle power at the 300 W cap;
    instruction count alone is not a lever: lane-vector loads (`lv`: one 16-byte load per lane covers two K substeps, A `global_load_b128` with immediate offsets, B `ds_load_b128`;
    both operands take K in the same permuted order, exact) cut SALU 409 -> 175 and waits 238 -> 98 and are 3-7 % SLOWER.
  - **`lvs` (adopted design): `lv` + the next K pair's loads issued before this pair's WMMAs.** Store GEMMs at 1024 rows -4.7..-7.6 % in four jobs (17408 x 5120 0.809 -> 0.766 ms,
    5120 x 10240 0.467 -> 0.444, x 12288 0.569 -> 0.542, x 6144 0.292 -> 0.271, 6144 x 5120 0.288 -> 0.274); the SwiGLU pair keeps v1 (its `lvs` at 64-row waves ties, at 32-row waves it spills).
    On the card (node with the choice set, DevLab 43875399332 / 43879845256): prefill +0.6 / +1.5 / +1.6 / +1.6 / +1.4 % at 1k / 2k / 8k / 16k / 32k, decode tie, ktest and
    golden / compare / spec / agent / vision / sampling gates pass, but the consistency gate FAILS deterministically (twice in one job; the base passes in the same job):
    a 392-token prompt run alone vs alone with 240 prefix-cached tokens picks '."' vs '.' at generated token 19 (reference gap 1.5 nat > the 1-nat near-tie rule).
    The changed summation order inside the fp8 WMMA moves a near-tie of the two (lossy fp8) prefill paths. Not enabled on the node: opt-in only.
  - Lost: wave-specialized producer / consumer (`ws`: 4 decode waves + 8 WMMA waves, double-buffered LDS, one barrier per step) +8..19 % (the VGPR budget is per kernel:
    12 waves x 200 VGPR = 2 workgroups / WGP = 16 WMMA waves instead of 24); 16-wave workgroups with 8 accumulators per wave (122-176 VGPR, 32 waves / WGP) tie or +8..17 %;
    next-step activation prefetch before the barrier (`lvpf`) +1..5 %; K step 256 with `lvs` tie. Rejected from the ISA without a run: `scf.for pipeline(2)` (duplicates the step,
    spills), split barriers `kernel.barrier.arrive/wait` (the compiler sinks the arrive onto the wait and doubles LDS).
  - Mixing a lane-vector operand with a plain fragment load of the other operand is wrong (the K orders differ): the kbench loom-test caught it in one round.

## Qwen3.8-27B on the R9700 (q38r9700 campaign, suite `gemm_q38`, e2e `devlab-perf`)
Prefill GEMM shapes of the 27B (hidden 5120, ffn 17408, gdn qkv 10240, attention q 12288, gate 6144, ssm_out/attn_out 6144 -> 5120), kernels as the model builds them:
- m256 (1024 rows): q4_k store 123-126 TFLOP/s (17408x5120 126, 5120x10240 124, 5120x6144 124), q4_k swiglu 115, q6_k store 115-118, q8_0 store 129. At 2048 rows the same
  kernels run 115-129 TFLOP/s (down 119, qkv 129): no tile-quantization tail at 1024 rows (4 x 40 = 160 workgroups on 64 slots), so a bigger chunk cannot gain GEMM time. m128 is 14-30 % slower than m256 everywhere.
- Dequant bound (`nodq`, zero weights, an upper bound): -13..-24 % (q4_k -14..-24, q6_k -21..-22, q8_0 -13). The generated ISA of the q4_k m256 loop already uses the f16 magic-number
  conversion (and 0x0f0f0f0f, `v_perm` with 0x6400 -> f16 1024+q, `v_pk_add` -1024, `v_pk_fma` scale/minimum): folding the -1024 into the fma bias is not exact in f16
  (1024*s + m needs more than 11 bits; the rounding error is about half a quantization step), so the 3 ops per 2 values stay. Removing the pk_add would be ~2 % of the kernel at best.
- Prefill chunk (`prefill_capacity`, execution override, context 65536 so the bigger chunk fits; 124928 context + chunk 2048 exhausts device memory): 1k/2k/8k/16k/32k tok/s
  card (1024) 1547/1634 (ABBA 1521/1609), 1708, 1645, 1521; 1536: 1462/1628 (1457/1624), 1691, 1634, 1512; 2048: 1454/1627 (1445/1616), 1715, 1651, 1529. 1k loses 5-6 %, 2k ties, 8k-32k +-0.5 %: keep 1024.

### Qwen3.8-27B on the R9700: prefill outside the GEMMs (gdnpf)
Doubling each kernel family in prefill (2k, ms/token of 0.608; single runs, noise +-0.01): GEMMs 0.485 (q4_k store 0.211, swiglu 0.167, q6_k 0.077, q8_0 0.030),
GDN recurrence 0.046, post-add norm 0.013, row scale 0.013, f32 mv (ssm alpha/beta) 0.012, conv 0.011, silu_mul 0.020 (one outlier), gated norm 0.007,
q/k norm 0.004, quant 0.003. Only the recurrence is above the noise: the chunked gfx12 kernel (facet hybrid-gdn prefill) takes it from 0.042 to 0.016
(+4.4 % prefill). The small kernels (together ~0.06) are individually below what a doubling resolves and below what one fusion could save (<= 1 % each).

## MXFP4 card on the R9700: prefill outside the GEMMs (pfrest, DevLab runs 43787206494 .. 43803850202)
- Doubling at 2k (ms/token of 0.381, chunk build): fp8 GEMMs 0.247 (swiglu 0.115, store 0.132), producers 0.032, GDN chunk 0.015, alpha/beta matvecs 0.012, conv + commit 0.012, norm_gate 0.008, qk_norm 0.004,
  split_qgate + rope 0.005, head 0.006, MTP warm 0.005; attention 0.012 at 2k, 0.054 at 16k, 0.105 at 32k. Doubling EVERY launch family is what finds the cost: the first pass named the GEMM, GDN, producer and attention
  families and left 0.045 ms/token (12 %) unexplained (the alpha/beta matvecs and the conv).
- **The producers' f16 operand copy was the biggest lever (+8..10 % prefill, one flag argument).** The fp8 twins of the row-scale producers wrote the f16 GEMM operand next to the fp8 one although the fp8 GEMM never reads it (the
  f16 GEMMs of the epochs whose rows are not a multiple of 256 do). The twin now takes `%w16` (`%huge` ? 0 : 1) and stores the f16 copy only then. A write of an unread copy costs far more than its share of the bytes.
- Chunked GDN (`hybrid.gdn.chunk`) +5..7 % (grouped value heads through `kh_line`). Flash trims (`prefill.attn.flash2_trim`) at head_dim 256: +0.5 / +0.9 / +1.7 % at 8k / 16k / 32k; `FLASH2_VP`, `FLASH2_PF`, trim + pf and 16-key tiles lose or tie.
- Prefill chunk: 1536 / 2048 gain +1..2 % (1k +13 %: the tail epoch of a 1140-token prompt is a 116-row epoch on the f16 m128 path) but cost context (neither fits at 113664); the epoch shapes are 1, 2, 4 .. 1024 (the next power of two of
  the ready tokens, capped at the chunk), so only prompts whose tail is 64..128 rows pay the slow tail.
