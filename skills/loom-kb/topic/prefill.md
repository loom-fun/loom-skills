# Topic: prefill (compute- and attention-bound prompt processing)

Levers for the WMMA GEMMs, flash attention, chunk size and fusions. Numbers come from the Flint (40B MoE) and Muse Glimmer (30B dense) ports on Strix Halo (RDNA3.5, gfx1151) against llama.cpp ROCm b1355 unless a lever says otherwise. Kernels are in `loom-port-llm/starter/gen_llm.py` / `gen_gemm.py` / `gen_flash2*.py`.

## Contents

- WMMA GEMM (`gen_gemm.py`)
- Q5_K / Q8_0 dequant (`gen_gemm.py`)
- Q6_K dequant (`gen_gemm.py`)
- flash2 (`gen_flash2.py`)
- flash2 at head_dim 256 and 512 (`gen_flash2s.py`)
- Last-layer output rows (`LAST_OUT`, llama.cpp `inp_out_ids`)
- Listed fallback
- Per-token RoPE + KV store (`qk_rope()`)
- Fusions
- Tile order per port (`cfg["gemm_n_fastest"]`)
- Settled for starter ports (do not re-run these A/Bs)
- RDNA4 (gfx12) GEMM twins: what bounds them (`gemm_r9700` kbench suite)
- Did not help (prefill)

## WMMA GEMM (`gen_gemm.py`)

- **Row-scaled f16 inputs.** `g_row_scale_f16` writes x/scale as f16 together with
  the per-row scale. The GEMM dequantizes weights to f16 fragments and accumulates
  in f32. The output is multiplied by the row scale (store, acc or swiglu modes).
- **Tiles.** 64, 128 or 256 rows × N tile: m128 from 128 tokens, m256 (`gemm_big256`) when the epoch's rows
  are a multiple of 256 (`GEMM_M256=0` turns it off).
- **m256: dequant per MMA halved.** On RDNA3 WMMA runs on the SIMD's VALU, so the weight dequant
  serializes with it: m128 spends ~190 VALU instructions per 32 WMMAs. m256 is the m128 layout with twice the
  rows (512 threads, 256 × 128 tile; swiglu interleaves 64 gate + 64 up rows in one weight stage, picking the
  buffer per wave with `scf.select` on `buffer`), so each dequantized weight slice feeds 256 rows (~115 VALU per
  32 WMMAs). 55 KB LDS (padded rows), 2 workgroups per WGP. Real-data microbenchmark (interleaved
  `loom-benchmark --compare`), Qwen3.8-27B shapes at 1024 rows: down +9.5 %, gate +5.4 %, gate/up swiglu
  +4.5 %; prefill 1k/2k/8k +5.8/+5.0/+6.0 % (this includes the GDN input prefetch below; the recurrence
  ablation puts that share near zero).
  - Tried: gufo's 256 × 256 tile with 1024 threads. LDS can't hold both padded operands (73 KB), so the
    stages were K-slice-major unpadded (`[4 x 256 rows] x 16`): 22.8 TFLOP/s vs 30.8 (one workgroup per WGP,
    2-way LDS read conflicts). A half-swap swizzle to remove the conflicts made the compiler keep one address
    register per fragment half (VGPR 184 → >192, did not fit 1024 threads; the 512-thread swizzled form ran
    256 VGPRs and 19-22 TFLOP/s). Moving m128's 8 KB inverse-scale tile into the activation stage (3 instead
    of 2 workgroups per WGP) was mixed: down -4 %, swiglu/gate +1.5/+2 %; not kept.
  - Tile order (`GEMM_N_FASTEST`) makes no difference for m256 on the 27B shapes.
- **m256 with 64 × 64 waves.** `gemm_big256` now runs 256 threads (8 waves, 4 × 2) of 64 rows × 64 columns
  instead of 16 waves of 32 × 64: each A fragment feeds 4 B fragments and each B fragment 4 A fragments (0.5
  LDS fragment reads per MMA instead of 0.75), VALU per WMMA 4.5 → 3.0 (store loop), 232-248 VGPRs (two
  workgroups per WGP still fit). Each thread stages two activation rows and one 32-value K group of a weight
  row. Real-data microbenchmark, 27B shapes at 1024 rows (REPS=7 interleaved): swiglu/gate/down
  +2.5/+2.1/+4.3 %. Qwen3.8-27B same-server A/B (two sessions): prefill 2k +2.0/+1.7 % (484.5 → 494.2,
  482.7 → 490.9), 1k +1.3 %, 8k +1.9 %; Bolt (gemma4 kit) 1k/2k +1.3/+1.1 %. All inside the noise band, but
  the sign held in every run.
  - Where the m256 time goes (store kernel, 27B shapes, same session): full kernel 32 TFLOP/s; no LDS
    fragment reads (fragments held in registers) 36; no staging (no global loads, dequant, barriers) 37-38;
    neither (WMMA from registers only) 40-41. Single ablations were each small: no dequant +6-8 %, no global
    loads +4.5 %, no barriers 0-5 %, 1 of 4 B fragments +1-3 %. The overheads are latency, not bandwidth.
  - Tried: loads of fragment step kk+1 placed before step kk's MMAs: the compiler reschedules anyway (tie).
    Forcing that order with `scf.schedule.fence`: 200-208 VGPRs, -15 %. Fences around each step's loads and
    MMAs: +2 % once, -6/-10 % on reruns (noise). The 64 × 64 layout with fences: swiglu -8/-10 %. A per-wave
    `buffer.view` at the B-fragment row so all B reads use immediate offsets (swiglu VGPR 208 → 160): tie,
    so occupancy is not the limit either. Sharing one Q4_K scale/min decode between a thread's two 16-value
    slices: tie (not kept). Load-time weight repack: capped at the dequant ablation's 6-8 %; not tried.
- **Register prefetch** of the next K-step, guarded by `scf.if step+1 < k_steps`
  (without the guard the prover rejects the access).
- `GEMM_N_FASTEST=1` puts output tiles on grid x, so consecutive workgroups reuse
  the same A rows from L2. This is model dependent. It won on Flint and Glimmer
  (large N, many experts), but on dense Qwen3-4B turning it off was 4–12 % faster
  at 8k–16k prefill. A/B it per port; `GEMM_N_FASTEST=0` disables it.
- `GEMM_SKIP_PADDING=1`: padding rows get scale 0 (`g_row_scale_f16_live`), and a tile whose
  rows all have scale 0 exits. Packed epochs have up to 50 % padding at odd lengths.
- **Ceiling.** We reached 25–33 TFLOP/s of a ~59 peak (~40 from registers with real data). Unexplored: LDS double
  buffering of B, and direct Q4_K→fragment dequant.

## Q5_K / Q8_0 dequant (`gen_gemm.py`)

- Q5_K: the fifth bit from the qh words (`(qh >> g) & 0x01010101`, shifted to bit 4) instead of byte vectors;
  Q8_0: int8 -> f16 and one f16 multiply (q * d is exact in f32, so the f16 result is bit-identical). Both
  ~+8 % in isolation on real weights (26.6 -> 28.8 / 28.7 TFLOP/s). Bolt-E4B (89 % Q6_K in the UD-Q5_K_XL
  file) prefill +18/+11/+6.5 % at 2k/8k/32k from the Q6_K change.

## Q6_K dequant (`gen_gemm.py`)

- Do the bit work on 32-bit words (masks 0x0F0F0F0F / 0x03030303; a byte's bits shifted in from its
  neighbour are masked off) and finish with one f16 FMA `codes * s + (-32 s)`; byte vectors plus f32 math
  ran 22 TFLOP/s, this 26-28 (Q4_K: ~35 on the same constant-data benchmark).

## flash2 (`gen_flash2.py`)

- The workgroup is one 16-token tile × one KV head, with G waves (one per query
  head). K and Vᵀ for 32 keys are staged once in LDS for all G heads.
- **Transposed form.** Sᵀ = K·Qᵀ. In the WMMA result, each lane's column is one
  query and its 8 elements are 8 keys, so the row max and row sum are in-lane plus
  one xor-16 shuffle. Pᵀ is assembled from the lane pair, then Oᵀ += Vᵀ·Pᵀ.
- Q stays in registers (loaded straight from global memory), so no LDS is needed
  for Q or for softmax scratch.
- Gains: 2.2× on Glimmer attention. 64-key tiles (`FLASH2_KT=64`) were *slower*.
- It only handles *uniform* tiles: all 16 tokens live, one row, consecutive
  positions (`g_tile_uniform`) that start at a position multiple of 16, so a tile depends on its own span, never on
  where the epoch packs it (epoch-aligned tiles made a token's path, and its rounding, depend on its neighbours: the
  Coder consistency failure). Every other token goes to the listed fallback.
- **Token subtiles at small G** (`FLASH2_TS`): each staged key feeds 16·G queries, so at G = 2 long prefill re-streams
  the KV cache per 32 queries and is bandwidth-bound (Qwen3-0.6B: 25 % of roofline at 1k, 13 % at 32k). Two tiles per
  workgroup share each staged K/V tile: Qwen3-0.6B / 1.7B 32k prefill 2.6x / 2.1x, 1k tie (default at G <= 2). At
  G = 3 / 4 it lost 3-16 %; 4 or 8 tiles lost to 2 (too few workgroups per chunk).

## flash2 at head_dim 256 and 512 (`gen_flash2s.py`)

- At head_dim 256, 256 dims of Q (as f16 fragments) plus 256 dims of output accumulators exceed 256 VGPRs, so
  gen_flash2.py reloads all of Q from global memory (f32, converted each time) every 32-key tile: ~6x the K/V
  tile's bytes. gen_flash2s.py gives each query head TWO waves of 128 dims: a wave holds its Q half in
  registers for the whole key loop, computes S^T over its dims, publishes it to LDS and adds its partner's
  (commutative add: both waves hold bit-identical S, softmax and P), then accumulates its 128 output dims.
- Staging: lane = key, workitem/32 = dim segment, so a wave's V^T writes hit 32 consecutive columns
  (conflict-free) and each lane does one paged-KV lookup per tile (V = K offset + one plane).
- Qwen3.8-27B (G = 6, 12 waves, 61 KB LDS): a 1024-token chunk at 32k depth runs 16.9 TFLOP/s; prefill
  2k/8k/32k 401.6/368.6/262.5 -> 429.1/426.7/378.1 tok/s. Attention had been 1.42 ms of 3.8 ms per token at 32k.
- Head_dim 512 (gemma4 global layers): four waves per head, 16-key tiles (K and V^T of 32 keys exceed 64 KiB), each
  wave sums the head's four S partials in wave order. Over the two-pass kernel that reloaded Q per tile and
  recomputed S per 256-dim output half: gemma-4-26B-A4B prefill 1k/8k/16k/32k 1.11/1.35/1.64/2.17x (local A/B).
- Test it standalone with real shapes: a flash2 harness (paged arena filled directly, float64 reference with
  the gate) plus a broken variant (partner sum dropped) as the negative control.

## Last-layer output rows (`LAST_OUT`, llama.cpp `inp_out_ids`)

- A prefill epoch only needs the last layer's post-attention rows that feed the head (one per span). gen_llm.py
  rewrites the last layer: gather those rows of the attention output and residual (`g_rows_gather`), run the
  output projection, residual, FFN and final norm on them (the decode matvecs, with their own int8
  activations), scatter the final rows back (`g_rows_scatter`). K/V are still written for every token.
- Off for drafters that read the trunk's final hidden rows (MTP, Gemma-4 assistant) or a DFlash tap after the
  last layer; the verify program switches it off. Qwen3.8-27B: greedy outputs identical; prefill +1 % (the last
  layer is ~1.5 % of the GEMM work).

## Listed fallback

- `g_attention_fallback_list` lists the tiles flash2 computes (their first tokens, which its workgroups read) and
  compacts every other token, padding included, into a list, using an atomic counter. The split-K attention then runs over
  that list with an indirect grid (`[buffer]` launch). Before compaction, padding
  tokens did full attention work.

## Per-token RoPE + KV store (`qk_rope()`)

- One 256-lane workgroup per token, with waves looping over all q and k heads. The
  angles are computed once per lane and reused for every head. The optional
  per-head RMSNorm uses a subgroup reduce. This was 6× faster than the per-head
  version.

## Fusions

- RMSNorm + f16 row scale in one pass (`g_row_scale_f16`).
- Residual add + next RMSNorm (`g_post_add_norm`), with optional Gemma-style post
  norms.

## Tile order per port (`cfg["gemm_n_fastest"]`)

- Columns fastest (the default) reuses each activation tile; rows fastest reuses each weight tile across
  the chunk's row tiles (gufo's activation-panel grouping is the same idea). It is shape dependent, so the
  reviewed cfg carries the winner: Muse-Glimmer-30B rows fastest +4 to +12 % (three server sessions: two wins,
  one tie; never a loss); Qwen3.8-27B and Bolt tie; Flint -4 % at 8k. Repeat the A/B across sessions.

## Settled for starter ports (do not re-run these A/Bs)

**Settled for starter ports (dense / Qwen3.5 hybrid, ≤ 4B): don't re-run these A/Bs.** Fresh-agent
trials re-tested them on Qwen3.5-0.8B and the defaults won or tied every time:
`GEMM_N_FASTEST=0` (slower prefill, 3 trials), `MV_WG1=1024` (no decode win), `--prefill 4096`
(short prefill −20–37 %; 2048 ≈ 1024 within noise). On the R9700 (64 CUs) the larger chunk wins for small models (hidden size ≤ 2048: Qwen3-0.6B +7..+24 %, Gemma-4-E2B +5..+15 %, Qwen3-1.7B +11..+18 % from 8k) and loses at 1k/2k on big ones (Qwen3-14B −8/−5 %): `build.py --prefill-r9700 N` (or `prefill_capacity_r9700` in llm_config.json) adds a second preset `<preset>-r9700` that only the R9700 Quick start fetches. On a starter port, spend the budget on fusion
(fewer launches per layer: fused QKV or norm fusion, which you write in a copy of the generator; pairing Qwen3.5's tiny GDN alpha/beta matvecs
measured neutral) or on the long
prefill path. Bigger or MoE models: the chunk size is worth one A/B measured at 1k too (4096 helped
one MoE port; 2048 and 4096 both cost 12-18 % at 1k on flint).

## RDNA4 (gfx12) GEMM twins: what bounds them (`gemm_r9700` kbench suite)

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

## Did not help (prefill)

- **64-key flash tiles.** Register pressure; occupancy dropped.
