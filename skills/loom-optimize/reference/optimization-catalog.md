# Optimization catalog (with the reasoning behind each)

Numbers come from the Flint (40B MoE) and Muse Glimmer (30B dense) ports on Strix
Halo, measured against llama.cpp ROCm (lemonade-sdk/llamacpp-rocm b1355) on the
same GGUF. The kernels named here are in `loom-port-llm/starter/gen_llm.py` unless a
lever says otherwise (MoE: `loom-port-llm/reference/{lfm2moe,flint}-example/gen_moe*.py`, and the
bucketed grouped GEMM idea in `flint-moe-example/gen_moe.py`).

## Decode

### Multi-token matrix-vector (`mv()` in gen_llm.py)
- **Layout.** A workgroup covers R=16 output rows of one weight matrix, with one
  lane per 32-value group of K, so lanes = K/32 rounded up to a wave. Each lane
  keeps its 32 x values for TG tokens in registers, dequantizes its weight group
  once, and does TG dot products. Wave partials go through
  `kernel.subgroup.reduce`, then one barrier and an LDS fold.
- **Why it works.** Decode is weight-bandwidth bound, so every byte of W should be
  read exactly once per token group. TG=4 serves up to 4 concurrent sequences for
  roughly the price of one.
- **Pitfall.** When K/32 ≤ 32 there is only one wave, so the R·TG outputs exceed
  the lane count. The writer must stride (`scf.for %wr0 = [%workitem to %outputs
  step %lanes]`). This bug produced garbage for hidden=1024 models.
- **Q4_K folding.** `y = Σ_groups (d_g · dot(codes_g, x_g) − m_g · Σx_g)`. Σx_g is
  computed once per token.

### Adaptive rows per workgroup (dense 4B-class decode: the biggest single win)
- **Problem.** The starter's fixed R=16 rows per workgroup leaves the GPU
  under-filled on narrow projections. For example, K/V with n = 512, giving only
  32 workgroups, or q with n = 4096. Meanwhile the long-K down projection
  (K = 10240) has few, very long workgroups.
- **Rule (1-token variant).** Start at R=16 and halve R while
  `n / R < WG_TARGET`, stopping at R=1. Use WG_TARGET ≈ 4096 by default, but
  ≈ 640 when K ≥ 8192 (the down projection), whose workgroups are long already.
  For the 4-token variant, use WG_TARGET ≈ 128: a larger value regresses c4.
  Emit one kernel variant per (format, K, swiglu, R, TG), named `_r{R}`.
- **Result.** On Nemotron-3.5-Bolt-E4B (Gemma-4 E4B, Q4_K_M), decode went from
  54.2 to 59.9 tok/s, half of the gap to the roofline in one change. Check c4
  afterwards: it regressed until the 4-token target was lowered to 128.
- **In the starter, on by default** (`pick_r()` in gen_llm.py; `MV_ADAPT=0` turns it off,
  `MV_WG1` / `MV_WG4` set the targets; `make_kernel_tests.py` tests the `_r{R}` variants
  the model launches). Hidden-1024 models gain the most: in fresh-agent trials decode went
  from 92 to 144 tok/s (Qwen3.5-0.8B) and from 124 to 157 tok/s (Qwen3-0.6B, where
  `MV_WG4=512` also beat 128 on 4 streams).
- **Keep 1- and 4-token R equal (the default).** R changes the summation order, so
  different R for the two variants makes a request's tokens depend on whether it was
  decoded alone or batched: `consistency.py` failed on Qwen3-0.6B. With one R per
  projection (the 1-token rule) decode keeps the whole gain and 4 streams most of it
  (interleaved medians: fixed R=16 118 / 249 tok/s, same R 131 / 368, separate R 131 / 514
  for decode / 4 streams). Setting `MV_WG4` opts into separate 4-token R; then the
  consistency gate can fail (it did on Qwen3-0.6B), so run it; on Bolt-E4B `MV_WG4=128` passed
  consistency and gave 4 streams +8.6 % with decode unchanged (Nov 2026 trial). Keep it only if the gate passes.

### Fused 1-token QKV and paired-row gated FFN
- **Fused QKV.** For a single decode token, one launch computes the q, k and v rows
  (one row range over n_q+n_k+n_v, dispatching on the row id). Launch it under
  `scf.if` for 1-token epochs only; multi-token epochs keep the separate kernels.
  It gained about 0.5 %.
- **Paired-row gate/up.** For the 1-token gated FFN (swiglu/geglu), have each
  workgroup process gate row i and up row i side by side, with R ≥ 2, so no lanes
  idle on the epilogue. Bolt went from 60.3 to 61.0.
- **Serial small kernels.** Any kernel launched with a single workgroup, such as
  per-layer-input prep or norms over many tokens, should get a (tokens, layers)
  grid. It is cheap to find by ablation, because doubling it costs a visible
  fraction of a millisecond.

### Argmax head (`head_mv()`)
- Vocabulary rows are split into 128-row chunks. Each workgroup writes
  (max, argmax) per chunk per span, and `g_head_select` reduces the chunks.
- No logits buffer is ever written, which saves V × 4 bytes per token plus a pass.
- This is only valid for greedy decoding. For sampling you'd need top-k partials.

### Many-stream drafted decode (4 streams x DFlash depth 3)
- **Attribute the epoch, not the token.** Under drafting, a streamed event is one epoch. Measure ms per event and
  tokens per event separately, so a doubled kernel that changes the drafts can't pass for a slower one.
  Doubling every launch of one kernel group in a copy of the package gave Muse-Glimmer-30B's 204 ms 4-stream epoch as:
  - 16-token trunk matvecs: +113 ms (one memory-bound 1-token pass is ~75 ms).
  - Proposals: +56 ms.
  - Head: +44 ms.
  - Warm/catch-up: +12 ms.
  - Decode attention: +5 ms.
- **The int8 head held one or two workgroups per CU.** Each 128-row chunk kept waves x 128 x TG F32 partials in
  LDS (28 KB at 8 spans, 57 KB at 16), so 8 spans took 11.7 ms against 5.2 ms for 1 span.
  - 16 verify rows ran as two 8-span groups (22 ms), and the 12 proposal rows took 18 ms, all per epoch.
  - Fix (`head_iq_body()`): the chunk runs in sub-blocks of 512/TG rows (partials ~14 KB at any TG). Lane j carries
    span j's running (max, argmax) across sub-blocks. Partials go through the matvecs' shared-tree butterfly.
  - Result: 8 spans 5.9 ms; a new 16-span variant does 16 spans in 8.7 ms and 12 in 8.6 ms.
  - Every variant rounds and breaks ties like the 1-span kernel (kernel tests `head_spans16` / `head_spans12`).
  - Muse-Glimmer 4 streams: 193 -> 160 ms per epoch.
  - The earlier "16-span head was slower" finding (Qwen3.8-27B) was this LDS limit.
- **DFlash block = depth + 1 tokens** (llama.cpp's `n_max + 1`), capped at the trained block size, instead of the
  full 16-token block. At depth 3, 4 streams propose 16 tokens (int8 matvec, one weight pass) instead of 64 (GEMM),
  and 1 stream proposes 4 instead of 16.
  - Muse-Glimmer: 4 streams -10 ms per epoch, 1 stream 109.6 -> 100.8 ms (+6.6 % decode).
  - Acceptance is unchanged within noise (2.6-2.7 tokens per step).
- **The drafter's catch-up (warm) program takes the GEMM from 5 tokens** (`DRAFT_MM_FROM`). It only feeds proposals,
  so rounding there never reaches the output. DFlash's 33280-wide fc ran on the f32 4-token matvec (K > 32768),
  which re-read 124 MB per 4 tokens: 4 streams 161.5 -> 157.6 ms per epoch.
  - The proposal program keeps the int8 matvec: the GEMM from 5 tokens there cost +12 ms per epoch.
- **What still separates the 16-token step from one weight pass** is VGPRs. 16 tokens of int8 activations per lane
  put the Q5_K gate/up kernel at 233 VGPRs (6 waves per SIMD), against 48 VGPRs and 15 waves for 1 token.
  - Measured: 1144 vs 817 us (gate/up 6656 x 19968), 588 vs 424 us (down 19968 x 6656).
  - See "Did not help" for what was tried.
- **Re-attributed after those changes** (Muse-Glimmer, 156 ms epoch, doubling in package copies, c4ep):
  - 16-token trunk matvecs: +114.5 ms.
  - Proposal program: +20.3 ms (its matvecs +11.8, its 12-row head +7.9).
  - Verify head: +9.6 ms. Warm: +5.0 ms. Decode attention: +4.5 ms.
  - The 16-token kernel is instruction-bound: its row loop is ~770 instructions per wave and row (gate/up, both
    weights), of which 256 are `v_dot4`. The rest is Q5_K decode (~125), F32 scaling (~150, needed to round like the
    1-token kernel) and the butterfly. Loads issued a row ahead (software prefetch) did not help (see "Did not help").
- **32 rows per workgroup at 16 tokens** for gate/up and for K >= 8192 (`mv_rows.iq_r`). Each workgroup loads its
  16 tokens' int8 activations for the whole K once, and 32 rows halve that load per weight row. The int8 kernels'
  arithmetic does not depend on R.
  - Interleaved microbenchmarks: down 19968 x 6656 -3.7 %, gate/up -1.8 % (-5.5 % with the permuted slots below).
  - K = 4096 / 6656 single-weight projections: flat or +4 %, so they keep 16 rows.
- **Butterfly without selects** (`BFLY_PERM`, gate/up at 8+ tokens and the 16-span head). Lane L holds token
  j ^ bt_idx(L) in slot j, where bt_idx is the token the lane ends the butterfly with. Every step then keeps slots
  [0, h) and sends slots [h, 2h), so the two selects per pair go away. The pairing tree per token is unchanged, so
  the results are bit-identical; kernel tests `mvsw_same_t8` / `mvsw_same_t16` and the `mv_*_q8_t*` cases now check
  the multi-token kernels against the 1-token kernel with zero tolerance.
  - Gate/up t16: -3.5 % at 16 rows, -5.5 % at 32. Head, 12 and 16 spans: 9.0 -> 8.5 ms. Head at 8 spans: flat.
  - Single-weight projections keep plain slots: the permuted activation loads are uncoalesced (down t16 +9 %).
  - Muse-Glimmer 4 streams, same server: 157.4 -> 152.8 ms per epoch, 67.2 -> 68.1 tok/s. Prefill and 1-stream
    decode kernels are unchanged (byte-identical).

### Split-K decode attention (`attention_text()`)
- The grid is (token, KV head, split), with G = heads/kv_heads waves per
  workgroup, one per query head. Each wave covers that head's 128 dims across 32
  lanes × 4.
- **Adaptive splits.** splits = min(64, ceil(keys / 32)). Splits beyond that exit
  immediately. Without the early exit, the 64-split grid regressed 4-stream
  throughput.
- A 16-key unroll with predicated tails. The V offset is the K offset plus
  pool × POS_BYTES, so `g_kv_offset` isn't called twice.
- The combine kernel folds (m, l, acc) across splits, then applies the optional
  sigmoid gate.

### F16 KV
- K and V are stored as f16 by the RoPE/KV kernel and read as `vector<4xf16>`.
  Decode at 32k improved from 59 to 63 tok/s on Flint, and memory halves.

## Prefill

### WMMA GEMM (`gen_gemm.py`)
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
- **Register prefetch** of the next K-step, guarded by `scf.if step+1 < k_steps`
  (without the guard the prover rejects the access).
- `GEMM_N_FASTEST=1` puts output tiles on grid x, so consecutive workgroups reuse
  the same A rows from L2. This is model dependent. It won on Flint and Glimmer
  (large N, many experts), but on dense Qwen3-4B turning it off was 4–12 % faster
  at 8k–16k prefill. A/B it per port; `GEMM_N_FASTEST=0` disables it.
- `GEMM_SKIP_PADDING=1`: padding rows get scale 0 (`g_row_scale_f16_live`), and a tile whose
  rows all have scale 0 exits. Packed epochs have up to 50 % padding at odd lengths.
- **Ceiling.** We reached 25–31 TFLOP/s of a ~59 peak. Unexplored: LDS double
  buffering of B, and direct Q4_K→fragment dequant.

### Gated delta-net recurrence (`gen_gdn.py`) and the f16 row scale
- **Loom's backend orders every global load after earlier global stores** (`s_waitcnt_vscnt 0` before the
  load), and waits for in-flight loads when it reuses their address registers. A loop that loads and stores
  every step pays a memory round trip per step. `buffer.assume.noalias` did not change the generated code.
  Batch the stores (or keep them out of the loop) and load ahead.
- Recurrence (Qwen3.8-27B: ~6.5 % of 2k prefill, 0.14 ms/token, VALU-bound: 1536 one-wave workgroups each
  recompute the token's decay/beta/norms for 4 columns): inputs loaded one step ahead (GDN_PREFETCH) and
  outputs stored in 32-step batches: +1 % (tie band).
  - Tried: gufo's column form (a lane pair owns one value column, a shuffle-xor instead of 32-lane reductions,
    per-token scalars once per token, chunks of 4 tokens staged in LDS): correct (kernel test) but -2.5 %
    (-18 % before batching stores): per-chunk load/store waits (above) and 240-256 VGPRs. Not kept.
- `g_row_scale_f16`: vector<4> steps instead of scalar ones (each scalar step waited on its load): +0.5 %.
- Tiny projections in prefill (Qwen3.5-style `ssm_alpha`/`ssm_beta`, 48 rows, F32, through `lin_mv`): the
  4-token matvec with the decode-tuned R=1 re-read each 4-token activation block once per output row
  (doubling them cost 5 % of 2k prefill). Prefill epochs now launch a 16-row variant: +2.4/+2.9 % at 2k/8k.
  Decode and verification keep their kernels (same summation order as before).
- `prefill_capacity` for the dense 27B, vs 1024 at 1k/2k/8k: 512 ties (-1 %), 2048 loses -41/-9/-8 % (the
  1k prompt is a little over 1024 tokens and pads to one 2048-row epoch). Keep 1024 for dense models.
- Qwen3.8-27B 2k prefill ablation after m256 + live row scale (doubling, same server): GEMMs ~79 % (q4_k store
  35, swiglu 27, q6_k store 12, q8_0 store 4.5; all ~31 TFLOP/s, so the weight format barely matters),
  post-add norm + row scale 2.8, row scale (o, f) 1.9, gdn_conv 1.8, flash2 1.6, silu_mul 0.8 %.
- `g_post_add_norm_rs[_live]` (gen_llm `fuse_post_add_row_scale`): the residual add + next norm also writes the
  next GEMM's f16 row and scale (one more workgroup max-reduce over values already in registers), replacing
  the separate `g_row_scale_f16` launch and its two re-reads of h. `g_gdn_conv`: 4 channels per loop step with
  every load before any store; `g_gdn_norm_gate`: stores after all loads (the store -> load ordering above).
  Together +0.6/+0.8/+0.6 % at 1k/2k/8k: tie band, so **not kept** (branch `issue-29b`, commit bdcf3cd, if a
  later GEMM speedup makes these few percent matter). What is left at 2k is the GEMM inner loop (~31 of ~59 TFLOP/s).
  - Tried: one 256-row swiglu kernel for layers whose gate and up formats differ (Q4_K gate + Q6_K up: waves
    0-7 dequant the gate format, waves 8-15 the up format, an scf.if on a wave-uniform condition; replaces two
    GEMMs + `g_silu_mul`). Correct (kernel test) but -1.1/-1.5 % at 2k/8k vs the two GEMMs: carrying both
    formats' prefetch registers costs more than `g_silu_mul` (0.8 %) and the %fu round trip save. Not kept.

### flash2 (`gen_flash2.py`)
- The workgroup is one 16-token tile × one KV head, with G waves (one per query
  head). K and Vᵀ for 32 keys are staged once in LDS for all G heads.
- **Transposed form.** Sᵀ = K·Qᵀ. In the WMMA result, each lane's column is one
  query and its 8 elements are 8 keys, so the row max and row sum are in-lane plus
  one xor-16 shuffle. Pᵀ is assembled from the lane pair, then Oᵀ += Vᵀ·Pᵀ.
- Q stays in registers (loaded straight from global memory), so no LDS is needed
  for Q or for softmax scratch.
- Gains: 2.2× on Glimmer attention. 64-key tiles (`FLASH2_KT=64`) were *slower*.
- It only handles *uniform* tiles: all 16 tokens live, one row, consecutive
  positions (`g_tile_uniform`). Every other token goes to the listed fallback.

### flash2 at head_dim 256 (`gen_flash2s.py`)
- At head_dim 256, 256 dims of Q (as f16 fragments) plus 256 dims of output accumulators exceed 256 VGPRs, so
  gen_flash2.py reloads all of Q from global memory (f32, converted each time) every 32-key tile: ~6x the K/V
  tile's bytes. gen_flash2s.py gives each query head TWO waves of 128 dims: a wave holds its Q half in
  registers for the whole key loop, computes S^T over its dims, publishes it to LDS and adds its partner's
  (commutative add: both waves hold bit-identical S, softmax and P), then accumulates its 128 output dims.
- Staging: lane = key, workitem/32 = dim segment, so a wave's V^T writes hit 32 consecutive columns
  (conflict-free) and each lane does one paged-KV lookup per tile (V = K offset + one plane).
- Qwen3.8-27B (G = 6, 12 waves, 61 KB LDS): a 1024-token chunk at 32k depth runs 16.9 TFLOP/s; prefill
  2k/8k/32k 401.6/368.6/262.5 -> 429.1/426.7/378.1 tok/s. Attention had been 1.42 ms of 3.8 ms per token at 32k.
- Test it standalone with real shapes: a flash2 harness (paged arena filled directly, float64 reference with
  the gate) plus a broken variant (partner sum dropped) as the negative control.

### Q5_K / Q8_0 dequant (`gen_gemm.py`)
- Q5_K: the fifth bit from the qh words (`(qh >> g) & 0x01010101`, shifted to bit 4) instead of byte vectors;
  Q8_0: int8 -> f16 and one f16 multiply (q * d is exact in f32, so the f16 result is bit-identical). Both
  ~+8 % in isolation on real weights (26.6 -> 28.8 / 28.7 TFLOP/s). Bolt-E4B (89 % Q6_K in the UD-Q5_K_XL
  file) prefill +18/+11/+6.5 % at 2k/8k/32k from the Q6_K change.

### Q6_K dequant (`gen_gemm.py`)
- Do the bit work on 32-bit words (masks 0x0F0F0F0F / 0x03030303; a byte's bits shifted in from its
  neighbour are masked off) and finish with one f16 FMA `codes * s + (-32 s)`; byte vectors plus f32 math
  ran 22 TFLOP/s, this 26-28 (Q4_K: ~35 on the same constant-data benchmark).

### Last-layer output rows (`LAST_OUT`, llama.cpp `inp_out_ids`)
- A prefill epoch only needs the last layer's post-attention rows that feed the head (one per span). gen_llm.py
  rewrites the last layer: gather those rows of the attention output and residual (`g_rows_gather`), run the
  output projection, residual, FFN and final norm on them (the decode matvecs, with their own int8
  activations), scatter the final rows back (`g_rows_scatter`). K/V are still written for every token.
- Off for drafters that read the trunk's final hidden rows (MTP, Gemma-4 assistant) or a DFlash tap after the
  last layer; the verify program switches it off. Qwen3.8-27B: greedy outputs identical; prefill +1 % (the last
  layer is ~1.5 % of the GEMM work).

### Listed fallback
- `g_attention_fallback_list` compacts the tokens of non-uniform tiles that are
  live into a list, using an atomic counter. The split-K attention then runs over
  that list with an indirect grid (`[buffer]` launch). Before compaction, padding
  tokens did full attention work.

### Per-token RoPE + KV store (`qk_rope()`)
- One 256-lane workgroup per token, with waves looping over all q and k heads. The
  angles are computed once per lane and reused for every head. The optional
  per-head RMSNorm uses a subgroup reduce. This was 6× faster than the per-head
  version.

### Fusions
- RMSNorm + f16 row scale in one pass (`g_row_scale_f16`).
- Residual add + next RMSNorm (`g_post_add_norm`), with optional Gemma-style post
  norms.

### Chunk size
- `prefill_capacity` 4096 helps MoE prefill: larger expert buckets mean fuller GEMM tiles. build.py now
  defaults MoE configs (`cfg["moe"]`) to 4096. Flint (256 experts, top-6: ~24 tokens per expert in a 1024
  chunk, 64-row tiles ~37 % full; the expert GEMMs were 85 % of prefill): 2k/8k/32k +22/+38/+28 % vs 1024
  (2048: +25/+27/+22 %). Dense models gain little beyond 1024–2048.
- **Padding is the price of a big chunk.** An epoch runs at the next power-of-two shape: a 1026-token prompt
  at chunk 4096 is one 2048-token epoch, half padding (at chunk 1024 it is 1024 + 2). Flint at 1k was 29 %
  slower at 4096 than at 1024. Doubling launches attributed it (seconds per 1k prompt, 2048-shape vs
  1024-shape epoch): router +0.05 (and 0.16 s even at 1024), dense GEMMs +0.07, elementwise kernels +0.07,
  grouped expert GEMMs +0.03; `g_moe_bucket` is free and already skips padding. Fixes (issue #26):
  - Router: the lane-per-32-value-chunk dot (strided loads, 32-way LDS bank conflicts) took 4.4 ms per
    2048-token layer; a coalesced layout (lane l takes 4-value chunk l of each 128-value block) takes 0.96 ms
    (`g_moe_route_tr`, prefill only). Per-wave top-k (subgroup reductions) replaces the workgroup ones.
    Several tokens per workgroup (weight reuse) did not matter. Decode keeps the old one-token-per-workgroup
    router: the new one is latency bound at t = 1 (decode -34 %).
  - Dense GEMMs: `GEMM_SKIP_PADDING` never fired, because the row-scale scale is a power of two and never
    0. `g_row_scale_f16_live` (the token-row scales in `@llm_advance`, given `%info`) writes scale 0 and
    zero rows for padding, and a GEMM row tile is skipped when all its scales are 0 (+9 % at 1k).
  - Result, Flint chunk 4096 vs the PR #25 build, one server, interleaved: 1k/2k/8k/32k +27/+15/+17/+13 %,
    decode tie; 1k now matches the 1024-chunk build (918 vs ~905 tok/s).
  - Not done: padding rows still run the elementwise kernels (norms, RoPE, combine; ~0.07 s at 1k).

### Tile order per port (`cfg["gemm_n_fastest"]`)
- Columns fastest (the default) reuses each activation tile; rows fastest reuses each weight tile across
  the chunk's row tiles (gufo's activation-panel grouping is the same idea). It is shape dependent, so the
  reviewed cfg carries the winner: Muse-Glimmer-30B rows fastest +4 to +12 % (three server sessions: two wins,
  one tie; never a loss); Qwen3.8-27B and Bolt tie; Flint -4 % at 8k. Repeat the A/B across sessions.

## Did not help
- **64-key flash tiles.** Register pressure; occupancy dropped.
- **Expert gate_up prefetch.** No gain; the kernel is compute-bound already.
- **Larger MoE tiles or wide GEMM tiles.** More padding waste and less occupancy.
- **LDS-staged x in the multi-token matvec.** Registers already hold x, so the
  extra barrier only cost time.
- **Fusing the residual add + RMSNorm into the next matvec's prologue (decode).**
  Every workgroup recomputes the norm, so the redundant reads cost more than the
  launch saved (Bolt: slower). Launches are nearly free under graph replay.
- **Unrolling the matvec row loop**, or head chunk sizes other than 128 rows.
  Neither was measurably different.
- **Cheaper 16-token int8 matvecs** (Muse-Glimmer Q5_K gate/up 6656 x 19968 at 1144 us, down 19968 x 6656 at
  588 us), each tried and measured:
  - Fewer rows per workgroup (R 4 / 8): 1352 / 642 us.
  - The 8-token kernel over 16 tokens: two token-group workgroups share no weight reads (1307 / 649 us).
  - A workgroup of two 8-token halves over the same weight groups: 120 VGPRs and 12 waves, but 1302 us.
  - Two rows per iteration, so both rows' loads issue first: 8-token down 579 -> 459 us in isolation, but
    4-stream epochs +4 ms and 2-stream epochs flat.
- **More on the 16-token int8 matvec** (Muse-Glimmer, interleaved `loom-benchmark --compare`):
  - Software prefetch: row r+1's weight loads issued before row r's dot products (codes split into load and
    decode functions, carried in the row loop). VGPRs 203 -> 240 (7 -> 6 waves). Gate/up 1105 -> 1142 us, down
    577 -> 598 us, q 117 -> 114 us: no gain.
  - 64 rows per workgroup: q 6656 x 4096 +20 %, o 4096 x 6656 +9 % (down does not fit: 80 KB of LDS).
  - Fewer rows for the small K/V projections (6656 x 256): R 8 is 18.0 vs 22.2 us, ~0.4 ms per epoch. Not
    adopted.
- **About 128 keys per decode-attention split.** Fewer splits under-filled the GPU
  at short context (−4 %).

Results per model (with ceilings and efficiency): `loom-roofline/reference/calibration.md`.
