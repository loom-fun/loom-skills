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
- **Tiles.** 64 or 128 rows × N tile; the m128 variant is used from 128 tokens.
- **Register prefetch** of the next K-step, guarded by `scf.if step+1 < k_steps`
  (without the guard the prover rejects the access).
- `GEMM_N_FASTEST=1` puts output tiles on grid x, so consecutive workgroups reuse
  the same A rows from L2. This is model dependent. It won on Flint and Glimmer
  (large N, many experts), but on dense Qwen3-4B turning it off was 4–12 % faster
  at 8k–16k prefill. A/B it per port; `GEMM_N_FASTEST=0` disables it.
- `GEMM_SKIP_PADDING=1`: padding rows get scale 0, and a tile whose first row has
  scale 0 exits. Packed epochs have up to 50 % padding at odd lengths.
- **Ceiling.** We reached 25–31 TFLOP/s of a ~59 peak. Unexplored: LDS double
  buffering of B, and direct Q4_K→fragment dequant.

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
  (2048: +25/+27/+22 %). The price is short prompts: a ~1k-token prompt prefills 29 % slower at 4096 (22 % at
  2048) than at 1024, a fixed per-epoch cost not yet attributed. Dense models gain little beyond 1024–2048.

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
- **About 128 keys per decode-attention split.** Fewer splits under-filled the GPU
  at short context (−4 %).

Results per model (with ceilings and efficiency): `loom-roofline/reference/calibration.md`.
