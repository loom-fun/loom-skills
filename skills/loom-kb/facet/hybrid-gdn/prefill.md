# Facet: hybrid-gdn: prefill and decode (gated delta-net recurrence)

## Gated delta-net recurrence (`gen_gdn.py`) and the f16 row scale

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
- **Row-split recurrence** (`_recurrence_rows`, default; `GDN_ROW_LANES=0` restores the 32-lane form): a 32-lane
  workgroup owns 16 value columns of one head; a group of RL = 8 lanes owns 4 columns and lane r of it the
  key rows [16r, 16r + 16) (64 state values in registers, 384 workgroups instead of 1536). S^T k and S^T q are
  per-lane `vector.dotf` over the lane's 16 rows plus a 3-step xor-shuffle inside the group (the 32-lane form
  ran two 32-lane reductions per 4 columns); the q/k L2 norms and the decay fold into per-column scalars
  (decay * S^T k = (S_old^T k) * decay), so each state element costs 4 VALU ops (dot, decay mul, fma
  update, dot); decay/beta/softplus once per 16 columns. Inputs loaded one step ahead, outputs stored in
  8-step batches (lane r keeps step r's 4 outputs). Same arithmetic, different summation order (kernel tests
  vs float64 pass at the old tolerances; new multi-span case `gdn_recurrence_spans`).
  Qwen3.8-27B same-server A/B vs r3 (5 rounds): prefill 1k/2k/8k +2.7/+2.8/+2.8 % (2k 496.9 -> 510.7 tok/s),
  32k +2.4 % (3 rounds), decode +1.8 % (tie). Doubling the recurrence at 2k: 0.124 -> 0.079 ms/token
  (6.1 -> 4.0 % of 2k prefill).
  - Variants (2k, 3 rounds each, vs the same baseline): RL 4 (32 rows/lane) -1.6 %, RL 16 +1.0 %, RL 8 with
    8 columns per lane (`GDN_COLS=8`) +2.6 %, RL 16 with 8 columns +1.7 %; RL 8 x 4 columns +2.9 % kept.
  - Not tried (time-box): the chunked WY / "chunked delta rule" form (llama.cpp `build_delta_net_chunking`,
    C = 64). In f32 it does about as many FLOPs per token as the recurrence (~3 DK*DV per token for S^T k,
    Q S and the state update, plus C*DK terms), so it only pays with f16/bf16 WMMA for the intra-chunk
    products, which changes precision of the state contribution; the recurrence is now 4 % of 2k prefill.
- `g_row_scale_f16`: vector<4> steps instead of scalar ones (each scalar step waited on its load): +0.5 %.
- Tiny projections in prefill (Qwen3.5-style `ssm_alpha`/`ssm_beta`, 48 rows, F32, through `lin_mv`): the
  4-token matvec with the decode-tuned R=1 re-read each 4-token activation block once per output row
  (doubling them cost 5 % of 2k prefill). Prefill epochs now launch a 16-row variant: +2.4/+2.9 % at 2k/8k.
  Decode and verification keep their kernels (same summation order as before).
- `prefill_capacity` for the dense 27B, vs 1024 at 1k/2k/8k: 512 ties (-1 %), 2048 loses -41/-9/-8 % (the
  1k prompt is a little over 1024 tokens and pads to one 2048-row epoch). Keep 1024 for dense models.
  - Re-measured after #26's live row scale + `GEMM_SKIP_PADDING` and the 64x64-per-wave m256 GEMM (same
    server, 5 rounds; 32k 3 rounds). 2048: -7.1/-1.0/-1.1 % at 1k/2k/8k, decode tie: padding is now cheap
    (-41 -> -7 % at 1k) but a bigger chunk gains nothing. 4096: -7.0/-12.4/-10.3/-7.7 % at 1k/2k/8k/32k.
    Doubling every GEMM at 8k: GEMM time is the same at both chunks (1.557 vs 1.549 ms/token; the m256 GEMM
    is already at its compute ceiling at 1024 rows, and the 17 GB of weights per 1024-token epoch is ~4 % of
    the bandwidth), while the rest grows 0.50 -> 0.67 ms/token. Doubling by family: flash2 0.095 vs 0.101
    ms/token (same), the `g_gdn_*` kernels 0.23 -> 0.44 ms/token: the whole loss is in the GDN layers (not
    split further: which GDN kernel grows, and why, is the next thing to measure if a dense hybrid ever needs
    bigger chunks). MoE models gain from 4096 (fuller expert buckets); dense models
    keep build.py's 1024 default. A dense non-hybrid model may behave differently (untested).
- Qwen3.8-27B 2k prefill ablation after m256 + live row scale (doubling, same server): GEMMs ~79 % (q4_k store
  35, swiglu 27, q6_k store 12, q8_0 store 4.5; all ~31 TFLOP/s, so the weight format barely matters),
  post-add norm + row scale 2.8, row scale (o, f) 1.9, gdn_conv 1.8, flash2 1.6, silu_mul 0.8 %.
- `g_post_add_norm_rs[_live]` (gen_llm `fuse_post_add_row_scale`): the residual add + next norm also writes the
  next GEMM's f16 row and scale (one more workgroup max-reduce over values already in registers), replacing
  the separate `g_row_scale_f16` launch and its two re-reads of h. `g_gdn_conv`: 4 channels per loop step with
  every load before any store; `g_gdn_norm_gate`: stores after all loads (the store -> load ordering above).
  Together +0.6/+0.8/+0.6 % at 1k/2k/8k (tie band; same values as before, kernel tests). Re-measured on top of
  the 64x64-per-wave m256 GEMM (7 rounds, same server): +0.3/+1.0/+0.6/+0.6 % at 1k/2k/8k/32k, decode tie;
  positive in both measurements, so kept (27B 2k 492.3 -> 497.1 tok/s).
  - Tried: one 256-row swiglu kernel for layers whose gate and up formats differ (Q4_K gate + Q6_K up: waves
    0-7 dequant the gate format, waves 8-15 the up format, an scf.if on a wave-uniform condition; replaces two
    GEMMs + `g_silu_mul`). Correct (kernel test) but -1.1/-1.5 % at 2k/8k vs the two GEMMs: carrying both
    formats' prefetch registers costs more than `g_silu_mul` (0.8 %) and the %fu round trip save. Not kept.
