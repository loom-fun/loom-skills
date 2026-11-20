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

## Producer fusions on the 27B hybrid, Strix Halo (switches `SILU_MUL_RS`, `GDN_NORM_RS`, `POST_NORM_NOH`, default on; `=0` = old path)
`g_silu_mul`+`g_row_scale_f16_live` -> `g_silu_mul_rs`; `g_gdn_norm_gate`+row scale -> `g_gdn_norm_gate_rs`; `g_post_add_norm_rs_live` without the
f32 h store where only h16 is read -> `..._noh`. Qwen3.8-27B Q4_K_M, DevLab ABBA (run 44647263024, 85 W runner), all three vs base prefill tok/s:
1k 530->534 (+0.8 %), 2k 531->537 (+0.9 %), 8k 537->545 (+1.5 %), 16k 511->517.5 (+1.3 %), 32k 462.5->468 (+1.1 %); decode unchanged (28.2, 22.9 at 32k).
Single switches +0.4..+0.7 % each at 8k (silu 539.6, gdn 540.5, noh 539.2 vs 537). Producer family 0.118 -> 0.090 ms/token (8k). ktest 160/160.
The 17408-wide f-scale (`g_row_scale_f16_live_lv`, 0.029 left) is f32-round-trip bandwidth bound: only a swiglu GEMM epilogue writing f16 removes it.

## gfx12 chunked (WY) prefill: `g_gdn_chunk_rdna4` (`gen_gdn_chunk.py`, choice `hybrid.gdn.chunk`, env `GDN_CHUNK`)
Qwen3.8-27B on the R9700, 2k prefill, ms/token (doubling each kernel in prefill, same server): the sequential recurrence 0.046 (7.5 %), conv 0.011,
gated norm 0.007, q/k norm 0.004 (the last three inside the run-to-run noise of one doubling, +-0.01). Nothing else of the GDN family is worth a kernel.
- **Row/column split of the sequential kernel does not move on the R9700**: the 8 x 4 form (MTP models' only form) is the best of 8x2, 8x8, 16x2, 16x4,
  16x8, 32x4, 4x4 (2k prefill -1.8, -2.9, -2.9, -2.4, -2.1, -6, -11.7 %; decode -0.4 .. -5 %).
- **The chunked form works on gfx12 with f16 WMMA and an f32 state** (the first kernel test run passed: 84/84): one workgroup of 8 waves per value head,
  chunks of CN = 32 tokens, wave w owns value columns [16w, 16w + 16) and keeps the state S^T as the 8 f32 result fragments of the state update
  (key rows 8 half + i of 16-row blocks); its f16 copy is the rhs fragment of the K S / Q S products as it is (the flash2_rdna4 layout identity).
  Per chunk: scan of the decays (threads), K K^T and Q K^T tiles (one tile per wave), T = (I + A)^-1 by forward substitution in f32 (one thread per column,
  32 rows), R = beta (v - exp(gc) K S), d = T R, O = exp(gc) Q S + P d, S <- exp(G) S + (K exp(G - gc))^T d. Spans of 64+ tokens run it; shorter
  spans (decode, verify, a prompt's tail) the sequential kernel's gfx12 twin (`g_gdn_recurrence_rdna4`: the same kernel text restricted to spans under
  64 tokens). A dual-group launch (no gfx11 counterpart) adds the chunk kernel in the prefill branch: a gfx11 build launches nothing there.
  Numerics: f16 operands (K, Q, S copy, T, P, d, R), f32 accumulation and state: relative error 5e-4 .. 9e-4 of the largest value against float64
  (numpy emulation of the exact dataflow) and the kernel test (`gdn_chunk_spans`: 130 / 64 / 97-token spans, one continuing a stored state) passes at 1e-2 relative.
- **Measured (R9700, same job, ABAB, 2 rounds)**: prefill 1k/2k 1536/1622 -> 1599/1694 tok/s (+4.1 / +4.4 %), decode unchanged (65.1 vs 64.9, tie). The kernel
  costs 0.016 ms/token (the card build's sequential recurrence 0.042): skipping the kernel altogether gives 0.5656 ms/token vs 0.5815 with it (card build 0.608).
  The K S / Q S products, the state update and the output stores are free in the ablation (the MMA work is not the cost; latency is).
- **What failed on the way, and why it matters for the next kernel here** (Loom compile-only checks catch most of it: `loom-compile --target=amdgpu:gfx1201`):
  - The scheduler hoists every independent load of a basic block to its top (scf.if regions are if-converted away, an earlier LDS store does not hold
    back a later LDS load from another buffer): the unrolled 64-row solve (2016 independent A loads) failed allocation at a peak of 600-1400 VGPRs.
    Fix: the A loads of row i depend on T[i - 2] through an index term that is zero at run time (the bits of T and an LDS zero, unknown to the compiler).
  - CN = 64 spilled (5.6 KB scratch per lane; every reload is a global-latency wait): the first working version was 9 % SLOWER than the sequential kernel
    (1498 vs 1644 tok/s at 2k). CN = 32 has no scratch (255 VGPRs, 53.7 KB LDS) and is the version above.
  - Prefetching the next chunk's global loads across the loop (K, Q, norms, alpha, beta in carried registers): 85 spills, 2k 1662 vs 1685 tok/s: not kept.
- Not done: fusing the q/k norm and the conv into the chunk kernel's staging (0.004 + 0.011 ms/token, each below the noise of a doubling); CN = 64 with the
  solve split in two 32-row blocks (T = [[T11, 0], [-T22 A21 T11, T22]] with one MMA each); a second chunk kernel for Strix (WMMA gfx11 form).

## gfx11 chunked prefill on Strix Halo (Qwen3.8-27B Q4_K_M, campaign q38pf, lane gdn)

Switches (all off by default; off = byte-identical programs): `GDN_CHUNK11` (g_gdn_chunk, RDNA3 WMMA, spans of 64+ tokens),
`GDN_CHUNK11_FUSE` (conv1d + SiLU + q/k norms in its staging), `GDN_CHUNK11_GATE` (gated RMSNorm epilogue), `GDN_CHUNK11_CN`
(16 | 32), `GDN_CHUNK11_OPT` (kbo, xpf, scan, acc4), `GDN_SKIP_LONG`, `GDN_FUSED11` (+ `GDN_FUSED11_REGNORM`), `GDN_CHUNK11_ALIAS`.
- **Winner: CHUNK11 + FUSE + GATE + CN=16 + OPT=kbo + SKIP_LONG**: one launch does conv, norms, the WY recurrence and the gated norm
  of the long spans. Local 123 W, same-job ABBA vs base: prefill 1k/2k/8k/16k/32k +3.0/+3.1/+3.3/+3.2/+2.8 % (8k 588.8 -> 608.5
  tok/s), decode tie; DevLab 85 W (44691375703) +3.0/+3.2/+3.3/+3.2/+2.8 % (8k 534.7 -> 552.2). GDN bucket at 8k 0.115 -> 0.054
  ms/token. Kernel tests vs float64 (gdn_chunkf_serve: serving-shaped multi-span epoch) pass. CN=16 alone (plain kernel) +1.2..1.6 %.
- **Why CN=16 wins on gfx11:** a 27B span has 48 value-head workgroups; CN=32 needs 253 VGPR + 53.7 KB LDS = 2 workgroups per WGP
  (40 slots), so 8 run in a second round (a 40-head ablation was -36 %). CN=16 (+ kbo: one state rhs copy live) is 196-200 VGPR and
  26 KB, 3 per WGP, one round, and the serial solve is 16 rows. Fused kernel per 1024-token span: CN32 3.6 ms, CN16+kbo 1.0 ms; the
  plain kernel 955 -> 729-800 us.
- **What does not pay:** the 32-row solve / scan latency (scan unrolled, 4 partial sums: <2 % at CN16), the kdt-in-Q LDS alias
  (occupancy unchanged), the fa/fb split (`GDN_FUSED11`, = the c11 chain: memory-bound on Strix, fa reads the f32 qkv and writes a
  ~34 KB-per-chunk workspace that fb reads back), fb's register norm epilogue (3 WG/WGP, -1 %), CN32 fused staging (-3 % e2e).
- The GDN chain is close to memory-bound on Strix: the f32 qkv projection is 40 KB/token/layer; a chain of separate kernels moves
  ~220 KB/token/layer, the fused kernel ~88 KB. `g_gdn_norm_gate_s` used to compute every long-span row and drop only the store
  (.012 ms/token at 8k) and `g_gdn_qk_norm_lv` normalized every row (.004): `GDN_SKIP_LONG` skips them.
- Timing ablation (fused CN16): no chunk-row global loads -22 % -> loads are exposed per chunk (opt `xpf` prefetches them).
