# Facet: moe: prefill

## Prefill chunk size and padding (`prefill_capacity`)

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
  - Padding rows in the per-row kernels (round 2): gen_llm `live_row_skip` gives every per-token-row kernel launched
    over `[%t]` in a prefill branch of `@llm_advance` a `_lv` copy that does nothing on a padding row (info word 2 = 0)
    except write the zero row scale the GEMM tile skip needs (residual add + norm, row scales, QK-norm + RoPE, MoE
    combine; any kit kernel with the one-workgroup-per-row pattern). Flint also: `g_moe_route_tr_live` skips
    all-padding router workgroups; `g_row_scale_f16_tiles` skips the unused tiles of the bucketed expert rows
    (max_tiles is sized for the worst case). Flint (DFlash d3), one server, interleaved, vs the final2 build:
    1k 870 -> 940 tok/s (+8 %; +9 % and +14 % in two other sessions, 947 and 963 tok/s), 2k/8k/32k and decode tie.
    Qwen3.8-27B (dense, chunk 1024: no padding at 1k): tie at 1k/2k/decode. `LIVE_ROW_SKIP=0` turns it off.
  - What is left at 1k is not padding: doubling attributes Flint's 1.09 s per 1k prompt to the expert grouped GEMMs
    0.80 s, per-row kernels 0.13 s, dense GEMMs 0.10 s, attention 0.03 s. With ~24 rows per expert, each expert's
    weights are dequantized once per (mostly single) row tile, so the expert GEMM cost barely depends on the token
    count (2k adds only 0.52 s). Skipping the WMMA of all-padding 16-row waves in the expert tiles: tie (1k +2 %,
    2k -2 %), so the MMA is not the limit; the per-k-step dequant/stage/barrier is. Faster 1k needs a cheaper
    expert weight path (e.g. int8/int4 WMMA on the codes), not less padding.
