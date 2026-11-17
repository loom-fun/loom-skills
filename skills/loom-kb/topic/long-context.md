# Topic: long-context (decode and verify attention at 8k-262k)

Split-K decode attention, F16 KV and chunk shapes are in topics decode and prefill; this file holds what is specific to long contexts and drafting.

## Long-context verify: a verify span's tokens share their KV loads (`attn_rules.ATTN_MT`)

- **Long-context verify: a verify span's tokens share their KV loads** (`attn_rules.ATTN_MT`, default 4; from position
  `ATTN_MT_FROM` 1024). When an aligned group of 4 epoch tokens is one row at consecutive positions, the group's first
  split-K workgroup computes all 4 (one K/V load per key for the 4 queries of its head; the others exit). Each token
  keeps its own split range, key masks, 16-key block partition and operation order, and a block holding none of its keys
  leaves its state untouched, so results are bit-identical to the per-token path (drafted == undrafted at 32k). Qwen3-8B
  at 32k, depth 3: verify attention 58 -> 40 ms per epoch (one token: 28). With the DFlash split-K draft attention below,
  the depth-3 epoch went 136 -> 85 ms: 12.5 -> 20.1 tok/s (undrafted 19.5); depth 7 7.7 -> 13.7.
  Only at 8 or more KV heads (`ATTN_MT_MIN_KV`): the span path leaves 1/4 of the workgroups, which run their 4 tokens
  serially, so a small grid loses (same-runner decode_32k, path on vs off: Llama-3.2-3B 8 KV +8 %, Qwen3-4B +21 %;
  Qwen2.5-7B 4 KV 0 %, Flint 4 KV -15 %; Qwen2.5-1.5B 2 KV -6 %, Qwen2.5-0.5B head_dim 64 -27 %; gemma-3-1b 1 KV -17 %
  at 2k).

## SDK 0.2.0: verify query tiling losses on Strix Halo

Qwen3-4B geometry (32 query heads, 8 KV heads, head_dim 128), DevLab Strix Halo, 2026-11-17: five interleaved kernel rounds per variant at a four-token span near 32k. These are isolated kernel results, not end-to-end decode measurements.

- Staging Q in 4 KiB LDS reduced WMMA registers 218 -> 168 but increased host time 5.43% ([initial run](https://github.com/loom-fun/loom-skills/actions/runs/44166215736)). Restoring LDS residency tier 6 by trimming V padding or K/V padding still lost 5.16% / 4.68%; each variant passed five exact partial tests ([follow-up](https://github.com/loom-fun/loom-skills/actions/runs/44170007939)). This parked WMMA control used 128 splits.
- Against shipped scalar attention (64 splits, GQA 8-key blocks), a full-wave 2-head x 2-token tile lost 10.34% host time / 6.54% device-replay p50. Removing only the old short-span wrapper reduced registers 216 -> 144 but tied at 32k: host +2.36%, device p50 +0.76% ([same-job control](https://github.com/loom-fun/loom-skills/actions/runs/44172967060)). All 14 exact assertions passed, covering spans 1/3/4/5/8, the position-4095 GQA handover and a short-context case.

Lower register counts or restored residency alone did not predict a win. No variant was adopted; no end-to-end or spec/consistency gate improvement is established, and these losses do not establish a current SDK blocker. Revisit with new profile evidence rather than repeating these configurations.

## DFlash v1 draft attention is split-K (`gen_dflash.DFL_SPLITS`)

- **DFlash v1 draft attention is split-K** (`gen_dflash.DFL_SPLITS`, 32; windowed DFlash2 and conv drafters keep the
  per-token kernel). The per-token kernel (one workgroup per block token and query head, one key per step) re-read
  the context once per block token and query head: 46 ms of the 136 ms 32k epoch above. `g_dfl_ctx_split` scores a
  chunk of 8 block tokens per wave against each key loaded once, per (KV head, split); `g_dfl_attention` merges the
  splits and adds the block's own keys.

## History: long-context verify paid decode attention per token

- **History: long-context verify paid decode attention per token.** The per-token split-K kernel costs the same per verified
  token as per decoded token (it streams the KV once per token: Qwen3.8-27B at 32k, 671 us per layer for 1 token,
  2058 us for 4, `loom-benchmark` dispatch time), so depth 7 falls behind depth 3 at long context and the 27B package
  uses depth 3. Two cheap fixes are in: a block's 16 key scores are reduced by one butterfly all-reduce (~31 shuffles
  instead of 16 subgroup reductions) and the 16-key blocks are absolute-aligned, so a block is one KV page and needs one
  page lookup (32k depth-3 epoch 135 -> 129 ms; at 32k the package now decodes 23.2 vs llama.cpp's 21.2 tok/s, three
  runs each). Tried and dropped: two or four independent softmax chains per block (no change); a workgroup per segment
  of same-row tokens staging each key block once in workgroup memory, double-buffered (exact; 4 tokens 1746 vs
  2058 us per layer in isolation, but slower in the model); and a WMMA split kernel with the span's tokens as MMA
  columns (32k epoch 210 -> 165 ms at depth 7) that is not exact: RDNA3.5's WMMA result for a column changes with the
  V rows of keys the column masks to P = 0 (1-ulp; verified in isolation by zeroing those rows), so a token's
  attention depended on its span and the spec gate failed (15/16). An exact design would run the 16-aligned key range
  a segment shares on WMMA and each token's ragged ends on the per-token path.

## Context gate (`SPEC_CTX_MAX`)

Context gate (`SPEC_CTX_MAX` or `cfg["drafter"]["ctx_max"]`, default 0 = off): `encode_epoch` also gates an epoch (all spans
speculative) whose deepest span's causal position (plan word 1) is >= ctx_max, through the same anchor-only path, because
verifying depth + 1 tokens per span costs depth + 1 attention passes over the whole context (Ornith-1.5-35B depth 3: short
rows 118 vs 62 tok/s, 16k 44.9 vs 53.5). tune.py retries the best depth of each kind with ctx_max 3500 when only the long
row loses, and ranks that candidate by the same margin; a card whose long row wins (80 % acceptance at 32k) keeps ctx_max 0.
`occ_min` and `ctx_max` are recorded in `cfg["drafter"]`; the env variables win.
