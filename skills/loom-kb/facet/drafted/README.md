# Facet: drafted

Speculative decoding present in the package (mtp, dflash/dflash2, assist, ngram). The kernels of the verify step, the drafter head and many-stream drafted decode. How a drafter is chosen and measured: topics speculative and spec-tuning.

## Contents

- Drafter bytes per step: load-time preparation (single-stream DFlash)
- Many-stream drafted decode (4 streams x DFlash depth 3)
- Did not help (drafted multi-token matvecs)

## Drafter bytes per step: load-time preparation (single-stream DFlash)

- **Count the drafting step's bytes beyond the trunk**: the drafter's weights and a second full head pass for the
  proposals (Bolt-E4B: 0.71 GB head of ~4.4 GB per step). Doubling the head launches attributes them exactly
  (+2.5-2.7 ms each); doubling small drafter matvecs measures nothing (they re-read from the 32 MB cache).
- **Weight policies convert at load, the card keeps the published files** (`starter/draft/prep_weights.py`): a BF16
  drafter becomes Q8_0 (int8-activation matvecs, half the bytes) and the proposals read a Q4_0 copy of the head
  (verification keeps the exact head, so outputs cannot change). Bolt-E4B depth 3 on DevLab: 82.6 -> 89.5 (Q8_0
  drafter) -> 98.0 tok/s (Q4_0 proposal head), +23 % over llama.cpp b1368's own drafting (79.9).
- A multi-model loom-serve is fine for A/B ratios but not absolute numbers (~2.5 ms more per step than served alone).

## Many-stream drafted decode (4 streams x DFlash depth 3)

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
- **Against llama.cpp at 4 streams the step time is now level; acceptance decides the throughput.** Same session
  (c4_steady x 3, medians): Loom 67.6 -> 67.7 tok/s, llama.cpp b1355 70.1 tok/s. Per step: Loom ~145 ms at 2.57
  tokens per stream, llama.cpp ~146 ms at 2.64 (its log's "mean len"). In an earlier session Loom 67.6 vs llama.cpp
  67.5. Tokens per step vary +-3 % between runs on both sides, so one c4 run cannot separate them; compare
  c4ep's ms per epoch for kernel changes.

- **Tokens per step: llama.cpp's DFlash2 proposal matches Loom's at temperature 0** (code comparison, llama.cpp
  `common/speculative.cpp`): same tapped residuals (`target_layers` t = input of layer t), anchor without its own
  feature, block = n_max + 1 at positions P..P+3, mask token, non-causal attention, conv taps, logit scale and softcap,
  top-16 selector with the rank-256 transition walked greedily from the anchor (the mt19937 path is only for
  temperature > 0), always 3 drafts (n_min = p_min = 0). Remaining differences are tie-breaking and precision.
  The 2.57 vs 2.64 tokens per step is run-to-run text variance: c4ep's tokens per event ranges 2.50-2.69 between
  rounds of the same Loom package, because 4-stream batching timing changes the text.

## Did not help (drafted multi-token matvecs)

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

- **Hoisting the int8 matvec's per-token F32 scaling out of the row loop.** The ~150 F32 instructions per row are
  (sitofp, x scale, minimum x code sum, subtract, x activation scale) per token and weight. Every product mixes a
  per-row with a per-token factor, so nothing is shared across rows without reassociating (which changes rounding
  in every int8 kernel, the 1-token decode included).
