# Family inkling2: long-context

Strix Halo gfx1151, FP16 KV, hrx-loom 0.2.0, 2026-11-22/23. Same-session cool A/B (start below ~65 C, sclk recorded).

## Prefill attention past 32k (`g_ink_attention_flash2_far`) — kept

- The `TS=8` far-key kernel spilled 88 B (38 `scratch_*` instructions) inside the key loop, which put an
  `s_waitcnt vmcnt(0)` between the two K/V prefetch loads (found in the compile report and `llvm-objdump` of the hsaco).
- Packing P to f16 before the half-wave XOR-16 shuffle (`vector<8xf32> -> vector<8xf16>`, bitcast to `vector<4xi32>`, 4
  shuffles instead of 8) removes ~24 live VGPRs and is bit-identical (`INK_FAR_PF16=1`). That made room for a 2-buffer LDS
  ping-pong (`INK_FAR_DB=1`: 37,888 B LDS, 225 VGPR, 0 spills, 1 barrier per 32 keys instead of 2). Technique credit:
  woct0rdho/feather-attn (MIT).
- Kernel at 66k tokens: 22.98 -> 26.4-26.8 TFLOP/s (+15-17 %). TTFT 64k / 100k / 150k / 200k: -2.1 / -4.5 / -5.6 / -4.4 %
  (-8.05 s at 200k). 8k-32k prefill and decode neutral. Not kept: `TS=4` (re-streams K/V, -49 %), a head-fast grid (-2 %:
  tile-fast shares a KV head in the 32 MB MALL), an `scf.if` guard around the prefetch (the compiler waited at the join),
  hoisting the LDS store indices (spilled again), `KT=64` (JIT preparation over 240 s).

## Prefill chunk size — 4096

Sweep at 8k-100k: 2048 is 5-5.5 % slower at 64k-100k (twice the KV passes, smaller MoE buckets), 8192 is 4.6 % slower at
64k and 9.9 % at 100k (bigger workspace, MALL pressure, +1.5 GiB). Keep 4096.

## Decode attention occupancy — raising it loses

Rolling the 8-row bodies of the span decode attention (`g_ink_attn_split_span_q{1,2,4}`) and the DSpark full-attention
kernel into `scf.for` loops cut VGPRs (q4 192 -> 128, spills gone) and raised waves per SIMD (q1 9 -> 12, q4 5 -> 8), and
lost on every row: q1 -24 to -28 % at 40k-162k, q4 -3.5 to -63 %, DSpark -25 to -125 %. The 64-split KV-streaming kernels
already run q1 at 202-218 GB/s; more concurrent waves thrash the caches and compete for outstanding loads, and fewer
up-front loads (`pipe=2`) starve memory-level parallelism. What q4 pays over q1 is 4x the per-key math, not occupancy.

