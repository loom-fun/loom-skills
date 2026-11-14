# Facet: drafted: decode

## Drafter bytes per step: load-time preparation (single-stream DFlash)

- **Count the drafting step's bytes beyond the trunk**: the drafter's weights and a second full head pass for the
  proposals (Bolt-E4B: 0.71 GB head of ~4.4 GB per step). Doubling the head launches attributes them exactly
  (+2.5-2.7 ms each); doubling small drafter matvecs measures nothing (they re-read from the 32 MB cache).
- **Weight policies convert at load, the card keeps the published files** (`starter/draft/prep_weights.py`): a BF16
  drafter becomes Q8_0 (int8-activation matvecs, half the bytes) and the proposals read a Q4_0 copy of the head
  (verification keeps the exact head, so outputs cannot change). Bolt-E4B depth 3 on DevLab: 82.6 -> 89.5 (Q8_0
  drafter) -> 98.0 tok/s (Q4_0 proposal head), +23 % over llama.cpp b1368's own drafting (79.9).
- A multi-model loom-serve is fine for A/B ratios but not absolute numbers (~2.5 ms more per step than served alone).

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
