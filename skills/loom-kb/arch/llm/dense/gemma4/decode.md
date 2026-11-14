# Family gemma4: decode

## Nemotron-3.5-Bolt-E4B drafting: BF16 drafter as published, Q8_0 at load, Q4_0 proposal head

BF16 drafters as published (Nemotron-3.5-Bolt-E4B Q4_K_M, NVIDIA's BF16 DFlash, same-session perf.py, two runs each,
c4_steady three runs; llama.cpp b1368 drafting from the same BF16 GGUF):

| | decode | @2k | @32k | prefill 1k / 8k / 32k | c4 (median) |
|---|---|---|---|---|---|
| Loom, BF16 drafter, depth 3 (package choice) | 81.9 | 70.6 | 51.3 | 2223 / 1835 / 929 | 197.4 |
| Loom, BF16 drafter, depth 7 | 81.8 | 61.0 | 45.1 | 2179 / 1811 / 907 | 154.1 |
| Loom, Q8_0 requantized drafter, depth 7 | 90.1 | 65.3 | 45.9 | 2208 / 1830 / 928 | 164.9 |
| Loom, no drafting | 58.7 | 56.2 | 44.9 | 2314 / 1872 / 937 | 211.3 |
| llama.cpp, DFlash n-max 7 (BF16) | 89.3 | 58.4 | 37.2 | 1312 / 1291 / 732 | 169.0 |
| llama.cpp, no drafting | 51.9 | 44.5 | 33.2 | 1350 / 1304 / 739 | 155.9 |

- The BF16 drafter costs ~9 % of drafted single-stream speed against the Q8_0 copy (twice the bytes per proposal on
  the f32-activation 4-token matvec); the copy cannot be referenced by a published card.
- Depth 3 is the package setting: same short decode as depth 7 on perf.py's essay, better with KV in use and with 4
  streams (verification of 4 x 8 tokens takes the GEMM). Four drafting streams still trail no drafting (197 vs 211):
  a 4 ms model step leaves little to win at 39 % acceptance.
- **Now (PR #42): the BF16 drafter is converted to Q8_0 at load and the proposals read a Q4_0 head copy** (both through
  weight policies, `prep_weights.py`; the card still references NVIDIA's BF16 file). DevLab Strix Halo, one runner,
  perf.py means of 4 runs, c4 median of 3, llama.cpp b1368 with the same BF16 drafter at n-max 7:

  | | decode | @2k | @32k | prefill 1k / 2k / 8k / 32k | c4 |
  |---|---|---|---|---|---|
  | Loom, depth 3 (this package) | **98.0** | **81.4** | **63.6** | 2310 / 2430 / 2073 / 1017 | **223.9** |
  | Loom main (BF16 drafter as published), depth 3 | 82.6 | 70.6 | 53.4 | 2289 / 2416 / 2078 / 1013 | 189.5 |
  | llama.cpp, DFlash n-max 7 | 79.9 | 60.4 | 37.7 | 1407 / 1400 / 1625 / 782 | 186.6 |

  Per step on the essay (perf.py's decode request): main 25.1 ms and 2.09 tokens; Q8_0 drafter 23.5 ms (same
  tokens); plus the Q4_0 proposal head 22.1 ms and 2.16 tokens. llama.cpp takes ~48 ms per step for 3.5-3.9 tokens
  (it decodes a different text: llama-server turns thinking on by default, the package follows the template's
  default off). llama.cpp's own decode is bimodal across runs on DevLab (~97 on the first request of a fresh
  server, 73-81 after), so compare means of several runs. The table above used each engine's default thinking.
  With perf.py's `--thinking` (now default off, the same `enable_thinking` sent to both engines; text_match.py
  shows the same text up to near-ties), one runner each, llama.cpp at its best n-max (3):
  thinking off 97.6 vs 61.4 tok/s (+59 %; n-max 7: 50.1), @32k 63.4 vs 49.6, c4 219 vs 165;
  thinking on 113.1 vs 83.5 (+35 %; n-max 7: 82.5), @32k 72.9 vs 51.9, c4 270 vs 198.
- Tried (Bolt, DevLab, no gain): depth 7 (93-94 tok/s short, but @2k 62-70 vs 81: 8-token verification costs ~4.5 ms
  more per step than 4 for +0.4 accepted tokens on this essay); fewer rows per workgroup for the 8/16-token int8
  matvecs on small outputs (MV_WG8 128-1024: flat to -5 %); the drafter's own widths in the R picker (flat); a Q4_0
  drafter (+0.5 %, not worth its own kernel path).
- Tried: proposals of 5+ tokens on the BF16 WMMA GEMM instead of the 4-token matvec (one weight pass): depth 7
  146.3 -> 137.2 tok/s, slower (64-row padded tiles for 8 tokens). Drafter FFN with SiLU instead of the gemma4 kit's
  GELU (llama.cpp's dflash graph uses SiLU unless `hidden_act` says otherwise): no change (145.0 vs 145.7).
