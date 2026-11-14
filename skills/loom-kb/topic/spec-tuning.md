# Topic: spec-tuning (choosing the drafter and depth; measured results)

## Choosing the drafter and depth (`tune.py`, qualify.sh stage 0)

Builds each available kind once at the full native context (build.py sizes each candidate's memory; a drafter adds
KV per position), serves each depth (a `model.json` setting; DFlash up to the drafter's block - 1) alone, times
decode-heavy requests (essay, code, a tool call, a reasoning answer; thinking off) plus one decode after the 32k document the card's decode_32k row uses
(`TUNE_LONG`, the largest document the context holds) and, with a multi-row preset, `rows` concurrent essays (`streams4`). A candidate's margin is its worst
row relative to depth 0; it is used only if no row is slower than 0.98 x depth 0 and its single-stream rate beats no
drafting by 3 %, and the best margin ranks first. Low-acceptance drafting loses on exactly the prose, 32k and 4-stream
rows (every span verifies depth + 1 tokens, padded to 16 at 4 streams): Qwen3.5-0.8B ngram depth 3 decode 223 vs 258 tok/s
without, 32k 125 vs 166, four streams 498 vs 787, so tune picks none; a span with real acceptance (Muse-Glimmer depth 3 over
none on 4 streams) keeps it, and the depth whose rows x (depth + 1) fits 16 is always a candidate. With several rows each drafter is also built ungated (`SPEC_OCC_MIN=0`, the occupancy gate above) and its streams4 measured per depth; a depth takes the better streams4 and the winner's `occ_min` goes to `cfg["drafter"]["occ_min"]` (build_draft.py reads it; the env var overrides): Muse-Glimmer depth 3 ungated 74 tok/s at 4 streams, gated 48.6. The best margin that also passes the spec gate's equivalence check (spec_check.py's requests at the
candidate depth vs draft_depth 0 of the same package: identical) and a concurrency check (consistency.py-style
~300-token prompts, two 0.5 s apart and four at once, twice each: agentic serving overlaps requests) wins; one that
fails is recorded with the reason (`checked` in the table) and the next is checked. The choice and the whole table go to `cfg["drafter"]` and
`bench/draft_tune.json` (put the table in the report). The llama.cpp baseline gets the same treatment over its own
`--spec-type draft-mtp` / `draft-dflash` (loom-benchmark `llama_spec_tune.py`), so reports compare drafting with
drafting. Measured on Strix Halo (greedy decode; tune.py requests, thinking off):

| Model | No drafting | MTP | DFlash2 | llama.cpp b1355 (same drafters) |
|---|---|---|---|---|
| Qwen3.8-27B Q4_K_M (bartowski), 262k context, vision | 12.6 tok/s | depth 3: 32.4 tok/s (depth 4: 32.6) | **depth 7: 47.7 tok/s**; depth 3: 35.4 (the published package's setting, better at long context; spec gate 16/16 incl. images) | none 12.0, MTP n-max 3 26.0, **DFlash2 n-max 3 28.1** |

Published Qwen3.8-27B package (DFlash2 depth 3) vs llama.cpp b1355 with its best drafting, same session, three runs
each (perf.py; c4_steady two runs): decode 26.9 vs 21.9 tok/s, @2k 27.9 vs 20.1, @32k 23.2 vs 21.2, prefill 1k-33k
372/393/365/324/264 vs 311/311/321/299/249, four streams 50.9 vs 48.3.
| Qwen3.8-27B Q8_0, 262k context | 7.9 tok/s | **14.6 tok/s** (49 % acceptance; 84 % in tool loops) | DFlash2 depth 7: 13.0 tok/s (53 %, 4.7 tokens/step) | - |
| Qwen3.5-0.8B Q4_K_M | **262 tok/s** | 208 tok/s | - |
| Qwen3-4B Q4_K_M | - | - | DFlash depth 7: 39 % acceptance, 3.7 tokens/step |

Other families (Nov 2026, this skill tree with int8 activations (`ACT_Q8`, "Limits"), re-tuned with the final kernels; tune.py and llama_spec_tune.py
on the same requests, llama.cpp b1364; every Loom setting below passed tune.py's equivalence check against
draft_depth 0 of the same package and its concurrency check):

| Model (arch, quant) | Drafter | Loom: none -> best | llama.cpp: none -> best | Loom before int8 activations |
|---|---|---|---|---|
| Nemotron-3.5-Bolt-E4B (gemma4, UD-Q5_K_XL) | NVIDIA DFlash, BF16 requantized to Q8_0 | 47.4 -> **115.9** (DFlash depth 7, x2.45; assistant depth 3: 108.8) | 45.3 -> 99.7 (DFlash n-max 7, x2.20) | 46.7 -> 73.4 |
| Nemotron-3.5-Bolt-E4B (gemma4, NVIDIA Q4_K_M; llama.cpp b1368) | NVIDIA DFlash **BF16 as published** (issue #31) | 57.6 -> **147.7** (depth 7, x2.56; depth 5 129.4, depth 3 116.2); the Q8_0 requantized copy: 161.8 | 51.9 -> 93.6 (same BF16 file, DFlash n-max 7; n-max 3/5: 89.6 / 93.4) | - |
| Qwen3.5-4B (qwen35 hybrid, UD-Q4_K_XL, vision with M-RoPE, F16 tensors) | AtomicChat DFlash | 62.8 -> **121.9** (DFlash depth 7, x1.94) | 57.2 -> 105.0 (DFlash n-max 7, x1.84) | 62.9 -> 80.2 |
| Muse-Glimmer-30B (muse-glimmer, UD-Q5_K_M) | z-lab DFlash2 | 12.4 -> **37.1** (DFlash2 depth 7, x3.00) | 11.6 -> 25.4 (DFlash2 n-max 3, x2.18) | 12.2 -> 20.5 |
| Flint (flint MoE, Q4_K_M) | incoai DFlash2 | 63.7 -> **102.6** (DFlash2 depth 3, x1.61) | 53.5 -> 86.0 (DFlash2 n-max 3, x1.61) | 65.6 -> 90.5 |
| LFM2.5-24B-A2B (lfm2moe, Q4_K_M) | none exists: n-gram only (short-conv replay) | **99.7** (n-gram 71.1: none kept) | **94.7** (ngram-simple 90.1, ngram-mod 93.6) | 102.7 |

Drafting works on every family, and with the int8-activation verify step Loom's drafted decode now leads llama.cpp's
on every model measured, in absolute speed and (but for Flint's tie) in speedup over no drafting; the multi-token step
was the gap ("Limits"). Depth 7 wins on these short requests where the drafter accepts long runs: up to 16 tokens
verify in one weight pass and round like 1-token decode. At long context each verified token still pays decode
attention (less since the multi-token split-K path, "Limits"), so a lower depth can win for long-context use
(Qwen3.8-27B above: depth 3 for its package).

Small models are launch-bound: drafting loses there and `tune.py` keeps ordinary decoding.

## Drafting as an optimization lever

**Speculative decoding** (on by default when the GGUF has an MTP block or a DFlash drafter exists;
topic speculative; n-gram is opt-in, `TUNE_NGRAM=1`, never on a card). `qualify.sh` chose the method and depth by
measurement (`bench/draft_tune.json`); card comparisons use the model's classified drafting choice on both engines
(`loom-benchmark/reference/methodology.md`), so decode targets are drafting vs drafting; re-run `draft/tune.py WORKDIR PORT MEMORY_GIB MODEL_ID` after a change that moves
the decode/verify balance (it rebuilds `pkg`). On a bandwidth-bound model it is the biggest decode lever
(Qwen3.8-27B Q4_K_M: 12.8 -> 21.0 tok/s with MTP depth 3); on small launch-bound models it loses (Qwen3.5-0.8B:
262 -> 208) and tune.py keeps it off. The multi-token target step that used to cap it is fixed in the starter (int8 activations: topic decode). Remaining levers: verify epochs
of 17+ tokens (many drafting streams) take the WMMA GEMM; long-context verify pays decode attention per token (topic long-context), drafter catch-up cost at prefill, and the MoE expert
matvecs (starter `gen_moe.py`, the lfm2moe kit), which still use f32 activations. Every change
here must keep the `spec` gate exact.
