# Speculative decoding in starter ports (MTP, Gemma-4 assistant, DFlash, DFlash2, n-gram)

Ports draft by default whenever their GGUF supports it, and stay token-for-token identical to the target. Nothing
needs configuring: `port.sh` enables a drafter, `qualify.sh` measures which drafter and depth actually decode
fastest (or none), and the gates prove the output did not change. `SPEC=off` opts out (port, build, qualify).
It works for every family the starter and the reference examples port (dense, gated delta-net and short-conv
hybrids, MoE, sinks/sliding window, Gemma-style scaled embeddings, vision): the verifier is the trunk itself, the
drafters only need the target's embedding, head and hidden features, and every model has at least the n-gram
drafter as a candidate.

## Drafters

| Kind | When it is available | What drafts |
|---|---|---|
| `mtp` | the GGUF carries NextN/MTP blocks (`blk.L.nextn.*`; Qwen3.5/3.6/3.8, unsloth `*-MTP-GGUF`; `inspect_gguf.py` records `cfg["mtp"]`) | the model's own NextN block, `draft_depth` autoregressive rounds |
| `dflash` | a DFlash (v1) drafter GGUF exists for the base model (`find_drafter.py`) | a small non-causal block-diffusion transformer over the target's hidden features; one pass proposes a whole block |
| `dflash` (v2 / DFlash2) | as above, drafter with `attn_conv_proj` / `selector_*` tensors | DFlash plus 2-tap conv modulation, a 2048-key context window and a low-rank candidate selector |
| `assist` | a gemma4-family target and a Gemma-4 "assistant" GGUF (arch `gemma4-assistant`, e.g. `AtomicChat/gemma-4-E4B-it-assistant-GGUF`; `find_drafter.py` records `cfg["assistant"]`) | Gemma 4's own MTP head as a separate ~100 MB model (`gen_assist.py`): `draft_depth` autoregressive rounds of 4 small layers that attend the target's own K/V (sliding layers: target layer L-2's cache, full: L-1's), fed by the target's embedding and post-norm hidden like MTP |
| `ngram` | always (`gen_ngram.py`; tune.py tries it for every model) | prompt lookup: the newest longest suffix match of the anchor in the row's own token history proposes what followed it. No weights; wins where output quotes context (code edits, tool arguments, JSON) |

`find_drafter.py` (port.sh stage 2c) searches HF by the base model's name (repo without -GGUF, `general.name`,
without -Instruct; a bare-name search also finds vendor variants such as `nvidia/<base>-4GB-DFlash`), ranks exact
name matches, then z-lab, then downloads, checks each candidate's header before downloading (arch `dflash`, hidden
size, tap layers inside the trunk, vocabulary, mask token, tensor formats, and no drafter feature `gen_dflash.py`
lacks: reduced draft vocabulary, own embedding/head, shared K/V, sinks, causal block, DSpark hyper-connections) and
records `cfg["drafter"]`. A BF16/F16-only drafter (NVIDIA publishes Nemotron's that way) is used as published:
the package's model.json references the vendor's file and a published card resolves it to the vendor repo (no
converted weights to host). loom-serve converts its matrices to Q8_0 once at load (`prep_weights.py`: the drafter
root's weight policy `draft_weights.loom` stages each BF16/F16 matrix and runs `llm_prep_<src>_q8_0`), so the drafter
runs the int8-activation matvecs at half the bytes (kernel tests `mv_dr_*_prep_t*`). `DRAFTER_PREP=0` keeps the
starter's BF16/F16 kernels (f32-activation 1/4-token matvecs, WMMA GEMM with the weights rounded to F16;
kernel tests `mv_dr_*` / `dr_*_gm*`). `--requantize`
(`DRAFTER_REQUANT=1`) writes a local Q8_0 copy instead (`drafter/*-Q8_0.gguf`, ~0.4 % max relative error; half the
bytes per proposal: Bolt-E4B depth 7 ~9 % faster, see below, but only for local use). `DRAFTER_REPO=org/name[:file]`
forces one.
Safetensors-only drafters are not converted.

## How a step runs (native contract: hrx-loom 0.1.0 text service, `draft_depth` 1-7)

1. **propose** (`@llm_mtp_propose` / `@llm_dflash_propose` / `@llm_ngram_propose`): for every span loom-serve marks PROPOSE, the drafter
   writes `draft_depth` tokens into the verifier's input slots on device (no host round trip). One stage per
   power-of-two cohort (1, 2, 4 ... spans).
2. **verify** (`@llm_verify`): the unchanged trunk runs anchor + drafts like a prefill chunk. The head predicts
   every position through synthetic records, so sampling draws at the true position: seeded sampling stays
   identical to ordinary decoding. `g_spec_accept` keeps the longest matching prefix (EOS and the output credit
   stop it), publishes `{consumed, outputs, tokens}` per span, and builds the committed and replay packets.
   Gated delta-net layers capture their inputs instead of publishing state for speculative spans; a replay pass
   recomputes conv + recurrence for the accepted prefix only and publishes that. K/V written past the accepted
   frontier is unreachable and overwritten later. Other recurrent layers use the same scheme through a hook (next
   section).
3. **warm** (`@llm_mtp_warm` / `@llm_dflash_warm` / `@llm_ngram_warm`, after every epoch, prefill included): the
   drafter's own state at every committed position (MTP: the NextN block's K/V from (token P, target hidden P-1)
   plus the per-row carry; DFlash: `fc(features)` -> norm -> each draft layer's context K/V; n-gram: the token id,
   4 bytes). The draft cache shares the target pool's page ids, so drafting covers the whole context, history
   reuse and checkpoints with no extra state.

**DFlash proposals read a Q4_0 copy of the head.** Proposing runs the target's head over every proposed position:
a second full head pass per step (Nemotron-3.5-Bolt-E4B: 262144 x 2560 Q8_0, 0.71 GB, ~2.5 ms of a ~25 ms step).
When the head is Q8_0/BF16/F16 and int8 activations are on, `prep_weights.py` gives the propose stage one more
parameter root, the target checkpoint through `head_weights.loom`, which converts the head to Q4_0 once at load
(llama.cpp's quantize_row_q4_0; +0.38 GB) for `g_head_mv*_q8_p40`. Verification still runs the target's own head, so
the output does not change; only which drafts are proposed can (Bolt-E4B: acceptance 36 % -> 39 %, step 23.5 -> 22.1 ms
on DevLab). `PROPOSE_HEAD=off` keeps the target's head. Kernel test `propose_head_q4`.

Files: `gen_draft.py` (verify, acceptance, GDN capture/replay, the generic replay hook, MTP), `gen_dflash.py`
(DFlash/DFlash2), `gen_assist.py` (Gemma-4 assistant), `gen_ngram.py` (prompt lookup), `prep_weights.py` (load-time weight
preparation: Q8_0 drafter, Q4_0 proposal head), `build_draft.py` (prepare/storage/control patches; ports without
a drafter are byte-identical to before), `find_drafter.py`, `tune.py`.

## Recurrent layers other than gated delta-net (kits)

A speculative span must not publish recurrent state for inputs that get rejected. Gated delta-net is built in. Any
other stateful mixer registers a replay in `gen_llm.MODEL_PARTS["replay"]` while `model()` emits its layers:
`{"name", "match": the prefix of the trunk launch that reads the span's inputs (it must contain %layer_N),
"buffers": [(ssa, f32 width per token)], "replay": f(layer) -> program text}`. The verifier captures the buffers
before every matching launch; after acceptance it restores them and appends `replay(layer)`, which re-runs the
state commit over the `%replay` packet (the accepted inputs, original cursors). The kit's commit kernel must skip
speculative spans (`@g_gdn_publish`, flag bit 4) when `LLM_DRAFT` is set. Example: LFM2 short-conv
(`reference/lfm2moe-example/gen_sconv.py` `replay_spec`, registered in its `sconv_layer`). A new stateful layer
without a replay corrupts its state on the first rejected draft; the `spec` gate catches it.

## Gemma-4 assistant (`gen_assist.py`)

llama.cpp src/models/gemma4-assistant.cpp, mapped onto gen_draft.py's MTP machinery (per-row carry of the target's
final normalized hidden, `g_mtp_prepare` / `g_mtp_gather` / `g_mtp_advance`). Each round: target embedding *
sqrt(H) concatenated with the hidden -> `nextn.pre_projection` -> 4 layers (query-only attention, GELU FFN, post
norms, `layer_output_scale`) -> the assistant's tied head (argmax) and `nextn.post_projection` (the next round's
hidden). The attention reuses the trunk's own decode kernels for the two target layers (parsed from the program text,
`target_attention`): the assistant's QA heads are placed in the target's QH-head layout so head a reads KV head
a // (QA/KVH), and `g_as_visible` clamps every round's queries to the anchor's context (positions < P). The propose
stage therefore binds the target's origins table and arena (build_draft.py); warm only saves the carry.
Nemotron-3.5-Bolt-E4B with the E4B assistant: acceptance equals llama.cpp's (depth 1: 75 % vs 77 %, depth 3: 57 % vs
57 %), spec gate 16/16 exact including images, 46.9 -> 72.7 tok/s at depth 3.

## Target-specific details DFlash handles

- Block embedding: llama.cpp's DFlash embeds the block with the target's raw token row times the drafter's own
  `dflash.embedding_scale` (Gemma-family drafters carry sqrt(hidden)), then the drafter's first norm with its own
  eps. When the target applies an embedding norm (Muse-Glimmer) or scale (Gemma), or the eps differs,
  `gen_dflash.py` emits `g_dfl_embed` for that; otherwise the trunk's `g_embed` is reused (Qwen packages unchanged).
- Features are the trunk's residual stream at the input of each tap layer (llama.cpp's layer inputs), captured at
  the `// ---- layer N` markers; kits must keep those markers and the residual in `%x`.
- The feature projection reads taps x hidden values per token: past 32768 (hidden > 6553 with 5 taps, e.g.
  Muse-Glimmer 6656) the decode matvec takes `mv_wide` (several 32-value groups per lane). The device plan copy marks speculative spans with flag bit 4 and the output
credit in bits 8-15; plan word 4 holds the packed cursor (`encode_epoch`).

## Choosing the drafter and depth (`tune.py`, qualify.sh stage 0)

Builds each available kind once at the full native context (build.py sizes each candidate's memory; a drafter adds
KV per position), serves each depth (a `model.json` setting; DFlash up to the drafter's block - 1) alone, times
decode-heavy requests (essay, code, a tool call, a reasoning answer; thinking off) and ranks candidates that beat
no drafting by 3 %. The fastest that also passes the spec gate's equivalence check (spec_check.py's requests at the
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
attention (less since the multi-token split-K path, "Limits"), so `tune.py` scores every candidate on a 16k-token
decode too (harmonic mean with the short requests): a depth that only wins on short text is not chosen.

Small models are launch-bound: drafting loses there and `tune.py` keeps ordinary decoding.

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

## Gates

- `spec` (gates.sh quick/full, and port.sh's smoke stage): the same package with `draft_depth 0`
  (`model_target.json`, same compiled program) is the reference; every completion (greedy, seeded sampling,
  tool calls, reasoning, a 1k-token prompt) must be identical. `spec_check.py compare` also prints acceptance.
- `agent` and `needle` run with drafting on: multi-turn tool loops with history reuse, and the full context.

## Limits / next levers

- **Fixed: the multi-token target step was the drafting bottleneck** (issue #21). Decode-shaped linears and the
  head kept f32 activations, so each lane of the 4-token matvec held 4 x 32 f32 values (192+ VGPRs, low occupancy)
  and decoded every weight group to f32: Nemotron-3.5-Bolt-E4B, no drafting, 1/2/4 streams took 21.5 / 26.2 / 35.0 ms
  per step (llama.cpp verifies 3-4 tokens for ~+5 ms), Qwen3.8-27B 78.6 / 90.8 / 131.9 ms. The starter now does what
  llama.cpp's `mul_mat_vec_q` does (`ACT_Q8=1`, the default): `g_quant_q8` quantizes each activation once per
  32-value group (`d = amax/127` rounded to F16, `q = round(x/d)`, llama.cpp's `quantize_q8_1` numerics), launched in
  the decode branch beside the GEMM's `g_row_scale_f16`; the `_q8` matvecs keep the starter's layout (one lane per
  group, R rows, TG tokens) but contract the raw weight codes with `vector.dot4i` (Q8_0, Q4_K, Q5_K, Q6_K; other
  formats keep the f32 kernel) and come in 1-, 4-, 8- and 16-token variants; the head (`g_head_mv*_q8`) in 1-, 4-, 8- and
  16-span variants (each 128-row chunk in sub-blocks of 512/TG rows: see below). A token's result is an exact integer dot followed by the same F32
  scaling, and its wave sum is an explicit xor-16/8/4/2/1 butterfly that the multi-token variants run by recursive
  halving (TG + 5 shuffles instead of 5 TG; Qwen3.8-27B with 4 drafting streams: 254 -> 230 ms per epoch) on the same
  pairing tree, so verification of up to 16 tokens rounds exactly like 1-token decode and verify epochs take the WMMA
  GEMM only from 17 tokens (`SPEC_MM_FROM`, was 5). Each token's integer dot is eight accumulating `dot4` steps
  (no vector reduce), and the 8/16-token variants take 16 rows per workgroup (a row's arithmetic does not depend on R:
  explicit ops and butterfly), which cuts the activation reloads: Qwen3.8-27B with 4 drafting streams at depth 3 went
  254 -> 230 (butterfly) -> 177 ms per epoch (chained dots, R = 16), steady c4 37 -> 51 tok/s, above llama.cpp's own
  drafting (48). Running a 16-token variant as two 8-token chunks in one workgroup was no faster. Nemotron 1/2/4
  streams: 21.4 / 22.5 / 24.6 ms per step; drafted results in the tables above. Kernel microbenchmark (`loom-benchmark` dispatch time, Q6_K 2560 x 40960): f32 1/4 tokens 438 / 643 us,
  int8 427 / 521 us. Quantizing inside the matvec instead (no separate launch) was slower: with the adaptive rows
  per workgroup (R = 1-4) every workgroup re-quantizes the whole activation (Q8_0 4 tokens at R=1: 1343 vs 735 us).
  `ACT_Q8=0` restores the f32 path (numerics of earlier packages). The quantize launches cost a fast model a little:
  LFM2.5-24B-A2B (10 ms steps, no drafting) 10.15 vs 9.75 ms per 1-stream step, but 20.3 vs 21.9 ms with 4 streams;
  big models do not notice (Qwen3.8-27B 1-stream decode 12.9 either way). A loom-optimize lever for small models:
  fuse the quantization into the producing norm / attention-combine kernels.
- Fixed: **DFlash2 crashed under concurrency on elastic backing** (`driver_failure DATA_LOSS gpu.event_wait`;
  Qwen3.8-27B, Muse-Glimmer-30B, Flint). A propose stage holds a power-of-two cohort of blocks; when fewer spans
  propose (one request still prefilling while others draft: tune.py's "two requests 0.5 s apart"), the trailing
  block tokens are not live and `g_token_info` gives them span -1. `g_dfl_conv` (DFlash2's conv modulation) read
  `records[span]` without checking liveness and ran far outside the metadata buffer: garbage on fixed backing,
  a fault on elastic. It now treats dead tokens as span 0 with no taps. Found by bisecting on Flint (selector off:
  still crashed; conv off: passed); after the fix the full DFlash2 passes tune.py's concurrency check 3 of 3.
  Kernels that index per-token metadata must check `info[t][2]` (live) first: cohort stages always carry dead
  tokens.
- **Fixed: 4 drafting streams trailed llama.cpp on Muse-Glimmer-30B** (issue #28: 51 vs 68 tok/s, depth 3).
  - Attribution by doubling kernel groups (ms per epoch, 204 ms total): 16-token trunk matvecs +113, proposals +56,
    head +44, warm +12, attention +5.
  - Three changes:
    - The int8 head kept waves x 128 x TG partials in LDS (57 KB at 16 spans), so it held one or two workgroups per
      CU: 16 verify rows took 22 ms and 12 proposal rows 18 ms. It now runs each chunk in sub-blocks of 512/TG rows,
      with a 16-span variant: 8.7 / 8.6 ms.
    - A DFlash block is depth + 1 tokens (llama.cpp's n_max + 1), not the trained 16: 4 streams propose 16 tokens
      on the int8 matvec instead of 64 on the GEMM.
    - The catch-up (warm) program takes the GEMM from 5 tokens. It only feeds proposals, and DFlash's 33280-wide fc
      was on the f32 4-token matvec.
  - Epoch 204 -> 158-165 ms (sessions vary). Same server: c4 +19 % (ab_perf), decode +6 %, prefill unchanged.
  - c4_steady, same session: 52.7 -> 63.5 tok/s, against llama.cpp's 69.7 (its own DFlash2, n-max 3).
  - Single-stream decode +6-7 % from the shorter block.
  - The remaining gap to one weight pass is the 16-token trunk matvec (233 VGPRs on Q5_K gate/up: 1144 vs 817 us
    for 1 token). The optimization catalog lists what was tried.
- Verification of more than 16 tokens per epoch (several drafting streams at depth 5+) takes the WMMA GEMM (rows
  padded to 64).
- **Long-context verify: a verify span's tokens share their KV loads** (`gen_llm.ATTN_MT`, default 4; from position
  `ATTN_MT_FROM` 1024). When an aligned group of 4 epoch tokens is one row at consecutive positions, the group's first
  split-K workgroup computes all 4 (one K/V load per key for the 4 queries of its head; the others exit). Each token
  keeps its own split range, key masks, 16-key block partition and operation order, and a block holding none of its keys
  leaves its state untouched, so results are bit-identical to the per-token path (drafted == undrafted at 32k). Qwen3-8B
  at 32k, depth 3: verify attention 58 -> 40 ms per epoch (one token: 28). With the DFlash split-K draft attention below,
  the depth-3 epoch went 136 -> 85 ms: 12.5 -> 20.1 tok/s (undrafted 19.5); depth 7 7.7 -> 13.7.
- **DFlash v1 draft attention is split-K** (`gen_dflash.DFL_SPLITS`, 32; windowed DFlash2 and conv drafters keep the
  per-token kernel). The per-token kernel (one workgroup per block token and query head, one key per step) re-read
  the context once per block token and query head: 46 ms of the 136 ms 32k epoch above. `g_dfl_ctx_split` scores a
  chunk of 8 block tokens per wave against each key loaded once, per (KV head, split); `g_dfl_attention` merges the
  splits and adds the block's own keys.
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
- Two-epoch continuation (`continuation_epochs: 2`) is not implemented (one device epoch per host turn).
- DFlash v1 drafters with full-context attention read the whole context per proposal (DFlash2 windows it); split-K
  (above) makes that one pass per 8 block tokens and KV head.
- DFlash RoPE uses KV positions; llama.cpp gives M-RoPE targets' drafters the temporal position (degenerate
  sections). Text after an image then proposes with shifted positions: lower acceptance, never wrong output.
  Gemma-4 (gemma4v) and Muse-Glimmer vision ports draft with DFlash and pass the image questions of the spec gate.
- **Wide verification trades exactness for speed.** Epochs of 9-63 tokens (several drafting streams) take the f16
  WMMA GEMM, one weight pass per step, while decode uses the int8 matvec. Near-ties can flip there (before the int8
  path the GEMM started at 5 tokens: Muse-Glimmer depth 7 changed 2 of 16 spec-gate completions; Flint depth 7 7 of 13,
  MoE routing amplifying the rounding). Single requests at every depth (up to 8 tokens) are exact by construction.
  `SPEC_EXACT=1` (build env) keeps the matvec at every width (decode's rounding), re-reading weights per 8 tokens;
  `SPEC_MM_FROM=N` moves the GEMM threshold. Always compare against draft_depth 0 of the SAME package
  (spec_check.py card): a reference recorded from an earlier build differs on its own (a stale Flint reference made the
  trunk look non-invariant above 4 tokens; it is not). Concurrent epochs that mix a 64+-token prefill with other rows'
  decode take the GEMM for the whole epoch, drafting or not: the mixed-epoch effect consistency.py documents, not a
  drafting property.
- **n-gram pays for every miss.** loom-serve 0.1.0 verifies a fixed width per speculative span, so a miss costs a
  4-token step (Nemotron: 33.7 vs 21.3 ms) where llama.cpp's ngram-simple/ngram-mod draft only after a long match
  and cost nothing on new text. Prompt lookup therefore wins on edit/quote-heavy output (Qwen3.5-0.8B code edit:
  72 % accepted, 212 vs ~140 tok/s) and loses on the standard mix; tune.py decides per model.
