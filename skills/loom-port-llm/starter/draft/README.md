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
records `cfg["drafter"]`. A BF16/F16-only drafter (NVIDIA publishes Nemotron's that way) is requantized to Q8_0
next to cfg.json (`drafter/*-Q8_0.gguf`, ~0.4 % max relative error). `DRAFTER_REPO=org/name[:file]` forces one.
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

Files: `gen_draft.py` (verify, acceptance, GDN capture/replay, the generic replay hook, MTP), `gen_dflash.py`
(DFlash/DFlash2), `gen_assist.py` (Gemma-4 assistant), `gen_ngram.py` (prompt lookup), `build_draft.py` (prepare/storage/control patches; ports without
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
| Qwen3.8-27B Q4_K_M (bartowski), 262k context | 12.9 tok/s | **depth 3: 21.3 tok/s** (74 % of proposals accepted in the spec gate) | depth 3: 21.9 tok/s, but fails the concurrency check (see Limits) | none 12.0, MTP n-max 3 26.0, **DFlash2 n-max 3 28.1** |
| Qwen3.8-27B Q8_0, 262k context | 7.9 tok/s | **14.6 tok/s** (49 % acceptance; 84 % in tool loops) | DFlash2 depth 7: 13.0 tok/s (53 %, 4.7 tokens/step) | - |
| Qwen3.5-0.8B Q4_K_M | **262 tok/s** | 208 tok/s | - |
| Qwen3-4B Q4_K_M | - | - | DFlash depth 7: 39 % acceptance, 3.7 tokens/step |

Other families (Nov 2026, this skill tree; tune.py and llama_spec_tune.py on the same requests, llama.cpp b1364;
acceptance from the spec gate's prompts at depth 3; every package below is token-exact at depth 3):

| Model (arch, quant) | Drafter | Loom: none -> best | llama.cpp: none -> best |
|---|---|---|---|
| Nemotron-3.5-Bolt-E4B (gemma4, UD-Q5_K_XL) | NVIDIA DFlash, BF16 requantized to Q8_0, 49 % accepted | 46.7 -> **73.4** (DFlash depth 3, x1.57) | 45.3 -> **99.7** (DFlash n-max 7, x2.20) |
| same | Gemma-4 E4B assistant (AtomicChat, Q8_0), 57 % accepted at depth 3 (llama.cpp: 57 %) | 46.9 -> **72.7** (assist depth 3, x1.55; ties DFlash) | 45.8 -> **98.8** (assist n-max 3, x2.16) |
| Qwen3.5-4B (qwen35 hybrid, UD-Q4_K_XL, vision with M-RoPE, F16 tensors) | AtomicChat DFlash, 68 % accepted | port.sh end to end, spec gate 16/16 incl. images | - |
| Muse-Glimmer-30B (muse-glimmer, UD-Q5_K_M) | z-lab DFlash2, 71 % accepted (61 % with the target's normed embedding) | 12.2 -> **20.5** (DFlash2 depth 3, x1.68) | 11.6 -> **25.4** (DFlash2 n-max 3, x2.18) |
| Flint (flint MoE, Q4_K_M) | incoai DFlash2, 65 % accepted | 65.6 -> **90.5** (DFlash2 depth 3, x1.38) | 53.5 -> **86.0** (DFlash2 n-max 3, x1.61) |
| LFM2.5-24B-A2B (lfm2moe, Q4_K_M) | none exists: n-gram only (short-conv replay; 13/13 exact) | **102.7** (n-gram 65.3: none kept) | **94.7** (ngram-simple 90.1, ngram-mod 93.6) |

Drafting now works on every family; Loom's relative gain trails llama.cpp's because its multi-token target step
costs more (Limits), while its absolute drafted speed already leads on Flint. tune.py chooses the bold Loom
setting on each and its concurrency check passes (Glimmer 20.4 vs 12.3, Flint 90.4 vs 64.7 after the DFlash2 conv
fix in Limits; before it, both were rejected for a crash).

Small models are launch-bound: drafting loses there and `tune.py` keeps ordinary decoding.

## Gates

- `spec` (gates.sh quick/full, and port.sh's smoke stage): the same package with `draft_depth 0`
  (`model_target.json`, same compiled program) is the reference; every completion (greedy, seeded sampling,
  tool calls, reasoning, a 1k-token prompt) must be identical. `spec_check.py compare` also prints acceptance.
- `agent` and `needle` run with drafting on: multi-turn tool loops with history reuse, and the full context.

## Limits / next levers

- **The multi-token target step is the drafting bottleneck.** Qwen3.8-27B Q4_K_M, no drafting, n concurrent
  streams: 78.6 ms per step (1 row), 90.8 (2), 130.1 (3), 131.9 (4). Drafting rounds follow it: 78 ms (depth 0),
  98 (1), 143 (2), 150 (3), 220 (4: width 5 takes the WMMA path). A proposal step costs ~6 ms; verifying 3-4
  tokens costs ~40 ms more than 2. That jump (not the drafter) is why Loom's drafted decode trails llama.cpp's
  with the same drafter and depth (21.3 vs 26.0 tok/s MTP) although its 1-token decode is faster (12.9 vs
  12.0). The 4-token matvec streams weights once for 2-4 tokens, so look elsewhere first (GDN recurrence/conv
  per span, decode attention, head per 4 rows, the stage chosen for the shape). rocprofv3 sees no kernels
  (Loom's AMD driver bypasses HSA): ablate with NO_GEMM-style build switches and per-stage timing.
  Same on other families: Nemotron-3.5-Bolt-E4B (gemma4), no drafting, 1/2/4 streams: 21.5 / 26.2 / 35.0 ms per
  step; an n-gram depth-3 epoch (verify 4 tokens, negligible drafter) 33.7 ms, a DFlash depth-3 epoch 39.6 ms, while
  llama.cpp's whole DFlash n-max-3 step is ~31.6 ms at the same tokens per step (2.9). Doubling ablation
  (loom-optimize `ablate.py`): the 4-token matvecs cost ~25 ms of the 4-stream step vs ~16 ms for the 1-token
  matvecs of a 1-stream step; attention, norms and head add < 3 ms. A separate 4-token R (`MV_WG4=16` or 128) gains
  only ~5 % (73.6 -> 77.9 tok/s DFlash) and may break the exactness the shared R guarantees. The lever is the 4-token
  matvec itself (each lane holds 4 x 32 f32 activations: register pressure, occupancy), for every family.
  With the Gemma-4 assistant the gap is the same at equal acceptance: llama.cpp's epoch at ~2.7 tokens per step is
  ~27 ms (+5 ms over its 1-token step), Loom's 37.6 ms (+16 ms). Tried: a 4-token matvec that reloads each token's
  activations per row instead of holding them in registers (`mv_wide`-style): slower (42.9 ms) and not bit-identical
  to 1-token decode, so dropped.
- Fixed: **DFlash2 crashed under concurrency on elastic backing** (`driver_failure DATA_LOSS gpu.event_wait`;
  Qwen3.8-27B, Muse-Glimmer-30B, Flint). A propose stage holds a power-of-two cohort of blocks; when fewer spans
  propose (one request still prefilling while others draft: tune.py's "two requests 0.5 s apart"), the trailing
  block tokens are not live and `g_token_info` gives them span -1. `g_dfl_conv` (DFlash2's conv modulation) read
  `records[span]` without checking liveness and ran far outside the metadata buffer: garbage on fixed backing,
  a fault on elastic. It now treats dead tokens as span 0 with no taps. Found by bisecting on Flint (selector off:
  still crashed; conv off: passed); after the fix the full DFlash2 passes tune.py's concurrency check 3 of 3.
  Kernels that index per-token metadata must check `info[t][2]` (live) first: cohort stages always carry dead
  tokens.
- Verification of more than 4 tokens per epoch (DFlash depth > 3, several streams) uses the 4-token matvec once
  per 4 tokens, so weights stream more than once per step. A small-M WMMA GEMM (16-row tiles) is the next lever;
  an f16 8-token matvec and an activation-streaming 16-token matvec were tried and were slower (register
  pressure / L2 traffic) and reduced Q4_K drafter acceptance.
- The head runs per 4 rows for verify and draft rows (the vocabulary matrix is re-read per 4 predictions).
- Two-epoch continuation (`continuation_epochs: 2`) is not implemented (one device epoch per host turn).
- DFlash v1 drafters with full-context attention read the whole context per proposal (DFlash2 windows it).
- DFlash RoPE uses KV positions; llama.cpp gives M-RoPE targets' drafters the temporal position (degenerate
  sections). Text after an image then proposes with shifted positions: lower acceptance, never wrong output.
  Gemma-4 (gemma4v) and Muse-Glimmer vision ports draft with DFlash and pass the image questions of the spec gate.
- **Wide verification trades exactness for speed.** Epochs of 5-63 tokens (depth > 3, or depth 3 with two or more
  drafting streams) take the f16 WMMA GEMM, one weight pass per step; 1-token decode the f32 matvec. Near-ties can
  flip: Muse-Glimmer depth 7 changed 2 of 16 spec-gate completions (llama.cpp gap 1.07 at the flip, just past the
  near-tie rule). `SPEC_EXACT=1` (build env) keeps the 4-token matvec at every width, decode's rounding (Glimmer depth
  7: 16/16 exact), but re-reads the weights per 4 tokens: Bolt-E4B with 4 drafting streams at depth 3 took 146 ms per
  epoch vs 69 (drafted c4 throughput 55 vs 116 tok/s), worse than not drafting, so it is not the default. tune.py's
  equivalence check rejects any depth whose single-request output changes; depth 3 (4 tokens, exact) won on every
  model measured. Flint additionally changes 7 of 13 at depth 7 even with SPEC_EXACT: its kit is not invariant above
  4 tokens per epoch on its own (single stream), an example-specific issue still to find. Concurrent epochs that mix a
  64+-token prefill with other rows' decode take the GEMM for the whole epoch, drafting or not (8 identical greedy
  prompts at once, no drafting: Glimmer 6/8 equal to the solo run, Flint 0/8): the mixed-epoch effect consistency.py
  documents, not a drafting property.
- **n-gram pays for every miss.** loom-serve 0.1.0 verifies a fixed width per speculative span, so a miss costs a
  4-token step (Nemotron: 33.7 vs 21.3 ms) where llama.cpp's ngram-simple/ngram-mod draft only after a long match
  and cost nothing on new text. Prompt lookup therefore wins on edit/quote-heavy output (Qwen3.5-0.8B code edit:
  72 % accepted, 212 vs ~140 tok/s) and loses on the standard mix; tune.py decides per model.
