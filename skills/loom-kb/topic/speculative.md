# Topic: speculative decoding (MTP, Gemma-4 assistant, DFlash, DFlash2, n-gram)

## Contents

- Using it (port.sh / build.py)
- Drafters
- How a step runs (native contract: hrx-loom 0.1.0 text service, `draft_depth` 1-7)
- Recurrent layers other than gated delta-net (kits)
- Target-specific details DFlash handles
- Gates
- Limits and next levers

Ports draft by default whenever their GGUF supports it, and stay token-for-token identical to the target. Nothing
needs configuring: `port.sh` enables a drafter, `qualify.sh` measures which drafter and depth actually decode
fastest (or none), and the gates prove the output did not change. `SPEC=off` opts out (port, build, qualify).
It works for every family the starter and the reference examples port (dense, gated delta-net and short-conv
hybrids, MoE, sinks/sliding window, Gemma-style scaled embeddings, vision): the verifier is the trunk itself, the
drafters only need the target's embedding, head and hidden features. Which drafter a *card* uses (and that n-gram never
is one) is fixed by `loom-benchmark/reference/methodology.md`, not by tune.py.

## Using it (port.sh / build.py)

Nothing to configure.
- `build.py --draft auto|mtp|dflash|assist|ngram|none` `[--drafter GGUF] [--draft-depth N]`; auto takes `cfg["drafter"]` (tune.py's measured choice), else MTP, else DFlash (kinds: the table above).
- Across families: the verifier is the trunk itself; DFlash embeds its block the way llama.cpp does (the target's
  raw row times the drafter's `embedding_scale`), so Gemma-style scaled or Glimmer-style normed embeddings work.
- Output is exactly the target's: verification, device acceptance and gated delta-net replay commit only target
  tokens, and seeded sampling draws at the true position. The `spec` gate compares with the same package at
  `draft_depth 0` (`pkg/model_target.json`); for a single request (up to 16 verified tokens: the int8 matvecs'
  1-/4-/8-/16-token variants round like 1-token decode) a difference is a bug, not a near-tie. Wider verification
  (many drafting streams, 17+ tokens) takes the f16 WMMA GEMM for speed and can flip near-ties (`SPEC_EXACT=1`
  builds trade that speed for decode's rounding); tune.py runs the equivalence check before it keeps a drafter and
  depth, so a depth that changes a request's output is never chosen.
- Drafting works to the full context and with history reuse and checkpoints: the draft K/V shares the pool's pages.
- Depth is a `model.json` setting (`execution.draft_depth`, 1-7); the method needs a rebuild.
- Small models (< ~2B) decode faster without it (launch-bound); `tune.py` measures and decides.

## Drafters

| Kind | When it is available | What drafts |
|---|---|---|
| `mtp` | the GGUF carries NextN/MTP blocks (`blk.L.nextn.*`; Qwen3.5/3.6/3.8, unsloth `*-MTP-GGUF`; `inspect_gguf.py` records `cfg["mtp"]`) | the model's own NextN block, `draft_depth` autoregressive rounds |
| `dflash` | a DFlash (v1) drafter GGUF exists for the base model (`find_drafter.py`) | a small non-causal block-diffusion transformer over the target's hidden features; one pass proposes a whole block |
| `dflash` (v2 / DFlash2) | as above, drafter with `attn_conv_proj` / `selector_*` tensors | DFlash plus 2-tap conv modulation, a 2048-key context window and a low-rank candidate selector |
| `assist` | a gemma4-family target and a Gemma-4 "assistant" GGUF (arch `gemma4-assistant`, e.g. `AtomicChat/gemma-4-E4B-it-assistant-GGUF`, or `mtp-*.gguf` in the target's own repo (Unsloth QAT); `find_drafter.py` records `cfg["assistant"]`) | Gemma 4's own MTP head as a separate ~100-300 MB model (`gen_assist.py`): `draft_depth` autoregressive rounds of 4 small layers that attend the target's own K/V (sliding layers: target layer L-2's cache, full: L-1's; KV heads per layer kind when `head_count_kv` is per layer), fed by the target's embedding and post-norm hidden like MTP |
| `ngram` | opt-in experiments only (`gen_ngram.py`; `TUNE_NGRAM=1`; never on a card) | prompt lookup: the newest longest suffix match of the anchor in the row's own token history proposes what followed it. No weights; wins where output quotes context (code edits, tool arguments, JSON) |

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
   Cost is small: Qwen3.8-27B Q4_K_M (NextN block Q4_0) local Strix pf8k, warm = g_gemm_q4_0_store_m256 .0035 +
   concat/norms/row-scale/embed/pair/rope share ~.0024 = ~.006 ms/token, 0.35 % of 1.68 busy, so a warm window (warm
   only the last W prompt positions) caps below 0.4 % and is not worth its acceptance loss; gufo also projects the
   MTP K/V of every prompt position (Flash-Next `MtpForward(kv_only)` in prefill). The profile's `g_gemm_q8*` family
   is the trunk's Q8_0 tensors (bartowski Q4_K_M keeps attn_k/v/output, some ssm_out at Q8_0), not warm.

**DFlash proposals read a Q4_0 copy of the head.** Proposing runs the target's head over every proposed position:
a second full head pass per step (Nemotron-3.5-Bolt-E4B: 262144 x 2560 Q8_0, 0.71 GB, ~2.5 ms of a ~25 ms step).
When the head is Q8_0/BF16/F16 and int8 activations are on, `prep_weights.py` gives the propose stage one more
parameter root, the target checkpoint through `head_weights.loom`, which converts the head to Q4_0 once at load
(llama.cpp's quantize_row_q4_0; +0.38 GB) for `g_head_mv*_q8_p40`. Verification still runs the target's own head, so
the output does not change; only which drafts are proposed can (Bolt-E4B: acceptance 36 % -> 39 %, step 23.5 -> 22.1 ms
on DevLab). `PROPOSE_HEAD=off` keeps the target's head. Kernel test `propose_head_q4`.

**Reduced draft vocabulary (`PROPOSE_HEAD_ROWS=N`, choice `draft.propose.head_rows`, default 0 = off).** A Q6_K head
(Qwen3.8-27B: 248320 x 5120, 1.04 GB) is read in full by every proposal round: depth 3 reads it 3 times per step on top
of verification's own pass. With N > 0 the proposals read a load-time Q4_0 copy (18 bytes per 32 values) of head rows
[0, N) plus the added-token rows (ids >= the tokenizer's base vocabulary: `<|im_end|>`, `<think>`, tool markers), so
end-of-turn and tool markers stay proposable (FR-Spec / EAGLE-3 style draft vocabulary). `prep_weights.py` converts
Q6_K (llama.cpp dequantize_row_q6_K, then quantize_row_q4_0) with a `split` / `skip` block map (`subset_entry`);
`g_head_select_p` maps the winning row back to its token id (row < N: itself; else + base - N; DFlash2's `g_dfl_topk`
does the same on its stored ids). Applies to `@llm_mtp_propose` and `@llm_dflash_propose`; verification keeps the
target's head, so the output never changes, only which drafts are accepted. N = every row is a precision-only copy. `PROPOSE_HEAD=q6_k` keeps the
rows as Q6_K (a byte copy of the subset, no re-rounding: more bytes, higher acceptance). A Q2_0 head (Ternary Bonsai, 2.25 bits per
weight) is always copied as it is (`g_prep_q2_0_q2_0`, `g_head_mv*_q8_pq2`): Q4_0 would be twice its bytes.
Kernel tests `propose_head_rows`, `propose_head_select`.

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

## Gates

- `spec` (gates.sh quick/full, and port.sh's smoke stage): the same package with `draft_depth 0`
  (`model_target.json`, same compiled program) is the reference; every completion (greedy, seeded sampling,
  tool calls, reasoning, a 1k-token prompt) must be identical. `spec_check.py compare` also prints acceptance.
- `agent` runs with drafting on: multi-turn tool loops with history reuse.

## Limits and next levers

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

- Verification of more than 16 tokens per epoch (several drafting streams at depth 5+) takes the WMMA GEMM (rows
  padded to 64).
