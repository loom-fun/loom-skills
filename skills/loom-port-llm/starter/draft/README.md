# Speculative decoding in starter ports (MTP, DFlash, DFlash2)

Ports draft by default whenever their GGUF supports it, and stay token-for-token identical to the target. Nothing
needs configuring: `port.sh` enables a drafter, `qualify.sh` measures which drafter and depth actually decode
fastest (or none), and the gates prove the output did not change. `SPEC=off` opts out (port, build, qualify).

## Drafters

| Kind | When it is available | What drafts |
|---|---|---|
| `mtp` | the GGUF carries NextN/MTP blocks (`blk.L.nextn.*`; Qwen3.5/3.6/3.8, unsloth `*-MTP-GGUF`; `inspect_gguf.py` records `cfg["mtp"]`) | the model's own NextN block, `draft_depth` autoregressive rounds |
| `dflash` | a DFlash (v1) drafter GGUF exists for the base model (`find_drafter.py`) | a small non-causal block-diffusion transformer over the target's hidden features; one pass proposes a whole block |
| `dflash` (v2 / DFlash2) | as above, drafter with `attn_conv_proj` / `selector_*` tensors | DFlash plus 2-tap conv modulation, a 2048-key context window and a low-rank candidate selector |

`find_drafter.py` (port.sh stage 2c) searches HF for `<base model>-DFlash*` GGUF repos (z-lab first), checks each
candidate's header before downloading (arch `dflash`, hidden size, tap layers inside the trunk, vocabulary, mask
token, tensor formats the kernels support) and records `cfg["drafter"]`. `DRAFTER_REPO=org/name[:file]` forces one.

## How a step runs (native contract: hrx-loom 0.1.0 text service, `draft_depth` 1-7)

1. **propose** (`@llm_mtp_propose` / `@llm_dflash_propose`): for every span loom-serve marks PROPOSE, the drafter
   writes `draft_depth` tokens into the verifier's input slots on device (no host round trip). One stage per
   power-of-two cohort (1, 2, 4 ... spans).
2. **verify** (`@llm_verify`): the unchanged trunk runs anchor + drafts like a prefill chunk. The head predicts
   every position through synthetic records, so sampling draws at the true position: seeded sampling stays
   identical to ordinary decoding. `g_spec_accept` keeps the longest matching prefix (EOS and the output credit
   stop it), publishes `{consumed, outputs, tokens}` per span, and builds the committed and replay packets.
   Gated delta-net layers capture their inputs instead of publishing state for speculative spans; a replay pass
   recomputes conv + recurrence for the accepted prefix only and publishes that. K/V written past the accepted
   frontier is unreachable and overwritten later.
3. **warm** (`@llm_mtp_warm` / `@llm_dflash_warm`, after every epoch, prefill included): the drafter's own K/V at
   every committed position (MTP: the NextN block's K/V from (token P, target hidden P-1) plus the per-row
   carry; DFlash: `fc(features)` -> norm -> each draft layer's context K/V). The draft cache shares the target
   pool's page ids, so drafting covers the whole context, history reuse and checkpoints with no extra state.

Files: `gen_draft.py` (verify, acceptance, GDN capture/replay, MTP), `gen_dflash.py` (DFlash/DFlash2),
`build_draft.py` (prepare/storage/control patches; ports without a drafter are byte-identical to before),
`find_drafter.py`, `tune.py`. The device plan copy marks speculative spans with flag bit 4 and the output
credit in bits 8-15; plan word 4 holds the packed cursor (`encode_epoch`).

## Choosing the drafter and depth (`tune.py`, qualify.sh stage 0)

Builds each available kind once, serves each depth (a `model.json` setting) alone, times decode-heavy requests
(essay, code, a tool call, a reasoning answer) and keeps the fastest if it beats no drafting by 3 %. The choice
and the whole table go to `cfg["drafter"]` and `bench/draft_tune.json` (put the table in the report).
Measured on Strix Halo (greedy decode, 300-token essay):

| Model | No drafting | MTP (depth 3) | DFlash |
|---|---|---|---|
| Qwen3.8-27B Q8_0, 262k context | 7.9 tok/s | **14.6 tok/s** (49 % acceptance; 84 % in tool loops) | DFlash2 depth 7: 13.0 tok/s (53 %, 4.7 tokens/step) |
| Qwen3.5-0.8B Q4_K_M | **262 tok/s** | 208 tok/s | - |
| Qwen3-4B Q4_K_M | - | - | DFlash depth 7: 39 % acceptance, 3.7 tokens/step |

Small models are launch-bound: drafting loses there and `tune.py` keeps ordinary decoding.

## Gates

- `spec` (gates.sh quick/full, and port.sh's smoke stage): the same package with `draft_depth 0`
  (`model_target.json`, same compiled program) is the reference; every completion (greedy, seeded sampling,
  tool calls, reasoning, a 1k-token prompt) must be identical. `spec_check.py compare` also prints acceptance.
- `agent` and `needle` run with drafting on: multi-turn tool loops with history reuse, and the full context.

## Limits / next levers

- Verification of more than 4 tokens per epoch (DFlash depth > 3, several streams) uses the 4-token matvec once
  per 4 tokens, so weights stream more than once per step. A small-M WMMA GEMM (16-row tiles) is the next lever;
  an f16 8-token matvec and an activation-streaming 16-token matvec were tried and were slower (register
  pressure / L2 traffic) and reduced Q4_K drafter acceptance.
- The head runs per 4 rows for verify and draft rows (the vocabulary matrix is re-read per 4 predictions).
- Two-epoch continuation (`continuation_epochs: 2`) is not implemented (one device epoch per host turn).
- DFlash v1 drafters with full-context attention read the whole context per proposal (DFlash2 windows it).
- DFlash RoPE uses KV positions; a vision (M-RoPE) port with a DFlash drafter is untested.
