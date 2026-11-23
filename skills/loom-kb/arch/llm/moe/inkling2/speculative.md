# Family inkling2: speculative (DSpark)

Strix Halo gfx1151, hrx-loom 0.2.0. Drafter `dspark-Inkling-2-Mini-Q8_0.gguf` (old layout, 390 MB; the mainline-format
drafter is in README).

## The drafter

- A 3-layer non-causal block transformer (block 8: anchor + 7 masks) fed by 4 taps of the target (logical visits 5, 23,
  41, 48), then a rank-256 autoregressive Markov head: step k feeds step k-1's token, so the head runs `depth` times in
  sequence. Layers 0 and 2 are sliding (f16 rings in the carry, 8 MiB per owner); layer 1 uses paged KV (4 KiB per position).
- Proposals read `output @ (h + hidden_out(latent))` in one head pass; the softcap is monotonic and skipped for argmax.
- `markov_w1` is [256, vocab] bf16 (103 MB) but only one row is read per step; `joint_proj` (7.9 MB bf16) is read every step.

## Kept: propose cost (round 4, bit-exact, acceptance unchanged)

Propose cost 3.16 -> 2.14 ms per step at depth 1, 6.02 -> ~3.20 at depth 3; spec check 13/13 identical; acceptance prose
0.556, code 0.872, 32k 0.728 before and after.
- Proposal-head shortlist of 65,536 rows ranked by `markov_w1` row L2 norm (over half its rows are untrained init:
  median 0.575 vs max 196), plus byte tokens 0-255 and every added token, ascending for tie parity: 99.93 % of real tokens
  (49,152 rows: 99.82 %). Stored as SoA Q4_0 (8 lanes per row, no barriers): head 260.5 -> 85 MB, 1,299 -> 463 us per call.
  A vocabulary-prefix shortlist does not work on this banded vocabulary (porting.md).
- Markov head: the static columns once per step before the loop (`vector<8xbf16>`), the dynamic 512 columns per draft
  token; state + out + Q8 quantize fused; select + commit 15.9 -> 9 us. Markov + select 133 -> 84 us at depth 1.
- Trunk: 3-matrix QKV multi-matvec (9 -> 3 launches, 244 -> 214 us), 2-matrix warm KV (6 -> 3), embed/add-norm with Q8
  output (-9 `g_quant_q8` per step).
- Not kept: Q4_0 drafter trunk (`DSPARK_Q4=all`: prose acceptance 0.556 -> 0.494, net slower; `ffn` only: -0.42 ms but
  12/13 parity).

## Depth

- Depth 2 verifies 3 tokens, which loom-serve pads to 4: slower than depth 3 (50.2 vs 47.5 ms per step). Depth 4 pads to 8.
- Fixed depth 3 wins code and math and loses prose (round 8, T = 1.0: essay 42.1 vs 46.1 at depth 1, code 64.7 vs 52.1).
- Adaptive depth (`SPEC_ADAPTIVE=1`, host `draft_depth` 3): per-row integer EMA of acceptance in the control state,
  ema = ((8 - trials) * ema + successes * 1024) >> 3; up at 640, down at 560 (tuned at T = 1.0; 760 / 670 tuned greedy),
  up threshold 850 past position 16384, multi-stream gate `SPEC_OCC_MIN=2`. At T = 1.0 vs fixed depth 1: essay 45.4 vs
  46.1, code 58.5 vs 52.1, reasoning 64.0 vs 56.6. The 640 / 560 thresholds cost greedy code ~10 % against 760 / 670
  (61.8 vs 68.5).
- A per-request `draft_depth` (omitted or -1 adaptive, 1 fixed, >= 2 full) does not turn drafting off at 0: the host still
  plans proposals. Only the deployment's `"draft_depth": 0` does.
- Greedy output across depths matches up to near-ties, not byte for byte (t = 1 / 2 / 4 matvec rounding).

## Hang in mixed prefill + decode epochs (fixed)

`g_dsp_attn_split_full_v2` read uninitialised span metadata for inactive spans in a mixed epoch and wrote out of bounds
(a VM fault that hung the GPU). `g_dsp_prepare` now zero-fills inactive slots and inactive spans write a disjoint slot. A
range assume placed before a select in `g_spec_rows` was violated by a 4,092-token prefill. Check: 16 mixed 1.2k-18k
requests, two waves, at adaptive depth 3 and depth 1.
