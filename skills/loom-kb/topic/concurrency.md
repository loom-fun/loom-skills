# Topic: concurrency (several rows at once)

What changes when several requests share an epoch: occupancy gates, cohort stages with dead tokens, batch invariance. The correctness gate is `consistency.py` (loom-benchmark).

## Occupancy gate (build_draft.py, `SPEC_OCC_MIN`, default 2; 0 = off)

The host plans depth + 1 tokens for every speculative span every step, so four streams verify a padded 16-token epoch even
at 3 % acceptance. `encode_epoch` therefore counts the speculative spans first; when ALL spans in the epoch are speculative
and there are at least `SPEC_OCC_MIN` of them it clamps every span to its anchor, `@epoch` skips propose and runs the
verify/advance/warm stage of the (s tokens, s spans) shape (the plain decode shape), and results publish as one token per
span. One stream still drafts (decode and long context unchanged); multi-stream decode runs at the undrafted epoch size.
The gate flag lives in control state (byte 1676, table at 1700): `encode_epoch` writes it and the same epoch's `@epoch` reads
it. It needs a `(s, s)` shape for the span count (otherwise the gate stays off for that count). Because the gated c4 rate
equals depth 0, tune.py's per-row margin keeps a drafter that wins single-stream without paying at four streams.

A gated epoch verifies its anchors as ordinary known spans in a separate verify stage per `(s, s)` shape (`llm.gated` 1, id packed with the shape index in the gate flag): the speculative flag and credit are not set on the device plan, and the stage skips the recurrent-state capture and the replay (every gated delta-net layer would otherwise run its conv and recurrence twice). Qwen3.5-9B four streams: gate 139.8 vs depth 0 142.8 tok/s (was 130.2).

## DFlash2 crashed under concurrency on elastic backing: kernels that index per-token metadata check liveness

- Fixed: **DFlash2 crashed under concurrency on elastic backing** (`driver_failure DATA_LOSS gpu.event_wait`;
  Qwen3.8-27B, Muse-Glimmer-30B, Flint). A propose stage holds a power-of-two cohort of blocks; when fewer spans
  propose (one request still prefilling while others draft: tune.py's "two requests 0.5 s apart"), the trailing
  block tokens are not live and `g_token_info` gives them span -1. `g_dfl_conv` (DFlash2's conv modulation) read
  `records[span]` without checking liveness and ran far outside the metadata buffer: garbage on fixed backing,
  a fault on elastic. It now treats dead tokens as span 0 with no taps. Found by bisecting on Flint (selector off:
  still crashed; conv off: passed); after the fix the full DFlash2 passes tune.py's concurrency check 3 of 3.
  Kernels that index per-token metadata must check `info[t][2]` (live) first: cohort stages always carry dead
  tokens.
