# Family muse-glimmer: decode

## Attention q + gate in one dual-weight launch at 8 and 16 tokens (`PAIR_PROJ`, `lin2()`)

- **Attention q + gate in one dual-weight launch at 8 and 16 tokens** (`PAIR_PROJ`, `lin2()`): two projections of
  the same input with the same format and shape share the activation load, the butterfly setup and the select-free
  butterfly, as gate/up do. Each output rounds exactly as the 1-token single-weight kernel (kernel tests
  `mvpair_attn_q_t8/t16`, zero tolerance). Prefill, 1-token decode and the 4-token variant keep separate launches.
  - Real-weight microbenchmark, Muse-Glimmer Q4_K 6656 x 4096 t16: 282 -> 251 us per layer (two launches vs one).
  - 4 streams, same server: -1 to -2 ms per ~153 ms epoch (paired rounds all faster). Single-stream decode and
    prefill unchanged (their kernels are byte-identical).
  - Only models with a separate `attn_gate` of the q format pair (Muse-Glimmer). Qwen3.8-27B (fused q+gate),
    Flint and Bolt generate byte-identical programs.
  - Same session, c4_steady x 4 each (medians, min-max): Loom before 68.1 (67.0-69.7), after 69.7 (68.2-70.7),
    llama.cpp b1355 69.4 (68.2-70.7) tok/s. Level with llama.cpp, not ahead in every run: the per-run spread (+-2 %)
    is larger than the remaining step-time difference.

## Did not help: pairing attention k + v (6656 x 256) into one launch (`PAIR_PROJ=kv`)

- **Pairing attention k + v (6656 x 256) into one launch** (`PAIR_PROJ=kv`): -16 us per layer in isolation
  (R 8; R 32 was +52 %), but with q + gate also paired the 4-stream epoch was +1.5 ms (interleaved, 3 rounds). The
  small separate k and v launches presumably overlap their neighbours. Opt-in only.

## Did not help: post-norm eps 1e-8

- **Muse-Glimmer post-norm eps 1e-8** (HF `post_norm_eps` and llama.cpp; the reviewed cfg uses 1e-5): tokens per
  step unchanged within noise (1 stream 2.57 vs 2.53, 4 streams 2.63 vs 2.60). Not changed here: it changes the trunk's
  numerics, so it needs its own gate run.
