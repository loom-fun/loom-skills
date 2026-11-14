# Topic: decode (bandwidth-bound token generation)

Levers for 1-token and multi-token decode matvecs, attention and the head. Numbers come from the Flint (40B MoE) and Muse Glimmer (30B dense) ports on Strix Halo, measured against llama.cpp ROCm (lemonade-sdk/llamacpp-rocm b1355) on the same GGUF unless a lever says otherwise. The kernels named here are in `loom-port-llm/starter/gen_llm.py` unless stated (MoE: `gen_moe.py`, `reference/lfm2moe-example/gen_moe_lfm.py`).

## Contents

- Multi-token matrix-vector (`mv()` in gen_llm.py)
- Adaptive rows per workgroup (dense 4B-class decode: the biggest single win)
- Fused 1-token QKV and paired-row gated FFN
- Argmax head (`head_mv()`)
- Split-K decode attention (`attention_text()`)
- F16 KV
- Did not help (decode)

## Multi-token matrix-vector (`mv()` in gen_llm.py)

- **Layout.** A workgroup covers R=16 output rows of one weight matrix, with one
  lane per 32-value group of K, so lanes = K/32 rounded up to a wave. Each lane
  keeps its 32 x values for TG tokens in registers, dequantizes its weight group
  once, and does TG dot products. Wave partials go through
  `kernel.subgroup.reduce`, then one barrier and an LDS fold.
- **Why it works.** Decode is weight-bandwidth bound, so every byte of W should be
  read exactly once per token group. TG=4 serves up to 4 concurrent sequences for
  roughly the price of one.
- **Pitfall.** When K/32 ≤ 32 there is only one wave, so the R·TG outputs exceed
  the lane count. The writer must stride (`scf.for %wr0 = [%workitem to %outputs
  step %lanes]`). This bug produced garbage for hidden=1024 models.
- **Q4_K folding.** `y = Σ_groups (d_g · dot(codes_g, x_g) − m_g · Σx_g)`. Σx_g is
  computed once per token.

## Adaptive rows per workgroup (dense 4B-class decode: the biggest single win)

- **Problem.** The starter's fixed R=16 rows per workgroup leaves the GPU
  under-filled on narrow projections. For example, K/V with n = 512, giving only
  32 workgroups, or q with n = 4096. Meanwhile the long-K down projection
  (K = 10240) has few, very long workgroups.
- **Rule (1-token variant).** Start at R=16 and halve R while
  `n / R < WG_TARGET`, stopping at R=1. Use WG_TARGET ≈ 4096 by default, but
  ≈ 640 when K ≥ 8192 (the down projection), whose workgroups are long already.
  For the 4-token variant, use WG_TARGET ≈ 128: a larger value regresses c4.
  Emit one kernel variant per (format, K, swiglu, R, TG), named `_r{R}`.
- **Result.** On Nemotron-3.5-Bolt-E4B (Gemma-4 E4B, Q4_K_M), decode went from
  54.2 to 59.9 tok/s, half of the gap to the roofline in one change. Check c4
  afterwards: it regressed until the 4-token target was lowered to 128.
- **In the starter, on by default** (`pick_r()` in gen_llm.py; `MV_ADAPT=0` turns it off,
  `MV_WG1` / `MV_WG4` set the targets; `make_kernel_tests.py` tests the `_r{R}` variants
  the model launches). Hidden-1024 models gain the most: in fresh-agent trials decode went
  from 92 to 144 tok/s (Qwen3.5-0.8B) and from 124 to 157 tok/s (Qwen3-0.6B, where
  `MV_WG4=512` also beat 128 on 4 streams).
- **Keep 1- and 4-token R equal (the default).** R changes the summation order, so
  different R for the two variants makes a request's tokens depend on whether it was
  decoded alone or batched: `consistency.py` failed on Qwen3-0.6B. With one R per
  projection (the 1-token rule) decode keeps the whole gain and 4 streams most of it
  (interleaved medians: fixed R=16 118 / 249 tok/s, same R 131 / 368, separate R 131 / 514
  for decode / 4 streams). Setting `MV_WG4` opts into separate 4-token R; then the
  consistency gate can fail (it did on Qwen3-0.6B), so run it; on Bolt-E4B `MV_WG4=128` passed
  consistency and gave 4 streams +8.6 % with decode unchanged (Nov 2026 trial). Keep it only if the gate passes.

## Fused 1-token QKV and paired-row gated FFN

- **Fused QKV.** For a single decode token, one launch computes the q, k and v rows
  (one row range over n_q+n_k+n_v, dispatching on the row id). Launch it under
  `scf.if` for 1-token epochs only; multi-token epochs keep the separate kernels.
  It gained about 0.5 %.
- **Paired-row gate/up.** For the 1-token gated FFN (swiglu/geglu), have each
  workgroup process gate row i and up row i side by side, with R ≥ 2, so no lanes
  idle on the epilogue. Bolt went from 60.3 to 61.0.
- **Serial small kernels.** Any kernel launched with a single workgroup, such as
  per-layer-input prep or norms over many tokens, should get a (tokens, layers)
  grid. It is cheap to find by ablation, because doubling it costs a visible
  fraction of a millisecond.

## Argmax head (`head_mv()`)

- Vocabulary rows are split into 128-row chunks. Each workgroup writes
  (max, argmax) per chunk per span, and `g_head_select` reduces the chunks.
- No logits buffer is ever written, which saves V × 4 bytes per token plus a pass.
- This is only valid for greedy decoding. For sampling you'd need top-k partials.

## Split-K decode attention (`attention_text()`)

- The grid is (token, KV head, split), with G = heads/kv_heads waves per
  workgroup, one per query head. Each wave covers that head's 128 dims across 32
  lanes × 4.
- **Adaptive splits.** splits = min(64, ceil(keys / 32)). Splits beyond that exit
  immediately. Without the early exit, the 64-split grid regressed 4-stream
  throughput.
- A 16-key unroll with predicated tails. The V offset is the K offset plus
  pool × POS_BYTES, so `g_kv_offset` isn't called twice.
- The combine kernel folds (m, l, acc) across splits, then applies the optional
  sigmoid gate.

## F16 KV

- K and V are stored as f16 by the RoPE/KV kernel and read as `vector<4xf16>`.
  Decode at 32k improved from 59 to 63 tok/s on Flint, and memory halves.

## Did not help (decode)

- **LDS-staged x in the multi-token matvec.** Registers already hold x, so the
  extra barrier only cost time.

- **Fusing the residual add + RMSNorm into the next matvec's prologue (decode).**
  Every workgroup recomputes the norm, so the redundant reads cost more than the
  launch saved (Bolt: slower). Launches are nearly free under graph replay.

- **Unrolling the matvec row loop**, or head chunk sizes other than 128 rows.
  Neither was measurably different.

- **About 128 keys per decode-attention split.** Fewer splits under-filled the GPU
  at short context (−4 %).
