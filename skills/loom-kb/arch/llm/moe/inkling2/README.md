# Family: inkling2

GGUF arch `inkling2` (the compact layout used by the Strix package at checkpoint revision `ea0c79b2`) and `inkling2_loop`
(the upstream layout used by that revision's `Mainline_Support/` assets). Model: thinkingmachines/Inkling-2-Mini
(17.6 GB UD-Q4_K_XL). The port uses its own kit beside the starter (kernels `gen_inkling2.py`, prefill attention
`gen_ink_flash2.py`, drafter `draft/gen_dspark.py`, channel chat template); it is not in `loom-port-llm/starter/` yet.
Facets: looped, moe, sliding-window; drafted (DSpark) and vision (`inkling2v` projector) when the package has them.

## Packages and SDK 0.3.0 qualification

- **Strix Halo `chat`**, frozen package `4e018e88`. The original SDK 0.2.0 card recorded greedy DSpark decode at
  48.5 / 45.2 / 39.1 tok/s for short / 2k / 32k prompts, with f16 KV and a 262,144-token context. Its b1383 reference
  predates the upstream loop-scale fix. The SDK 0.3.0 migration kept the measured programs. On a single Strix runner,
  matched SDK 0.2.0 / 0.3.0 timing changed short decode -0.67%, 8k prefill -0.10%, 32k decode +0.28%, and four-stream
  throughput -0.19%. The full suites reproduced concurrent output divergence on the same two prompts and drafted
  versus undrafted equivalence on 3 of 13 requests per SDK. The native-context admission succeeded and cancellation
  exceeded 60 seconds. Production promotion is held. [Original card](https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom/tree/4e018e88a3aa9826286c71dfe2f6c905a491659b)
  [SDK 0.3.0 staging](https://huggingface.co/loom-fun-staging/unsloth-Inkling-2-Mini-loom/tree/bd021e649b158e608b389eca9bfed60a0ec2d225)
  [Paired correctness](https://github.com/loom-fun/loom-skills/actions/runs/45267522252/job/138018261220)
  [Matched timing](https://github.com/loom-fun/loom-skills/actions/runs/45277848206/job/138054186652)
- **Radeon R9700S**, frozen package `6213fd3c`, already pinned to SDK 0.3.0. `chat-r9700` uses undrafted Q4 with a
  131,072-token preset. `chat-q8-r9700` uses undrafted Q8 with a 32,768-token preset. The fixed b1384 reference
  matched native chat and passed the compare and agent gates for both. Q8 passed all five full gates. Q4 passed four
  full gates and passed the corrected sampling check in a separate run. Q4 native-context cancellation exceeded 60
  seconds and remains a promotion hold. [Original card](https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom-r9700/tree/6213fd3ce3a15ea2bab98d4adf09ec783c5ad5dc)
  [SDK 0.3.0 staging](https://huggingface.co/loom-fun-staging/unsloth-Inkling-2-Mini-loom-r9700/tree/90c125c8b0686030e33ee899c7a5098a83a25a46)
  [Fixed-reference gates](https://github.com/loom-fun/loom-skills/actions/runs/45285206955)
  [Sampling proof](https://github.com/loom-fun/loom-skills/actions/runs/45288818385/job/138091995682)

## Graph (verified against llama.cpp's trace)

- 24 physical blocks run as **48 logical layers**: `blk.0..5` prelude, `blk.6..17` looped 3 times, `blk.18..23` epilogue.
  Per-layer costs and the roofline count the 48 logical layers (facet looped).
- `H=2304`, 24 query / 8 KV heads, head_dim 128, no RoPE, final logit softcap 30, vocab 200,058 (padded 201,024).
- Attention: block index `% 6 != 5` is sliding (window 1024): 40 sliding and 8 full logical layers (blocks 5, 11, 17,
  23). Learned relative-position bias (d_rel 16, extent 1024), Q/K L2 clip 22.6274, R clip 8.0, weightless V RMSNorm,
  per-query sink (`attn_sink_proj`), elementwise sigmoid output gate. Full layers scale Q and R by
  `1 + 0.1*log(max(1, (pos+1)/8192))` and run dynamic conv projections on K and V.
- ShortConv (`l_cache=4`): 4 causal convs per logical layer (K, V, attn_post, ffn_post); 3 frames of history per logical
  layer, 3.66 MiB per row of recurrent state.
- Block attention residuals (`attn_res`, block size 6) and loop message controllers (rank 64) on loop passes 1 and 2.
- MoE: 128 experts, top-8, sigmoid router plus `exp_probs_b` bias, normalised (eps 1e-20), scale 8.0, fused
  `ffn_gate_up_exps` with SiTU (beta 4.0, linear beta 25.0). Experts are NVFP4 in `blk.0..19` (per-expert `.scale`
  tensors) and Q8_0 in `blk.20..23`; attention projections and the head are Q8_0.

## NVFP4 (type 36, 64-value superblocks; only this family so far)

- E2M1 codes with a UE4M3 scale per 16 values plus the tensor's per-expert `.scale`. Bit-exact f16 dequant: the scale is
  2 ops (`byte << 7` and a 128x LUT), the 16 codes go through a single 8-byte `v_perm_b32` LUT, a sign OR and a
  `v_pk_mul_f16` (a gfx11-generic `low.func.def` helper called by `low.invoke`; bit-exact on all 256 payloads).
- The dequant must be inline in the GEMM's K loop: a `func.call` to it stopped `gen_moe._prefetch_stages` from
  software-pipelining the weight loads (`--compile-report`: 19 and 14 full waits in the grouped gate_up / down GEMMs).
- Decode matvecs use int8 activations against the codes (`vector.dot4i`, two half-scales per block, as in `q6_k`).

## Bf16-exact f32 tensors

The router (`ffn_gate_inp`) and all loop-controller matrices are stored f32 but have all 16 low mantissa bits zero (168
tensors, ~160 MB; checked at build). An exact load-time bf16 copy (weights.loom `f32e -> bf16c`) halves their bytes with
bit-identical outputs: -1.7 % per decode step, 4 streams +5-10 %. Also bf16-exact: `attn_rel_bias`, conv weights, norms,
the q/rel clips. Not exact: `*_res_query`, `exp_probs_b`, `*_exps.scale`.

## Two GGUF layouts

`inkling2_loop` renames metadata (`inkling2.*` -> `inkling2_loop.*`, `num_loops` -> `loop.count`, `loop.core_start/end`
-> `loop.start/end`) and loop tensors (`loop_{attn,k,v,mlp}_{down,forget,gain}` -> `loop_{q,k,v,ffn}_ctrl_*`), and re-slices
the two packed `loop_res_query` / `loop_kv_res_query` (2304x24) into 12 per-layer `loop_msg_query` (2304x4). The header
comparison identified 588 tensors outside that re-slicing. Fifteen random 64 KB payload samples matched on 2026-11-23.
The new root drafter is the mainline DSpark
format (unpadded `markov_w1` [256, 200058], explicit `markov_w2`, softmax-normalised `ctx_mix`); the old one is
`*_Q8_0_Old.gguf`. A kit written for one layout needs a name map for the other.
