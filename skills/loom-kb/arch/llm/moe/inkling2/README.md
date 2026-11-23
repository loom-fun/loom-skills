# Family: inkling2

GGUF arch `inkling2` (llama.cpp PR #32695 schema; the layout of unsloth/Inkling-2-Mini-GGUF up to revision `ea0c79b2`, now
`Old_Support/`) and `inkling2_loop` (mainline llama.cpp >= b12611, #32704; Unsloth's root file since 2026-11-23). Model:
thinkingmachines/Inkling-2-Mini (17.6 GB UD-Q4_K_XL). Published package: `loom-fun/unsloth-Inkling-2-Mini-loom` (pinned
`ea0c79b2`, gfx1151). The port uses its own kit beside the starter (kernels `gen_inkling2.py`, prefill attention
`gen_ink_flash2.py`, drafter `draft/gen_dspark.py`, channel chat template); it is not in `loom-port-llm/starter/` yet.
Facets: looped, moe, sliding-window; drafted (DSpark) and vision (`inkling2v` projector) when the package has them.

Published card (2026-11-23, Strix Halo, DSpark, f16 KV cache, 256k context) vs llama.cpp ROCm b1383: decode short / 2k / 32k
48.5 / 45.2 / 39.1 vs 15.7 / 14.9 / 10.7 tok/s; prefill 1k-32k 1,589-1,857 vs 158-293 tok/s; 4 streams 95.8 vs 39.4.
Mainline llama.cpp HIP builds are faster than b1383 on this model (24.3 tok/s tg128, 383 pp2048 reported in gufo PR #1340).

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

## Two GGUF layouts (same weight bytes)

`inkling2_loop` renames metadata (`inkling2.*` -> `inkling2_loop.*`, `num_loops` -> `loop.count`, `loop.core_start/end`
-> `loop.start/end`) and loop tensors (`loop_{attn,k,v,mlp}_{down,forget,gain}` -> `loop_{q,k,v,ffn}_ctrl_*`), and re-slices
the two packed `loop_res_query` / `loop_kv_res_query` (2304x24) into 12 per-layer `loop_msg_query` (2304x4). The other 588
tensors are byte-equal (header diff plus 15 random 64 KB samples, 2026-11-23). The new root drafter is the mainline DSpark
format (unpadded `markov_w1` [256, 200058], explicit `markov_w2`, softmax-normalised `ctx_mix`); the old one is
`*_Q8_0_Old.gguf`. The mmproj files are unchanged. A kit written for one layout needs a name map for the other.
