# Family inkling2: decode

Strix Halo gfx1151, hrx-loom 0.2.0, FP16 KV, rounds 2-9 (2026-11-21..23). Every kept change below is bit-identical to the
build before it unless it says otherwise.

## Contents

- Where a verify step goes
- Kept: dispatch count and fusions
- Kept: Loom low-level controls (`command.concurrent`, `pipeline`, Low helpers)
- Kept: sampler (`g_head_select`)
- Did not help
- Measuring drafted decode

## Where a verify step goes (DSpark depth 1, t = 2, ~35 ms before the low-level round)

- Bandwidth-bound work is ~27 ms and at or near the 228 GB/s roof: NVFP4 MoE matvecs 13 ms (~15 expert-equivalents of
  the 16 (token, expert) pairs), dense q/gate/k/v/rel 5.0 ms (93 % of bandwidth after multi-matrix matvecs), target head
  2.36 ms (92 %), proposal head 1.3 ms, Q8_0 MoE 2.1 ms, sliding-window attention 0.74 ms (4 MB per layer).
- The rest, ~7 ms in round 2 and ~3.7 ms by round 7, is ~15 families of 3-9 us kernels (loop controllers, mixes,
  shortconv, norms, glue) over ~1,350-1,900 dispatches per step. Each fusion was worth 0.3-0.8 ms. HAL busy time equals
  wall time on decode (no host gaps), so the tail is kernel latency, not launch gaps.

## Kept: dispatch count and fusions

- Multi-matrix int8 matvec (`gen_pstep.pmv`, generalised to single-plane Q8_0) for q, gate, k, v, rel at 2-4 tokens
  (`INK_PMV_QKV=1`): k2304 dense 5.49 -> 5.0 ms per step. Separate small k/v launches reach only ~153 GB/s. At t = 1, the
  `g_pmv_q8_0_k2304_t1_q8_r1_m{3,5}` launches removed 136 dispatches per step (1,257 -> 1,121). Credit: shawnshekari/strixite.
- Exact bf16 copies of the bf16-exact router and controller matrices (README): -1.7 % per step, 4 streams +5-10 %.
- RMSNorm + Q8 quantize in one kernel (`g_ink_rms_norm_plane_q8f`), mix + norm + quantize (`g_ink_mix{2..5}_norm_q8f`),
  shortconv span + KV prep + sliding-window ring store in one kernel, Q8-output NVFP4 gate_up (`g_ink_moe_gu_nvfp4_w1_q8o`,
  -0.86 ms), MoE combine + shortconv + add: 1,622 -> 1,382 dispatches per step in round 3.
- Live-token early return in the 1-wave MoE matvecs (`g_ink_moe_{gu,dn}_{nvfp4,q8_0}_w1*`): padding rows read no expert
  weights. -7.71 ms per step (+28.5 % MoE) on a 3-row depth-2 verify, and no 4 -> 5 row padding cliff with several streams.
  Fused Q8-out Q8_0 MoE for the 4 Q8_0 layers: -0.29 ms per step at depth 3. Credit: Annieren (llama.cpp #26301), shawnshekari.
- One 96-thread workgroup per (row, token) for the decode bf16 projection: 4 streams +6 %. A 1-wave subgroup-reduce
  version was faster but reordered the sum (short-prompt compare 8 -> 6 full matches): not kept.

## Kept: Loom low-level controls (round 7)

Before round 7 `llm.loom` had 8,349 `kernel.launch` in program order and no `low.invoke`, `schedule`, `pipeline` or
`command.concurrent`; all pipelining was hand-unrolled in the generators (I-cache overflow, VGPR tier cliffs, spills).

- `command.concurrent` around independent launches (1,116 regions, `INK_CMD_CONC=1`: q/k/v controllers on decode-only
  scratch, mixes, 3-way norms, attention projections, qr_prep + shortconv, attention split + partial): prose decode
  44.97 -> 47.46 tok/s (+5.5 %, -1.83 ms per step), code +4.1 %, 4 streams +4.0 %. The HAL records no barrier between
  siblings; summed dispatch time stayed the same while wall time dropped.
- Target head `g_head_mv_q8` with `scf.for pipeline(3) unroll(2)` at t = 1: 2,355 -> 2,263 us (-4.4 %), 56 VGPR.
- Paged-KV split attention with `pipeline(3) unroll(2) schedule(recurrence)` instead of hand unrolling, plus prefetch of
  the next page's map entry and a lazy O-rescale (skip the rescale unless the max grew by more than tau = 8,
  `kernel.subgroup.vote.any`): q2 1,650 -> 1,559 us per layer at 32k-74k (~85 % of 228 GB/s), q1 48.7 -> 24.6 us; q2 now
  fits the 16 KiB I-cache with 0 spills (q4 still spills 68 B).
- SDK facts found on the way (gfx1151, 0.2.0): `scf.for pipeline` lowers with prologue loads and partial `vmcnt` waits
  across the back edge; `low.func.def` on gfx11-generic, `schedule(phased)` and `low.invoke` work (the helper's
  instructions appear inside the caller's loop in the compile report); `scf.schedule.fence` is a full scheduling barrier
  and blocked prefetch interleaving; indirect `kernel.launch @k[%cnt]` from a `view<3xi32>` lowers to a dynamic indirect
  dispatch. Not on gfx11: f16 `vector.fragment.repack<lhs>` (bf16 only, `AMDGPU/041`), split barriers
  (`kernel.barrier.arrive`, gfx12+), `kernel.async.tensor.load.to.lds` (gfx1250+). `{slc=1}` compiles but MALL allocation
  on the APU is set by the page tables, so it did nothing.

## Kept: sampler (`g_head_select`, round 8)

At T > 0 the old `g_head_select` made 64 serial passes over the 200k f32 logits in one workgroup: 5,760 us per token, the
whole T=1 vs greedy gap. An exact two-stage top-k reuses the per-128-token chunk maxima already in `%partials`, keeps the
top <= 64 chunks and extracts from those only: 5,760 -> 199 us undrafted, 6,190 -> 230 us on a verify step; undrafted
decode is now the same at T = 1.0 (44.0) and T = 0 (43.8). Bit-exact on 80 adversarial cases (8 distributions x 10 seeds).
Loom's `top_k=0` means 64 candidates; there is no `min_p`.

## Did not help

- Raising decode attention occupancy (long-context.md).
- Expert-grouped decode MoE (`INK_MOE_GRP`): +0.9 ms per step at t = 2, +2.3 ms at depths 2-3. `MOE_W1_R=2`: +0.19 ms.
- Rows per workgroup for the dense matvecs (`MV_WG1` 256 / 1024): tie. `MV_PIPE=3 MV_WG1=512`: -1.2 % on k2304 but
  +2.5 % on k3072 (wave tail at 576 workgroups).

## Measuring drafted decode

- Drafted tok/s swings ±5-10 % with acceptance: any sub-argmax change in the hidden state (even a prefill-only kernel on
  a short prompt) moves DSpark acceptance while the text stays identical. A/B on ms per verify step,
  (tokens / tok_s) / (tokens - accepted), and check text and draft counters are identical.
- Short prompts (< 64 tokens) never take the prefill GEMM path: a short-prompt divergence means a decode kernel changed.
- With adaptive depth, `/metrics` `draft_proposed_tokens` divides by the host `draft_depth` (3), so reported acceptance
  is ~1/3 of the per-token rate during depth-1 steps.
- Strix Halo reaches STAPM at 85-87 C: a sweep that heats up shows 32k decode at ~15 instead of ~38 tok/s. Start runs
  below ~65 C and record sclk.
