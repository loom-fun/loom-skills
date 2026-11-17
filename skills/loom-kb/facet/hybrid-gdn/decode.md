# Facet: hybrid-gdn: decode

## Decode vs prefill split of the recurrence kernels

  - Decode vs prefill split (MiniCPM-V-4.7 / Qwen3.5-35B-A3B, DevLab, active wait): spans shorter than 64 tokens
    (decode, verify) run `g_gdn_recurrence` with RL 16 x 2 columns, `GDN_CONV_UNROLL=8`; longer spans run
    `g_gdn_recurrence_pf` (RL 8 x 4 columns, the 2k winner above). Uniform 16 x 2 gave decode +5-8 % but prefill -12 to
    -16 % on the MoE hybrids (-3 % on Qwen3.5-9B), hence two kernels; a span runs in exactly one of them, chosen by its own length, so batching never changes its numerics. `GDN_ROW_LANES`/`GDN_COLS` set in the environment
    force one form for both.

## Fusing the GDN front of a drafted step does not pay (R9700, gfx12, Qwen3.8-27B, q38r9700)

  - Doubling (run 43770401446, MTP 3 step 35.4 ms): conv_commit 1.08 ms, conv + commit + qk_norm 1.60 ms, recurrence 0.05 (verify) /
    0.03 (replay), norm_gate 0.14 per step. The capture / restore copies were the cheap win (#866, `decode.spec.gdn_copy_split`, +2 %).
  - A fused gfx12 front (conv + history commit + q/k norms in one launch; ktest, gates and text match pass) LOST 4 % decode (v1 128-thread
    blocks, run 43771517926; v2 one wave per 128 channels, run 43774217524): the fused kernel costs ~18 us per launch, the three legacy
    kernels ~16.6 us together. The chain is latency-bound (dependent metadata / origin / history loads), not launch-bound. Not merged.

## Fold instead of capture / restore / replay (`decode.spec.gdn_fold`, R9700, MXFP4 Qwen3.8-27B, mxfp4st)

  - The replay of a drafted step re-runs conv + q/k norms + the recurrence over the accepted inputs: it re-reads every layer's state
    (3 MiB) a second time and costs ~240 tiny launches. With `GDN_FOLD` the verification records what the fold needs (raw conv input
    and key per step, the update scalar dk per value column and the decay per head; banks of 8 rows per speculative span, indexed by its
    rank), and after acceptance ONE `g_gdn_fold` launch (all layers, S = fma(k, dk, S * decay) per accepted step, the recurrence's own
    op order) plus ONE `g_gdn_fold_hist` launch publish the state and the conv history. Bit-identical to the replay (ktest
    `gdn_fold_vs_replay`, kbench `gdn_fold`). Drafted step 33.94 -> 33.45 ms, GDN 2.00 -> 1.48 ms/step, decode +2 %.
  - Size per-span scratch with a cap: the runtime prepares @llm_verify at every packing shape up to (1024, 1024) (schedule.c), so
    `spans x 8` rows were 29 GB there (RESOURCE_EXHAUSTED at serve); `min(spans, 32) x 8` keeps the largest verify scratch below the capture's.
  - With a fused prefill GDN (`prefill.gdn.fused2`), @llm_verify keeps only the chain branch (`gen_draft.drop_fused_gdn`): a verify epoch
    can pack another request's prompt rows with the speculative spans (GEMM-shape, t >= 64), and the fused pair records no fold factors
    and runs a 4-token span in the 32-token chunk form. The chain records them with decode's sequential kernel and chunks the prompt
    spans; @llm_advance (every prompt-only epoch) keeps the fused pair. ktest `gdn_fold_serve`: fold == replay bit-exact on a 210-row
    mixed epoch (prompt 150 + 40, anchor, three speculative spans, rows out of packet order, gaps, stale packets).
  - Deferred fold (`decode.spec.gdn_fold_defer`, opt-in, off everywhere): the accepted steps' factors become the writer slot's
    pending fold (a block per layer after every layer's state, +8 % slot); the next verification's recurrence applies them in
    registers and writes the folded state back (one state pass per step instead of two); every other reader (advance epochs,
    publishing spans, forks from a retained slot) runs `g_gdn_flush` + `g_gdn_pend_clear` first. Bit-identical to the eager fold
    (ktest `gdn_fold_defer`: two verifications, a fork, a slot read by two spans). R9700 MXFP4 card, DevLab, base pinned: GDN
    1.05 -> 0.79 ms/step, but drafted decode ties (75.38 vs 75.11, 4 ABBA sessions; device step 32.15 vs 32.21 ms) and 4-stream
    ties. Measured lesson: the folded-state write, issued inside the per-layer stream with a regular cache policy, slowed the next
    weight matvecs by ~0.35 ms/step (dirty lines written back during the weight stream); a non-temporal store removes that, and what
    is left is the extra write the latency-bound recurrence now carries (+0.15-0.2 ms). ninfer reports the same (net ~0 at one
    stream). Not a decode lever on this hardware; a last-use cache hint on the state read made the recurrence slower (0.68 -> 0.88).
