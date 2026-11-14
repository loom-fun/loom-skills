# Facet: hybrid-gdn: decode

## Decode vs prefill split of the recurrence kernels

  - Decode vs prefill split (MiniCPM-V-4.7 / Qwen3.5-35B-A3B, DevLab, active wait): spans shorter than 64 tokens
    (decode, verify) run `g_gdn_recurrence` with RL 16 x 2 columns, `GDN_CONV_UNROLL=8`; longer spans run
    `g_gdn_recurrence_pf` (RL 8 x 4 columns, the 2k winner above). Uniform 16 x 2 gave decode +5-8 % but prefill -12 to
    -16 % on the MoE hybrids (-3 % on Qwen3.5-9B), hence two kernels; a span runs in exactly one of them, chosen by its own length, so batching never changes its numerics. `GDN_ROW_LANES`/`GDN_COLS` set in the environment
    force one form for both.
