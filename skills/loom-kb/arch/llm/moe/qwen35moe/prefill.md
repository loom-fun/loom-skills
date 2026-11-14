# Family qwen35moe: prefill

## Prefill at 1k (hybrid MoE): where the time went and what moved it

  - Hybrid MoE at 1k (qwen35moe, 256 experts, top-8; doubling on Ornith-1.0-35B, ms per token of 0.55): expert
    GEMMs 0.19, dense GEMMs 0.11, `g_moe_combine_sorted` 0.06, `g_gdn_norm_gate` 0.04, recurrence 0.03. The two
    small kernels were launch-shape bound: combine ran a workgroup per (token, 256 channels) with tokens fastest,
    so each 8 KB expert row was read in 8 pieces far apart (now one workgroup per token); norm_gate ran 32-lane
    workgroups, tokens x 32 of them per layer (now 8 heads per workgroup). The grouped GEMMs also prefetch the
    next K step into registers and skip the MMAs of empty 16-row waves (`MOE_PREFETCH`, `MOE_SKIP_EMPTY`; together
    +5 %, the skip alone ties as on Flint). Same outputs bit for bit. DevLab, prefill 1k: Ornith-1.0-35B 1838 ->
    2423 tok/s, Sage-A1.5 1768 -> 2197, Qwen3.5-35B-A3B 1740 -> 2120 (+22-32 %; 2k +24-42 %), decode tie.
