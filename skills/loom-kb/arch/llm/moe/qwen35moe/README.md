# Family: qwen35moe

GGUF arch `qwen35moe` (Qwen3.5/3.6 35B-A3B, Ornith, Sage, MiniCPM-V 4.7 35B-A3B, 122B): hybrid gated-delta-net + MoE (256 experts, top-8), vision, usually drafted. Facets: moe, hybrid-gdn, vision, drafted, kv-le2 (2 KV heads).

## Prefill at 1k (hybrid MoE): where the time went and what moved it

  - Hybrid MoE at 1k (qwen35moe, 256 experts, top-8; doubling on Ornith-1.0-35B, ms per token of 0.55): expert
    GEMMs 0.19, dense GEMMs 0.11, `g_moe_combine_sorted` 0.06, `g_gdn_norm_gate` 0.04, recurrence 0.03. The two
    small kernels were launch-shape bound: combine ran a workgroup per (token, 256 channels) with tokens fastest,
    so each 8 KB expert row was read in 8 pieces far apart (now one workgroup per token); norm_gate ran 32-lane
    workgroups, tokens x 32 of them per layer (now 8 heads per workgroup). The grouped GEMMs also prefetch the
    next K step into registers and skip the MMAs of empty 16-row waves (`MOE_PREFETCH`, `MOE_SKIP_EMPTY`; together
    +5 %, the skip alone ties as on Flint). Same outputs bit for bit. DevLab, prefill 1k: Ornith-1.0-35B 1838 ->
    2423 tok/s, Sage-A1.5 1768 -> 2197, Qwen3.5-35B-A3B 1740 -> 2120 (+22-32 %; 2k +24-42 %), decode tie.

## Fleet models (published packages, llm_config.json)

- `InternScience-Sage-A1.5-loom` (qwen35moe): 40L h2048, 16h/2kv hd256, moe 256e top-8, kv_layers 10, vision, drafter mtp d3, ctx 262144
- `bartowski-MiniCPM-V-4.7-35B-A3B-loom` (qwen35moe): 40L h2048, 16h/2kv hd256, moe 256e top-8, kv_layers 10, ctx 262144
- `bartowski-Ornith-1.5-122B-loom` (qwen35moe): 48L h3072, 32h/2kv hd256, moe 256e top-8, kv_layers 12, vision, drafter dflash d1, ctx 262144
- `bartowski-Ornith-1.5-35B-A3B-loom` (qwen35moe): 40L h2048, 16h/2kv hd256, moe 256e top-8, kv_layers 10, vision, drafter dflash d1, ctx 262144
- `ggml-org-Qwen3.6-35B-A3B-loom` (qwen35moe): 40L h2048, 16h/2kv hd256, moe 256e top-8, kv_layers 10, vision, drafter dflash d3, ctx 262144
- `lmstudio-community-Ornith-1.0-35B-loom` (qwen35moe): 40L h2048, 16h/2kv hd256, moe 256e top-8, kv_layers 10, vision, ctx 262144
- `unsloth-Qwen3.5-35B-A3B-loom` (qwen35moe): 40L h2048, 16h/2kv hd256, moe 256e top-8, kv_layers 10, vision, drafter dflash d3, ctx 262144
