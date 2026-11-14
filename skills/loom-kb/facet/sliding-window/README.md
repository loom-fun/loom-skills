# Facet: sliding-window

Sliding-window attention layers (gemma3/gemma4, spark2_5, lfm3, glimmer, laguna, flint, mellum, gpt-oss): window, per-layer pattern, RoPE variants and flash tiles.

## Declaring sliding layers (cfg review)

- Sliding-window layers: list them in `sliding_layers` and set `sliding_window` (from the GGUF's pattern, else
  `inspect_gguf.SWA_PERIOD`: `arch: (n, dense_first)`, the period llama.cpp hard-codes in `load_swa_pattern`).
  Sliding layers with their own RoPE at the same head_dim (`rope_theta_swa`, `rope_dims_swa` from
  `rope.freq_base_swa` / `rope.dimension_count_swa`, spark2_5) get `g_qk_rope_store_swa`.
