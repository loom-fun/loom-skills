# Family: deepseek2

The `deepseek2` example uses absorbed multi-head latent attention (MLA) for the JoyAI-LLM-Flash port. The cache stores one shared row of 512 latent values and 64 rotary values per layer and token. Prefill expands selected segments into per-head keys and values before flash attention.

The reference kernels support 32 query heads, a positive query rank divisible by 256, 512 KV latent values, 64 rotary dimensions, 128 non-rotary key dimensions, and 128 value dimensions. They require separate `attn_k_b` and `attn_v_b` projections and full attention layers. The JoyAI-specific choices are in its model node.
