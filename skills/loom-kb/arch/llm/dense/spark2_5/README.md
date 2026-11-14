# Family: spark2_5

GGUF arch `spark2_5` (Spark-X 2.5/2.6 4B): head_dim 256, 4 query heads per KV head, 27 of 36 layers sliding, fused qkv, per-head attention gate, GEGLU. Facets: sliding-window.

## cfg review: `attn_qkv`, `attn_gate`, `ffn_act` and per-layer head counts

- `attn_qkv` (one fused q|k|v projection, split by `g_split_qkv`), `attn_gate: "per_head"` (one sigmoid logit
  per query head, broadcast by `g_gate_heads`; `attn_gate_act: "softplus"` applies softplus there instead, laguna),
  `ffn_act: "gelu"` (GEGLU; `inspect_gguf.FFN_ACT` lists the architectures llama.cpp hard-codes it for). spark2_5 uses
  all three. A per-layer `attention.head_count` gives `layer_heads` (`heads` is the widest): the attention kernels are
  generated again for each narrower count (`_h<heads>`) and kernel-tested at it.

## Fleet models (published packages, llm_config.json)

- `XHToken-Spark-X2.5-4B-loom` (spark2_5): 36L h2560, 16h/4kv hd256, swa 512, ctx 1048576
- `bartowski-Spark-X2.6-4B-loom` (spark2_5): 36L h2560, 16h/4kv hd256, swa 512, ctx 1048576
