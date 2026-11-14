# Topology: hybrid

Gated-delta-net (recurrent, constant state) layers interleaved with full-attention layers: qwen35 (dense FFN). The recurrent state lives in the runtime's recurrent slots (facet hybrid-gdn); only the attention layers (`kv_layers`) have a KV cache, so long-context and prefix reuse behave differently from dense models. The MoE hybrids (qwen35moe, qwen4exp) sit under topology moe and also match facet hybrid-gdn.
