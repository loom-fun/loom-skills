# Family: llama

GGUF arch `llama` (Llama-3.x, Nanbeige, HyperCLOVAX): neox-less `norm` RoPE pairing, Llama-3 frequency factors, json tool calls.

## Llama 3.x chat template (cfg review)

  - Llama 3.x (`call_format` `json`: `{"name": ..., "parameters": ...}` without a wrapper, one call per turn):
    `system_preamble`/`system_preamble_tools` (the always-printed system block's "Cutting Knowledge Date ... Today
    Date ..." text, "Environment: ipython" with tools), `tools_in_user` (the tools open the first user message),
    `tools_indent` (tojson(indent=4)), `calls_drop_content`, `tool_content_json` (results as JSON strings);
    `templates/control_json.loom` holds their control functions. The date: `strftime_now` renders the request day in
    llama.cpp, which a control program can't; Loom renders the templates' own no-clock date, 26 Jul 2024
    (`chat_probe.CLOCK`), so prompts are the same every day. The architecture side is `rope_freq_factors`
    (`rope_freqs.weight`: llama.cpp divides each rotary pair's frequency by it).

## Fleet models (published packages, llm_config.json)

- `bartowski-Nanbeige4.3-33B-loom` (llama): 52L h7168, 56h/8kv hd128, ctx 524288
- `rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom` (llama): 24L h2048, 16h/8kv hd128, ctx 131072
- `unsloth-Llama-3.2-3B-Instruct-loom` (llama): 28L h3072, 24h/8kv hd128, ctx 131072
