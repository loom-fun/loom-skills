## Fleet program diff: 6 of 51 models touched

GPUs: gfx11, gfx12; rows that can move: decode, decode_2k, decode_32k, streams4

| model | files | blocks (+/-/~) | GPUs | rows | stratum |
|---|---|---|---|---|---|
| ggml-org-Qwen3-1.7B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| ggml-org-Qwen3-14B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-0.6B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-4B-Instruct-2507-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-VL-4B-Instruct-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3vl hd128 kv>=8 none dense |
| unsloth-Qwen3-VL-8B-Instruct-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3vl hd128 kv>=8 none dense |

### Changed kernel families (models)

- `kernel:g_attention_split`: 6 model(s)
