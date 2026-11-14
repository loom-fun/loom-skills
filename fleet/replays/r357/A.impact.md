## Fleet program diff: 17 of 51 models touched

GPUs: gfx11, gfx12; rows that can move: decode, decode_2k, decode_32k, streams4

| model | files | blocks (+/-/~) | GPUs | rows | stratum |
|---|---|---|---|---|---|
| bartowski-LFM3-1.7B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | lfm3 hd128 kv4 none dense |
| bartowski-Mellum2.1-12B-A2.5B-Thinking-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | mellum hd128 kv4 none moe |
| bartowski-Nanbeige4.3-33B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | llama hd128 kv>=8 none dense |
| bartowski-Qwen2.5-1.5B-Instruct-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen2 hd128 kv<=2 none dense |
| bartowski-Qwen2.5-32B-Instruct-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen2 hd128 kv>=8 none dense |
| bartowski-Qwen2.5-7B-Instruct-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen2 hd128 kv4 none dense |
| ggml-org-LFM3-Omni-2B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | lfm3 hd128 kv4 none dense |
| ggml-org-Qwen2.5-VL-7B-Instruct-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen2vl hd128 kv4 none dense |
| ggml-org-Qwen3-1.7B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| ggml-org-Qwen3-14B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | llama hd128 kv>=8 none dense |
| unsloth-Llama-3.2-3B-Instruct-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | llama hd128 kv>=8 none dense |
| unsloth-Qwen3-0.6B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-30B-A3B-Instruct-2507-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3moe hd128 kv4 none moe |
| unsloth-Qwen3-4B-Instruct-2507-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-VL-4B-Instruct-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3vl hd128 kv>=8 none dense |
| unsloth-Qwen3-VL-8B-Instruct-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3vl hd128 kv>=8 none dense |

### Changed kernel families (models)

- `kernel:g_attention_split`: 17 model(s)
