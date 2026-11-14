## Fleet program diff: 8 of 51 models touched

GPUs: gfx11, gfx12; rows that can move: decode, decode_2k, decode_32k, streams4

| model | files | blocks (+/-/~) | GPUs | rows | stratum |
|---|---|---|---|---|---|
| bartowski-Ornith-1.5-9B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35 hd256 kv4 dflash dense |
| bartowski-Qwen3.8-27B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35 hd256 kv4 mtp dense |
| ggml-org-Flint-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | flint hd128 kv4 dflash moe |
| ggml-org-Qwen3-4B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 dflash dense |
| ggml-org-Qwen3.6-27B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35 hd256 kv4 dflash dense |
| unsloth-Qwen3-8B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen3 hd128 kv>=8 dflash dense |
| unsloth-Qwen3.5-4B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35 hd256 kv4 dflash dense |
| unsloth-Qwen3.5-9B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35 hd256 kv4 dflash dense |

### Changed kernel families (models)

- `kernel:g_attention_split`: 8 model(s)
