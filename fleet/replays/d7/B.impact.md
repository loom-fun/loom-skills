## Fleet program diff: 5 of 51 models touched

GPUs: gfx11, gfx12; rows that can move: decode, decode_2k, decode_32k, streams4

| model | files | blocks (+/-/~) | GPUs | rows | stratum |
|---|---|---|---|---|---|
| InternScience-Sage-A1.5-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35moe hd256 kv<=2 mtp moe |
| bartowski-Ornith-1.5-122B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| bartowski-Ornith-1.5-35B-A3B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| ggml-org-Qwen3.6-35B-A3B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| unsloth-Qwen3.5-35B-A3B-loom | program/llm.loom | 0/0/1 | gfx11+gfx12 | decode, decode_2k, decode_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |

### Changed kernel families (models)

- `kernel:g_attention_split`: 5 model(s)
