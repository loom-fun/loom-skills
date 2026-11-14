## Fleet program diff: 50 of 51 models touched

GPUs: gfx11, gfx12; rows that can move: prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k

| model | files | blocks (+/-/~) | GPUs | rows | stratum |
|---|---|---|---|---|---|
| InternScience-Sage-A1.5-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35moe hd256 kv<=2 mtp moe |
| XHToken-Spark-X2.5-4B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | spark2_5 hd256 kv4 none dense |
| bartowski-LFM3-1.7B-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | lfm3 hd128 kv4 none dense |
| bartowski-Mellum2.1-12B-A2.5B-Thinking-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | mellum hd128 kv4 none moe |
| bartowski-MiniCPM-V-4.7-35B-A3B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35moe hd256 kv<=2 none moe |
| bartowski-Nanbeige4.3-33B-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | llama hd128 kv>=8 none dense |
| bartowski-Ornith-1.5-122B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35moe hd256 kv<=2 dflash moe |
| bartowski-Ornith-1.5-35B-A3B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35moe hd256 kv<=2 dflash moe |
| bartowski-Ornith-1.5-9B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 dflash dense |
| bartowski-Qwen2.5-0.5B-Instruct-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2 hd64 kv<=2 none dense |
| bartowski-Qwen2.5-1.5B-Instruct-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2 hd128 kv<=2 none dense |
| bartowski-Qwen2.5-32B-Instruct-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2 hd128 kv>=8 none dense |
| bartowski-Qwen2.5-7B-Instruct-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2 hd128 kv4 none dense |
| bartowski-Qwen3.8-27B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 mtp dense |
| bartowski-Spark-X2.6-4B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | spark2_5 hd256 kv4 none dense |
| ggml-org-Flint-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | flint hd128 kv4 dflash moe |
| ggml-org-LFM2.5-24B-A2B-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | lfm2moe hd64 kv>=8 none moe |
| ggml-org-LFM3-Omni-2B-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | lfm3 hd128 kv4 none dense |
| ggml-org-Laguna-XS-2.1-loom | program/llm.loom | 0/0/12 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | laguna hd128 kv>=8 dflash moe |
| ggml-org-Muse-Glimmer-30B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | muse-glimmer hd128 kv<=2 dflash dense |
| ggml-org-Qwen2.5-VL-7B-Instruct-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2vl hd128 kv4 none dense |
| ggml-org-Qwen3-1.7B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 none dense |
| ggml-org-Qwen3-14B-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 none dense |
| ggml-org-Qwen3-4B-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 dflash dense |
| ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3moe hd128 kv4 dflash moe |
| ggml-org-Qwen3.6-27B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 dflash dense |
| ggml-org-Qwen3.6-35B-A3B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35moe hd256 kv<=2 dflash moe |
| ggml-org-gemma-3-1b-it-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma3 hd256 kv<=2 none dense |
| lmstudio-community-Ornith-1.0-35B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35moe hd256 kv<=2 none moe |
| lmstudio-community-gemma-4-26B-A4B-it-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv>=8 dflash moe |
| nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv<=2 dflash dense |
| rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | llama hd128 kv>=8 none dense |
| unsloth-Llama-3.2-3B-Instruct-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | llama hd128 kv>=8 none dense |
| unsloth-Ornith-1.0-9B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 none dense |
| unsloth-Qwen3-0.6B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-30B-A3B-Instruct-2507-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3moe hd128 kv4 none moe |
| unsloth-Qwen3-4B-Instruct-2507-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-8B-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 dflash dense |
| unsloth-Qwen3-VL-4B-Instruct-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3vl hd128 kv>=8 none dense |
| unsloth-Qwen3-VL-8B-Instruct-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3vl hd128 kv>=8 none dense |
| unsloth-Qwen3.5-0.8B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv<=2 none dense |
| unsloth-Qwen3.5-2B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv<=2 none dense |
| unsloth-Qwen3.5-35B-A3B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35moe hd256 kv<=2 dflash moe |
| unsloth-Qwen3.5-4B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 dflash dense |
| unsloth-Qwen3.5-9B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 dflash dense |
| unsloth-gemma-4-12B-it-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv>=8 dflash dense |
| unsloth-gemma-4-31B-it-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv>=8 dflash dense |
| unsloth-gemma-4-E2B-it-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv<=2 assist dense |
| unsloth-gemma-4-E4B-it-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv<=2 assist dense |
| unsloth-gpt-oss-20b-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gpt-oss hd64 kv>=8 none moe |

### Changed kernel families (models)

- `kernel:g_attention_flash2`: 45 model(s)
- `kernel:g_attention_flash2_dN`: 6 model(s)
- `kernel:g_attention_flash2_dN_partial`: 6 model(s)
- `kernel:g_attention_flash2_deep`: 20 model(s)
- `kernel:g_attention_flash2_deep_hN`: 1 model(s)
- `kernel:g_attention_flash2_hN`: 1 model(s)
- `kernel:g_attention_flash2_partial`: 45 model(s)
- `kernel:g_attention_flash2_partial_hN`: 1 model(s)
