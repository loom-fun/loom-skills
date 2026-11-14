## Fleet program diff: 50 of 51 models touched

GPUs: gfx11, gfx12; rows that can move: decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4

| model | files | blocks (+/-/~) | GPUs | rows | stratum |
|---|---|---|---|---|---|
| InternScience-Sage-A1.5-loom | program/llm.loom, program/vision_encoder.loom | 105/0/23 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 mtp moe |
| XHToken-Spark-X2.5-4B-loom | program/llm.loom | 74/0/12 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | spark2_5 hd256 kv4 none dense |
| bartowski-LFM3-1.7B-loom | program/llm.loom | 120/0/19 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | lfm3 hd128 kv4 none dense |
| bartowski-Mellum2.1-12B-A2.5B-Thinking-loom | program/llm.loom | 60/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | mellum hd128 kv4 none moe |
| bartowski-MiniCPM-V-4.7-35B-A3B-loom | program/llm.loom | 81/0/16 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 none moe |
| bartowski-Nanbeige4.3-33B-loom | program/llm.loom | 120/0/19 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | llama hd128 kv>=8 none dense |
| bartowski-Ornith-1.5-122B-loom | program/llm.loom, program/vision_encoder.loom | 208/0/34 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| bartowski-Ornith-1.5-35B-A3B-loom | program/llm.loom, program/vision_encoder.loom | 177/0/32 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| bartowski-Ornith-1.5-9B-loom | program/llm.loom, program/vision_encoder.loom | 157/0/27 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv4 dflash dense |
| bartowski-Qwen2.5-0.5B-Instruct-loom | program/llm.loom | 73/0/17 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen2 hd64 kv<=2 none dense |
| bartowski-Qwen2.5-1.5B-Instruct-loom | program/llm.loom | 78/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen2 hd128 kv<=2 none dense |
| bartowski-Qwen2.5-32B-Instruct-loom | program/llm.loom | 78/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen2 hd128 kv>=8 none dense |
| bartowski-Qwen2.5-7B-Instruct-loom | program/llm.loom | 78/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen2 hd128 kv4 none dense |
| bartowski-Qwen3.8-27B-loom | program/llm.loom, program/vision_encoder.loom | 211/0/30 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv4 mtp dense |
| bartowski-Spark-X2.6-4B-loom | program/llm.loom | 86/0/15 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | spark2_5 hd256 kv4 none dense |
| ggml-org-Flint-loom | program/llm.loom | 151/0/24 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | flint hd128 kv4 dflash moe |
| ggml-org-LFM2.5-24B-A2B-loom | program/llm.loom | 90/0/16 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | lfm2moe hd64 kv>=8 none moe |
| ggml-org-LFM3-Omni-2B-loom | program/llm.loom | 87/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | lfm3 hd128 kv4 none dense |
| ggml-org-Laguna-XS-2.1-loom | program/llm.loom | 149/0/41 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | laguna hd128 kv>=8 dflash moe |
| ggml-org-Muse-Glimmer-30B-loom | program/llm.loom, program/vision_encoder.loom | 149/0/26 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | muse-glimmer hd128 kv<=2 dflash dense |
| ggml-org-Qwen2.5-VL-7B-Instruct-loom | program/llm.loom, program/vision_encoder.loom | 93/0/17 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen2vl hd128 kv4 none dense |
| ggml-org-Qwen3-1.7B-loom | program/llm.loom | 74/0/12 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| ggml-org-Qwen3-14B-loom | program/llm.loom | 78/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| ggml-org-Qwen3-4B-loom | program/llm.loom | 129/0/22 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3 hd128 kv>=8 dflash dense |
| ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom | program/llm.loom | 116/0/22 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3moe hd128 kv4 dflash moe |
| ggml-org-Qwen3.6-27B-loom | program/llm.loom, program/vision_encoder.loom | 141/0/23 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv4 dflash dense |
| ggml-org-Qwen3.6-35B-A3B-loom | program/llm.loom, program/vision_encoder.loom | 116/0/20 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| ggml-org-gemma-3-1b-it-loom | program/llm.loom | 78/0/16 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma3 hd256 kv<=2 none dense |
| lmstudio-community-Ornith-1.0-35B-loom | program/llm.loom, program/vision_encoder.loom | 105/0/20 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 none moe |
| lmstudio-community-gemma-4-26B-A4B-it-loom | program/llm.loom | 159/0/25 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma4 hd512 kv>=8 dflash moe |
| nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom | program/llm.loom, program/vision_encoder.loom | 191/0/33 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma4 hd512 kv<=2 dflash dense |
| rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom | program/llm.loom | 74/0/12 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | llama hd128 kv>=8 none dense |
| unsloth-Llama-3.2-3B-Instruct-loom | program/llm.loom | 78/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | llama hd128 kv>=8 none dense |
| unsloth-Ornith-1.0-9B-loom | program/llm.loom, program/vision_encoder.loom | 123/0/20 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv4 none dense |
| unsloth-Qwen3-0.6B-loom | program/llm.loom | 74/0/12 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-30B-A3B-Instruct-2507-loom | program/llm.loom | 60/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3moe hd128 kv4 none moe |
| unsloth-Qwen3-4B-Instruct-2507-loom | program/llm.loom | 78/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-8B-loom | program/llm.loom | 129/0/22 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3 hd128 kv>=8 dflash dense |
| unsloth-Qwen3-VL-4B-Instruct-loom | program/llm.loom, program/vision_encoder.loom | 105/0/20 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3vl hd128 kv>=8 none dense |
| unsloth-Qwen3-VL-8B-Instruct-loom | program/llm.loom, program/vision_encoder.loom | 97/0/18 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3vl hd128 kv>=8 none dense |
| unsloth-Qwen3.5-0.8B-loom | program/llm.loom, program/vision_encoder.loom | 131/0/22 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv<=2 none dense |
| unsloth-Qwen3.5-2B-loom | program/llm.loom, program/vision_encoder.loom | 131/0/22 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv<=2 none dense |
| unsloth-Qwen3.5-35B-A3B-loom | program/llm.loom, program/vision_encoder.loom | 152/0/28 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| unsloth-Qwen3.5-4B-loom | program/llm.loom, program/vision_encoder.loom | 182/0/31 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv4 dflash dense |
| unsloth-Qwen3.5-9B-loom | program/llm.loom, program/vision_encoder.loom | 178/0/30 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv4 dflash dense |
| unsloth-gemma-4-12B-it-loom | program/llm.loom | 154/0/17 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma4 hd512 kv>=8 dflash dense |
| unsloth-gemma-4-31B-it-loom | program/llm.loom | 158/0/18 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma4 hd512 kv>=8 dflash dense |
| unsloth-gemma-4-E2B-it-loom | program/llm.loom, program/vision_encoder.loom | 263/0/46 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma4 hd512 kv<=2 assist dense |
| unsloth-gemma-4-E4B-it-loom | program/llm.loom, program/vision_encoder.loom | 203/0/34 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma4 hd512 kv<=2 assist dense |
| unsloth-gpt-oss-20b-loom | program/llm.loom | 51/0/13 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gpt-oss hd64 kv>=8 none moe |

### Changed kernel families (models)

- `decl:device`: 50 model(s)
- `decl:rdna3`: 50 model(s)
- `decl:rdna4`: 50 model(s)
- `func:g16_split_count`: 1 model(s)
- `func:g16_split_count_hN`: 1 model(s)
- `func:g16d256_split_count`: 1 model(s)
- `func:g16d512_split_count`: 1 model(s)
- `func:g_dfl_kv_offset`: 1 model(s)
- `func:g_f32_values`: 1 model(s)
- `func:g_pair_deep`: 1 model(s)
- `func:g_pair_deep_hN`: 1 model(s)
- `func:g_split_count`: 1 model(s)
- `func:g_split_count_hN`: 1 model(s)
- `func:g_tile_uniform`: 2 model(s)
- `func:gd256_split_count`: 1 model(s)
- `func:gd512_split_count`: 1 model(s)
- `kernel:g16_attention_split_listed`: 1 model(s)
- `kernel:g16_attention_split_listed_hN`: 1 model(s)
- `kernel:g16d256_attention_split`: 1 model(s)
- `kernel:g16d512_attention_split`: 1 model(s)
- `kernel:g_attention_flash2`: 50 model(s)
- `kernel:g_attention_flash2_dN`: 6 model(s)
- `kernel:g_attention_flash2_dN_partial`: 6 model(s)
- `kernel:g_attention_flash2_deep`: 20 model(s)
- `kernel:g_attention_flash2_deep_hN`: 1 model(s)
- `kernel:g_attention_flash2_hN`: 1 model(s)
- `kernel:g_attention_flash2_partial`: 50 model(s)
- `kernel:g_attention_flash2_partial_hN`: 1 model(s)
- `kernel:g_attention_split`: 1 model(s)
- `kernel:g_attention_split_hN`: 1 model(s)
- `kernel:g_dfl_kv_store`: 1 model(s)
- `kernel:g_gemm_bf16_store`: 3 model(s)
- `kernel:g_gemm_bf16_store_mN`: 3 model(s)
- `kernel:g_gemm_f32_store`: 2 model(s)
- `kernel:g_gemm_f32_store_mN`: 2 model(s)
- `kernel:g_gemm_q4_0_store`: 1 model(s)
- `kernel:g_gemm_q4_0_store_mN`: 1 model(s)
- `kernel:g_gemm_q4_0_swiglu`: 1 model(s)
- `kernel:g_gemm_q4_0_swiglu_mN`: 1 model(s)
- `kernel:g_gemm_q4_k_store`: 47 model(s)
- `kernel:g_gemm_q4_k_store_mN`: 47 model(s)
- `kernel:g_gemm_q4_k_swiglu`: 42 model(s)
- `kernel:g_gemm_q4_k_swiglu_f16o`: 1 model(s)
- `kernel:g_gemm_q4_k_swiglu_mN`: 42 model(s)
- `kernel:g_gemm_q4_k_swiglu_mN_f16o`: 1 model(s)
- `kernel:g_gemm_q5_0_store`: 4 model(s)
- `kernel:g_gemm_q5_0_store_mN`: 4 model(s)
- `kernel:g_gemm_q5_0_swiglu`: 2 model(s)
- `kernel:g_gemm_q5_0_swiglu_mN`: 2 model(s)
- `kernel:g_gemm_q5_k_store`: 11 model(s)
- `kernel:g_gemm_q5_k_store_mN`: 11 model(s)
- `kernel:g_gemm_q6_k_store`: 46 model(s)
- `kernel:g_gemm_q6_k_store_mN`: 46 model(s)
- `kernel:g_gemm_q6_k_swiglu`: 1 model(s)
- `kernel:g_gemm_q6_k_swiglu_mN`: 1 model(s)
- `kernel:g_gemm_q8_0_store`: 26 model(s)
- `kernel:g_gemm_q8_0_store_mN`: 26 model(s)
- `kernel:g_gemm_q8_0_swiglu`: 14 model(s)
- `kernel:g_gemm_q8_0_swiglu_f16o`: 1 model(s)
- `kernel:g_gemm_q8_0_swiglu_mN`: 14 model(s)
