## Fleet program diff: 50 of 51 models touched

GPUs: gfx11, gfx12; rows that can move: decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4

| model | files | blocks (+/-/~) | GPUs | rows | stratum |
|---|---|---|---|---|---|
| InternScience-Sage-A1.5-loom | program/llm.loom, program/vision_encoder.loom | 18/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 mtp moe |
| XHToken-Spark-X2.5-4B-loom | program/llm.loom | 11/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | spark2_5 hd256 kv4 none dense |
| bartowski-LFM3-1.7B-loom | program/llm.loom | 18/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | lfm3 hd128 kv4 none dense |
| bartowski-Mellum2.1-12B-A2.5B-Thinking-loom | program/llm.loom | 12/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | mellum hd128 kv4 none moe |
| bartowski-MiniCPM-V-4.7-35B-A3B-loom | program/llm.loom | 15/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 none moe |
| bartowski-Nanbeige4.3-33B-loom | program/llm.loom | 18/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | llama hd128 kv>=8 none dense |
| bartowski-Ornith-1.5-122B-loom | program/llm.loom, program/vision_encoder.loom | 28/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| bartowski-Ornith-1.5-35B-A3B-loom | program/llm.loom, program/vision_encoder.loom | 27/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| bartowski-Ornith-1.5-9B-loom | program/llm.loom, program/vision_encoder.loom | 22/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv4 dflash dense |
| bartowski-Qwen2.5-0.5B-Instruct-loom | program/llm.loom | 16/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2 hd64 kv<=2 none dense |
| bartowski-Qwen2.5-1.5B-Instruct-loom | program/llm.loom | 12/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2 hd128 kv<=2 none dense |
| bartowski-Qwen2.5-32B-Instruct-loom | program/llm.loom | 12/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2 hd128 kv>=8 none dense |
| bartowski-Qwen2.5-7B-Instruct-loom | program/llm.loom | 12/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2 hd128 kv4 none dense |
| bartowski-Qwen3.8-27B-loom | program/llm.loom, program/vision_encoder.loom | 25/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 mtp dense |
| bartowski-Spark-X2.6-4B-loom | program/llm.loom | 14/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | spark2_5 hd256 kv4 none dense |
| ggml-org-Flint-loom | program/llm.loom | 19/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | flint hd128 kv4 dflash moe |
| ggml-org-LFM2.5-24B-A2B-loom | program/llm.loom | 15/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | lfm2moe hd64 kv>=8 none moe |
| ggml-org-LFM3-Omni-2B-loom | program/llm.loom | 12/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | lfm3 hd128 kv4 none dense |
| ggml-org-Laguna-XS-2.1-loom | program/llm.loom | 23/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | laguna hd128 kv>=8 dflash moe |
| ggml-org-Muse-Glimmer-30B-loom | program/llm.loom, program/vision_encoder.loom | 20/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | muse-glimmer hd128 kv<=2 dflash dense |
| ggml-org-Qwen2.5-VL-7B-Instruct-loom | program/llm.loom, program/vision_encoder.loom | 15/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen2vl hd128 kv4 none dense |
| ggml-org-Qwen3-1.7B-loom | program/llm.loom | 11/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 none dense |
| ggml-org-Qwen3-14B-loom | program/llm.loom | 12/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 none dense |
| ggml-org-Qwen3-4B-loom | program/llm.loom | 18/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 dflash dense |
| ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom | program/llm.loom | 17/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3moe hd128 kv4 dflash moe |
| ggml-org-Qwen3.6-27B-loom | program/llm.loom, program/vision_encoder.loom | 18/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 dflash dense |
| ggml-org-Qwen3.6-35B-A3B-loom | program/llm.loom, program/vision_encoder.loom | 14/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| ggml-org-gemma-3-1b-it-loom | program/llm.loom | 15/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma3 hd256 kv<=2 none dense |
| lmstudio-community-Ornith-1.0-35B-loom | program/llm.loom, program/vision_encoder.loom | 18/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 none moe |
| lmstudio-community-gemma-4-26B-A4B-it-loom | program/llm.loom | 21/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma4 hd512 kv>=8 dflash moe |
| nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom | program/llm.loom, program/vision_encoder.loom | 26/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv<=2 dflash dense |
| rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom | program/llm.loom | 11/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | llama hd128 kv>=8 none dense |
| unsloth-Llama-3.2-3B-Instruct-loom | program/llm.loom | 12/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | llama hd128 kv>=8 none dense |
| unsloth-Ornith-1.0-9B-loom | program/llm.loom, program/vision_encoder.loom | 18/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 none dense |
| unsloth-Qwen3-0.6B-loom | program/llm.loom | 11/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-30B-A3B-Instruct-2507-loom | program/llm.loom | 12/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen3moe hd128 kv4 none moe |
| unsloth-Qwen3-4B-Instruct-2507-loom | program/llm.loom | 12/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 none dense |
| unsloth-Qwen3-8B-loom | program/llm.loom | 18/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3 hd128 kv>=8 dflash dense |
| unsloth-Qwen3-VL-4B-Instruct-loom | program/llm.loom, program/vision_encoder.loom | 18/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3vl hd128 kv>=8 none dense |
| unsloth-Qwen3-VL-8B-Instruct-loom | program/llm.loom, program/vision_encoder.loom | 16/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen3vl hd128 kv>=8 none dense |
| unsloth-Qwen3.5-0.8B-loom | program/llm.loom, program/vision_encoder.loom | 20/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv<=2 none dense |
| unsloth-Qwen3.5-2B-loom | program/llm.loom, program/vision_encoder.loom | 20/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv<=2 none dense |
| unsloth-Qwen3.5-35B-A3B-loom | program/llm.loom, program/vision_encoder.loom | 23/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35moe hd256 kv<=2 dflash moe |
| unsloth-Qwen3.5-4B-loom | program/llm.loom, program/vision_encoder.loom | 26/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | qwen35 hd256 kv4 dflash dense |
| unsloth-Qwen3.5-9B-loom | program/llm.loom, program/vision_encoder.loom | 25/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | qwen35 hd256 kv4 dflash dense |
| unsloth-gemma-4-12B-it-loom | program/llm.loom | 13/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv>=8 dflash dense |
| unsloth-gemma-4-31B-it-loom | program/llm.loom | 14/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma4 hd512 kv>=8 dflash dense |
| unsloth-gemma-4-E2B-it-loom | program/llm.loom, program/vision_encoder.loom | 35/0/3 | gfx11+gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gemma4 hd512 kv<=2 assist dense |
| unsloth-gemma-4-E4B-it-loom | program/llm.loom, program/vision_encoder.loom | 29/0/0 | gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv<=2 assist dense |
| unsloth-gpt-oss-20b-loom | program/llm.loom | 12/0/0 | gfx12 | decode, decode_2k, decode_32k, prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k, streams4 | gpt-oss hd64 kv>=8 none moe |

### Changed kernel families (models)

- `kernel:g_attention_flash2`: 50 model(s)
- `kernel:g_attention_flash2_dN`: 6 model(s)
- `kernel:g_attention_flash2_dN_partial`: 6 model(s)
- `kernel:g_attention_flash2_deep`: 20 model(s)
- `kernel:g_attention_flash2_deep_hN`: 1 model(s)
- `kernel:g_attention_flash2_hN`: 1 model(s)
- `kernel:g_attention_flash2_partial`: 50 model(s)
- `kernel:g_attention_flash2_partial_hN`: 1 model(s)
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
- `kernel:g_gemm_q8_0_swiglu_mN_f16o`: 1 model(s)
- `kernel:g_moe_gdn_mxfp4_kN`: 1 model(s)
- `kernel:g_moe_gdn_q4_k_kN`: 11 model(s)
- `kernel:g_moe_gdn_q5_0_kN`: 2 model(s)
- `kernel:g_moe_gdn_q5_k_kN`: 3 model(s)
- `kernel:g_moe_gdn_q6_k_kN`: 7 model(s)
- `kernel:g_moe_gdn_q8_0_kN`: 4 model(s)
- `kernel:g_moe_ggu_mxfp4_kN`: 1 model(s)
- `kernel:g_moe_ggu_q4_k_fused_kN`: 1 model(s)
- `kernel:g_moe_ggu_q4_k_kN`: 13 model(s)
- `kernel:g_moe_ggu_q4_k_q5_k_kN`: 1 model(s)
- `kernel:g_moe_ggu_q4_k_q6_k_kN`: 3 model(s)
- `kernel:g_moe_ggu_q6_k_q8_0_kN`: 3 model(s)
- `kernel:g_mv_q4_k_kN_tN_q8`: 2 model(s)
- `kernel:g_mv_q8_0_kN_tN_q8`: 1 model(s)
- `kernel:v_gemm_bf16_store_1152x1152`: 3 model(s)
- `kernel:v_gemm_bf16_store_1152x3456`: 3 model(s)
- `kernel:v_gemm_bf16_store_1536x1536`: 1 model(s)
- `kernel:v_gemm_bf16_store_1536x8960`: 1 model(s)
- `kernel:v_gemm_bf16_store_4096x4096`: 1 model(s)
- `kernel:v_gemm_bf16_store_4096x6656`: 1 model(s)
- `kernel:v_gemm_bf16_store_4608x2048`: 2 model(s)
- `kernel:v_gemm_bf16_store_4608x4608`: 3 model(s)
