## Fleet program diff: 7 of 51 models touched

GPUs: gfx11, gfx12; rows that can move: prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k

| model | files | blocks (+/-/~) | GPUs | rows | stratum |
|---|---|---|---|---|---|
| XHToken-Spark-X2.5-4B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | spark2_5 hd256 kv4 none dense |
| bartowski-LFM3-1.7B-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | lfm3 hd128 kv4 none dense |
| bartowski-Spark-X2.6-4B-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | spark2_5 hd256 kv4 none dense |
| ggml-org-LFM3-Omni-2B-loom | program/llm.loom | 0/0/6 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | lfm3 hd128 kv4 none dense |
| nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv<=2 dflash dense |
| unsloth-gemma-4-E2B-it-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv<=2 assist dense |
| unsloth-gemma-4-E4B-it-loom | program/llm.loom | 0/0/4 | gfx11+gfx12 | prefill_1k, prefill_2k, prefill_8k, prefill_16k, prefill_32k | gemma4 hd512 kv<=2 assist dense |

### Changed kernel families (models)

- `kernel:g_attention_flash2`: 5 model(s)
- `kernel:g_attention_flash2_dN`: 3 model(s)
- `kernel:g_attention_flash2_dN_partial`: 3 model(s)
- `kernel:g_attention_flash2_deep`: 2 model(s)
- `kernel:g_attention_flash2_partial`: 5 model(s)
