# Metric census (measure 1)

Generated 2026-11-14 22:19 by `eng/guard10/census.py` over 51 models; re-run it for fresh numbers.
Current = loom-fun-staging main. Reference: Strix = official loom-fun card; R9700 = the pre-campaign staging revision in fleet/manifest.json.
Rows: decode x3, prefill x5, streams4 (acceptance not counted). Bands per row from methodology.json (prefill_8k 2.0%, prefill_16k 2.0%, prefill_32k 2.0%, prefill_1k 3.0%, prefill_2k 3.0%, decode 3.5%, decode_32k 3.0%, decode_2k 4.0%, streams4 4.0%). better > 1+band, worse < 1-band.

## Totals per GPU

| GPU | rows | better | within | worse | not worse |
|---|---|---|---|---|---|
| strix-halo | 457 | 26 | 413 | 18 | 96.1% |
| radeon-r9700 | 429 | 9 | 408 | 12 | 97.2% |

## strix-halo: per model (better / within / worse)

- InternScience-Sage-A1.5-loom: 4 / 5 / 0
- XHToken-Spark-X2.5-4B-loom: 0 / 9 / 0
- bartowski-LFM3-1.7B-loom: 0 / 9 / 0
- bartowski-Mellum2.1-12B-A2.5B-Thinking-loom: 3 / 6 / 0
- bartowski-MiniCPM-V-4.7-35B-A3B-loom: 1 / 8 / 0
- bartowski-Nanbeige4.3-33B-loom: 1 / 8 / 0
- bartowski-Ornith-1.5-122B-loom: 0 / 9 / 0
- bartowski-Ornith-1.5-35B-A3B-loom: 0 / 9 / 0
- bartowski-Ornith-1.5-9B-loom: 0 / 9 / 0
- bartowski-Qwen2.5-0.5B-Instruct-loom: 1 / 8 / 0
- bartowski-Qwen2.5-1.5B-Instruct-loom: 0 / 9 / 0
- bartowski-Qwen2.5-32B-Instruct-loom: 0 / 9 / 0
- bartowski-Qwen2.5-7B-Instruct-loom: 0 / 9 / 0
- bartowski-Qwen3.8-27B-loom: 0 / 9 / 0
- bartowski-Spark-X2.6-4B-loom: 0 / 9 / 0
- ggml-org-Flint-loom: 4 / 5 / 0
- ggml-org-LFM2.5-24B-A2B-loom: 0 / 9 / 0
- ggml-org-LFM3-Omni-2B-loom: 0 / 9 / 0
- ggml-org-Laguna-XS-2.1-loom: 0 / 9 / 0
- ggml-org-Muse-Glimmer-30B-loom: 0 / 9 / 0
- ggml-org-Qwen2.5-VL-7B-Instruct-loom: 0 / 9 / 0
- ggml-org-Qwen3-1.7B-loom: 0 / 9 / 0
- ggml-org-Qwen3-14B-loom: 0 / 9 / 0
- ggml-org-Qwen3-4B-loom: 1 / 8 / 0
- ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom: 0 / 9 / 0
- ggml-org-Qwen3.6-27B-loom: 0 / 0 / 9
- ggml-org-Qwen3.6-35B-A3B-loom: 0 / 9 / 0
- ggml-org-gemma-3-1b-it-loom: 4 / 3 / 0
- lmstudio-community-Ornith-1.0-35B-loom: 0 / 9 / 0
- lmstudio-community-gemma-4-26B-A4B-it-loom: 0 / 9 / 0
- nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom: 0 / 9 / 0
- rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom: 0 / 9 / 0
- unsloth-Llama-3.2-3B-Instruct-loom: 0 / 0 / 9
- unsloth-Ornith-1.0-9B-loom: 0 / 9 / 0
- unsloth-Qwen3-0.6B-loom: 0 / 9 / 0
- unsloth-Qwen3-30B-A3B-Instruct-2507-loom: 0 / 9 / 0
- unsloth-Qwen3-4B-Instruct-2507-loom: 2 / 7 / 0
- unsloth-Qwen3-8B-loom: 0 / 9 / 0
- unsloth-Qwen3-VL-4B-Instruct-loom: 2 / 7 / 0
- unsloth-Qwen3-VL-8B-Instruct-loom: 0 / 9 / 0
- unsloth-Qwen3.5-0.8B-loom: 1 / 8 / 0
- unsloth-Qwen3.5-2B-loom: 0 / 9 / 0
- unsloth-Qwen3.5-35B-A3B-loom: 0 / 9 / 0
- unsloth-Qwen3.5-4B-loom: 1 / 8 / 0
- unsloth-Qwen3.5-9B-loom: 0 / 9 / 0
- unsloth-Qwen3.8-Flash-Next-loom: 0 / 9 / 0
- unsloth-gemma-4-12B-it-loom: 1 / 8 / 0
- unsloth-gemma-4-31B-it-loom: 0 / 9 / 0
- unsloth-gemma-4-E2B-it-loom: 0 / 9 / 0
- unsloth-gemma-4-E4B-it-loom: 0 / 9 / 0
- unsloth-gpt-oss-20b-loom: 0 / 9 / 0

### strix-halo: worse rows (18)

| model | row | reference | current | ratio | band | reason |
|---|---|---|---|---|---|---|
| ggml-org-Qwen3.6-27B-loom | decode | 22.8 | - | - | 3.5% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| ggml-org-Qwen3.6-27B-loom | decode_2k | 21.3 | - | - | 4.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| ggml-org-Qwen3.6-27B-loom | decode_32k | 18.4 | - | - | 3.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| ggml-org-Qwen3.6-27B-loom | prefill_16k | 510.8 | - | - | 2.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| ggml-org-Qwen3.6-27B-loom | prefill_1k | 527.3 | - | - | 3.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| ggml-org-Qwen3.6-27B-loom | prefill_2k | 533.2 | - | - | 3.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| ggml-org-Qwen3.6-27B-loom | prefill_32k | 462.7 | - | - | 2.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| ggml-org-Qwen3.6-27B-loom | prefill_8k | 535.4 | - | - | 2.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| ggml-org-Qwen3.6-27B-loom | streams4 | 43.9 | - | - | 4.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| unsloth-Llama-3.2-3B-Instruct-loom | decode | 98.7 | - | - | 3.5% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| unsloth-Llama-3.2-3B-Instruct-loom | decode_2k | 85.1 | - | - | 4.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| unsloth-Llama-3.2-3B-Instruct-loom | decode_32k | 37.6 | - | - | 3.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| unsloth-Llama-3.2-3B-Instruct-loom | prefill_16k | 2291.4 | - | - | 2.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| unsloth-Llama-3.2-3B-Instruct-loom | prefill_1k | 3711.9 | - | - | 3.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| unsloth-Llama-3.2-3B-Instruct-loom | prefill_2k | 3761.5 | - | - | 3.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| unsloth-Llama-3.2-3B-Instruct-loom | prefill_32k | 1530.3 | - | - | 2.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| unsloth-Llama-3.2-3B-Instruct-loom | prefill_8k | 3064.0 | - | - | 2.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |
| unsloth-Llama-3.2-3B-Instruct-loom | streams4 | 347.4 | - | - | 4.0% | current card has no results for this GPU (no performance session; not yet measured on staging) |

## radeon-r9700: per model (better / within / worse)

- InternScience-Sage-A1.5-loom: 1 / 8 / 0
- XHToken-Spark-X2.5-4B-loom: 0 / 9 / 0
- bartowski-LFM3-1.7B-loom: 0 / 9 / 0
- bartowski-Mellum2.1-12B-A2.5B-Thinking-loom: 0 / 9 / 0
- bartowski-MiniCPM-V-4.7-35B-A3B-loom: 0 / 9 / 0
- bartowski-Nanbeige4.3-33B-loom: 0 / 9 / 0
- bartowski-Ornith-1.5-122B-loom: no reference results (not counted)
- bartowski-Ornith-1.5-35B-A3B-loom: 1 / 8 / 0
- bartowski-Ornith-1.5-9B-loom: 0 / 9 / 0
- bartowski-Qwen2.5-0.5B-Instruct-loom: 0 / 9 / 0
- bartowski-Qwen2.5-1.5B-Instruct-loom: 0 / 6 / 3
- bartowski-Qwen2.5-32B-Instruct-loom: 0 / 7 / 0
- bartowski-Qwen2.5-7B-Instruct-loom: 0 / 9 / 0
- bartowski-Qwen3.8-27B-loom: 0 / 9 / 0
- bartowski-Spark-X2.6-4B-loom: 0 / 9 / 0
- ggml-org-Flint-loom: 0 / 7 / 2
- ggml-org-LFM2.5-24B-A2B-loom: 0 / 9 / 0
- ggml-org-LFM3-Omni-2B-loom: 0 / 9 / 0
- ggml-org-Laguna-XS-2.1-loom: 0 / 7 / 0
- ggml-org-Muse-Glimmer-30B-loom: 0 / 9 / 0
- ggml-org-Qwen2.5-VL-7B-Instruct-loom: 0 / 9 / 0
- ggml-org-Qwen3-1.7B-loom: 0 / 8 / 1
- ggml-org-Qwen3-14B-loom: 0 / 9 / 0
- ggml-org-Qwen3-4B-loom: 0 / 8 / 1
- ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom: 0 / 9 / 0
- ggml-org-Qwen3.6-27B-loom: 0 / 9 / 0
- ggml-org-Qwen3.6-35B-A3B-loom: 0 / 9 / 0
- ggml-org-gemma-3-1b-it-loom: 0 / 7 / 0
- lmstudio-community-Ornith-1.0-35B-loom: 0 / 9 / 0
- lmstudio-community-gemma-4-26B-A4B-it-loom: 0 / 7 / 0
- nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom: 0 / 9 / 0
- rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom: 0 / 9 / 0
- unsloth-Llama-3.2-3B-Instruct-loom: 0 / 9 / 0
- unsloth-Ornith-1.0-9B-loom: 0 / 9 / 0
- unsloth-Qwen3-0.6B-loom: 0 / 8 / 1
- unsloth-Qwen3-30B-A3B-Instruct-2507-loom: 0 / 9 / 0
- unsloth-Qwen3-4B-Instruct-2507-loom: 0 / 9 / 0
- unsloth-Qwen3-8B-loom: 1 / 8 / 0
- unsloth-Qwen3-VL-4B-Instruct-loom: 0 / 9 / 0
- unsloth-Qwen3-VL-8B-Instruct-loom: 0 / 9 / 0
- unsloth-Qwen3.5-0.8B-loom: 0 / 5 / 4
- unsloth-Qwen3.5-2B-loom: 0 / 9 / 0
- unsloth-Qwen3.5-35B-A3B-loom: 1 / 8 / 0
- unsloth-Qwen3.5-4B-loom: 0 / 9 / 0
- unsloth-Qwen3.5-9B-loom: 0 / 9 / 0
- unsloth-Qwen3.8-Flash-Next-loom: no reference results (not counted)
- unsloth-gemma-4-12B-it-loom: 3 / 6 / 0
- unsloth-gemma-4-31B-it-loom: 1 / 4 / 0
- unsloth-gemma-4-E2B-it-loom: 0 / 9 / 0
- unsloth-gemma-4-E4B-it-loom: 0 / 9 / 0
- unsloth-gpt-oss-20b-loom: 1 / 8 / 0

### radeon-r9700: worse rows (12)

| model | row | reference | current | ratio | band | reason |
|---|---|---|---|---|---|---|
| unsloth-Qwen3-0.6B-loom | prefill_8k | 33323.2 | 31370.8 | 0.9414 | 2.0% | same programs as the reference: runner/session noise |
| ggml-org-Flint-loom | decode_2k | 186.6 | 175.8 | 0.9421 | 4.0% | same programs as the reference: runner/session noise |
| ggml-org-Qwen3-4B-loom | decode_2k | 184.2 | 175.2 | 0.9511 | 4.0% | same programs as the reference: runner/session noise |
| ggml-org-Qwen3-1.7B-loom | prefill_8k | 19847.4 | 18944.2 | 0.9545 | 2.0% | same programs as the reference: runner/session noise |
| unsloth-Qwen3.5-0.8B-loom | prefill_2k | 31069.6 | 29857.7 | 0.961 | 3.0% | same programs as the reference: runner/session noise |
| bartowski-Qwen2.5-1.5B-Instruct-loom | prefill_8k | 23177.8 | 22321.3 | 0.963 | 2.0% | same programs as the reference: runner/session noise |
| unsloth-Qwen3.5-0.8B-loom | prefill_8k | 33407.6 | 32269.4 | 0.9659 | 2.0% | same programs as the reference: runner/session noise |
| ggml-org-Flint-loom | decode_32k | 138.8 | 134.2 | 0.9669 | 3.0% | same programs as the reference: runner/session noise |
| unsloth-Qwen3.5-0.8B-loom | prefill_16k | 29989.7 | 29060.8 | 0.969 | 2.0% | same programs as the reference: runner/session noise |
| bartowski-Qwen2.5-1.5B-Instruct-loom | prefill_16k | 18596.2 | 18078.2 | 0.9721 | 2.0% | same programs as the reference: runner/session noise |
| unsloth-Qwen3.5-0.8B-loom | prefill_32k | 25013.2 | 24341.3 | 0.9731 | 2.0% | same programs as the reference: runner/session noise |
| bartowski-Qwen2.5-1.5B-Instruct-loom | prefill_32k | 12887.2 | 12546.5 | 0.9736 | 2.0% | same programs as the reference: runner/session noise |
