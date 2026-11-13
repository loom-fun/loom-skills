# Drafting classification of all 51 cards (final benchmark rule)

Choice per model, identical on Loom and llama.cpp ROCm+Vulkan: MTP head in the card's own GGUF > DFlash/assistant drafter published for this exact model (maker or Z-Lab, GGUF = faithful conversion) > no drafting. No ngram. Counts: {'mtp': 5, 'dflash': 17, 'assist': 2, 'none': 27}; Loom rebuilds needed: 20.

Evidence for MTP: the checkpoint GGUF header was read (`nextn_predict_layers`, nextn tensors; `work/baselines/nextn.json`). Drafter provenance: HF repo + its upstream release (`classify_data.json`). llama.cpp n-max: the drafter README's stated value (GGUF repo `--spec-draft-n-max`, else upstream `num_speculative_tokens` / block size - 1), else llama.cpp's default (flag omitted). Loom keeps its tuned depth. MTP GGUF variants published separately upstream (e.g. unsloth/Qwen3.5-9B-MTP-GGUF) are different checkpoints and are NOT used (the card's GGUF has no NextN head) - noted as a possible follow-up.

| card | choice | live Loom | llama.cpp spec (ROCm and Vulkan, 1 stream; streams4 none) | n-max source | provenance / evidence | Loom action |
|---|---|---|---|---|---|---|
| bartowski-Qwen3.8-27B-loom | mtp | dflash 3 | `--spec-type draft-mtp` | llama.cpp default | https://huggingface.co/bartowski/Qwen3.8-27B-GGUF qwen35.nextn_predict_layers=1, 4 nextn tensors | REBUILD: dflash -> mtp |
| ggml-org-Qwen3.6-35B-A3B-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 7 -ngld 99` | z-lab/Qwen3.6-35B-A3B-DFlash README (dflash block size - 1) | https://huggingface.co/magnitudedev/Qwen3.6-35B-A3B-DFlash-GGUF; https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| bartowski-Ornith-1.5-35B-A3B-loom | mtp | dflash 1 | `--spec-type draft-mtp` | llama.cpp default | https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF qwen35moe.nextn_predict_layers=1, 4 nextn tensors | REBUILD: dflash -> mtp |
| bartowski-Ornith-1.5-9B-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 8 -ngld 99` | ornith-ai/Ornith-1.5-9B-DFlash README (num_speculative_tokens 8) | https://huggingface.co/abenzerps/Ornith-1.5-9B-DFlash-GGUF; https://huggingface.co/ornith-ai/Ornith-1.5-9B-DFlash drafter published for this exact model by its maker; GGUF = conversion of that release | ok: keep |
| unsloth-gemma-4-12B-it-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 16 -ngld 99` | Anbeeld/gemma4-12B-it-DFlash-GGUF README (--spec-draft-n-max) | https://huggingface.co/Anbeeld/gemma4-12B-it-DFlash-GGUF; https://huggingface.co/z-lab/gemma4-12B-it-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| unsloth-gemma-4-E4B-it-loom | assist | assist 3 | `--spec-type draft-mtp --spec-draft-model {drafter} -ngld 99` | not stated: llama.cpp default | https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF; https://huggingface.co/google/gemma-4-E4B-it-assistant drafter published for this exact model by its maker; GGUF = conversion of that release | ok: keep |
| unsloth-Qwen3.5-9B-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 7 -ngld 99` | z-lab/Qwen3.5-9B-DFlash README (dflash block size - 1) | https://huggingface.co/AtomicChat/Qwen3.5-9B-DFlash-GGUF; https://huggingface.co/z-lab/Qwen3.5-9B-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| lmstudio-community-gemma-4-26B-A4B-it-loom | dflash | dflash 1 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 15 -ngld 99` | z-lab/gemma-4-26B-A4B-it-DFlash README (num_speculative_tokens) | https://huggingface.co/Anbeeld/gemma-4-26B-A4B-it-DFlash-GGUF; https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| unsloth-Ornith-1.0-9B-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| bartowski-Ornith-1.5-122B-loom | mtp | dflash 1 | `--spec-type draft-mtp` | llama.cpp default | https://huggingface.co/bartowski/Ornith-1.5-122B-GGUF qwen35moe.nextn_predict_layers=1 (part 1 header; tensors in later part: helper verifies) | REBUILD: dflash -> mtp |
| unsloth-Qwen3.5-4B-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 7 -ngld 99` | z-lab/Qwen3.5-4B-DFlash README (dflash block size - 1) | https://huggingface.co/magnitudedev/Qwen3.5-4B-DFlash-GGUF; https://huggingface.co/z-lab/Qwen3.5-4B-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| ggml-org-LFM2.5-24B-A2B-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| XHToken-Spark-X2.5-4B-loom | none | ngram 1 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| unsloth-Qwen3-VL-8B-Instruct-loom | none | ngram 3 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| InternScience-Sage-A1.5-loom | mtp | mtp 3 | `--spec-type draft-mtp` | llama.cpp default | https://huggingface.co/InternScience/Sage-A1.5-Q4_K_M-GGUF qwen35moe.nextn_predict_layers=1, 4 nextn tensors | ok: keep |
| unsloth-gemma-4-E2B-it-loom | assist | assist 2 | `--spec-type draft-mtp --spec-draft-model {drafter} -ngld 99` | not stated: llama.cpp default | https://huggingface.co/AtomicChat/gemma-4-E2B-it-assistant-GGUF; https://huggingface.co/google/gemma-4-E2B-it-assistant drafter published for this exact model by its maker; GGUF = conversion of that release | ok: keep |
| ggml-org-Qwen3-4B-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 15 -ngld 99` | z-lab/Qwen3-4B-DFlash-b16 README (num_speculative_tokens) | https://huggingface.co/AtomicChat/Qwen3-4B-DFlash-GGUF; https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16 drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| lmstudio-community-Ornith-1.0-35B-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| unsloth-Qwen3-8B-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 15 -ngld 99` | z-lab/Qwen3-8B-DFlash-b16 README (num_speculative_tokens) | https://huggingface.co/AtomicChat/Qwen3-8B-DFlash-GGUF; https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16 drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom | none | ngram 3 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| unsloth-Qwen3-VL-4B-Instruct-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| ggml-org-Muse-Glimmer-30B-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 15 -ngld 99` | z-lab/Muse-Glimmer-30B-DFlash2-GGUF README (--spec-draft-n-max) | https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2-GGUF; https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2-GGUF drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| ggml-org-Laguna-XS-2.1-loom | dflash | none  | `--spec-type draft-dflash --spec-draft-model {hf:dev7a/Laguna-XS-2.1-DFlash-GGUF/Laguna-XS-2.1-DFlash-Q8_0.gguf} --spec-draft-n-max 15 -ngld 99` | poolside/Laguna-XS-2.1-DFlash README (num_speculative_tokens) | https://huggingface.co/dev7a/Laguna-XS-2.1-DFlash-GGUF; https://huggingface.co/poolside/Laguna-XS-2.1-DFlash drafter published for this exact model by its maker; GGUF = conversion of that release | REBUILD: none -> dflash |
| unsloth-gpt-oss-20b-loom | dflash | none  | `--spec-type draft-dflash --spec-draft-model {hf:Anbeeld/gpt-oss-20b-DFlash-GGUF/FILE} --spec-draft-n-max 7 -ngld 99` | z-lab/gpt-oss-20b-DFlash README (num_speculative_tokens) | https://huggingface.co/Anbeeld/gpt-oss-20b-DFlash-GGUF; https://huggingface.co/z-lab/gpt-oss-20b-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | REBUILD: none -> dflash |
| bartowski-Qwen2.5-7B-Instruct-loom | none | ngram 2 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| bartowski-Spark-X2.6-4B-loom | none | ngram 3 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| bartowski-LFM3-1.7B-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| unsloth-Qwen3-4B-Instruct-2507-loom | none | none  | `none` |  |  the Qwen3-4B DFlash drafter was trained for Qwen3-4B, not -Instruct-2507 | ok: keep |
| ggml-org-LFM3-Omni-2B-loom | none | ngram 1 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| ggml-org-Flint-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} -ngld 99` | not stated: llama.cpp default | https://huggingface.co/incoai/Flint-DFlash2-GGUF; https://huggingface.co/incoai/Flint-DFlash2 drafter published for this exact model by its maker; GGUF = conversion of that release | ok: keep |
| ggml-org-Qwen3.6-27B-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 15 -ngld 99` | z-lab/Qwen3.6-27B-DFlash README (num_speculative_tokens) | https://huggingface.co/Anbeeld/Qwen3.6-27B-DFlash-GGUF; https://huggingface.co/z-lab/Qwen3.6-27B-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| unsloth-Qwen3.5-0.8B-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| unsloth-gemma-4-31B-it-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 15 -ngld 99` | z-lab/gemma-4-31B-it-DFlash README (num_speculative_tokens) | https://huggingface.co/Anbeeld/gemma-4-31B-it-DFlash-GGUF; https://huggingface.co/z-lab/gemma-4-31B-it-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| bartowski-Qwen2.5-32B-Instruct-loom | none | ngram 1 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| bartowski-MiniCPM-V-4.7-35B-A3B-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| bartowski-Nanbeige4.3-33B-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| unsloth-Qwen3.5-2B-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| unsloth-Qwen3-0.6B-loom | none | ngram 2 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| ggml-org-Qwen3-1.7B-loom | none | ngram 2 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom | dflash | none  | `--spec-type draft-dflash --spec-draft-model {hf:Anbeeld/Qwen3-Coder-30B-A3B-DFlash-GGUF/Qwen3-Coder-30B-A3B-DFlash-Q4_K_M.gguf} --spec-draft-n-max 15 -ngld 99` | z-lab/Qwen3-Coder-30B-A3B-DFlash README (num_speculative_tokens) | https://huggingface.co/Anbeeld/Qwen3-Coder-30B-A3B-DFlash-GGUF; https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | REBUILD: none -> dflash |
| unsloth-Qwen3-30B-A3B-Instruct-2507-loom | none | none  | `none` |  |  the only DFlash drafter on file is for Qwen3-Coder-30B-A3B, a different model | ok: keep |
| unsloth-Qwen3.5-35B-A3B-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 7 -ngld 99` | z-lab/Qwen3.5-35B-A3B-DFlash README (dflash block size - 1) | https://huggingface.co/Anbeeld/Qwen3.5-35B-A3B-DFlash-GGUF; https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash drafter published for this exact model by Z-Lab; GGUF = conversion of that release | ok: keep |
| ggml-org-gemma-3-1b-it-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| ggml-org-Qwen3-14B-loom | none | ngram 1 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom | dflash | dflash 3 | `--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 7 -ngld 99` | nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash README (--spec-draft-n-max) | https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash; https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4 drafter published for this exact model by its maker; GGUF = conversion of that release | ok: keep |
| bartowski-Qwen2.5-1.5B-Instruct-loom | none | ngram 2 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| ggml-org-Qwen2.5-VL-7B-Instruct-loom | none | ngram 1 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| bartowski-Qwen2.5-0.5B-Instruct-loom | none | ngram 3 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| unsloth-Llama-3.2-3B-Instruct-loom | none | ngram 2 | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | REBUILD: ngram -> none |
| bartowski-Mellum2.1-12B-A2.5B-Thinking-loom | none | none  | `none` |  |  no MTP head in the GGUF; no drafter from the model maker or Z-Lab for this exact model | ok: keep |
| unsloth-Qwen3.8-Flash-Next-loom | mtp | mtp 3 | `--spec-type draft-mtp --spec-draft-model {hf:unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf} -ngl 99` | llama.cpp default | https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF separate MTP GGUF (qwen4exp.nextn_predict_layers=1, 6 nextn tensors) from the same release | ok: keep |

## Rebuilds (20)

- bartowski-Qwen3.8-27B-loom: REBUILD: dflash -> mtp
- bartowski-Ornith-1.5-35B-A3B-loom: REBUILD: dflash -> mtp
- bartowski-Ornith-1.5-122B-loom: REBUILD: dflash -> mtp
- XHToken-Spark-X2.5-4B-loom: REBUILD: ngram -> none
- unsloth-Qwen3-VL-8B-Instruct-loom: REBUILD: ngram -> none
- rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom: REBUILD: ngram -> none
- ggml-org-Laguna-XS-2.1-loom: REBUILD: none -> dflash
- unsloth-gpt-oss-20b-loom: REBUILD: none -> dflash
- bartowski-Qwen2.5-7B-Instruct-loom: REBUILD: ngram -> none
- bartowski-Spark-X2.6-4B-loom: REBUILD: ngram -> none
- ggml-org-LFM3-Omni-2B-loom: REBUILD: ngram -> none
- bartowski-Qwen2.5-32B-Instruct-loom: REBUILD: ngram -> none
- unsloth-Qwen3-0.6B-loom: REBUILD: ngram -> none
- ggml-org-Qwen3-1.7B-loom: REBUILD: ngram -> none
- ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom: REBUILD: none -> dflash
- ggml-org-Qwen3-14B-loom: REBUILD: ngram -> none
- bartowski-Qwen2.5-1.5B-Instruct-loom: REBUILD: ngram -> none
- ggml-org-Qwen2.5-VL-7B-Instruct-loom: REBUILD: ngram -> none
- bartowski-Qwen2.5-0.5B-Instruct-loom: REBUILD: ngram -> none
- unsloth-Llama-3.2-3B-Instruct-loom: REBUILD: ngram -> none
