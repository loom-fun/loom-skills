---
base_model: ggml-org/Flint-GGUF
tags:
- loom
- loom-serve
- gguf
- strix-halo
- speculative-decoding
- dflash
---

# Flint (Q4_K_M) for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint [ggml-org/Flint-GGUF](https://huggingface.co/ggml-org/Flint-GGUF) (`Flint-Q4_K_M.gguf`), generated on an AMD Ryzen AI Max+ 395 "Strix Halo" with the [loom-skills](https://github.com/loom-fun/loom-skills) workflow.

## Quick start

```bash
pip install "hrx-loom[hf]"
loom-fetch https://huggingface.co/loom-fun/ggml-org-Flint-loom/resolve/main/model.json --materialize=copy --output=flint
loom-serve --model=flint/model.json --device=amdgpu --memory_bytes=67000000000
```

The server listens on `127.0.0.1:8080` and answers OpenAI-style `/v1/chat/completions` requests.

## Performance vs llama.cpp

| metric | llama.cpp | Loom | margin | roofline | Loom % of roofline |
|---|---|---|---|---|---|
| decode tok/s, short prompt | 70.6 | 92.4 ¹ † | +31 % | 177.8 | 52 % |
| decode tok/s, 2k context | 54.7 | 76.3 | +39 % | 173.4 | 44 % |
| decode tok/s, 32k context | 49.5 | 47.8 | -3 % | 141.3 | 34 % |
| prefill tok/s, 1k document | 1038.0 | 956.9 ¹ † | -8 % | 12248.4 | 8 % |
| prefill tok/s, 2k document | 1079.0 | 1316.4 | +22 % | 11945.5 | 11 % |
| prefill tok/s, 8k document | 1094.1 | 1424.7 | +30 % | 10484.7 | 14 % |
| prefill tok/s, 16k document | 929.3 | 1359.0 | +46 % | 9014.9 | 15 % |
| prefill tok/s, 32k document | 667.6 | 1193.9 | +79 % | 7040.8 | 17 % |
| 4 streams, aggregate decode tok/s | 124.2 | 134.2 | +8 % | 301.8 | 44 % |
| speculative decoding, drafted tokens accepted | — | 68 % | — | n/a | n/a |

- llama.cpp: build `llamacpp-rocm/b1355`, server arguments `-ngl 99 -cram 0 --spec-type draft-dflash --spec-draft-model ~/drafter.gguf --spec-draft-n-max 3 -ngld 99 -c 65536 -kvu -np 4`.
- Loom: speculative decoding on (DFlash, depth 3).
- Same GGUF on both engines, tokens per second. Loom: client-side timing; llama.cpp: server-side timings. Runs per value: decode and prefill 2 (mean), 4 streams 3 (median).
- Measured 2026-11-06; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 26.04; hrx-loom `0.1.0` (hrx-system `d01d1c1`); source: local run.
- ¹ Measured 2026-11-07; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 26.04; hrx-loom `0.1.0` (hrx-system `d01d1c1`); source: local run.
- † Measured on programs other than this repository's (sha256 `aef7c52`, this repository's `38d15af`): the value describes another build of the package until a session re-measures it.
- Drafted tokens accepted: the share of the drafter's proposed tokens the model accepted during the `spec` gate's requests.
- Roofline: the hardware ceiling for this GGUF on AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151) at 216 GB/s and 59 TFLOP/s, from loom-roofline `roofline.py` (loom-skills `0123456`); decode and 4 streams with speculative decoding: verify passes × (1 + acceptance × depth 3) tokens per pass, the drafter's cost not counted; the `spec` gate's acceptance 68 % (× 3.04 tokens per pass), as the row's session recorded none: the gate's text can accept more or less than the row's.
- — not measured; n/a not applicable.

## Features

| feature | status | detail |
|---|---|---|
| API | ✅ | OpenAI-style `/v1/chat/completions` |
| tool calling | ✅ | OpenAI-style `tools` and `tool_calls` |
| reasoning / thinking | ✅ | always on |
| reasoning effort | ❌ | not supported |
| speculative decoding | ✅ | DFlash, depth 3, on by default; drafter [incoai/Flint-DFlash2-GGUF](https://huggingface.co/incoai/Flint-DFlash2-GGUF) |
| image input | ❌ | not supported |
| audio input / output | ❌ | not supported |
| sampling | ✅ | `temperature`, `top_p`, `top_k`, `seed` |
| context size | ✅ | 262,144 tokens per request |
| concurrency | ✅ | 4 concurrent requests |
| max output tokens | ✅ | 4,096 (preset `max_tokens`) |

## Correctness

| gate | description | result |
|---|---|---|
| `kernel_tests` | Unit tests of the generated kernels against reference implementations | ❌ no evidence |
| `golden` | llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents | ✅ pass ¹ |
| `compare` | Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie | ✅ pass ¹ |
| `vision` | Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties | — not applicable |
| `consistency` | Concurrent requests give the same output as the same requests run alone | ✅ pass ¹ |
| `spec` | Speculative decoding on gives exactly the output of speculative decoding off | ✅ pass ¹ |
| `agent` | Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp | ✅ pass ¹ |
| `agent_think_off` | The tool-calling loop with thinking off | — not applicable |
| `sampling` | temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp | ✅ pass ¹ |
| `needle` | Needle-in-a-haystack prompt at up to half the context, compared with llama.cpp; over-capacity prompt refused | ✅ pass ¹ |
| `needle_full` | Needle-in-a-haystack prompt at the full context is served | ✅ pass ² |
| `decision` | Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts | — not applicable |

- ¹ local run, LEVEL=quick, 2026-11-06; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 26.04; hrx-loom `0.1.0` (hrx-system `d01d1c1`).
- ² [DevLab Dispatch run 1000](https://github.com/loom-fun/loom-skills/actions/runs/1000/job/2000), LEVEL=full, 2026-11-06; device not recorded; hrx-loom `0.1.0` (hrx-system `d01d1c1`, recovered).
- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.

## Files

- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).
- `program/*.loom`, `weights.loom`: the generated Loom programs.
- `tokenizer.json`: the tokenizer.
- `llm_config.json`: the configuration the programs were generated from.
- `info.json`: the measurements, gate evidence and versions this card shows; everything else comes from the files above.
- `REPORT.md`: how the values above were measured: setup, commands and the value of every run.

Versions: card schema `loom-card/2`; loom-skills `af9f2a0`; hrx-loom `0.1.0` (hrx-system `d01d1c1`); card generator `0123456`.

Source files, fetched by `loom-fetch` from their own repositories:

- checkpoint: [ggml-org/Flint-GGUF](https://huggingface.co/ggml-org/Flint-GGUF) @ `468307a`, `Flint-Q4_K_M.gguf`
- drafter: [incoai/Flint-DFlash2-GGUF](https://huggingface.co/incoai/Flint-DFlash2-GGUF) @ `74e6ac0`, `Flint-DFlash2-Q4_K_M.gguf`
