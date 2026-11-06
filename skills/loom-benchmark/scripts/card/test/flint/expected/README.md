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

| metric | llama.cpp | Loom | margin |
|---|---|---|---|
| decode tok/s, short prompt | 67.3 | 96.9 | +44 % |
| decode tok/s, 2k context | 54.7 | 76.3 | +39 % |
| decode tok/s, 32k context | 49.5 | 47.8 | -3 % |
| prefill tok/s, 1k document | 1081.8 | 900.0 ¹ | -17 % |
| prefill tok/s, 2k document | 1079.0 | 1316.4 | +22 % |
| prefill tok/s, 8k document | 1094.1 | 1424.7 | +30 % |
| prefill tok/s, 16k document | 929.3 | 1359.0 | +46 % |
| prefill tok/s, 32k document | 667.6 | 1193.9 | +79 % |
| 4 streams, aggregate decode tok/s | 124.2 | 134.2 | +8 % |

- llama.cpp: ROCm build b1355 (lemonade-sdk/llamacpp-rocm), speculative decoding on (draft-dflash, n-max 3).
- Loom: speculative decoding on (DFlash, depth 3).
- Same GGUF on both engines, tokens per second. Loom: client-side timing; llama.cpp: server-side timings. Runs per value: decode and prefill 2 (mean), 4 streams 3 (median).
- Measured 2026-11-06; source: local run.
- ¹ Loom value from [DevLab Dispatch run 1000](https://github.com/loom-fun/loom-skills/actions/runs/1000/job/2000), 2026-11-06.
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

- ¹ local run, LEVEL=quick, 2026-11-06.
- ² [DevLab Dispatch run 1000](https://github.com/loom-fun/loom-skills/actions/runs/1000/job/2000), LEVEL=full, 2026-11-06.
- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.

## Files

- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).
- `program/*.loom`, `weights.loom`: the generated Loom programs.
- `tokenizer.json`: the tokenizer.
- `llm_config.json`: the configuration the programs were generated from.
- `info.json`: the data this card is rendered from (schema `loom-card/1`).

Source files, fetched by `loom-fetch` from their own repositories:

- checkpoint: [ggml-org/Flint-GGUF](https://huggingface.co/ggml-org/Flint-GGUF) @ `468307a`, `Flint-Q4_K_M.gguf`
- drafter: [incoai/Flint-DFlash2-GGUF](https://huggingface.co/incoai/Flint-DFlash2-GGUF) @ `74e6ac0`, `Flint-DFlash2-Q4_K_M.gguf`
