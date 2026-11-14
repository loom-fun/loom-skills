---
base_model: unsloth/Qwen3-0.6B-GGUF
tags:
- loom
- loom-serve
- gguf
- strix-halo
- radeon-r9700
---

# Qwen3-0.6B (Q4_K_M) for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) (`Qwen3-0.6B-Q4_K_M.gguf`), for the AMD Ryzen AI Max+ 395 "Strix Halo" and AMD Radeon AI PRO R9700 GPUs, generated with the [loom-skills](https://github.com/loom-fun/loom-skills) workflow. Learn more on the [Loom ecosystem website](https://loom-fun.github.io/loom-dev/).

## Quick start

```bash
pip install "hrx-loom[hf]==0.1.0"
```

Fetch and serve on the AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151):

```bash
loom-fetch https://huggingface.co/loom-fun-staging/unsloth-Qwen3-0.6B-loom/resolve/main/model.json --preset=chat --materialize=copy --output=Qwen3-0.6B
loom-serve --model=Qwen3-0.6B/model.json '--device=amdgpu://?completion_spin_ns=4294967295' --memory_bytes=17000000000
```

Fetch and serve on the AMD Radeon AI PRO R9700 (gfx1201, 32 GB):

```bash
loom-fetch https://huggingface.co/loom-fun-staging/unsloth-Qwen3-0.6B-loom/resolve/main/model.json --preset=chat-r9700 --materialize=copy --output=Qwen3-0.6B
loom-serve --model=Qwen3-0.6B/model.json '--device=amdgpu://?completion_spin_ns=4294967295' --pool_backing=fixed
```

The server listens on `127.0.0.1:8080` and answers OpenAI-style `/v1/chat/completions` requests.

On the AMD Radeon AI PRO R9700, the package's preset sizes the memory pool (`--pool_backing=fixed` commits it up front): this package serves up to 20,480 tokens per request there (its native context is 40,960); the package files are the same on every GPU.

On the AMD Radeon AI PRO R9700, the Quick start fetches the `chat-r9700` preset of `model.json`: the default preset with `prefill_capacity` 4,096 for its 32 GB. The programs, weights and tokenizer are the same.

## Performance vs llama.cpp

### Strix Halo (gfx1151)

AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151): the results for this package are not yet re-measured.

### Radeon AI PRO R9700 (gfx1201, 32 GB)

| metric | llama.cpp | Loom | margin | roofline | Loom % of roofline | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| decode tok/s, short prompt | 781.9 | 1023.4 | +31 % | 1535.5 | 67 % | 744.4 |
| decode tok/s, 2k context | 638.9 | 841.0 | +32 % | 967.5 | 87 % | 609.6 |
| decode tok/s, 32k context | doesn't fit | doesn't fit | doesn't fit | doesn't fit | doesn't fit | doesn't fit |
| prefill tok/s, 1k document | 17154.6 | 21764.2 | +27 % | 77348.7 | 28 % | 9164.3 |
| prefill tok/s, 2k document | 18192.9 | 24133.0 | +33 % | 69358.1 | 35 % | 9447.7 |
| prefill tok/s, 8k document | 12104.5 | 18302.3 | +51 % | 42818.0 | 43 % | 6150.7 |
| prefill tok/s, 16k document | 8161.1 | 14047.5 | +72 % | 28352.5 | 50 % | 3931.4 |
| prefill tok/s, 32k document | doesn't fit | doesn't fit | doesn't fit | doesn't fit | doesn't fit | doesn't fit |
| 4 streams, aggregate decode tok/s | 1710.4 | 2350.8 | +37 % | 3603.6 | 65 % | 1204.7 |
| speculative decoding, drafted tokens accepted | n/a | n/a | n/a | n/a | n/a | n/a |

**Methodology**

- Compared: Loom against llama.cpp ROCm `llamacpp-rocm/b1374`, llama.cpp Vulkan `llamacpp-vulkan/b12211`; the same GGUF, the same DevLab runner and job, the same prompts; Loom served with the active wait. Values are the mean of 2 rounds (4 streams: median of 3); Loom is timed by the client, llama.cpp by its server.
- Drafting, 1-stream rows (decode, decode_2k, decode_32k, prefill): no drafting on either engine (no MTP head in the GGUF and no drafter published for this exact model). Each engine is loaded once for all these rows.
- 4 streams: both engines loaded with drafting off, 4 parallel requests, aggregate tokens per second.
- Rows: decode = short-context decode; decode_2k / decode_32k = decode after a ~2k / ~32k-token context; prefill_Nk = prompt processing of an N k-token prompt; streams4 = 4-stream aggregate decode. A margin within +-3 % (the measured same-runner noise) is a tie.
- llama.cpp ROCm server arguments: `-ngl 99 -cram 0 -c 65536 -kvu -np 4`.
- llama.cpp Vulkan server arguments: `-ngl 99 -cram 0 -c 65536 -kvu -np 4`.
- Measured on AMD DevLab Dispatch R9700 runners (`devlab-perf` jobs); runners differ between sessions.
- Measured 2026-11-14; device TEST FIXTURE: made-up numbers, not a measurement, AMD Radeon AI PRO R9700S (gfx1201), Ubuntu 24.04; hrx-loom `0.1.0` (hrx-system `d01d1c1`); source: [DevLab Dispatch run 9000001](https://github.com/loom-fun/loom-skills/actions/runs/9000001).
- llama.cpp Vulkan: measured 2026-11-14; source: [DevLab Dispatch run 9000001](https://github.com/loom-fun/loom-skills/actions/runs/9000001). Archive sha256 `aabbccd`.
- Roofline: the hardware ceiling for this GGUF on AMD Radeon AI PRO R9700S (gfx1201, 32 GB, 2.35 GHz) at 600 GB/s and 77 TFLOP/s, from loom-roofline `roofline.py` (loom-skills `0123456`).
- doesn't fit (decode_32k, prefill_32k): the document is longer than the 20,480-token context this GPU serves; not expected to be measured, not a failure.
- — not measured; n/a not applicable.

## Features

| feature | status | detail |
|---|---|---|
| API | ✅ | OpenAI-style `/v1/chat/completions` |
| tool calling | ✅ | OpenAI-style `tools` and `tool_calls` |
| reasoning / thinking | ✅ | on by default; `chat_template_kwargs: {"enable_thinking": false}` turns it off |
| reasoning effort | — | not applicable |
| speculative decoding | — | not applicable |
| image input | ? | not determined |
| audio input / output | ? | not determined |
| sampling | ✅ | `temperature`, `top_p`, `top_k`, `seed` |
| context size | ✅ | 40,960 tokens per request; Radeon AI PRO R9700 up to 20,480 (its memory) |
| concurrency | ✅ | 4 concurrent requests |
| max output tokens | ✅ | 4,096 (preset `max_tokens`) |

## Correctness

| gate | description | Strix Halo | Radeon AI PRO R9700 |
|---|---|---|---|
| `kernel_tests` | Unit tests of the generated kernels against reference implementations | — not yet re-measured | ❌ no evidence |
| `golden` | llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents | — not yet re-measured | ✅ pass ¹ |
| `compare` | Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie | — not yet re-measured | ✅ pass ¹ |
| `vision` | Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties | — not yet re-measured | — not applicable |
| `consistency` | Concurrent requests give the same output as the same requests run alone | — not yet re-measured | ✅ pass ¹ |
| `spec` | Speculative decoding on gives exactly the output of speculative decoding off | — not yet re-measured | — not applicable |
| `agent` | Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp | — not yet re-measured | ✅ pass ¹ |
| `agent_think_off` | The tool-calling loop with thinking off | — not yet re-measured | ✅ pass ¹ |
| `sampling` | temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp | — not yet re-measured | ✅ pass ¹ |
| `needle` | Needle-in-a-haystack prompt at up to half the context, compared with llama.cpp; over-capacity prompt refused | — not yet re-measured | ✅ pass ¹ |
| `needle_full` | Needle-in-a-haystack prompt at the full context is served | — not yet re-measured | doesn't fit (needs more context than the 20,480 tokens this GPU serves) |
| `decision` | Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts | — not yet re-measured | — not applicable |

- ¹ [DevLab Dispatch run 9000001](https://github.com/loom-fun/loom-skills/actions/runs/9000001), LEVEL=quick, 2026-11-14; device TEST FIXTURE: made-up numbers, not a measurement, AMD Radeon AI PRO R9700S (gfx1201), Ubuntu 24.04; hrx-loom `0.1.0` (hrx-system `d01d1c1`).
- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.

## Files

- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).
- `program/*.loom`, `weights.loom`: the generated Loom programs.
- `tokenizer.json`: the tokenizer.
- `llm_config.json`: the configuration the programs were generated from.
- `info.json`: the measurements, gate evidence and versions this card shows; everything else comes from the files above.
- `REPORT.md`: how the values above were measured: setup, commands and the value of every run.

Versions: card schema `loom-card/3`; loom-skills `c6506b9`; hrx-loom `0.1.0` (hrx-system `d01d1c1`); card generator `0123456`.

Source files, fetched by `loom-fetch` from their own repositories:

- checkpoint: [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) @ `50968a4`, `Qwen3-0.6B-Q4_K_M.gguf`
