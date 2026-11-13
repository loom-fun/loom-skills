---
base_model: unsloth/Qwen3-0.6B-GGUF
tags:
- loom
- loom-serve
- gguf
- strix-halo
---

# Qwen3-0.6B (Q4_K_M) for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) (`Qwen3-0.6B-Q4_K_M.gguf`), generated on an AMD Ryzen AI Max+ 395 "Strix Halo" with the [loom-skills](https://github.com/loom-fun/loom-skills) workflow. Learn more on the [Loom ecosystem website](https://loom-fun.github.io/loom-dev/).

## Quick start

```bash
pip install "hrx-loom[hf]==0.1.0"
loom-fetch https://huggingface.co/loom-fun/unsloth-Qwen3-0.6B-loom/resolve/main/model.json --materialize=copy --output=Qwen3-0.6B
loom-serve --model=Qwen3-0.6B/model.json '--device=amdgpu://?completion_spin_ns=4294967295' --memory_bytes=17000000000
```

The server listens on `127.0.0.1:8080` and answers OpenAI-style `/v1/chat/completions` requests.

AMD Radeon AI PRO R9700 (gfx1201, 32 GB): this package does not fit its 32 GB (it needs about 82 GB).

## Performance vs llama.cpp

| metric | llama.cpp | Loom | margin | roofline | Loom % of roofline | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| decode tok/s, short prompt | 266.4 | 330.3 | +24 % | 552.8 | 60 % | 313.2 |
| decode tok/s, 2k context | 201.8 | 221.3 | +10 % | 348.3 | 64 % | 229.1 |
| decode tok/s, 32k context | 50.1 | 50.3 | +0 % | 53.2 | 95 % | 47.2 |
| prefill tok/s, 1k document | 11911.2 | 15193.5 | +28 % | 59267.2 | 26 % | 10027.6 |
| prefill tok/s, 2k document | 11778.6 | 15420.9 | +31 % | 53144.5 | 29 % | 9227.8 |
| prefill tok/s, 8k document | 8287.9 | 9466.5 | +14 % | 32808.6 | 29 % | 4885.3 |
| prefill tok/s, 16k document | 4979.2 | 6763.8 | +36 % | 21724.6 | 31 % | 2855.5 |
| prefill tok/s, 32k document | 2716.6 | 4024.9 | +48 % | 12964.7 | 31 % | 1625.5 |
| 4 streams, aggregate decode tok/s | 734.4 | 963.3 | +31 % | 1297.3 | 74 % | 838.1 |
| speculative decoding, drafted tokens accepted | n/a | n/a | n/a | n/a | n/a | n/a |

**Methodology**

- Compared: Loom against llama.cpp ROCm `llamacpp-rocm/b1374`, llama.cpp Vulkan `llamacpp-vulkan/b12211`; the same GGUF, the same DevLab runner and job, the same prompts; Loom served with the active wait. Values are the mean of 3 rounds (4 streams: median of 3); Loom is timed by the client, llama.cpp by its server.
- Drafting, 1-stream rows (decode, decode_2k, decode_32k, prefill): no drafting on either engine (no MTP head in the GGUF and no drafter published for this exact model). Each engine is loaded once for all these rows.
- 4 streams: both engines loaded with drafting off, 4 parallel requests, aggregate tokens per second.
- Rows: decode = short-context decode; decode_2k / decode_32k = decode after a ~2k / ~32k-token context; prefill_Nk = prompt processing of an N k-token prompt; streams4 = 4-stream aggregate decode. A margin within +-3 % (the measured same-runner noise) is a tie.
- llama.cpp ROCm server arguments: `1-stream rows: -ngl 99 -cram 0 -c 65536 -np 1; 4 streams: -ngl 99 -cram 0 -c 65536 -kvu -np 4`.
- llama.cpp Vulkan server arguments: `1-stream rows: -ngl 99 -cram 0 -c 36864 -np 1; 4 streams: -ngl 99 -cram 0 -c 147456 -np 4`.
- Measured on AMD DevLab Dispatch Strix Halo runners (`devlab-perf` jobs); runners differ between sessions.
- Measured 2026-11-13; device AMD Strix Halo (DevLab Dispatch runner X2-LA03-S03-D03), Linux; hrx-loom `0.1.0` (hrx-system `d01d1c1`, recovered); source: [DevLab Dispatch run 43392598780](https://github.com/loom-fun/loom-skills/actions/runs/43392598780/job/131775918531).
- llama.cpp Vulkan: measured 2026-11-13; source: [DevLab Dispatch run 43392598780](https://github.com/loom-fun/loom-skills/actions/runs/43392598780/job/131775918531). Archive sha256 `9759acd`.
- Roofline: the hardware ceiling for this GGUF on AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151) at 216 GB/s and 59 TFLOP/s, from loom-roofline `roofline.py` (loom-skills `acd9ac7`).
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
| context size | ✅ | 40,960 tokens per request |
| concurrency | ✅ | 4 concurrent requests |
| max output tokens | ✅ | 4,096 (preset `max_tokens`) |

## Correctness

| gate | description | result |
|---|---|---|
| `kernel_tests` | Unit tests of the generated kernels against reference implementations | ✅ pass (59/59) ¹ |
| `golden` | llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents | ✅ pass ² |
| `compare` | Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie | ✅ pass ² |
| `vision` | Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties | — not applicable |
| `consistency` | Concurrent requests give the same output as the same requests run alone | ✅ pass ² |
| `spec` | Speculative decoding on gives exactly the output of speculative decoding off | — not applicable |
| `agent` | Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp | ✅ pass ² |
| `agent_think_off` | The tool-calling loop with thinking off | ✅ pass ² |
| `sampling` | temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp | ✅ pass ² |
| `needle` | Needle-in-a-haystack prompt at up to half the context, compared with llama.cpp; over-capacity prompt refused | ✅ pass ² |
| `needle_full` | Needle-in-a-haystack prompt at the full context is served | ✅ pass ³ |
| `decision` | Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts | — not applicable |

- ¹ [DevLab Dispatch run 43376610754](https://github.com/loom-fun/loom-skills/actions/runs/43376610754), 2026-11-13; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 24.04; hrx-loom `0.1.0` (hrx-system `d01d1c1`).
- ² [DevLab Dispatch run 43376610754](https://github.com/loom-fun/loom-skills/actions/runs/43376610754), LEVEL=full, 2026-11-13; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 24.04; hrx-loom `0.1.0` (hrx-system `d01d1c1`).
- ³ [DevLab Dispatch run 42445355675](https://github.com/loom-fun/loom-skills/actions/runs/42445355675/job/128623039814), LEVEL=full, 2026-11-08; device not recorded; hrx-loom unknown.
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
