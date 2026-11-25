---
base_model: ggml-org/Flint-GGUF
license: mit
tags:
- loom
- loom-serve
- gguf
- strix-halo
- speculative-decoding
- dflash
---

# Flint (Q4_K_M) for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint [ggml-org/Flint-GGUF](https://huggingface.co/ggml-org/Flint-GGUF) (`Flint-Q4_K_M.gguf`), generated on an AMD Ryzen AI Max+ 395 "Strix Halo" with the [loom-skills](https://github.com/loom-fun/loom-skills) workflow. Learn more on the [Loom ecosystem website](https://loom-fun.github.io/loom-dev/).

## Quick start

```bash
pipx install "hrx-loom[hf]==0.3.0"
loom-fetch huggingface:loom-fun/ggml-org-Flint-loom:chat --output=flint
loom-serve --model=flint '--device=amdgpu://?completion_spin_ns=4294967295' --pool_backing=fixed
```

- `completion_spin_ns=4294967295` enables active waiting for GPU completions (uses CPU while GPU work is in flight); the card's performance measurements use this setting.
- `--pool_backing=fixed` commits the preset's memory pool up front instead of growing elastic backing.

## Send a request

In another terminal:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "flint", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Performance vs llama.cpp

| metric | llama.cpp | Loom | margin vs best llama.cpp | roofline | Loom % of roofline |
|---|---|---|---|---|---|
| decode tok/s, short prompt | 67.3 | 93.0 | +38 % | 177.8 | 52 % |
| decode tok/s, 2k context | 54.7 | 76.3 | +39 % | 173.4 | 44 % |
| decode tok/s, 32k context | 49.5 | 47.8 | -3 % | 141.3 | 34 % |
| prefill tok/s, 1k document | 1081.8 | 813.6 | -25 % | 12248.4 | 7 % |
| prefill tok/s, 2k document | 1079.0 | 1263.8 | +17 % | 11945.5 | 11 % |
| prefill tok/s, 8k document | 1094.1 | 1424.7 | +30 % | 10484.7 | 14 % |
| prefill tok/s, 16k document | 929.3 | 1359.0 | +46 % | 9014.9 | 15 % |
| prefill tok/s, 32k document | 667.6 | 1193.9 | +79 % | 7040.8 | 17 % |
| 4 streams, aggregate decode tok/s | 124.2 | 134.2 | +8 % | 301.8 | 44 % |
| speculative decoding, drafted tokens accepted | — | 1.25 per pass ¹ | — | n/a | n/a |

**Methodology**

- Compared: Loom against llama.cpp ROCm `llamacpp-rocm/b1355`; the same GGUF, the same DevLab runner and job, the same prompts; Loom served with the active wait. Values are the median of 2 sessions (4 streams: median of 3); Loom is timed by the client, llama.cpp by its server.
- Out of the box on both engines: Loom serves the preset shipped in this package; llama.cpp runs its build defaults plus only the server arguments listed below (offload, context and slots). No tuning sweep on either engine.
- Drafting, 1-stream rows (decode, decode_2k, decode_32k, prefill): Loom DFlash depth 3; llama.cpp ROCm no drafting (NOT the same choice on both engines). Each engine is loaded once for all these rows.
- 4 streams: both engines loaded with drafting off, 4 parallel requests, aggregate tokens per second.
- Rows: decode = short-context decode; decode_2k / decode_32k = decode after a ~2k / ~32k-token context; prefill_Nk = prompt processing of an N k-token prompt; streams4 = 4-stream aggregate decode. A margin within +-3 % (the measured same-runner noise) is a tie.
- llama.cpp ROCm server arguments: not recorded.
- Measured on AMD DevLab Dispatch Strix Halo runners (`devlab-perf` jobs); runners differ between sessions.
- Margin vs best llama.cpp: (Loom / max(ROCm, Vulkan) - 1) × 100 %, using the available measured tokens/s for each row on this GPU. With only one valid backend, that backend is used; with neither, the margin is not measured. Raw backend values and their measurement sources are retained below and in the table.
- Measured 2026-11-25; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 26.04; hrx-loom `0.3.0` (hrx-system `e2c41a7`); source: DevLab Dispatch run 3101.
- ¹ Measured 2026-11-25; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 26.04; hrx-loom `0.3.0` (hrx-system `e2c41a7`); source: DevLab Dispatch run 3102.
- Drafted tokens accepted per verification pass during the measuring session's requests: accepted drafted tokens / verification passes, from loom-serve's counters (passes = decoded positions - accepted drafted tokens). hrx-loom 0.3.0 reports no drafter proposal count for this package's service ABI 1 controller (`proposed_tokens` is null), so the share of proposals accepted is not measured; it is never derived from the configured draft depth, which is a ceiling.
- Roofline: the hardware ceiling for this GGUF on AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151) at 216 GB/s and 59 TFLOP/s, from loom-roofline `roofline.py` (loom-skills `0123456`); single-pass ceiling (no acceptance measured on this package) for decode, decode at 2k, decode at 32k; 4 streams run without drafting, so its ceiling is single-pass.
- — not measured; n/a not applicable.

## Features

| feature | status | detail |
|---|---|---|
| API | ✅ | OpenAI-style `/v1/chat/completions` |
| tool calling | ✅ | OpenAI-style `tools` and `tool_calls` |
| reasoning / thinking | ✅ | always on |
| reasoning effort | — | not applicable |
| speculative decoding | ✅ | DFlash, depth 3, on by default; drafter [incoai/Flint-DFlash2-GGUF](https://huggingface.co/incoai/Flint-DFlash2-GGUF) |
| image input | ? | not determined |
| audio input / output | ? | not determined |
| sampling | ✅ | `temperature`, `top_p`, `top_k`, `seed` |
| context size | ✅ | 262,144 tokens per request |
| concurrency | ✅ | 4 concurrent requests |
| max output tokens | ✅ | 4,096 (preset `max_tokens`) |

## Correctness

| gate | description | result |
|---|---|---|
| `kernel_tests` | Unit tests of the generated kernels against reference implementations | ❌ fail (120/121; failed: moe_gemm_12) ¹ |
| `golden` | llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents | ✅ pass ² |
| `compare` | Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie | ✅ pass ² |
| `vision` | Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties | — not applicable |
| `consistency` | Concurrent requests give the same output as the same requests run alone | ✅ pass ² |
| `spec` | Speculative decoding on gives exactly the output of speculative decoding off | ✅ pass ² |
| `agent` | Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp | ✅ pass ² |
| `agent_think_off` | The tool-calling loop with thinking off | — not applicable |
| `sampling` | temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp | ✅ pass ² |
| `logits` | Host logits readback (llm.logits on): at T=0 every returned row's argmax is the device's token, the top-k matches llama.cpp, greedy text is unchanged | ✅ pass (128/128 steps, top-5 within 0.05) ² |
| `decision` | Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts | — not applicable |

- ¹ DevLab Dispatch run 3102, 2026-11-25; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 26.04; hrx-loom `0.3.0` (hrx-system `e2c41a7`).
- ² DevLab Dispatch run 3102, LEVEL=full, 2026-11-25; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 26.04; hrx-loom `0.3.0` (hrx-system `e2c41a7`).
- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.

## Files

- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).
- `program/*.loom`, `weights.loom`: the generated Loom programs.
- `tokenizer.json`: the tokenizer.
- `llm_config.json`: the configuration the programs were generated from.
- `info.json`: the measurements, gate evidence and versions this card shows; everything else comes from the files above.
- `REPORT.md`: how the values above were measured: setup, commands and the value of every run.

Versions: card schema `loom-card/3`; loom-skills `af9f2a0`; hrx-loom `0.3.0` (hrx-system `e2c41a7`); programs generated with hrx-loom `0.1.0`; card generator `0123456`.

Source files, fetched by `loom-fetch` from their own repositories:

- checkpoint: [ggml-org/Flint-GGUF](https://huggingface.co/ggml-org/Flint-GGUF) @ `468307a`, `Flint-Q4_K_M.gguf`
- drafter: [incoai/Flint-DFlash2-GGUF](https://huggingface.co/incoai/Flint-DFlash2-GGUF) @ `74e6ac0`, `Flint-DFlash2-Q4_K_M.gguf`
