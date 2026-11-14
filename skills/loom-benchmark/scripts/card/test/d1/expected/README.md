---
base_model: LiquidAI/d1-2B-GGUF
license: other
tags:
- loom
- loom-serve
- gguf
- strix-halo
---

# d1-2B (Q4_K_M) for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint [LiquidAI/d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) (`d1-2B-Q4_K_M.gguf`), generated on an AMD Ryzen AI Max+ 395 "Strix Halo" with the [loom-skills](https://github.com/loom-fun/loom-skills) workflow. Learn more on the [Loom ecosystem website](https://loom-fun.github.io/loom-dev/).

## Quick start

```bash
pip install "hrx-loom[hf]"
loom-fetch https://huggingface.co/loom-fun/LiquidAI-d1-2B-loom/resolve/main/model.json --materialize=copy --output=d1-2B-GGUF
loom-serve --model=d1-2B-GGUF/model.json '--device=amdgpu://?completion_spin_ns=4294967295'
```

The server listens on `127.0.0.1:8080` and answers SystemOne decision requests at `/v1/systemone`.

## Performance vs llama.cpp

| metric | llama.cpp | Loom | margin | roofline | Loom % of roofline |
|---|---|---|---|---|---|
| decode tok/s, short prompt | n/a | n/a | n/a | n/a | n/a |
| decode tok/s, 2k context | n/a | n/a | n/a | n/a | n/a |
| decode tok/s, 32k context | n/a | n/a | n/a | n/a | n/a |
| prefill tok/s, 1k document | 5926.4 | 7041.3 | +19 % | — | — |
| prefill tok/s, 2k document | 5684.0 | 7138.1 | +26 % | — | — |
| prefill tok/s, 8k document | 4204.3 | 5488.4 | +31 % | — | — |
| prefill tok/s, 16k document | 3128.7 | 4637.4 | +48 % | — | — |
| prefill tok/s, 32k document | — | — | — | — | — |
| 4 streams, aggregate decode tok/s | n/a | n/a | n/a | n/a | n/a |
| speculative decoding, drafted tokens accepted | n/a | n/a | n/a | n/a | n/a |

**Methodology**

- Compared: Loom against llama.cpp ROCm `llamacpp-rocm/b1357`; the same GGUF, the same DevLab runner and job, the same prompts; Loom served with the active wait. Values are the mean of not recorded rounds (4 streams: median of not recorded); Loom is timed by the client, llama.cpp by its server.
- Out of the box on both engines: Loom serves the preset shipped in this package; llama.cpp runs its build defaults plus only the server arguments listed below (offload, context and slots). No tuning sweep on either engine.
- Drafting, 1-stream rows (decode, decode_2k, decode_32k, prefill): no drafting on either engine (no MTP head in the GGUF and no drafter published for this exact model). Each engine is loaded once for all these rows.
- 4 streams: both engines loaded with drafting off, 4 parallel requests, aggregate tokens per second.
- Rows: decode = short-context decode; decode_2k / decode_32k = decode after a ~2k / ~32k-token context; prefill_Nk = prompt processing of an N k-token prompt; streams4 = 4-stream aggregate decode. A margin within +-3 % (the measured same-runner noise) is a tie.
- llama.cpp ROCm server arguments: `-ngl 99 -np 4 -c 48000`.
- Measured on AMD DevLab Dispatch Strix Halo runners (`devlab-perf` jobs); runners differ between sessions.
- Measured 2026-10-29; device not recorded; hrx-loom `0.1.0` (hrx-system `dd5ad00`, recovered); source: [loom-factory run 40842422037](https://github.com/loom-fun/loom-skills/actions/runs/40842422037).
- Roofline: not computed.
- — not measured; n/a not applicable.

## Features

| feature | status | detail |
|---|---|---|
| API | ✅ | `/v1/systemone`: choice, score and noul decisions |
| tool calling | — | not applicable |
| reasoning / thinking | — | not applicable |
| reasoning effort | — | not applicable |
| speculative decoding | — | not applicable |
| image input | — | not applicable |
| audio input / output | ? | not determined |
| sampling | — | not applicable |
| context size | ? | not determined |
| concurrency | ✅ | 64 pending requests |
| max output tokens | — | not applicable |

## Correctness

| gate | description | result |
|---|---|---|
| `kernel_tests` | Unit tests of the generated kernels against reference implementations | ❌ no evidence |
| `golden` | llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents | — not applicable |
| `compare` | Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie | — not applicable |
| `vision` | Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties | — not applicable |
| `consistency` | Concurrent requests give the same output as the same requests run alone | ✅ pass ¹ |
| `spec` | Speculative decoding on gives exactly the output of speculative decoding off | — not applicable |
| `agent` | Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp | — not applicable |
| `agent_think_off` | The tool-calling loop with thinking off | — not applicable |
| `sampling` | temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp | — not applicable |
| `needle` | Needle-in-a-haystack prompt at up to half the context, compared with llama.cpp; over-capacity prompt refused | — not applicable |
| `needle_full` | Needle-in-a-haystack prompt at the full context is served | — not applicable |
| `decision` | Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts | ✅ pass ¹ |

- ¹ [loom-factory run 40842422037](https://github.com/loom-fun/loom-skills/actions/runs/40842422037), 2026-10-29; device not recorded; hrx-loom `0.1.0` (hrx-system `dd5ad00`, recovered).
- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.

## Files

- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).
- `program/*.loom`, `weights.loom`: the generated Loom programs.
- `tokenizer.json`: the tokenizer.
- `llm_config.json`: the configuration the programs were generated from.
- `info.json`: the measurements, gate evidence and versions this card shows; everything else comes from the files above.
- `REPORT.md`: the report written when this package was made.

Versions: card schema `loom-card/3`; loom-skills `b0f5897` (recovered); hrx-loom `0.1.0` (hrx-system `dd5ad00`, recovered); card generator `0123456`.

Source files, fetched by `loom-fetch` from their own repositories:

- checkpoint: [LiquidAI/d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) @ `e968f65`, `d1-2B-Q4_K_M.gguf`
