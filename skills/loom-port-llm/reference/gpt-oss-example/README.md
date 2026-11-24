# gpt-oss (arch `gpt-oss`) port: worked example

A port of `unsloth/gpt-oss-20b-GGUF` (Q4_K_M: the experts stay MXFP4, attention Q5_0/Q8_0/Q4_K, embedding Q5_0,
head Q8_0). Make a kit with `../apply_example.sh gpt-oss W/kit` (port.sh does it by itself for arch `gpt-oss`).

| File | Adds |
|---|---|
| `chat/harmony.loom`, `chat/harmony_rodata.json` | OpenAI's harmony channel format: `chat_begin` renders the whole conversation (system block with the reasoning effort, the developer block with the first system message and the tools as a TypeScript namespace, analysis / final / commentary turns, tool results framed with the last call's name), on `starter/templates/control_channels.loom` (shared with the ATEM kit: `output_next`, `parse_tools`): `@ch_header_state` maps a header to a channel (analysis -> reasoning, final and commentary preambles -> content, a `to=functions.NAME` header -> the call) and `@ch_calls_text` turns the call into hermes text for the generic `@hermes_calls` |
| `starter.patch` | `chat_probe.py`: the harmony model (`harmony_simulate`, `raw_from_chat`, `raw_alternatives`, `derive` and `verify` for harmony templates); `inspect_gguf.py`: `<|call|>` as an end-of-generation id |
| (card) | the reviewed config: `loom-fun/unsloth-gpt-oss-20b-loom`'s `llm_config.json` (`examples.json`) |

The kernel list and chat notes of this family: `kb.py route --arch gpt-oss --topic porting`.
