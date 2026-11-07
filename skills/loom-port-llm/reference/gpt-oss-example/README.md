# gpt-oss (arch `gpt-oss`) port: worked example

A port of `unsloth/gpt-oss-20b-GGUF` (Q4_K_M: the experts stay MXFP4, attention Q5_0/Q8_0/Q4_K, embedding Q5_0,
head Q8_0). Make a kit with `../apply_example.sh gpt-oss W/kit` (port.sh does it by itself for arch `gpt-oss`).

The kernels are the starter's, driven by the config `inspect_gguf.py` writes:
- MoE (`gen_moe.py`): 32 experts top-4, router with its own bias, llama.cpp's SOFTMAX_WEIGHT gating (`moe.gating` 3:
  top-k of the biased logits, softmax over the 4 picked), per-expert gate/up/down biases (`moe.expert_bias`) and the
  clamped SwiGLU (`moe.act` "swiglu_oai": x = min(g, 7), y = clamp(u, -7, 7), x sigmoid(1.702 x) (y + 1));
- MXFP4 expert weights (`formats.py`, `gen_llm.MXFP4`: E8M0 scale byte + 16 bytes of E2M1 nibbles, decode matvecs and
  the grouped prefill GEMMs' dequant);
- attention sinks (decode split-K, the listed fallback and flash2), sliding window 128 on the even layers
  (`inspect_gguf.SWA_PERIOD`: llama.cpp hard-codes the period, the GGUF has none), q/k/v biases and the output
  projection's bias (`attn_out_bias`, `g_bias_add`), YaRN RoPE (factor 32 from a 4k original context: `rope_scaling`,
  `gen_llm.yarn_constants`, cos/sin times 1 + 0.1 ln 32), the FFN norm stored as `post_attention_norm`.

| File | Adds |
|---|---|
| `chat/harmony.loom`, `chat/harmony_rodata.py` | OpenAI's harmony channel format: `chat_begin` renders the whole conversation (system block with the reasoning effort, the developer block with the first system message and the tools as a TypeScript namespace, analysis / final / commentary turns, tool results framed with the last call's name), on `starter/templates/control_channels.loom` (shared with the ATEM kit: `output_next`, `parse_tools`): `@ch_header_state` maps a header to a channel (analysis -> reasoning, final and commentary preambles -> content, a `to=functions.NAME` header -> the call) and `@ch_calls_text` turns the call into hermes text for the generic `@hermes_calls` |
| `starter.patch` | `chat_probe.py`: the harmony model (`harmony_simulate`, `raw_from_chat`, `raw_alternatives`, `derive` and `verify` for harmony templates); `inspect_gguf.py`: `<|call|>` as an end-of-generation id |
| (card) | the reviewed config: `loom-fun/unsloth-gpt-oss-20b-loom`'s `llm_config.json` (`examples.json`) |

Chat notes:
- `reasoning_content` plays the template's `thinking` (llama.cpp's gpt-oss handler copies it there and drops a call
  turn's content). The template renders a past answer that carries reasoning in the analysis channel; Loom follows it.
- Reasoning effort: `reasoning_effort` low / medium (default) / high selects the system block's `Reasoning:` line
  (`chat.reasoning_instructions`); thinking cannot be turned off.
- Deliberate deviation: no `Current date:` line (the control program has no clock), as in the glimmer example.
- The template renders a call as `<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>`, the model
  writes `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>`: a tool-loop turn's history is
  re-rendered, not reused (`raw_alternatives` lists both spellings for the agent gate's golden).
- Tool schemas: string (enum, nullable), number/integer, boolean, type lists, arrays of those and objects of those
  (one level) render as the template does; deeper schemas and `oneOf` render as `any` (the template recurses).
