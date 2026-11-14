# Family: gpt-oss

GGUF arch values: gpt-oss, gptoss.
Kit example: `skills/loom-port-llm/reference/gpt-oss-example/` (kit docs stay there; model-family knowledge lives here).

## Kernels and config driven by the starter (from the gpt-oss kit)

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

## Chat notes: harmony format, reasoning effort, tool schemas

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
