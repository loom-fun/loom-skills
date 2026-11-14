# Topic: cfg-review (the cfg.json review checklist, port.sh step 2)

## cfg.json review checklist (step 2)

Compare with llama.cpp's graph builder for the arch (`src/models/<arch>.cpp` in
recent llama.cpp; older trees have it in `src/llama-model.cpp`). Check these fields:
- `rope_type`:
  - `neox` pairs (d, d+64). Used by qwen2/qwen3/gemma/phi3/…; llama.cpp's
    `LLAMA_ROPE_TYPE_NEOX`.
  - `norm` pairs (2d, 2d+1). Used by llama/mistral; `LLAMA_ROPE_TYPE_NORM`.
  - A wrong value still gives plausible-looking text for a few tokens, then degrades.
- `rope_theta`; Llama-3 frequency factors (`rope_freqs.weight` -> `rope_freq_factors`) and YaRN scaling (`rope_scaling`,
  when the GGUF says `rope.scaling.type` yarn, as llama.cpp does: gpt-oss, laguna) are applied; linear scaling is **not**
  implemented, so add it in `qk_rope()` if the model uses it.
- `qk_norm` (per-head RMSNorm on q/k before RoPE), `post_norms` (Gemma-style
  post-attention/FFN norms), `attn_gate` (sigmoid output gate), `embed_scale`
  (Gemma multiplies embeddings by √hidden), `embed_norm`.
- Chat templates whose first system block always opens with the default prompt and append the request's
  system message (`system_lead`, `system_lead_separator`; reasoning instructions appended to it:
  `reasoning_after_lead`), that render a turn without reasoning as a bare think close (`history_think: "close"`)
  or print tools as compact JSON (`tools_compact`) are derived by `chat_probe.py`; so are templates that close each
  role with its own tag (`system_suffix`, `assistant_suffix` beside the user's `suffix`; the terminal is the
  assistant's) and that show reasoning in history only with thinking on, a bare think close in every turn with it off
  (`history_think_off: "close"`). build.py patches control.loom only for blocks that use them.
  `nope_layers` are layers without RoPE.
- `extra_eos_ids`: every id that should stop generation. For ChatML models that is
  `<|im_end|>` and `<|endoftext|>`.
- `chat`: derived and verified automatically (all `chat OK` lines). It matches the
  template byte for byte with **one deliberate exception**: when the model emits
  `<think>` itself (Qwen3 hybrid), `<think>\n` is appended to the thinking generation
  prompt so output starts in the reasoning channel. llama.cpp's *chat* endpoint then
  sees fewer prompt tokens than Loom; the goldens render with Loom's prompt, so they
  are exact. Check the printed summary against the model card:
  - `thinking`: `toggle` (Qwen3, Qwen3.5: `enable_thinking`), `always`, or `never`
    (`*-Instruct-2507`); `thinking_default` (Qwen3: on; Qwen3.5 small models: **off**);
  - `call_format`: `hermes` (Qwen2.5/Qwen3), `qwen_xml` (Qwen3.5/Qwen3-Coder), `glm`, `pythonic`
    (LFM2: `<|tool_call_start|>[f(a='x', n=3)]<|tool_call_end|>`, parsed in control.loom);
  - `default_system` (Qwen2.5 inserts "You are Qwen, created by Alibaba Cloud…"); `default_system_tools` when the
    template swaps in another default when tools are present (Nanbeige: a "tool-calling expert" prompt; the text
    beside the tools block, separator included; absent otherwise);
  - `history_think_before: "empty"` (MiniCPM-V): turns before the last user query keep an empty think block.
  - A template that renders reasoning in history reads as thinking `always`, also for models that never reason
    (Qwen3-*-Instruct-2507); no template signal tells them from Flint or Jan-v1. port.sh's smoke test decides: two
    finished replies that never leave the reasoning channel switch the block to `never` (`chat_probe.py GGUF --never
    cfg.json`, verified against the template), rebuild and smoke again.
  - `bos` only when the template prints `bos_token` (a GGUF whose bos id is the role tag `<|im_start|>` is not).
  - `reasoning_instructions` (Qwen3.8: "Reasoning effort is set to xhigh. …" per `reasoning_effort`, only
    with thinking on, `reasoning_separator` before the tools text / system message): build.py advertises
    those efforts (plus none = thinking off) and `chat_begin` renders the request effort's text.

  **History reuse depends on the canonical rendering.** `complete_text` must store the turn as the
  template re-renders it, not as the model generated it. Two trials failed `reuse_ok` here: a model
  that omits `<|eom|>` before a tool call (the template renders it), and one that writes newlines
  after a call that the template drops. If `reuse_ok` fails after a tool call, diff Loom's
  `input_tokens` rendering of the next turn against `chat_probe.simulate()` and canonicalize those bytes.
  GLM-style templates: the model ends a call turn with `<|observation|>` and an answer with `<|user|>`;
  those stop tokens stay in the KV row, so set `chat.stop_after_calls` / `chat.stop_after_answer`:
  `complete_text` publishes them, and `g_embed` maps the service terminal loom-serve feeds back at the
  next turn to the real stop token (`stop_restore()`; flint reused turns diverged at gaps up to 2.7 without it).
  Templates the probe can't model (e.g. role tags with no closing suffix, tools
  inside the first user message, channel formats): fill the block by hand, keep
  `simulate()` in step, and run `chat_probe.py MODEL.gguf --verify cfg.json` (your kit's copy) until every line is OK. If the
  control program must parse structural tokens out of the generated text (e.g.
  channel tags), list them in `chat.nonspecial_tokens`: loom-serve strips special
  tokens from decoded text.
