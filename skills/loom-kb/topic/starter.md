# Topic: starter (what the starter generates and its fixed assumptions)

## What the starter generates

| File | Role |
|---|---|
| `inspect_gguf.py` | GGUF metadata → `cfg.json`: hidden, layers, heads/kv_heads, head_dim, ffn, vocab, **context_length**, rms_eps, rope_theta/type, **rope_dims** (partial RoPE), qk_norm, attn_bias (q/k/v projection biases, Qwen2; added by the QK-norm + RoPE kernel), attn_gate (`true`: separate tensor; `"fused_q"`: packed in attn_q), post_norms, sliding window/layers, NoPE layers, tied embeddings, eos ids, **recurrent_layers / gdn** (hybrids), the chat block (via `chat_probe.py`), and every tensor's format and shape. |
| `chat_probe.py` | Renders the model's Jinja chat template on probe conversations and cuts out what the control program needs: role prefixes/suffix, BOS, default system prompt, generation prompts with thinking on/off, the tools block and its order relative to the system message, the tool-call format (`hermes` JSON, `qwen_xml`, `glm`, `pythonic`) and separators, tool-response framing, reasoning-in-history policy, trimming. A template that ignores `tools` (plain ChatML chat models) gives a chat-only block (`call_format` `none`): a request's tools are ignored as the template does, no call is parsed, the card says tool calling "no" and gates.sh skips the agent gates (not applicable); probes with tool calls or results print SKIP. `verify()` re-renders a dozen conversations exactly as `control.loom` will (`simulate()`) and diffs them against the template. `simulate()` is also what the benchmark scripts use to build golden prompts. |
| `gen_llm.py` | `cfg.json` → `llm.loom`: all kernels plus the `@llm_advance` command program. The command program is a **static per-layer unroll**, so each layer's kind (attention or gated delta-net) and tensor formats come from the GGUF (Q4_K_M mixes Q4_K/Q5_K/Q6_K/Q8_0 per tensor). Decode-shaped linears (< 64 tokens) and the head use int8 activations (`g_quant_q8` + `_q8` matvecs with integer dot4, llama.cpp's Q8_1 numerics; 1-, 4-, 8- and 16-token variants that round alike) for Q8_0/Q4_K/Q5_K/Q6_K/Q5_0 weights (`formats.ICODES`); `ACT_Q8=0` keeps f32 activations. |
| `gen_moe.py` | MoE FFN kernels (router + top-k, expert matvecs, shared-expert gate, combine; bucketing + grouped WMMA GEMMs for prefill), for `cfg["moe"]` models. |
| `gen_gdn.py` | Gated delta-net kernels (conv1d + SiLU, history commit, q/k L2 norms, register recurrence, gated RMSNorm) and the fused q/gate split, for hybrids. Recurrent state lives in the runtime's recurrent slots (see porting-notes). |
| `gen_gemm.py` | WMMA prefill GEMMs per (format, mode). Formats: q4_k, q5_k, q6_k, q8_0, and through `gen_llm.py`'s value helpers (`HELPER`) q4_0, q5_0, iq4_nl, iq4_xs, q3_k, iq3_s, f16. Modes: store, acc, swiglu. |
| `gen_flash2.py` | Prefill flash attention (transposed, K/V staged once per KV head for `FLASH2_TS` 16-token tiles: 2 by default at head_dim <= 128 with <= 2 query heads per KV head, else 1), head_dim 64 or 128 (and 256 with `FLASH2_SPLIT=0`). |
| `gen_flash2s.py` | Prefill flash attention for head_dim 256 and 512 (used automatically): two (four) waves per query head own 128 dims each, so Q stays in registers for the whole key loop; they sum S partials through workgroup memory. |
| `tile_uniform.loom` | Helpers that classify 16-token tiles (and their neighbours) for flash2. |
| `templates/*.loom` | Serving glue for the hrx-loom 0.1.0 text ABI (`text_*_v1` exports). `prepare` stages one program per packing shape. `storage` sets the KV geometry (2 planes per attention layer, F16) and the recurrent slot size. `control` is the chat policy: rendering, tools, reasoning, thinking switch, tokenization, output channels, tool-call parsing, canonical history for reuse. `service` maps the ABI exports. `weights.loom` is the parameter policy. |
| `build.py` | Fills the templates from `cfg.json["chat"]`, sizes context/pool/memory, then writes `model.json`, `tokenizer.json`, the GGUF symlink and `llm_config.json` (read by the benchmark scripts). |
| `pick_gguf.py` | Chooses the GGUF in a repo from the headers only (HTTP range read): quant preference, tensor formats the kit supports, and the memory rule (the file must fit twice: a conservative margin, kept until quant choice accounts for the KV pool at full context). |
| `llama_trace.py` | llama.cpp's intermediate tensors and compute graph for a prompt (`llama-eval-callback`): the spec and the node-by-node reference for a new architecture. |
| `draft/` | Speculative decoding: `gen_draft.py` (verify stage, device acceptance, GDN capture/replay and the generic replay hook, MTP drafter), `gen_dflash.py` (DFlash/DFlash2 drafters), `gen_assist.py` (Gemma-4 assistant), `gen_ngram.py` (prompt lookup), `build_draft.py` (prepare/storage/control glue), `find_drafter.py`, `tune.py`; see its README. |
| `make_kernel_tests.py`, `run_kernel_tests.sh` | `check.case` unit tests with numpy references (gguf dequantize) on the real tensor shapes: embed, matvecs (every format/K in use), QK-norm + RoPE, the prefill GEMMs (every format/mode in use, 64-, 128- and 256-row kernels), the f16 row scale, the attention chain (q/gate split → RoPE + paged KV store → split-K attention → gated combine), the output head argmax, and the gated delta-net chain on a span that continues a stored state. |

Kernel inventory:
- **Decode:**
  - one-lane-per-32-value-group matvecs, 1-token and 4-token variants, with fused SwiGLU;
  - an argmax output head that writes no logits to memory;
  - split-K decode attention with up to 64 splits and early exit.
- **Prefill:**
  - f16 row-scaled activations feeding WMMA GEMMs;
  - transposed flash attention;
  - a split-K fallback for tokens in mixed (multi-sequence) tiles.
- **Shared:** a per-token QK-norm + RoPE + F16 KV store kernel, and fused add + RMSNorm.

**Fixed assumptions:**
- head_dim 64, 128 or 256 (attention); gated delta-net head dims multiples of 32; query-head groups that are not a
  power of two (Qwen3.8: 6) use flash prefill with staging sized for the power of two below (`gen_flash2.py`);
  `NO_FLASH2=1` forces the listed split-K path for every prefill token (diagnostics; quadratic at long prompts);
- K-quant / Q8_0 / Q4_0 / Q5_0 / IQ4_NL / IQ4_XS / Q3_K / IQ3_S / MXFP4 (experts) weights (`starter/formats.py`, `quant_values.loom`; add dequant helpers
  for other formats); f32 norms and conv weights;
- hidden ≤ 8192; context ≤ 262,144 (kernel bounds; raise the `range(..., 262144)` facts for 1M-context models);
- **sampling**: `temperature`, `top_p`, `top_k` and `seed` per request. Temperature 0 (the default
  unless `build.py --temperature/--top-p/--top-k` set other defaults) is exact greedy. Otherwise the
  head draws from the top-k candidates (k ≤ 64; `top_k` 0 = 64) after temperature softmax and top-p,
  with a counter-based RNG keyed by (seed, position): a fixed `seed` reproduces a completion, and a
  request without one draws from its own nonce. Thinking models want sampling (Qwen: temperature
  0.6, top_p 0.95, top_k 20 for thinking; 0.7/0.8/20 without); greedy can end reasoning with EOS or
  loop, which `agent_golden.py` flags. Gates and goldens always send temperature 0.
- text only (image content parts are rejected; text parts are joined).

**Chat policy facts (control.loom):**
- Tools follow the template exactly (verified by `chat_probe.py`). Tool definitions are printed as
  the request sent them; send them serialized like Jinja `tojson` (`json.dumps` defaults) for
  byte parity with the template.
- Hermes-JSON calls are parsed in source without a JSON parser, so a malformed call never fails
  the request: its arguments go back as generated for the harness to handle. Qwen-XML and GLM
  calls go through the host `tools.parse_xml` codec (a malformed XML frame fails the request; a
  frame cut off by `max_tokens` is dropped). Text after the last complete call is dropped.
  Content before a call ends without `content_call_separator`: the template prints it between content and
  calls, so returning it doubled it in the next turn's rendering and broke history reuse (`@content_until`;
  streaming withholds trailing whitespace until it is known not to precede a call).
- Reasoning is rendered for assistant turns after the last user message, as the Qwen templates do.
  A new user turn therefore re-renders earlier turns without reasoning; inside a tool loop the previous
  turn is reused entirely (`prompt_tokens_details.cached_tokens`). loom-serve resumes a retained row only
  when the new prompt extends its history (`complete_text`) exactly: it cannot cut a row back to a common
  prefix, so a turn the template re-renders differently (dropped reasoning, an added newline) is prefilled
  again (the prefix cache only keeps intermediate frontiers of long prefills).
- loom-serve rejects `"reasoning_content": null` in a message: omit the field instead.
- `usage.prompt_tokens` of a turn that reuses history is one lower than `/v1/chat/completions/input_tokens`
  (the previous turn's EOS is already in the row); compare renderings with `input_tokens`.
- `input_tokens` above `chat_probe.simulate()` while the rendered text is right (a debug `@require(0, %prefixed)`
  in chat_begin returns it as the error message) is loom-serve's tokenizer: it differs from HF tokenizers after
  64 pre-tokenizer pieces when a later `Sequence` stage splits them further. build.py rewrites an
  individual-digits `Digits` stage (Spark-X, Mellum) as the equivalent Isolated `\p{N}` Split; check another shape with
  HF vs loom-serve counts on a long text.
