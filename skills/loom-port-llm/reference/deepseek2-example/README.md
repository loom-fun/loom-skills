# deepseek2 (MLA) port: worked example

A port of `jdopensource/JoyAI-LLM-Flash-GGUF` (IQ4_XS, 48B MoE, 3B active, 128K context) written by agents from
these skills (Nov 2026; Strix Halo). Make a kit with `../apply_example.sh deepseek2 W/kit W/cfg.json`.

| File | Adds |
|---|---|
| `gen_mla.py` | absorbed multi-head latent attention: q_a -> rms(q_a_norm) -> q_b, kv_a_mqa split into the 512-wide latent (rms(kv_a_norm)) and the 64-wide RoPE key, NORM RoPE, the cache store (ONE 576-value f16 row per token and layer, V = its first 512 values), per-head `k_b` absorb (q_abs = k_b[h] q_nope) and `v_b` (o2 = v_b[h] o) as matvecs for decode and WMMA for prefill |
| `ktest_mla.py` | kernel tests: every MLA kernel alone, the decode / listed / flash chains (dirty KV pool, padding tokens, > 2048 keys) and layer 0 with the program's launch order on real embeddings, vs numpy |
| `starter.patch` | `inspect_gguf.py`: the `mla` block, deepseek2 rope type NORM, MoE defaults; `gen_llm.py`: MLA wiring, decode split-K at D=576 (keys) / 512 (values) with 4 keys per block; `gen_flash2s.py`: an MLA mode (576-dim keys, 512-dim values staged from one row); a fix of the NO_FLASH2 fallback list (runs of 9+ tokens got no attention); IQ4_XS MoE experts (`gen_moe.py` FMT; int8 path with IQ4_Q8); chat: `call_lead` (text before a call marker in a call-only turn) in `chat_probe.py`, `control.loom`, `build.py` |
| `cfg.json` | JoyAI's reviewed config (chat block: `<|User|>`/`<|Assistant|>`, `<|end_of_thought|>` generation prompt, Qwen-XML tool calls with a leading newline, `<|end▁of▁sentence|>` after every assistant turn) |

Spec facts (checked against `llama_trace.py` + a numpy forward): RoPE NORM (interleaved), scale 1/sqrt(192),
kv norm on the latent before caching, layer 0's q_nope is exactly zero in this GGUF. Strix Halo only (no gfx12 flash).
