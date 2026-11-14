# Family: lfm3

GGUF arch values: lfm3.
Kit example: `skills/loom-port-llm/reference/lfm2moe-example/` (kit docs stay there; model-family knowledge lives here).

## LFM3 on the lfm2moe kit (from the lfm2moe kit README)

**LFM3 (arch `lfm3`, LFM3-1.7B / LFM3-Omni-2B text) uses this kit too** (`examples.json` `archs`: port.sh applies it
to an `lfm3` GGUF; the LFM2.5 reviewed config is never merged into one). Over lfm2 (llama.cpp `src/models/lfm3.cpp`)
the kit adds only a kernel-7 short conv (`shortconv.l_cache`; `gen_sconv.py` is generic in K) and sandwich norms at the
model's rms eps (`post_norm_eps`). The rest is the starter's: the per-head output gate (`attn_gate: "per_head"`), the
sliding layers' rope base (`rope_theta_swa`, `g_qk_rope_store_swa`) and the global layers' `rope_freqs` factors
(`rope_freq_factors`: 1 / 1e30, only the first 32 of 64 rotary pairs rotate), and the empty `<think></think>` before
the last user message (`history_think_before`).
