# Facet: moe

Mixture-of-experts FFN: expert matvecs and grouped GEMMs, the router, shared experts, chunk size. Applies to every model whose config has `moe` (also gemma4 26B-A4B, the qwen35moe hybrids).

## Starter MoE support (`gen_moe.py`, `cfg["moe"]`)

- **MoE** is in the starter (`gen_moe.py`, wired by `gen_llm.py` from `cfg["moe"]`): router top-k (softmax,
  sigmoid + selection bias, or llama.cpp's SOFTMAX_WEIGHT with the router's own bias; weights kept in registers, times
  `expert_weights_scale`),
  per-expert output biases and the gpt-oss clamped SwiGLU (`moe.expert_bias`, `moe.act`), per-assignment expert matvecs for decode, tokens bucketed
  by expert and grouped WMMA GEMMs for prefill, shared experts (with the Qwen sigmoid gate), combine; MoE layers
  after `leading_dense` dense ones (an MoE MTP block generates too; not yet validated on the GPU). `inspect_gguf.py` fills `cfg["moe"]` from the GGUF;
  routing choices llama.cpp hard-codes per architecture instead of writing them (`expert_gating_func`,
  `expert_weights_norm`) come from `MOE_DEFAULTS`: add a row for a new MoE architecture from its
  `src/models/<arch>.cpp` (`build_moe_ffn` arguments). Expert formats: Q4_K/Q5_K/Q6_K/Q8_0/MXFP4, gate and up may differ
  (`GATE+UP` kernels, e.g. bartowski's MiniCPM-V-4.7 Q4_K_M: Q4_K gate, Q6_K up) (`gen_moe.FMT`; an MTP
  block in another format is left out). Kernel tests cover the decode and prefill chains on the real tensors.
  Ported this way: Qwen3.6-35B-A3B (`qwen35moe`, hybrid + MoE, 256 experts top-8), gpt-oss-20b (MXFP4 experts,
  `reference/gpt-oss-example` for its chat format).
