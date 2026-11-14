# flint (arch `flint`) port: worked example

A port of `unsloth/flint-GGUF` (UD-Q5_K_XL, 40B MoE) written by fresh agents from these skills (Nov 2026;
latest run 58 min end to end, full gates and audit PASS, agent loop 6/6). Built on `../lfm2moe-example`
(MoE) + the starter. Make a kit with `../apply_example.sh flint W/kit`.

| File | Adds |
|---|---|
| (starter) `gen_moe.py` | written for this port, now the starter's MoE: 256-expert top-6 router (bf16 router weights), shared experts, expert decode matvecs, grouped WMMA prefill GEMM. **The routing keeps the top-k weights in registers**: writing them to global memory and reading them back after a barrier made prefill nondeterministic (neighbouring tokens' workgroups share cache lines) |
| `starter.patch` | `inspect_gguf.py`: NoPE layers from the GGUF's RoPE pattern, end-of-generation ids, the EOS terminal. `make_kernel_tests.py`: the stop-token restore. (Written for this port, now in the starter: attention sinks, a learned per-head logit added to the softmax denominator, in decode split-K, the listed fallback and flash2, with their kernel tests incl. flash2 with sinks on a span continuing stored KV.) |
| reviewed config | the published package's `llm_config.json` ([loom-fun/ggml-org-Flint-loom](https://huggingface.co/loom-fun/ggml-org-Flint-loom), category A, pinned in `../examples.json`; merged by `apply_example.sh flint KIT CFG`) |
