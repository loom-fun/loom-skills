# Topology: moe

Mixture-of-experts FFN: qwen35moe, qwen3moe, mellum, gpt-oss, laguna, flint, lfm2moe, qwen4exp (gemma4 26B-A4B is a dense-topology family with the moe facet). The MoE kernels, chunk size, and what did not help are in facet moe; the family nodes hold routing choices llama.cpp hard-codes per architecture (`gen_moe.MOE_DEFAULTS`), expert formats and chat formats.
