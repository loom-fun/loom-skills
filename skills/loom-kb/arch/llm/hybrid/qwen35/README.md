# Family: qwen35

GGUF arch values: qwen35.

PrismML Ternary Bonsai 27B is this family with `prism.hadamard` (llm_config `hadamard`): rotating activation quantizers in front of every
folded projection, a latent token embedding (D (H z) after the lookup), and the gated delta-net `ssm_out` input permuted from the
tiled head order to the grouped order of the fold (`gdn_v_grouped`); alpha / beta (F32) and the Q4_0 MTP block are not folded and read
the f32 activation in the model basis. Code and rationale: `starter/ternary.py`; weights: loom-kb `quant/ptq1_0`.
