# Facet: wide-head

Attention head_dim 256 or more (qwen35 / qwen35moe hybrids, gemma3/gemma4, spark2_5): the two-wave flash2 prefill kernel (`gen_flash2s.py`) instead of `gen_flash2.py`, whose Q fragments no longer fit in registers. Its findings are the task file for prefill.
