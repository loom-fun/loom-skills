# Model: unsloth-Qwen3.8-27B-UD-Q4_K_XL

Qwen3.8-27B from unsloth's UD-Q4_K_XL GGUF (the quant gufo serves). Same architecture and geometry as
`bartowski-Qwen3.8-27B-loom`; the format mix differs (trunk bytes: q5_k 50 %, iq4_xs 30 %, q4_k 8 %, q6_k 7 %, iq4_nl 3 %,
q8_0 1 %, q3_k 0.4 %; ssm_alpha/ssm_beta q8_0; 22 FFN layers mix gate/up formats), so the sibling's GEMM rules are re-evaluated here.
