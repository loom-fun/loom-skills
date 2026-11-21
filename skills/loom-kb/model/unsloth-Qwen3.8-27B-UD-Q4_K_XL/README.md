# Model: unsloth-Qwen3.8-27B-UD-Q4_K_XL

Qwen3.8-27B from unsloth's UD-Q4_K_XL GGUF (the quant gufo serves). Same architecture and geometry as
`bartowski-Qwen3.8-27B-loom`; the format mix differs (trunk bytes: q5_k 50 %, iq4_xs 30 %, q4_k 8 %, q6_k 7 %, iq4_nl 3 %,
q8_0 1 %, q3_k 0.4 %; ssm_alpha/ssm_beta q8_0; 22 FFN layers mix gate/up formats), so the sibling's GEMM rules are re-evaluated here.

## Format support found at the port (q38xl-port, 2026-11-20)
- Decode matvecs: every format of the file has kernels (IQ4_XS/IQ4_NL/Q3_K on f32 activations, the K-quants and Q8_0 on int8).
- Prefill GEMMs: IQ4_XS / IQ4_NL / Q3_K went through the `@g_<fmt>_values` helper, so they had only the 64-row tile and no mixed-format
  swiglu; `prefill.gemm.iq4_native` gives IQ4_XS/IQ4_NL their own 128/256-row kernels (+15 % prefill). Q3_K (one ffn_up, 0.4 % of bytes)
  keeps the helper tile (14.9 TFLOP/s in the pf8k profile); layers pairing IQ4_NL or Q3_K with another format keep two projections.
- `prefill.fuse.qg_split` wraps a layer's split only when its q GEMM has the m256 epilogue (5 of 16 q projections are IQ4_XS).
- GDN alpha/beta are Q8_0 here: `prefill.gdn.ab_wmma` runs `g_gemm_ab_q8_0w` (codes * d in F16 while staging), 0.007 ms/token at 8k.
- With this format mix the kernel library is at the hrx-loom encoding limit: `prefill.gemm.prune` drops the swiglu m256 kernels the f16o
  rewrite replaced (enc kernels 4.06 -> 3.83 MB of the 4 MB ENC_BYTES budget in `dual_target.encodings_first`); a further kernel variant may
  need a split library (protected `build.py` / `dual_target.py`).
