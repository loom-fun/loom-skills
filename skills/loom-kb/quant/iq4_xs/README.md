# IQ4_XS

256 values per block, 136 bytes per block (4.25 bits per weight). Dequant helper: quant_values.loom (appended when used).
Supports: low-bit. Devices verified (rdna3 / rdna4): yes / yes.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.

## Dequant in the WMMA GEMM (`gen_gemm.py`)

Through the `@g_iq4_xs_values` helper (a call whose loads the m128/m256 pipelines cannot prefetch) IQ4_XS only had the 64-row tile at
every size and no mixed-format swiglu. `prefill.gemm.iq4_native` (opt-in, model nodes) decodes the 16-value half-group inline
(`gen_gemm.iq4_dequant16`: header + 16 code bytes loaded first, `vector.table.lookup` on kvalues, `d * (ls - 32)` per 32-group in F32,
fptrunc: the helper's values) and gets the 128/256-row kernels, the qg-split / residual-epilogue m256 variants and the mixed m256 swiglu
with q4_k/q5_k/q6_k. Qwen3.8-27B UD-Q4_K_XL (30 % of trunk bytes IQ4_XS), local Strix 123 W: see `model/unsloth-Qwen3.8-27B-UD-Q4_K_XL`.
