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

## Decode matvecs: int8 activations (`decode.iq4.act_q8`, IQ4_Q8)

Without the choice IQ4_XS / IQ4_NL decode and verify matvecs take f32 activations: at the 4-token verify of a drafted step they ran at
138 GB/s against 225 for the int8 K-quant matvecs (179 VGPR, 762 instructions at K=5120 t4), and 4 streams re-read the weights per
4 tokens. `@g_iq4_<fmt>_icodes` gives the kvalues codes and one scale per 32 values (llama.cpp's vec_dot_iq4_xs_q8_1), so the dot4 path
applies as for Q8_0 (67 VGPR, 438 instructions). Qwen3.8-27B UD-Q4_K_XL (33 % IQ4 bytes): decode +10.5 %, 32k +7.9 %, 4 streams +12.7 %
(model node). Opt-in per model: it changes the verify numerics (int8 activations), so the model's gates run with it.
