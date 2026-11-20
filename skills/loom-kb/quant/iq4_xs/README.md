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

The F32 dequant of `iq4_dequant16` (sign-extend + cvt_f32 + mul + cvt_f16 per value, a scale decode per 16) put the m256 loop at 467
VALU per 64 WMMA and 227-229 VGPRs with scratch spills whose `vmcnt(0)` waits drained the prefetched loads (store 17, acc 20, swiglu 29
TFLOP/s in the pf8k profile). `prefill.gemm.iq4_g32` (gfx11 m256: the thread's whole 32-group, one scale decode per 32) and
`prefill.gemm.iq4_f16lut` (the 16-value path) keep the book as kvalue + 128 (unsigned), so `vector.uitofp` i8 -> f16 is the 0x64 magic
(v_perm + pk_add) and the scale is one pk_fma per pair: fma(u, f16(dl), -128 f16(dl)) = round_f16(kvalue * f16(dl)) (one rounding, as
the K-quant kernels; `-128 dl` must be computed in F32 then fptrunc). Loop VALU 467 -> ~180, no spills. Two codegen pitfalls (hrx-loom
0.2.0): a `vector.table.lookup` index masked as i32 words then bitcast to i8 lowers to 480 cmp + cndmask (mask on `vector<16xi8>` so the
index range is provable); a scalar `scalar.mulf ... : f16` feeding a splat was miscompiled (result into a `.h` half of a live VGPR, the
fma read an unwritten register: every G32 m256 ktest failed).
