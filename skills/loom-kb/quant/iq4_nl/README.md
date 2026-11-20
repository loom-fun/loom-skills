# IQ4_NL

32 values per block, 18 bytes per block (4.5 bits per weight). Dequant helper: quant_values.loom (appended when used).
Supports: low-bit. Devices verified (rdna3 / rdna4): yes / yes.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.

## Dequant in the WMMA GEMM (`gen_gemm.py`)

`prefill.gemm.iq4_native` (with IQ4_XS, see its node): the 16-value half-block inline (`gen_gemm.iq4_dequant16`: F16 `d * kvalues[q]`,
one rounding as the helper + fptrunc) -> the 128/256-row kernels. 32-value blocks: not in the mixed m256 swiglu (it needs 256-value
superblocks), so a layer pairing IQ4_NL with another format keeps two projections + `g_silu_mul`.
