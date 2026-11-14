# Q8_0

32 values per block, 34 bytes per block (8.5 bits per weight). Dequant helper: gen_llm.py.
Supports: int8 activation path, MoE experts. Devices verified (rdna3 / rdna4): yes / yes.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.

## Dequant in the WMMA GEMM (`gen_gemm.py`)

int8 -> f16 and one f16 multiply (q * d is exact in f32, so the f16 result is bit-identical): ~+8 % in isolation on real weights
(26.6 -> 28.7 TFLOP/s).
