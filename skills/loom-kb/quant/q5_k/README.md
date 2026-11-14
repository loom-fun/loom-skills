# Q5_K

256 values per block, 176 bytes per block (5.5 bits per weight). Dequant helper: gen_llm.py.
Supports: int8 activation path, MoE experts. Devices verified (rdna3 / rdna4): yes / yes.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.

## Dequant in the WMMA GEMM (`gen_gemm.py`)

The fifth bit comes from the qh words (`(qh >> g) & 0x01010101`, shifted to bit 4) instead of byte vectors: ~+8 % in isolation on real
weights (26.6 -> 28.8 TFLOP/s).
