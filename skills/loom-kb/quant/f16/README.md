# F16

32 values per block, 64 bytes per block (16.0 bits per weight). Dequant helper: gen_llm.py (@g_f16_values).
Supports: dense matmuls (f16 dequant in LDS before the WMMA). Devices verified (rdna3 / rdna4): yes / yes.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.
