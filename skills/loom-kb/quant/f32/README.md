# F32

32 values per block, 128 bytes per block (32.0 bits per weight). Dequant helper: gen_llm.py.
Supports: dense matmuls (f16 dequant in LDS before the WMMA). Devices verified (rdna3 / rdna4): yes / yes.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.
