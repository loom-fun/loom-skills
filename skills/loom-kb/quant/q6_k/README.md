# Q6_K

256 values per block, 210 bytes per block (6.5625 bits per weight). Dequant helper: gen_llm.py.
Supports: int8 activation path, MoE experts. Devices verified (rdna3 / rdna4): yes / yes.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.

## Dequant in the WMMA GEMM (`gen_gemm.py`)

Do the bit work on 32-bit words (masks 0x0F0F0F0F / 0x03030303; a byte's bits shifted in from its neighbour are masked off) and finish
with one f16 FMA `codes * s + (-32 s)`; byte vectors plus f32 math ran 22 TFLOP/s, this 26-28 (Q4_K: ~35 on the same constant-data
benchmark). Bolt-E4B (89 % Q6_K in the UD-Q5_K_XL file) prefill +18/+11/+6.5 % at 2k/8k/32k from this change.
