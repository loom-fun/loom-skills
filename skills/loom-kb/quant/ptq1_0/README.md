# PTQ1_0 (PrismML ternary)

128 trits {-1, 0, 1} and one F16 scale per 28-byte block (1.75 bits per weight): `qs[24]` holds 5 trits per byte (base 3, stored
scaled to 0..255 so trit n of byte q is `((q * 3^n mod 256) * 3) >> 8`), `qh[2]` 4 trits per byte, then `d`. Value order: v < 80 is byte
v % 16 trit v / 16; v < 120 is byte 16 + (v - 80) % 8 trit (v - 80) / 8; the last 8 are byte 24 + v % 2 trit (v - 120) / 2. GGML type
143 exists only in the PrismML llama.cpp fork: stock llama.cpp (and the lemonade ROCm builds) cannot read it; Ternary Bonsai repos
also ship the same weights as Q2_0 (stock type 42, 2 bits per trit, group 64, same scales), which the llama.cpp reference loads
(`REF_GGUF` / `LLAMA_REF_GGUF`). `starter/gguf_ext.py` registers the type with the gguf package and gives the exact numpy oracle.

Helpers (`starter/ternary.py`): `@g_ptq1_0_icodes` returns one 32-value group's codes and `d` (one scale, as q8_0: the decode matvecs
and the head take int8 activations); `@g_ptq1_0_values` the f32 values (GEMM dequant, embedding). The 32-value group branches on the
group index (0..3), so lanes of one wave that hold different groups of a block diverge: a dedicated kernel decodes whole blocks.

Ternary Bonsai checkpoints are also rotated (`prism.hadamard`, llm_config `hadamard`): see `starter/ternary.py` and the port notes in
loom-kb `arch/llm/hybrid/qwen35` (rotating activation quantizers, latent embedding, grouped `ssm_out` input).

Published ternary-inference ideas and how they map to RDNA3.5 / RDNA4 (#1416 research):
- BitNet b1.58 / bitnet.cpp: I2_S (2-bit packing, int8 multiply), TL1 / TL2 (lookup tables of activation partial sums indexed by 2 or 3
  packed trits). On a GPU the int8 dot (`v_dot4_i32_iu8`) is cheaper than table indexing; the cost is unpacking trits to int8.
- ggml TQ1_0 / TQ2_0: base-3 (the same `* 3^n, * 3 >> 8` decode as PTQ1_0) and 2-bit packings; TQ2_0 trades 25 % more bytes for a shift
  and mask decode.
- T-MAC: bit-serial table lookup instead of multiplies (a CPU technique: `pshufb` / `tbl`); on RDNA, `v_perm_b32` is a 8-entry byte table.
- Decode is bandwidth-bound at 1.75 bits per weight (Strix Halo ceiling 38 tok/s, R9700 103 tok/s for the 27B, loom-roofline) only
  while unpacking stays under the memory time: about 3 VALU ops per weight is the budget on Strix Halo.
- Prefill: weights are exact in int8 (and int4) and activations are 8-bit, so an int8 WMMA GEMM is exact on the weight side; RDNA4 has
  twice the int8 WMMA rate of f16, RDNA3 the same.
