# Q2_0 (upstream ggml)

64 values per 18-byte block (2.25 bits per weight): F16 `d`, then 16 bytes of 2-bit codes, value j in byte j / 4 at bits 2 (j % 4);
value (q - 1) d (codes 00 = -1, 01 = 0, 10 = +1, 11 = +2). Ternary Bonsai ships its ternary weights in this packing for stock llama.cpp
(each 128-weight ternary group's scale repeated in both 64-value halves; tools in #1416 proved it identical to the PTQ1_0 file).
hrx-loom 0.3.0's GGUF parser knows GGML types below 43, so this is the file a Loom package can reference (PTQ1_0, type 143, is refused
at load: loom-kb topic `sdk`).

Helpers (`starter/ternary.py`): `@g_q2_0_icodes` (codes 0..3, scale d, minimum d: int8 matvecs and the head subtract the activation
code sum) and `@g_q2_0_values`. A 32-value group is 8 bytes: four shift-and-mask steps give the codes of values 4i + k in byte i, one
byte shuffle restores value order (about 1 VALU op per weight).
