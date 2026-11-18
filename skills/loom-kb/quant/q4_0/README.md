# Q4_0

32 values per block, 18 bytes per block (4.5 bits per weight). Dequant helper: gen_llm.py (proposal head copy; weights too).
Supports: low-bit. Devices verified (rdna3 / rdna4): yes / yes.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.

Q4_0 experts use F32 decode matvecs with the shared value helper and grouped
WMMA prefill with direct 16-value half-block staging (`starter/moe_q4_0.py`).
The block has a signed F16 scale, then 16 packed bytes: low nibbles are columns
0–15 and high nibbles 16–31; each code is centered at eight. Grouped gate/up
supports the fused per-expert layout and different gate/up formats.

Dense Q4_0 (this node's `scope.json` turns both on for every model with Q4_0 tensors; Qwen3.8-27B, whose MTP block is Q4_0,
first failed registration with both on: its attention kernels crossed the SDK's 16-bit SSA encoding ids, fixed in the kit;
topic `sdk`):
- `prefill.gemm.q4_0_native`: dense GEMMs stage with the same `moe_q4_0.dequant16` snippet, so Q4_0 gets the
  128/256-row kernels (before: a helper format, the 64-row tile at every size).
- `decode.q4_0.act_q8`: matvecs, head and expert matvecs take int8 activations (`@g_q4_0_icodes`, minimum 8 d),
  as llama.cpp's `vec_dot_q4_0_q8_1`. The f32 path paid one subgroup reduction per row and token: ALU-bound at 4 tokens.

Measured (DevLab run 44306088328, same-job ABBA, base 33c407da): Gemma 4 31B QAT on the R9700 prefill 1k/2k/8k
871/888/868 -> 1337/1421/1362 tok/s (+54..60 %; the GEMMs now 77 % of a 2k prefill, the rate of the Q4_K_M 31B card),
4 streams 58.5 -> 121.2, decode +3 %. 26B-A4B QAT 4 streams 318 -> 406 (R9700), 153 -> 182 (Strix Halo), prefill +8..16 %,
decode -1..-3 % (within the band). The expert matvecs of gemma4 (input `%h2`, not the router input) stay f32; the
rows above did not need them.

The `moe_q4_0` DevLab suite uses independent F64 references at Gemma expert
widths (2816 and 704), full and partial gathered tiles, GEGLU, down scales,
sentinel fills and explicit wrong-result controls. CPU generation/compilation
alone does not establish real-weight or device qualification.
