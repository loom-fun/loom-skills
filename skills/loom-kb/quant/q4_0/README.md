# Q4_0

32 values per block, 18 bytes per block (4.5 bits per weight). Dequant helper: gen_llm.py (proposal head copy; weights too).
Supports: low-bit. Devices verified (rdna3 / rdna4): yes / yes.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.

Q4_0 experts use F32 decode matvecs with the shared value helper and grouped
WMMA prefill with direct 16-value half-block staging (`starter/moe_q4_0.py`).
The block has a signed F16 scale, then 16 packed bytes: low nibbles are columns
0–15 and high nibbles 16–31; each code is centered at eight. Grouped gate/up
supports the fused per-expert layout and different gate/up formats. Dense
Q4_0 kernels and all previously supported expert formats keep their old paths.

The `moe_q4_0` DevLab suite uses independent F64 references at Gemma expert
widths (2816 and 704), full and partial gathered tiles, GEGLU, down scales,
sentinel fills and explicit wrong-result controls. CPU generation/compilation
alone does not establish real-weight or device qualification.
