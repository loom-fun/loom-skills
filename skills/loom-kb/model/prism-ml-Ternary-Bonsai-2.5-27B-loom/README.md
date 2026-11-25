# Model: prism-ml-Ternary-Bonsai-2.5-27B-loom

Generator choices of the Ternary Bonsai 2.5 27B package (qwen35 hybrid with `prism.hadamard` rotations; Q2_0 ternary weights, MTP depth 3),
measured on DevLab in the #1416 optimization lane; resolved per device family by `resolve.py`.

- Prefill: the Q2_0 GEMM stages its weights natively (16-value quarter-blocks to F16, `prefill.gemm.q2_0_native`), plus the per-family prefill
  settings the same-architecture Qwen3.8-27B node measured (gfx11 chunked GDN and 16-key flash tiles; gfx12 chunked GDN and flash trims). That node's
  fused epilogues (residual add in the output GEMM, q/gate split, GDN alpha/beta WMMA) are not set: with them the compare and consistency
  gates fail on the R9700 (their gfx12 twins run there), without them they pass.
- Decode: MTP proposals read a Q2_0 row copy of the head's first 81920 rows (`draft.propose.head_rows`); verification keeps the full head.
- Not kept (measured): load-time Q2_0 -> PTQ1_0 repack and table-decode (T-MAC style) matvecs (slower on both GPUs: the Q2_0 matvecs already
  stream at ~95 % of DRAM bandwidth, the base-3 decode makes them ALU-bound), the register/shuffle Hadamard (tie), a 32768-row proposal head.
