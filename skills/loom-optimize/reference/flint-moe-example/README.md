# Flint (RecursiveSI/flint, 40B MoE): hand-tuned MoE prefill generator

`gen_moe.py` is the one piece of the original hand-written Flint port (int4 Q4_1 weights) whose idea
is not in the starter or in `loom-port-llm/reference/{flint,lfm2moe}-example` (start a MoE port from
those). Ideas, not a base:

- router top-k -> one-workgroup bucketing of the token x slot assignments into 32-row tiles per
  expert (`tile_info` = (expert, rows), a `rowmap` of each padded row's assignment, `pos` of each
  assignment's row) -> grouped WMMA gate/up and down GEMMs over the tiles -> weighted combine;
- `MOE_TILE` > 32 buckets into bigger tiles that the down GEMM runs as 32-row virtual tiles (`down_virtual`; empty halves exit).

`python gen_moe.py > moe.loom` (env: `MOE_TILE` rows per tile, default 32; `FLINT_MOE_FMT` q4_k|q4_1;
`DOWN_GROUPS`). It imports `dequant16` from `loom-port-llm/starter/gen_gemm.py` (q4_1 support there exists for it).

The rest of that port (adaptive split-K decode attention, fused RMSNorm + f16 row scale, the listed
prefill fallback, per-token RoPE + KV store) has been folded into the starter's `gen_llm.py`.
Its measured numbers (int4 build vs llama.cpp b1355: decode 99-104 vs 62 tok/s) are in
`loom-roofline/reference/calibration.md`.
