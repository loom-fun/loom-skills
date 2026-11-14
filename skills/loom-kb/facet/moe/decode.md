# Facet: moe: decode

## MoE decode: expert matvec workgroups, int8 expert activations, router loads (`gen_moe.py`)

Qwen3-30B-A3B-Instruct-2507 (128 experts, top-8, 48 layers; DevLab, interleaved `ab` rounds in one server, two runners
agreeing within 0.4 %): main build 14.39 ms per token, all levers below 12.97 ms (1.11x), each measured on top of the
previous. Decode had no kernel at its bandwidth: a 1-token launch is KU x FFN / R workgroups, and with R=8 that is only
768 for gate/up.
- **Rows per workgroup `MOE_R`** (auto: aim for ~3000 workgroups per gate/up launch, powers of two within 2..8): R=8 -> 4
  +4 %, 4 -> 2 +1.5 %, R=1 slower (13.13 vs 12.99 ms).
- **Router `MOE_ROUTE_U`** (default 4): `g_moe_route` is one latency-bound workgroup per token (34 us per layer: 1.6 ms of
  the 14.4); scoring 4 experts per wave in the same loop (4 independent weight loads in flight, summation order
  unchanged, outputs bit-identical) is +2 %; 8 ties.
- **int8 expert activations `MOE_Q8`** (default on with `ACT_Q8`): the router input's `g_quant_q8` copy (made beside the
  row scale) feeds the gate/up matvec, a `g_quant_q8` launch quantizes the gate/up output for the down matvec, and both
  contract with the weight codes by integer dot4 (llama.cpp's mul_mat_vec_q for experts). +2 %; alone, at R=8, it
  tied (the f32 kernel was latency-bound, not bandwidth-bound). The kernel-test chain emulates the Q8_1 rounding.
- The ablation by doubling under-reports bandwidth-bound expert launches (a 14 MB layer of experts re-read from the MALL):
  the route and small-op doublings (+0.8 ms, +0.6 ms per 48 layers) are the believable ones.

## Did not help (MoE)

- **Unrolling the expert matvec's row loop** (R straight-line copies, all weight loads before the first wave reduction;
  Qwen3-30B-A3B decode): -2 % at R=2 and -6 % at R=4 against the same R rolled. Occupancy (more, smaller workgroups) beats per-wave memory parallelism.

- **Expert gate_up prefetch.** No gain; the kernel is compute-bound already.

- **Larger MoE tiles or wide GEMM tiles.** More padding waste and less occupancy.
