# Qwen3.5 and Qwen3.6 35B-A3B expert GEMM experiment

```loom-scope
kind: optimize
scopes:
  - unsloth-Qwen3.5-35B-A3B-loom/strix-halo
  - ggml-org-Qwen3.6-35B-A3B-loom/strix-halo
kernels:
  - g_moe_ggu_q4_k_k2048
```

The two suites use independent exact staging revisions and verify that the default generator produces the deployed gfx11 gate/up body. They compare the same 64×64 output tile and K64 accumulation order. The experimental body assigns half of the workgroup to each of the gate and up weights, stages 32 contiguous Q4_K codes per producer, and shares one header and scale decode per group. It keeps prefetch for the A gather and B weights. The fixture uses sector-varied finite weights and arbitrary row scales, and checks exact agreement on 63- and 65-token edge cases, 1024 and 4096 tokens, partially occupied expert tiles and unused tiles. The expected-failure case checks that the populated result differs from its nonzero sentinel.

Select exactly one suite per hardware measurement. Wait for independent serving profiles before projecting each generation's end-to-end impact. The Qwen3.5 32K profile attributes 12.4% of device time to gate/up, so about 16% kernel time reduction is needed to recover 2% of the total. Kernel-only results do not establish an end-to-end win or change any catalog default.

The paired-row alternative handles two adjacent 64-row tiles in one workgroup only if they use the same expert. The workgroup stages B once and accumulates each tile separately with its original MMA order. A tile without an eligible adjacent partner keeps the original single-tile path. This adds one A scratch tile and retains the original scheduling and prefetch for the primary A and shared B. At 4096 input tokens the average expert receives 128 assignments, which gives the pairing a larger potential payoff than at 1024 tokens.
