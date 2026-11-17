# qwen35: decode

## Small-model residual RMSNorm + Q8 publication experiment (2026-11-17)

Measured scope: Qwen3.5-0.8B (hidden 1024), DevLab, SDK 0.2.0. Do not extrapolate these results to the larger family.
The issue #1106 experiment published the existing planar Q8 representation from the residual/RMSNorm producer's retained
F32 values, keeping its original F32 result and residual update. No model rule was enabled and no serving win was adopted.

- [Standalone kernel A/B 44167771341](https://github.com/loom-fun/loom-skills/actions/runs/44167771341), source e647a41f:
  all 12 exact/reference cases passed. The complete two-kernel baseline versus fused publication improved case wall time
  11.6–20.5% across 9 width/row/device combinations (Strix: 1024; R9700: 1024/5120; rows: 1/4/63), five interleaved runs each.
  Compare the whole baseline chain, not the norm kernel alone. Count HAL `dispatch_function` rows only: command-operation
  rows repeat those samples.
- [Paired real-model Strix profile 44173422464](https://github.com/loom-fun/loom-skills/actions/runs/44173422464),
  fixed base 5f590bc2/head 6782988a, 400 tokens each: named norm+quant time 160.705→191.198ms; total device busy
  1496.810→1528.090ms. The producer bucket explains 97.5% of that device-time increase. Exactly 47×401 norm/quant
  pairs became fused launches; all other kernel launch counts were unchanged. The fused kernel averaged 6.131 us here
  versus 3.526 us in its isolated profiled replay. This localizes a cost; it does not establish a register-pressure or cache cause.
- The earlier [serving A/B 44170483430](https://github.com/loom-fun/loom-skills/actions/runs/44170483430) emitted only
  three sessions per side with `repeat_rows=1`, below the fast-model rule of six sessions and three repeats. Its rate
  estimates are diagnostic only, never promotion evidence. Profile request wall time is also not a throughput verdict.
- [Chunk-publication kernel A/B 44174119125](https://github.com/loom-fun/loom-skills/actions/runs/44174119125), e226fc0f:
  all 12 exact/reference cases passed. On Strix, chunk 8 versus fused 32 case medians were 2.586/2.631 us at one row and
  2.592/2.659 us at four rows, with separated five-run ranges, but 5.187/4.990 us at 63 rows. Chunk 16 had no clear
  all-shape gain. Lower VGPR use 132→104→96 for 32/16/8-value publication did not prove a serving improvement.

Revisit only with a producer design that wins in the model and a verified fast-model session plan. Preserve the original
path and numerical gates; an isolated kernel win is insufficient to enable a model scope.
