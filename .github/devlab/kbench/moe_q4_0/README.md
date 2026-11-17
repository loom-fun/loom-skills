# Q4_0 expert qualification pilot

Run `.github/devlab/moe_q4_0-kbench.json` through `devlab-kbench.yml`. It pins
SDK 0.2.0 and has one Strix Halo and one R9700 job. The generator builds the
current Gemma kit and extracts its GEGLU transforms; no model weights or
previous checkpoint numerics are used.

Nine independent NumPy reference cases cover the Gemma dimensions H=2816,
expert FFN=704, fused gate/up per-expert offsets, Q4_0 nibble order and signed
scales, one/four decode tokens, 63/65 gathered input rows, full/partial/dead
expert tiles, and per-expert down scales before weighted combine. Eight
experts retain nonzero expert offsets without allocating the full 128-expert
model. Real-weight model kernel tests and end-to-end gates remain required.

A tenth case executes the actual old F32-helper/F16-cast snippet and new F16
staging snippet for all 63,488 finite F16 scale bit patterns. Both nibble
halves contain all 16 codes, in different orders. Each path stores the F16
result as I16 bits and must match the independently rounded CPU reference
and the other path exactly, including signed zeros, subnormals, normal-range
rounding boundaries and overflow infinities. This checks the GPU's DAZ/FTZ
behavior directly; CPU multiplication equivalence alone cannot establish it.

Output buffers begin at 42, so missing writes fail the reference check. Two
separate negative modules deliberately omit the decode launch or swap Q4_0
nibble halves; both must fail the approximate comparator with EXPECT/006,
not compilation, setup or a different error. Use the campaign's shared strict
negative-control harness when it is integrated. No pass or performance claim
is established by CPU generation alone.

The benchmark file measures eight expert cases over three repetitions.
Inspect case-specific raw device timestamps; kernel names recur at different
row counts and an aggregate kernel-name summary loses that distinction.
