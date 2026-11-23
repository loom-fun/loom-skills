# Facet: looped: decode

## Strix Halo: the per-pass small-kernel tail

Each pass adds its own controller, mixer, norm and glue launches, so a looped model carries several times the small-kernel
tail of a plain one. Inkling-2-Mini (3 passes over 12 of 24 blocks, DSpark depth 1): ~1,350-1,900 dispatches per verify
step; the tail of 3-9 us kernels was ~7 ms of ~35 ms, with HAL busy time equal to wall time (kernel latency, not host gaps).
What paid, in order of size: `command.concurrent` around the independent per-pass launches (-1.83 ms per step, +5.5 %),
fusing norm + quantize and the mixers into neighbouring kernels (round 3: 1,622 -> 1,382 dispatches per step), and
multi-matrix matvecs for the per-pass projections. Each single fusion was worth 0.3-0.8 ms. Profile the tail by kernel
family (HAL dispatch timestamps) before fusing: it is spread thin, with no single dominant kernel.
