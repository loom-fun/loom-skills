# Facet: looped: roofline

## `roofline.py` counts physical blocks

`roofline.py` reads each GGUF tensor once, so it misses the extra passes and overstates the ceilings of a looped model.
Inkling-2-Mini on Strix Halo: 98.5 tok/s decode from the physical tensors, ~56 tok/s from the 48 logical layers (4.0-4.1
GB per token: head 0.49, Q8_0 attention 1.37, active experts 1.98, f32 router and controllers 0.25). Prefill: ~10 GFLOP per
token, ~4,380 tok/s at 43.8 TFLOP/s (85 W). Until the script counts passes, compute the ceilings by hand over the logical
layers and set targets and report efficiency against those: against the script's ceiling a port looks half as efficient
as it is.

## Strix Halo: the looped weights stream from DRAM on every pass

The looped core is gigabytes, against a 32 MB MALL, so each pass re-reads its weights from LPDDR5X. Inkling's measured
decode matvecs sit at 92-93 % of 228 GB/s on the logical-layer byte count (MoE experts at ~15 of 16 expert-equivalents per
(token, expert) pair), which is consistent with no cross-pass reuse.
