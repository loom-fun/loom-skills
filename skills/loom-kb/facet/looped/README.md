# Facet: looped

Looped (recurrent-depth) transformers: a contiguous range of physical blocks runs several passes per token with shared
weights, plus small per-pass tensors (loop controllers, message queries). Today: inkling2 (24 physical blocks, `blk.6..17`
looped 3 times: 48 logical layers). Matched when a family's `scope.json` lists `looped`, or a cfg has a `loop` key.

## Count logical layers, not physical blocks

- Every per-layer quantity scales with the logical count: weight bytes and FLOPs per token, kernel launches, KV planes,
  recurrent state per row, and the layer indices a drafter taps. Only the weights on disk count physical blocks.
- Passes are sequential (a pass reads the previous pass's output), so the shared weights cannot be batched across passes:
  each pass is its own matvec in decode and its own GEMM over the chunk in prefill.
- The static `@llm_advance` program is emitted per logical layer, with a (logical layer -> physical block, pass) map
  choosing the shared weights and the per-pass tensors. A loop of 3 over 12 blocks makes every launch-site edit hit 36
  launches.
