# Facet: wide-head: prefill

## flash2 at head_dim 256 and 512 (`gen_flash2s.py`)

- At head_dim 256, 256 dims of Q (as f16 fragments) plus 256 dims of output accumulators exceed 256 VGPRs, so
  gen_flash2.py reloads all of Q from global memory (f32, converted each time) every 32-key tile: ~6x the K/V
  tile's bytes. gen_flash2s.py gives each query head TWO waves of 128 dims: a wave holds its Q half in
  registers for the whole key loop, computes S^T over its dims, publishes it to LDS and adds its partner's
  (commutative add: both waves hold bit-identical S, softmax and P), then accumulates its 128 output dims.
- Staging: lane = key, workitem/32 = dim segment, so a wave's V^T writes hit 32 consecutive columns
  (conflict-free) and each lane does one paged-KV lookup per tile (V = K offset + one plane).
- Qwen3.8-27B (G = 6, 12 waves, 61 KB LDS): a 1024-token chunk at 32k depth runs 16.9 TFLOP/s; prefill
  2k/8k/32k 401.6/368.6/262.5 -> 429.1/426.7/378.1 tok/s. Attention had been 1.42 ms of 3.8 ms per token at 32k.
- Head_dim 512 (gemma4 global layers): four waves per head, 16-key tiles (K and V^T of 32 keys exceed 64 KiB), each
  wave sums the head's four S partials in wave order. Over the two-pass kernel that reloaded Q per tile and
  recomputed S per 256-dim output half: gemma-4-26B-A4B prefill 1k/8k/16k/32k 1.11/1.35/1.64/2.17x (local A/B).
- Test it standalone with real shapes: a flash2 harness (paged arena filled directly, float64 reference with
  the gate) plus a broken variant (partner sum dropped) as the negative control.
