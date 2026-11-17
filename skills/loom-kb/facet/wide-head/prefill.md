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

## Key-split flash at head_dim 256 / 6 query heads per KV head (`FLASH2_B=2`, `flash2_b2.py`; choice `prefill.attn.flash2_b2`, gfx12 twin only)

- **Why.** `FLASH2_ABL` ablations of the staged twin on the Qwen3.8-27B MXFP4 card (R9700, run 43803038899, prefill tok/s base -> ablated at 16k / 32k): no
  workgroup barriers 2520 -> 2730 / 2255 -> 2505, no V^T staging stores 2714 / 2477, no global K/V loads 2592 / 2359, no PV MMAs 2568 / 2356, no S MMAs
  2625 / 2421, no exp 2696 / 2427. Barriers plus staging are about a fifth of 32k prefill; the WMMAs are not the bound.
- **What.** The d512 key-split loop of gemma4 E2B ported to head_dim 256: a workgroup is the four waves of a head pair (three pairs per KV head: grid
  tiles x 12), a block is 64 keys, wave j computes the S^T of keys 16 j.. over all 256 dims for both heads from K fragments loaded straight from the paged
  arena (Q^T of both heads staged once in 17 KB of workgroup memory), one softmax per head and block (two barriers per block), and accumulates O^T of its 64
  dims from V^T fragments straight from the arena. 180 VGPRs and 23.5 KB of workgroup memory (staged twin: 210 and 62 KB). `FLASH2_B=3` (three heads per
  wave, two triples per KV head, 256 VGPRs, the partial kernel spills) is slower than 2.
- **Measured** (one DevLab job, ABBA, merged base with the chunked GDN and fp8 producers, run 43808920153): prefill 2k / 8k / 16k / 32k 2681 / 2853 / 2691 / 2392
  -> 2690 / 2897 / 2760 / 2495 (+0.3 / +1.5 / +2.6 / +4.3 %), 1k tie. The attention bucket (every flash launch doubled, ms/token) 0.0141 / 0.0295 / 0.0533 /
  0.0994 -> 0.0074 / 0.0212 / 0.0419 / 0.0828. ktest 93/93 (the flash cases run the key-split kernel); an earlier job (run 43803453515) 89/89.
- **Fused q|gate RoPE (`ROPE_QG=1`, `rope_qg.py`, choice `prefill.attn.rope_qg`, opt-in): no change.** `g_qk_rope_store_qg` reads the query heads from the
  projection rows and copies the gate half, replacing `g_split_qgate` + `g_qk_rope_store`: doubling of the pair 0.0040 / 0.0029 ms/token at 2k / 8k vs 0.0053 /
  0.0046 fused; prefill alone within +-0.6 % at every length. The split is a small memory-bound copy; not in the AMD model's scope.
