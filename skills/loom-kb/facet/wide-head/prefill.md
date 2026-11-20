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
- **gfx11, head_dim 256, G = 6: 16-key tiles (`prefill.attn.flash2_kt256 = 16`, set for Qwen3.8-27B on Strix Halo).** 32-key tiles take 255
  VGPRs and 61 KB LDS: one 12-wave workgroup per WGP, so every phase's latency is exposed (kbench ablations of the 32-key kernel: no global K/V
  loads -57..-62 %, no V^T stores -44 %, no barriers -41 % at 8k, exp ~0). 16-key tiles: 208 VGPRs / 33 KB, two workgroups per WGP that
  overlap each other's loads: kernel (1024-token chunk) -12 / -21 % at 8k / 32k depth locally (123 W), -26 / -27 % on DevLab (85 W);
  prefill 8k/16k/32k +0.3 / +1.3 / +2.8 % (local, ABBA). The K/V loads are still ~55-60 % of the 16-key kernel.
  - Not kept (kernel, vs 16-key tiles): two query tiles per 24-wave workgroup (`FLASH2_MT256=2`, half the K/V traffic) +22 % on DevLab,
    +18 % locally at 8k: halving the traffic made the load phase slower (latency, not bandwidth, bound; one big workgroup loses the
    cross-workgroup overlap); its double-buffered form needs a 4-half V^T row pad to fit 64 KiB, and that pad alone costs +60 % (LDS
    conflicts). Double-buffered staging (`FLASH2_G11=db`, at = b2) ties (-2 %). gfx12 trims on gfx11: `lazy` spills (684 B scratch,
    +120 %); register prefetch `pf` ties (its loads sit behind an scf.if and are waited at once). Query-head groups (`FLASH2_NG256` 2 / 3,
    6 / 4 waves): -23 % / -5 % at 8k but +218 % / +147 % at 32k on DevLab, +23 % / +168 % locally. Paired-key V^T stores (`FLASH2_G11=vp`, 256 of
    384 lanes stage two keys each) +26 % at 32k. The non-lazy trims (`FLASH2_G11=kvh,vtf,hot`): kernel -1.4..-3.8 % at 32k, prefill tie
    (+0.0/+0.2/+0.4 % at 8k/16k/32k).
  - Ablations of the 16-key kernel (32k / 8k): no global K/V loads -39 / -39 %, no V^T stores -16 / -22 %, no barriers +92 / -21 %: without
    the barriers the waves drift apart and lose the workgroup's lockstep K/V cache sharing at 32k, so a barrier-free design where every
    wave reads its own K/V (flash2_g6-style) is expected to lose there.

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
