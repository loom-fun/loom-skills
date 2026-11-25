# Model: unsloth-Qwen3.5-35B-A3B-loom

Radeon AI PRO R9700 serving settings of `unsloth-Qwen3.5-35B-A3B-loom` (prefill_capacity 2048, context_capacity 56320, pool_capacity 56320, context_cap 56320); resolved per device by `resolve.py`, written into the `<first>-r9700` preset of model.json by `make_info.py build --fit`.

## SDK 0.3.0 optimization lane (#1418): what moved and what did not

Where the time goes (DevLab HAL profiles, kit build on 0.3.0): Strix pf8k 0.33 ms/token = trunk Q8_0 GEMMs 30 % (~28 TFLOP/s, near the
real-weight WMMA rate), grouped expert GEMMs 26 % (~23 TFLOP/s), flash2 attention 11 % (36 % at 32k, ~16 TFLOP/s at head_dim 256),
prefill router 5-6 %, sorted combine 3.5-5 %. Drafted decode at 32k (Qwen3.6, Strix): 47 ms per verify step of 2.2 tokens, of which the
split attention 12 ms (the 4 verify tokens each re-read the 68 MB/layer KV: no K/V-once body at head_dim 256) and the DFlash drafter's
window attention 6 ms (fixed by `draft.dflash.split_window`).

Kept (scope.json rules on this node): gfx11 qg_split + chunk11 cn16, gfx12 GDN chunk + copy split 8 + flash2 trims, f16 swiglu m256
output, f16 expert rows (`moe.prefill.store16`), 8-token router, split-K window drafter attention, Strix prefill chunk 8192.
Measured and dropped: flash2_kt256 16 (16k/32k -5..-10 % at 8 query heads per KV head; the 27B's G = 6 gain does not transfer),
FLASH2_NG256 2 and the gfx11 flash prefetch transforms (ties or losses once kt256 is off), prefill.gdn.fused2 (its scratch does not fit
the R9700 at the card's context), R9700 prefill chunk 4096 (workspace does not fit beside 21.6 GB of weights).
Open levers: a multi-query WMMA decode attention at head_dim 256 (decode_32k), cheaper short-prompt prefill (1k/2k at 67-71 % of the
roofline target on Strix: fixed per-request cost and epoch padding), expert GEMM efficiency.
