# Model: ggml-org-Qwen3.6-35B-A3B-loom

Radeon AI PRO R9700 serving settings of `ggml-org-Qwen3.6-35B-A3B-loom` (prefill_capacity 2048, context_capacity 84992, pool_capacity 84992, context_cap 84992); resolved per device by `resolve.py`, written into the `<first>-r9700` preset of model.json by `make_info.py build --fit`.

## Attention on Radeon (SDK 0.3.0)

This model has no rule of its own for `decode.attn.kv_once.d256`: the drafted-qwen35moe family rule (`false`, split attention) applies on both GPUs.
Under SDK 0.2.0 a model rule kept the K/V-once body on gfx12 (it raised R9700 decode_32k 20-23 %, cost decode_2k 7-10.5 %). On 0.3.0 that body,
`g_attention_split_rdna4`, takes 1109.92 ms instead of 427.80 ms for the 1290 launches of a 32K decode (2.59x; frozen-source depth-0 profile), so R9700 decode_32k
fell from 121.92 to 83.16 tok/s. Split attention gives 121.18 tok/s at 32K and 161.24 vs 161.05 tok/s at 2K ([DevLab run 45297807464](https://github.com/loom-fun/loom-skills/actions/runs/45297807464)).

## SDK 0.3.0 optimization lane (#1418)

Same rules and findings as `model/unsloth-Qwen3.5-35B-A3B-loom` (same architecture and drafter geometry); this package drafts at every
context (tuned ctx_max 0), so `draft.dflash.split_window` also moves its decode_32k (R9700 +9 %, Strix +7 % with the set).
