# Family qwen35moe: decode

## Drafted decode attention at 32k (D7 scope rule)

`scope.json` sets `decode.attn.kv_once.d256 = false` for drafted models (`ATTN_GQA_D256=0`). With 2 KV heads and head_dim 256 the K/V-once decode body
(`g_attention_split`, 252 VGPRs) ran decode_32k at 0.74-0.78x of the loom-fun package on Strix (Qwen3.5-35B-A3B, Qwen3.6-35B-A3B, Ornith-1.5-35B-A3B, Sage-A1.5);
undrafted qwen35moe (Ornith-1.0-35B, MiniCPM-V-4.7) never had the body and are unchanged. The rule restores the loom-fun kernel (byte-identical `g_attention_split`).
The key is per_target: gfx11 and gfx12 may differ (a twin `g_attention_split_rdna4`, emitted only where they do). Without a model rule both targets follow it. DevLab A/B of the rule (run 43631671557): Strix decode_32k vs loom-fun 0.98-1.04,
R9700 decode_32k vs the previous staging package 1.22-1.41, every other row within its band. Fleet diff vs `refactor`: 5 models (the four, plus Ornith-1.5-122B), `g_attention_split` only.

Qwen3.6-35B-A3B kept the K/V-once body on gfx12 under SDK 0.2.0 because D7 raised R9700 decode_32k 20–23% and reduced decode_2k 7–10.5%. With SDK 0.3.0, the frozen-source 32K profile showed a 2.59× increase in the gfx12 attention split kernel's time. The model now uses the split attention path on Radeon. On 0.3.0 it recovered 32K decode from 83.16 to 121.18 tok/s, compared with 121.92 tok/s under 0.2.0, and kept 2K decode within 0.12% of the old baseline. [Radeon comparison and profile](https://github.com/loom-fun/loom-skills/actions/runs/45297807464)

## Read-once verify spans at 32k (decode.attn.segment.d256 + .wmma)

With 2 KV heads x 8 query heads at head_dim 256 and DFlash depth 3, the per-token split kernel re-read the 66 MB/layer K/V for each of the 4
verify tokens (Qwen3.6 Strix: 12.2 ms of a 47 ms step at 32k). The segment path at head_dim 256 reads each 32-key block once for the span: the
scalar body (chunk-outer score loop) halved the kernel (1212 -> 637 us/layer, ~45 % of the KV-once DRAM floor); the WMMA body (32 query columns =
4 tokens x 8 heads, K/V staged once per block, one-block prefetch) reaches 348 us (~83 %). Strix decode_32k 56.2 -> 72.4 (Qwen3.6), and Qwen3.5's
3500-token draft gate (set because drafted 32k lost to undrafted) is no longer needed: drafting at every context, 48.3 -> 63.6. R9700 gains the same
(Qwen3.6 134.8 -> 163.5). Runs 45583608766, 45591666122, 45619373718. Starting the span at 1k (SEG_MIN 1024) tied decode_2k and failed ktest: keep 8192.
