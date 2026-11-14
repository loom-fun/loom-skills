# Facet: kv-le2

At most 2 KV heads (G >= 8 query heads per KV head): decode and verify attention shapes where the 8-KV-head span path loses.

## Long-context verify at 2 or fewer KV heads: one workgroup per span

- **Long-context verify at 2 or fewer KV heads: one workgroup per span** (inside `g_attention_split`, `SEG_MIN` 8192;
  head_dim 64 and 128; drafting packages). The span path above needs 8 KV heads; below that, a token at position >=
  `SEG_MIN` on a full-attention layer that leads a verification span runs the span body, which reads each key block once
  for all the span's tokens (keys over lanes for the scores); the span's other tokens exit. Exact per token (drafted ==
  undrafted). It is folded into the split kernel's prologue (no leader-list kernel, no extra launch per layer): a
  separate list plus an indirect launch cost 4 to 14 % of short-context 4-stream decode on DevLab. Same-runner DevLab,
  with vs without (`SEG_ATTN=0`): Qwen2.5-0.5B depth 2, decode at 32k +21 to +35 %, 4 streams 98.6 % of `SEG_ATTN=0`
  (90 runs per variant per runner against a 2 % spread between two `SEG_ATTN=0` builds), short decode equal; locally
  +37 % at 32k. Qwen2.5-1.5B +9 to +18 % at 32k. It lost at 8 KV heads (Qwen3-4B DFlash, -17 % locally) and at head_dim
  256 with one KV head (Gemma-3-1B, 8k/16k -40 %). Tried: a span workgroup with lanes over dims (slower: 45
  instructions per key and head at head_dim 64 for the per-key 32-lane reductions).
