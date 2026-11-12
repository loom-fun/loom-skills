"""Decode/verify attention variants, shared by gen_llm.py (which emits them) and make_kernel_tests.py (which must
test the variant the model launches). Env: ATTN_GQA, ATTN_GQA_FROM, ATTN_UNROLL, ATTN_BFLY, ATTN_PAGED, ATTN_ILP, ATTN_MT, ATTN_MT_FROM, ATTN_MT_MIN_KV (see gen_llm.py)."""
import os

UNROLL = int(os.environ.get("ATTN_UNROLL", "16"))   # keys per decode-attention block (gen_llm.py)
ATTN_BFLY = os.environ.get("ATTN_BFLY", "1") == "1"   # decode attention: a block's key scores in one butterfly all-reduce
ATTN_PAGED = os.environ.get("ATTN_PAGED", "1") == "1"   # decode attention: 16-aligned key blocks, one page lookup each
ATTN_ILP = int(os.environ.get("ATTN_ILP", "1"))   # independent softmax chains per block (2, 4: no gain on Qwen3.8-27B at 32k)
# Decode split kernel (non-drafting packages): K/V loaded once for the G query heads of a KV head (gen_llm.gqa_split); the
# per-head body stays for contexts under ATTN_GQA_FROM visible keys. Nodraft Qwen3-4B (kv8, d128), same runner: decode_32k
# 24.5 -> 28.3 (+15 %), decode/2k/4 streams within noise. ATTN_GQA=0 keeps the per-head kernel.
ATTN_GQA = os.environ.get("ATTN_GQA", "1") == "1"
ATTN_GQA_FROM = int(os.environ.get("ATTN_GQA_FROM", "4096"))
# Drafting packages (verify spans, ctx-gated and plain epochs): the K/V-once body also serves the tokens that do not take the
# multi-token path (1-token epochs, the tail of a span, every token when MT is off) past ATTN_GQA_FROM keys. ATTN_GQA_DRAFT=0: nodraft only.
ATTN_GQA_DRAFT = os.environ.get("ATTN_GQA_DRAFT", "1") == "1"
# Drafting packages, head_dim 128, more than 2 KV heads: tokens at >= ATTN_MMA_FROM visible keys on full-attention layers take
# g_attention_split_mma (gen_attn_mma.py: K/V read once per KV head for a whole verify span x G heads, QK^T and PV on WMMA).
ATTN_MMA = os.environ.get("ATTN_MMA", "1") == "1"
ATTN_MMA_FROM = int(os.environ.get("ATTN_MMA_FROM", "4096"))
ATTN_MT = int(os.environ.get("ATTN_MT", "4"))   # verify-span tokens per wave (gen_llm.multi_token_split)
ATTN_MT_FROM = int(os.environ.get("ATTN_MT_FROM", "1024"))   # ... from this position on
# The span path leaves one workgroup per (4 tokens, KV head, split), each running its tokens serially: with fewer KV
# heads the grid is too small to hide that latency. Same-runner decode_32k, path on vs off (ngram depth 2-3 unless noted):
# 8 KV heads +8 % (Llama-3.2-3B), +21 % (Qwen3-4B DFlash), 2x (Qwen3-8B); 4 KV heads 0 % (Qwen2.5-7B), -15 % (Flint DFlash);
# 2 KV heads -6 % (Qwen2.5-1.5B) to -27 % (Qwen2.5-0.5B, head_dim 64); 1 KV head -17 % decode_2k (gemma-3-1b).
GQA_U = int(os.environ.get("ATTN_GQA_U", "8"))   # keys per block of the K/V-once body (decode_32k, same runner, nodraft: Qwen3-30B-A3B kv4 35.2 -> 36.5 (+3.7 %), Qwen3-8B kv8 22.9 -> 23.2 (+1.5 %) vs 16)
ATTN_MT_MIN_KV = int(os.environ.get("ATTN_MT_MIN_KV", "8"))   # KV heads from which the span path is taken


def paged(U):
    return ATTN_PAGED and 16 % U == 0


def butterfly(U):
    return ATTN_BFLY and U & (U - 1) == 0 and 2 <= U <= 32


def ilp(U):
    return ATTN_ILP if U % ATTN_ILP == 0 else 1


def gqa(draft_on, seg):
    """Does the decode split kernel carry the K/V-once body? Drafting packages: not with the segment (KVH <= 2) path."""
    return ATTN_GQA and (not draft_on or (ATTN_GQA_DRAFT and not seg and not ATTN_MMA))


def mma_tokens(draft_depth, d, kv_heads, g):
    """Run length of g_attention_split_mma (gen_attn_mma.py) for a drafting build (draft_depth > 0): the verify span, at most 8
    tokens, at most 32 query columns (tokens x G heads); 0 when the scalar split kernel keeps every token (head_dim != 128, or
    at most 2 KV heads: the segment path of gen_llm.seg_tokens, or fewer than 2 tokens)."""
    if not (ATTN_MMA and draft_depth and d == 128 and kv_heads > 2):
        return 0
    t = min(draft_depth + 1, 8, 32 // g)
    return t if t >= 2 else 0


def multi_token(U, kv_heads):
    """Does the decode attention with U keys per block and `kv_heads` KV heads take the multi-token path?"""
    return 1 < ATTN_MT <= U and kv_heads >= ATTN_MT_MIN_KV and paged(U) and butterfly(U) and ilp(U) == 1
