"""Decode/verify attention variants, shared by gen_llm.py (which emits them) and make_kernel_tests.py (which must
test the variant the model launches). Env: ATTN_UNROLL, ATTN_BFLY, ATTN_PAGED, ATTN_ILP, ATTN_MT, ATTN_MT_FROM (see gen_llm.py)."""
import os

UNROLL = int(os.environ.get("ATTN_UNROLL", "16"))   # keys per decode-attention block (gen_llm.py)
ATTN_BFLY = os.environ.get("ATTN_BFLY", "1") == "1"   # decode attention: a block's key scores in one butterfly all-reduce
ATTN_PAGED = os.environ.get("ATTN_PAGED", "1") == "1"   # decode attention: 16-aligned key blocks, one page lookup each
ATTN_ILP = int(os.environ.get("ATTN_ILP", "1"))   # independent softmax chains per block (2, 4: no gain on Qwen3.8-27B at 32k)
ATTN_MT = int(os.environ.get("ATTN_MT", "4"))   # verify-span tokens per wave (gen_llm.multi_token_split)
ATTN_MT_FROM = int(os.environ.get("ATTN_MT_FROM", "1024"))   # ... from this position on


def paged(U):
    return ATTN_PAGED and 16 % U == 0


def butterfly(U):
    return ATTN_BFLY and U & (U - 1) == 0 and 2 <= U <= 32


def ilp(U):
    return ATTN_ILP if U % ATTN_ILP == 0 else 1


def multi_token(U):
    """Does the decode attention with U keys per block take the multi-token path?"""
    return 1 < ATTN_MT <= U and paged(U) and butterfly(U) and ilp(U) == 1
