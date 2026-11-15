#!/usr/bin/env python3
"""Bench files of the gemm_q38 kbench suite: the prefill GEMMs of Qwen3.8-27B (hidden 5120, ffn 17408, attention q/gate/qkv/out, ssm out) on the R9700,
as the model builds them (gfx12 twins, GEMM_SKIP_PADDING and GEMM_N_FASTEST on), at 1024 rows (the chunk today: m128 and m256 kernels) and 2048 rows
(m256: a 2048 chunk); variants as gen_gemm_bench (base, nodq, nowst, ...).

usage: gen_q38.py ROOT OUTDIR [VARIANT ...]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-port-llm", "starter"))
sys.path.insert(0, HERE)
import gen_gemm_bench as B

SHAPES = [  # (fmt, mode, K, N): the model's GEMMs with their weight formats (Q4_K_M mixes q4_k / q6_k / q8_0 per tensor)
    ("q4_k", "swiglu", 5120, 17408),   # ffn gate + up
    ("q4_k", "store", 17408, 5120),    # ffn down
    ("q4_k", "store", 5120, 10240),    # gdn qkv
    ("q4_k", "store", 5120, 12288),    # attention q (+ gate)
    ("q4_k", "store", 5120, 6144),     # attn_gate / gdn z
    ("q6_k", "store", 6144, 5120),     # ssm_out
    ("q6_k", "store", 5120, 10240),    # gdn qkv (q6_k layers)
    ("q8_0", "store", 6144, 5120),     # attn_output / ssm_out (q8_0 layers)
]
B.NO_EXTRA = True
B.ROOT = ROOT
B.SHAPES = [(f, m, 1024, K, N) for f, m, K, N in SHAPES]
B.ROWS = [(f, m, 2048, K, N) for f, m, K, N in SHAPES]
B.EXTRA = []
B.ALL_SHAPES = []
_build = B.build

def build(variant=""):
    out = _build(variant)
    return out
if len(sys.argv) > 2:
    d = sys.argv[2]
    os.makedirs(d, exist_ok=True)
    os.environ["KB_OUTDIR"] = d
    for v in (sys.argv[3:] or ["base"]):
        v = "" if v == "base" else v
        open(os.path.join(d, "bench%s.loom" % ("_" + v if v else "")), "w").write(B.HEAD + "\n".join(B.build(v)) + ("\n" + B.PEAK if not v else ""))
