#!/usr/bin/env python3
"""Interleaved A/B of two builds served by ONE loom-serve: the reliable way to decide on a shared GPU.

usage: ab_perf.py URL MODEL_A MODEL_B [--rounds 5] [--metrics decode,prefill_2k] [--log ab_log.jsonl] [--label NAME]
  metrics: decode (300-token essay, tok/s), decode_2k (256 tokens after a 2k doc),
           prefill_1k / prefill_2k / prefill_8k / prefill_16k / prefill_32k (prompt tok/s, max_tokens 4),
           c4 (4 concurrent 200-token streams, aggregate tok/s)
Serve both packages in one server (build each with its own --model-id, e.g. a small --context 8192
for A/B packages: same kernels, faster to serve):
  serve.sh ab PORT amdgpu MEM A/model.json B/model.json
Requests alternate A, B, B, A, ... for --rounds rounds (load swings hit both sides), every prompt
carries a nonce (no prefix cache). Prints per-metric medians, B/A, and a verdict per metric:
  B WINS / B LOSES when the medians differ by more than the noise band (3 %, or the spread of A's
  own samples if larger), else TIE. Appends one JSON line per run to --log.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loom-benchmark", "scripts"))
from bench_common import doc, stream  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--metrics", default="decode,prefill_2k")
ap.add_argument("--log", default="ab_log.jsonl")
ap.add_argument("--label", default="")
a = ap.parse_args()
DOCS = {"1k": 1000, "2k": 2000, "8k": 8000, "16k": 16000, "32k": 32000}


def measure(model, metric):
    n = f"[{time.time()} {random.random()}] "
    if metric == "decode":
        return stream(a.url, model, [{"role": "user", "content": n + "Write a detailed 300-word essay about the history of the printing press."}], 300)["decode_tps"]
    if metric.startswith("decode_"):
        d = doc(DOCS[metric.split("_")[1]])
        return stream(a.url, model, [{"role": "user", "content": n + "Write a long, detailed essay discussing this documentation.\n\n" + d}], 256)["decode_tps"]
    if metric.startswith("prefill_"):
        s = stream(a.url, model, [{"role": "user", "content": n + "Summarize this documentation in one sentence.\n\n" + doc(DOCS[metric.split("_")[1]])}], 4)
        return s["prompt_tokens"] / s["ttft"]
    if metric == "c4":
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(4) as ex:
            toks = list(ex.map(lambda i: stream(a.url, model, [{"role": "user", "content": n + f"Write a short story ({i}) about a lighthouse keeper, about 200 words."}], 200)["tokens"], range(4)))
        return sum(toks) / (time.perf_counter() - t0)
    raise SystemExit(f"unknown metric {metric}")


metrics = a.metrics.split(",")
for m in (a.a, a.b):  # warm-up: lazy weight load
    stream(a.url, m, [{"role": "user", "content": "Say hi."}], 8)
res = {m: {"A": [], "B": []} for m in metrics}
for r in range(a.rounds):
    order = [("A", a.a), ("B", a.b)] if r % 2 == 0 else [("B", a.b), ("A", a.a)]
    for metric in metrics:
        for side, model in order:
            res[metric][side].append(measure(model, metric))
    print(f"round {r + 1}: " + "  ".join(f"{m} A {res[m]['A'][-1]:.1f} B {res[m]['B'][-1]:.1f}" for m in metrics), flush=True)
out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": a.label, "A": a.a, "B": a.b, "rounds": a.rounds, "metrics": {}}
for m in metrics:
    A, B = res[m]["A"], res[m]["B"]
    ma, mb = statistics.median(A), statistics.median(B)
    band = max(0.03, (max(A) - min(A)) / ma / 2 if len(A) > 2 else 0.03)
    ratio = mb / ma
    verdict = "B WINS" if ratio > 1 + band else "B LOSES" if ratio < 1 - band else "TIE"
    out["metrics"][m] = {"A": round(ma, 1), "B": round(mb, 1), "ratio": round(ratio, 3), "band": round(band, 3), "verdict": verdict}
    print(f"{m:12} A {ma:9.1f}  B {mb:9.1f}  B/A {ratio:5.3f}  (noise ±{band:.0%})  {verdict}")
with open(a.log, "a") as f:
    f.write(json.dumps(out) + "\n")
