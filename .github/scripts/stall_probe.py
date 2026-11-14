#!/usr/bin/env python3
"""Diagnostic: when does a prefill request stall? Sends the perf.py documents in controlled orders against one loom-serve
and logs client TTFT, prompt tokens and (when readable) the GPU's current sclk/mclk/fclk states before each request.
usage: stall_probe.py URL MODEL OUT.jsonl"""
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "skills", "loom-benchmark", "scripts"))
from bench_common import doc, stream  # noqa: E402

url, model, outp = sys.argv[1:4]


def clocks():
    out = {}
    for d in glob.glob("/sys/class/drm/card*/device"):
        for f in ("pp_dpm_sclk", "pp_dpm_mclk", "pp_dpm_fclk", "gpu_busy_percent"):
            try:
                t = open(os.path.join(d, f)).read()
            except OSError:
                continue
            m = re.findall(r"^\d+: (\S+)\s*\*", t, re.M)
            out[f] = m[0] if m else t.strip()
        if out:
            break
    return out


nonce = [str(time.time())]
n_req = [0]


def req(tag, n, mx=4):
    n_req[0] += 1
    c = clocks()
    text = "Write a long, detailed essay discussing this documentation." if mx > 4 else "Summarize this documentation in one sentence."
    s = stream(url, model, [{"role": "user", "content": f"[{nonce[0]} {n_req[0]} {n}] {text}\n\n" + doc(n)}], mx)
    row = {"i": n_req[0], "tag": tag, "doc": n, "prompt_tokens": s["prompt_tokens"], "ttft": round(s["ttft"], 4) if s["ttft"] else None,
           "tps": round(s["prompt_tokens"] / s["ttft"]) if s["ttft"] else None, "total": round(s["total"], 3), "clocks": c, "t": round(time.time() % 100000, 1)}
    print(json.dumps(row), flush=True)
    open(outp, "a").write(json.dumps(row) + "\n")


req("warm", 1000)
for _ in range(4):
    req("8k-b2b", 8000)
time.sleep(5)
req("8k-after-5s-idle", 8000)
req("8k", 8000)
req("16k", 16000)
req("32k-decode", 32000, 128)
for _ in range(3):
    req("8k-after-32k", 8000)
req("16k", 16000)
req("32k", 32000)
req("8k", 8000)
time.sleep(15)
req("8k-after-15s-idle", 8000)
req("8k", 8000)
for _ in range(3):
    req("2k", 2000)
req("1k", 1000)
