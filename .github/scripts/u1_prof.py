"""U1 experiment workload (sdk020 campaign, never merged): a fixed number of identical requests against one loom-serve, so a
profiled session's aggregate HAL statistics can be differenced between two request counts (N2 - N1 cancels load, warm-up
and first-use kernels).

usage: u1_prof.py URL MODEL WORKLOAD N --log FILE
  WORKLOAD p1k | p2k: perf.py's prefill_1k / prefill_2k prompt (fresh nonce each), max_tokens 4; logs prompt tokens, TTFT
           dec:        perf.py's decode essay (short prompt), max_tokens 256; logs tokens and decode tok/s
           asr:PCM:    /v1/audio/transcriptions of the 16 kHz mono PCM16 file PCM; logs the wall time per request
  One untimed warm-up request of the same shape runs first (also in the profile: the difference N2 - N1 removes it).
"""
import argparse
import json
import os
import random
import statistics as st
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "skills", "loom-benchmark", "scripts"))
from bench_common import doc, stream, thinking_extra  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("workload")
ap.add_argument("n", type=int)
ap.add_argument("--log", required=True)
a = ap.parse_args()
TH = thinking_extra("off")
ESSAY = "Write a detailed 300-word essay about the history of the printing press."


def chat(content, max_tokens):
    s = stream(a.url, a.model, [{"role": "user", "content": content}], max_tokens, extra=TH, timeout=600)
    if s["status"] != 200:
        raise SystemExit(f"HTTP {s['status']}: {s.get('error')}")
    return s


def asr(path):
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        r = requests.post(a.url + "/v1/audio/transcriptions", data={"model": a.model}, timeout=120,
                          files={"file": ("recording.pcm", f, "audio/pcm;rate=16000;channels=1")})  # no Expect header: an empty one makes loom-serve drop the connection
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        raise SystemExit(f"HTTP {r.status_code}: {r.text[:300]}")
    return dt, r.json().get("text", "")


def one(i):
    tag = f"[{time.time()} {i} {random.random()}] "
    if a.workload in ("p1k", "p2k"):
        n = 1000 if a.workload == "p1k" else 2000
        s = chat(tag + "Summarize this documentation in one sentence.\n\n" + doc(n), 4)
        return {"prompt_tokens": s["prompt_tokens"], "ttft": s["ttft"], "prefill_tps": s["prompt_tokens"] / s["ttft"], "total": s["total"]}
    if a.workload == "dec":
        s = chat(tag + ESSAY, 256)
        return {"tokens": s["tokens"], "prompt_tokens": s["prompt_tokens"], "ttft": s["ttft"], "decode_tps": s["decode_tps"], "total": s["total"]}
    if a.workload.startswith("asr:"):
        dt, text = asr(a.workload[4:])
        return {"total": dt, "text": text[:120]}
    raise SystemExit(f"unknown workload {a.workload}")


warm = one("warm")
rows = [one(i) for i in range(a.n)]
summ = {"workload": a.workload, "n": a.n, "warmup": warm, "rows": rows}
for k in ("prefill_tps", "decode_tps", "total", "ttft"):
    v = [r[k] for r in rows if r.get(k)]
    if v:
        summ[f"median_{k}"] = st.median(v)
summ["tokens_sum"] = sum(r.get("tokens", 0) for r in rows) + warm.get("tokens", 0)
summ["prompt_tokens_sum"] = sum(r.get("prompt_tokens", 0) for r in rows) + warm.get("prompt_tokens", 0)
json.dump(summ, open(a.log, "w"), indent=1)
print("U1", a.workload, a.n, {k: round(v, 4) for k, v in summ.items() if k.startswith("median_")}, flush=True)
