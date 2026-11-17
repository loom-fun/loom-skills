#!/usr/bin/env python3
"""The requests of one profiled serve session (devlab_perf.sh step profile:V): the server runs with --print_device_statistics,
so its per-kernel statistics cover exactly this script's requests (plus one tiny warm-up, which loads weights and runs first-use work).

usage: profile_requests.py URL MODEL SCENARIO [--reps K] [--depth D] [--tokens N]
  probe   one tiny request (is the statistics output there at all?)
  pf2k / pf8k   K fresh-nonce prompts of ~2000 / ~8000 tokens, max_tokens 2 (prefill only)
  dec     one essay request (short prompt), N (default 400) tokens: a drafted decode, the counters give its verify steps
Prints one JSON line: scenario, requests (prompt/completion tokens, ttft, decode tok/s), draft counter deltas."""
import argparse
import json
import time

from bench_common import doc, server_draft_counters, stream, thinking_extra

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("scenario", choices=["probe", "pf2k", "pf8k", "dec"])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--tokens", type=int, default=400)
a = ap.parse_args()
TH = thinking_extra("off")
nonce = str(time.time())


def go(content, n):
    s = stream(a.url, a.model, [{"role": "user", "content": content}], n, extra=TH)
    if s["status"] != 200:
        raise SystemExit(f"HTTP {s['status']}: {s.get('error')}")
    return {"prompt_tokens": s["prompt_tokens"], "completion_tokens": s["tokens"], "ttft_s": round(s["ttft"], 4),
            "decode_tps": s["decode_tps"] and round(s["decode_tps"], 2), "total_s": round(s["total"], 3)}


out = {"scenario": a.scenario, "warmup": go(f"[warm-up {nonce}] Say hi.", 2)}
if a.scenario != "probe":
    before = server_draft_counters(a.url)
    reqs = []
    if a.scenario in ("pf2k", "pf8k"):
        n = 2000 if a.scenario == "pf2k" else 8000
        for i in range(a.reps):
            reqs.append(go(f"[{nonce} {i}] Summarize this documentation in one sentence.\n\n" + doc(n), 2))
    else:
        reqs.append(go(f"[{nonce}] Write a detailed 300-word essay about the history of the printing press.", a.tokens))
    after = server_draft_counters(a.url)
    out["requests"] = reqs
    out["draft"] = {k: after.get(k, 0) - before.get(k, 0) for k in after}
print(json.dumps(out), flush=True)
