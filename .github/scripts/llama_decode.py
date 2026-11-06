"""perf.py's decode request on llama-server (non-streamed, like perf.py --kind llama), N times: decode tok/s from the
server's timings plus its draft counters, so tokens per step can be compared with Loom's (ab_decode.py).

usage: python llama_decode.py URL [N=2] [--think on|off] [--log FILE.jsonl]
"""
import argparse
import json

import requests

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("n", type=int, nargs="?", default=2)
ap.add_argument("--think", choices=["on", "off"])
ap.add_argument("--log")
a = ap.parse_args()
P = "Write a detailed 300-word essay about the history of the printing press."
ex = {"chat_template_kwargs": {"enable_thinking": a.think == "on"}} if a.think else {}
requests.post(a.url + "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0})
for i in range(a.n):
    d = requests.post(a.url + "/v1/chat/completions", json={"messages": [{"role": "user", "content": P}], "max_tokens": 300,
                                                           "temperature": 0, "cache_prompt": False, **ex}).json()
    t, m = d["timings"], d["choices"][0]["message"]
    n, dn, da = t["predicted_n"], t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    steps = max(1, n - da)
    r = {"think": a.think or "default", "prompt_n": t["prompt_n"], "predicted_n": n, "decode_tps": round(t["predicted_per_second"], 2),
         "draft_n": dn, "draft_n_accepted": da, "steps": steps, "tok_per_step": round(n / steps, 3),
         "ms_per_step": round(t["predicted_ms"] / steps, 2), "reasoning": (m.get("reasoning_content") or "")[:80],
         "content": (m.get("content") or "")[:80]}
    print(json.dumps(r), flush=True)
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(r) + "\n")
