#!/usr/bin/env python3
"""gufo_exact.py URL OUT.json TEXT: greedy completions with top-5 logprobs for fixed prompts (short, ~2k, ~8k tokens of
WikiText), saved verbatim (content + logprobs) so two builds can be compared byte for byte."""
import json
import sys
import urllib.request

url, out, text = sys.argv[1], sys.argv[2], sys.argv[3]
wiki = open(text, encoding="utf-8").read()
prompts = [
    "What is the capital of France? Answer in one word.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain what a hash map is in two sentences.",
    "Summarize this text in three sentences:\n\n" + wiki[:9000],
    "Summarize this text in three sentences:\n\n" + wiki[20000:56000],
]
res = []
for p in prompts:
    body = {"model": "gufo-ab", "messages": [{"role": "user", "content": p}], "temperature": 0,
            "max_tokens": 48, "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    c = r["choices"][0]
    res.append({"prompt_tokens": r.get("usage", {}).get("prompt_tokens"), "content": c["message"].get("content"),
                "logprobs": c.get("logprobs")})
    print(res[-1]["prompt_tokens"], repr((res[-1]["content"] or "")[:60]), flush=True)
json.dump(res, open(out, "w"), indent=1)
