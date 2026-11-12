"""Occupancy transitions: four greedy requests of different lengths start together, so the server runs 4, 3, 2, then 1
streams (gated -> ungated drafting). Saves the texts; with --ref compares them to another variant's file (token-exact
expected when both paths are exact; a divergence is reported with its first differing character for near-tie review).

usage: python transition.py URL --model ID --out FILE [--ref FILE]"""
import argparse
import concurrent.futures as cf
import json
import requests

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ref", default=None)
a = ap.parse_args()
TOPICS = ["the history of the printing press", "how a transistor works", "the water cycle and its effect on climate",
          "the rules and strategy of chess"]
LENS = [30, 120, 260, 480]


def run(i):
    p = f"[transition {i}] Write a long, detailed essay about {TOPICS[i]}."
    r = requests.post(a.url + "/v1/chat/completions", headers={"Content-Type": "application/json"}, timeout=600,
                      json={"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": LENS[i], "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False}}).json()
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")


with cf.ThreadPoolExecutor(4) as ex:
    texts = list(ex.map(run, range(4)))
json.dump(texts, open(a.out, "w"))
print("lengths", [len(t) for t in texts])
if a.ref:
    ref = json.load(open(a.ref))
    same = [x == y for x, y in zip(texts, ref)]
    first = [next((k for k, (c, d) in enumerate(zip(x, y)) if c != d), None if x == y else min(len(x), len(y))) for x, y in zip(texts, ref)]
    print("identical to ref:", same, "first differing char:", first)
    print("TRANSITION", "EXACT" if all(same) else "DIVERGES")
