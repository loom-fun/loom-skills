"""Interleaved A/B of short-prompt decode on one loom-serve that serves several packages (one model id each):
perf.py's decode request (300-token essay from a one-sentence prompt, greedy, the template's default thinking
unless --think), ROUNDS rounds in alternating order. Per run: decode tok/s (perf.py's formula), streamed events
(= epochs), tokens per event, ms per event, the draft counters' change in /metrics (proposed / accepted) and a
hash of the text, so every variant can be checked to decode the same output.

usage: python ab_decode.py URL MODEL [MODEL...] [--rounds 3] [--think on|off] [--log ab.jsonl]
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "skills", "loom-benchmark", "scripts"))
from bench_common import stream  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("models", nargs="+")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--think", choices=["on", "off"])
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--log")
a = ap.parse_args()
P = "Write a detailed 300-word essay about the history of the printing press."


def counters():
    try:
        text = urllib.request.urlopen(a.url + "/metrics", timeout=30).read().decode()
    except Exception:
        return {}
    v = {}
    for line in text.splitlines():
        for k in ("draft_proposed_tokens", "draft_accepted_inputs"):
            if f'field="{k}"' in line:
                v[k] = v.get(k, 0.0) + float(line.split()[-1])
    return v


def one(m):
    ev = []
    ex = {"chat_template_kwargs": {"enable_thinking": a.think == "on"}} if a.think else None
    c0 = counters()
    s = stream(a.url, m, [{"role": "user", "content": P}], a.max_tokens, extra=ex, on_token=ev.append)
    c1 = counters()
    n, e = s["tokens"], len(ev)
    d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
    return dict(tps=s["decode_tps"], ev=e, tpe=(n - 1) / max(1, e - 1), ms=(ev[-1] - ev[0]) * 1000 / max(1, e - 1), n=n,
                proposed=d.get("draft_proposed_tokens"), accepted=d.get("draft_accepted_inputs"),
                h=hashlib.sha1(s["text"].encode()).hexdigest()[:8])


for m in a.models:   # warm-up: weights load lazily on the first request
    stream(a.url, m, [{"role": "user", "content": "Say hi."}], 8)
res = {m: [] for m in a.models}
for r in range(a.rounds):
    for m in (a.models if r % 2 == 0 else a.models[::-1]):
        x = one(m)
        res[m].append(x)
        print(f"r{r} {m}: {x['tps']:.1f} tok/s  {x['ev']} ev  {x['tpe']:.2f} tok/ev  {x['ms']:.2f} ms/ev  n={x['n']} "
              f"proposed={x['proposed']} accepted={x['accepted']} h={x['h']}", flush=True)
base = None
for m in a.models:
    t = st.mean(x["tps"] for x in res[m])
    ms = st.median(x["ms"] for x in res[m])
    p = st.median(x["tpe"] for x in res[m])
    base = base or t
    hs = sorted({x["h"] for x in res[m]})
    print(f"SUMMARY {m}: {t:.1f} tok/s ({t / base:.3f}x)  {ms:.2f} ms/ev  {p:.2f} tok/ev  text {','.join(hs)}", flush=True)
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps({"model": m, "think": a.think, "tps": t, "ms": ms, "tpe": p, "texts": hs, "raw": res[m]}) + "\n")
