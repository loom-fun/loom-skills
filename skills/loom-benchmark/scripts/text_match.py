"""Same workload check between engines: perf.py's prompts (the decode essay and the 2k-document request), greedy,
the first N generated tokens on Loom and on llama.cpp, compared token by token (Loom's tokenizer on both texts).
Run it once per engine (one server at a time); the second run compares.

usage: python text_match.py save URL OUT.json --model MODEL_ID --tokenizer tokenizer.json [--thinking off|on] [--n 50]
       python text_match.py compare URL OUT.json --tokenizer tokenizer.json [--thinking off|on] [--n 50]   (llama-server)
Prints per prompt the number of leading tokens that match (of N) and the first divergence. A divergence after a
long match is a near-tie (int8 vs f32 rounding); a divergence at token 0-2 means the engines decode different texts
(a template or thinking-setting mismatch).
"""
import argparse
import json

import requests
from tokenizers import Tokenizer

from bench_common import doc, stream, thinking_extra

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["save", "compare"])
ap.add_argument("url")
ap.add_argument("file")
ap.add_argument("--model")
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--thinking", default="off", choices=["off", "on", "template"])
ap.add_argument("--n", type=int, default=50)
a = ap.parse_args()
tok = Tokenizer.from_file(a.tokenizer)
PROMPTS = ["Write a detailed 300-word essay about the history of the printing press.",
           "Write a long, detailed essay discussing this documentation.\n\n" + doc(2000),
           "Explain how a refrigerator works, step by step."]
TH = thinking_extra(a.thinking)


def gen(p):
    msgs = [{"role": "user", "content": p}]
    if a.mode == "save":
        return stream(a.url, a.model, msgs, a.n + 8, extra=TH)["text"]
    d = requests.post(a.url + "/v1/chat/completions", json={"messages": msgs, "max_tokens": a.n + 8, "temperature": 0,
                                                           "cache_prompt": False, "logprobs": True, "top_logprobs": 10, **TH}).json()
    m = d["choices"][0]["message"]
    LP.append(((d["choices"][0].get("logprobs") or {}).get("content") or []))
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


LP = []
texts = [gen(p) for p in PROMPTS]
if a.mode == "save":
    json.dump({"thinking": a.thinking, "texts": texts}, open(a.file, "w"))
    print("saved", [t[:60] for t in texts])
else:
    ref = json.load(open(a.file))
    assert ref["thinking"] == a.thinking
    ok = 0
    for i, (x, y) in enumerate(zip(ref["texts"], texts)):
        tx, ty = tok.encode(x, add_special_tokens=False).ids[:a.n], tok.encode(y, add_special_tokens=False).ids[:a.n]
        k = next((j for j, (u, v) in enumerate(zip(tx, ty)) if u != v), min(len(tx), len(ty)))
        ok += k >= min(a.n, len(tx), len(ty)) or k >= 10
        print(f"prompt {i}: {k}/{min(a.n, len(tx), len(ty))} leading tokens match"
              + ("" if k >= min(len(tx), len(ty)) else f"; first divergence: Loom {tok.decode(tx[k:k + 3])!r} vs llama.cpp {tok.decode(ty[k:k + 3])!r}"))
        if k < min(len(tx), len(ty)):   # llama.cpp's log-probs at the divergence: a near-tie has both tokens in its top-10
            want, done = len(tok.decode(ty[:k])), ""
            for st in LP[i]:   # llama.cpp's step that produced its token k (step texts concatenate to its output)
                if len(done) >= want and st.get("token", "").strip():
                    top = {c["token"]: c["logprob"] for c in st.get("top_logprobs", [])}
                    lo, ll = tok.decode(tx[k:k + 1]), tok.decode(ty[k:k + 1])
                    print(f"  llama.cpp top-10 at the divergence ({st.get('token')!r}): Loom's {lo!r} {top.get(lo)}, "
                          f"llama.cpp's {ll!r} {top.get(ll)}" + (f", gap {abs(top[ll] - top[lo]):.3f} nats" if lo in top and ll in top else ""))
                    break
                if "<" not in st.get("token", ""):   # thinking-channel markers are not in the parsed text
                    done += st.get("token", "")
        print(f"  Loom:      {x[:100]!r}\n  llama.cpp: {y[:100]!r}")
    print(f"TEXT_MATCH thinking={a.thinking}: {ok}/{len(texts)} prompts on the same text (>= 10 leading tokens identical)")
