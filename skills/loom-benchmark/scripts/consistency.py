"""Batching correctness: prompts run concurrently must give the same greedy
output as the same prompts run alone. Concurrent prefill packs several
sequences into one epoch, so 16-token attention tiles straddle sequences and
exercise the mixed/fallback paths that single requests never hit.

usage: python consistency.py URL --model MODEL_ID [--n 4] [--pkg PKG --llama URL]
  With --pkg (and --llama), a mismatch is diagnosed: the first divergent token and llama.cpp's
  log-prob gap between the two choices there (the near-tie evidence consistency_note.md needs).
"""
import argparse
import concurrent.futures as cf

import requests

from bench_common import doc

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--pkg", default=None)
ap.add_argument("--llama", default=None)
a = ap.parse_args()
body = doc(1000)
prompts = [f"Question {i}: summarize part {i} of this text.\n\n" + body[i * 700:(i * 700) + 700 + 157 * i] for i in range(a.n)]


def run(p):
    r = requests.post(a.url + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                      json={"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": 40, "temperature": 0}).json()
    m = r["choices"][0]["message"]
    run.raw[p] = run.raw.get(p, []) + [(m, r["choices"][0]["finish_reason"])]
    return (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""), r["usage"]["prompt_tokens"]


run.raw = {}


alone = [run(p) for p in prompts]
with cf.ThreadPoolExecutor(a.n) as ex:
    together = list(ex.map(run, prompts))
# Run-to-run determinism alone (a kernel race, e.g. a router reading back its own writes across a
# barrier, shows up here before it shows up anywhere else).
again = [run(p) for p in prompts]
repeat = [x[0] == y[0] for x, y in zip(alone, again)]
same = [x[0] == y[0] and r for x, y, r in zip(alone, together, repeat)]
print("prompt tokens", [x[1] for x in alone], "identical when concurrent:", [x[0] == y[0] for x, y in zip(alone, together)],
      "identical when repeated alone:", repeat)
if not all(repeat):
    print("NOT DETERMINISTIC: the same prompt alone gave different greedy output (a race in a kernel, not a near-tie)")
if a.pkg and not all(same):
    from bench_common import Package, llama_completion
    P = Package(a.pkg)
    for i, ok in enumerate(same):
        if ok:
            continue
        (m1, f1), (m2, f2) = run.raw[prompts[i]][:2]
        t1, t2 = P.ids(P.raw_from_chat(m1, f1 == "stop")), P.ids(P.raw_from_chat(m2, f2 == "stop"))
        k = next((j for j, (x, y) in enumerate(zip(t1, t2)) if x != y), min(len(t1), len(t2)))
        msg = f"prompt {i}: first divergence at generated token {k}: alone {P.tok.decode(t1[k:k + 1])!r} vs concurrent {P.tok.decode(t2[k:k + 1])!r}"
        if a.llama and k < min(len(t1), len(t2)):
            ids = P.ids(P.render(prompts[i])) + t1[:k]
            g = llama_completion(a.llama, {"prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": 10, "cache_prompt": False}, P)
            top = {t["id"]: t["logprob"] for t in g["completion_probabilities"][0]["top_logprobs"]}
            if t1[k] in top and t2[k] in top:
                msg += f"; llama.cpp log-prob gap between them {abs(top[t1[k]] - top[t2[k]]):.3f}"
            else:
                msg += "; one of them is outside llama.cpp's top-10 (not a near-tie)"
        print(msg)
print("PASS" if all(same) else "FAIL")
from campaign import log_gate, run_audit
log_gate("consistency", "PASS" if all(same) else "FAIL", identical=same)
run_audit()
