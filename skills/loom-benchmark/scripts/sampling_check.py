"""Sampling gate: temperature / top_p / top_k / seed behave as specified.

Checks, on one short prompt with thinking off when the template allows it:
  1. temperature 0 == a request without sampling fields (greedy), and top_k 1 == greedy;
  2. the same seed reproduces the same completion; different seeds vary;
  3. the empirical first-token distribution at temperature 1 (N draws, distinct seeds) matches
     llama.cpp's probabilities for the same prompt ids (total-variation distance), when --llama is given.

usage: sampling_check.py PKG LOOM_URL [--llama URL] [--draws 200]
"""
import argparse
import collections
import math
import sys

import requests

from bench_common import Package, llama_greedy

PROMPT = "Pick a random fruit: apple, banana, cherry or grape. Answer with just the fruit name in lowercase."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("url")
    ap.add_argument("--llama", default=None)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    P = Package(a.pkg)
    thinking = False if P.chat.get("thinking") == "toggle" else None
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if thinking is False else {}
    msgs = [{"role": "user", "content": PROMPT}]

    def ask(max_tokens=16, **kw):
        r = requests.post(a.url + "/v1/chat/completions", timeout=600,
                          json={"model": P.model_id, "messages": msgs, "max_tokens": max_tokens, **extra, **kw}).json()
        if "error" in r:
            raise SystemExit(f"Loom error: {r['error']}")
        m = r["choices"][0]["message"]
        return (m.get("reasoning_content") or "") + (m.get("content") or "")

    fails = 0
    greedy = ask()
    checks = [("temperature 0 == greedy", ask(temperature=0) == greedy),
              ("top_k 1 == greedy", ask(temperature=1.0, top_k=1, seed=7) == greedy)]
    same = [ask(temperature=1.0, seed=1234) for _ in range(2)]
    checks.append(("same seed reproduces", same[0] == same[1]))
    varied = {ask(temperature=1.5, seed=s) for s in range(12)}
    # A prompt this model answers almost deterministically says nothing about seeds (Muse-Glimmer:
    # llama.cpp itself gave 2 distinct of 12): skip the check when llama.cpp's top token is near-certain.
    top_p1 = None
    if a.llama:
        ids0 = P.ids(P.render_messages(msgs, None, thinking))
        g0 = llama_greedy(a.llama, ids0, 1, P, n_probs=20)[0]
        top_p1 = max(math.exp(t["logprob"]) for t in g0["completion_probabilities"][0]["top_logprobs"])
    if len(varied) > 1 or top_p1 is None or top_p1 < 0.9:
        checks.append((f"different seeds vary ({len(varied)} distinct of 12)", len(varied) > 1))
    else:
        print(f"SKIP different seeds vary ({len(varied)} distinct of 12): llama.cpp's top first token has p={top_p1:.3f}")
    for name, ok in checks:
        fails += not ok
        print(("OK  " if ok else "FAIL"), name)
    if a.llama:
        first = collections.Counter()
        for s in range(a.draws):
            t1 = P.ids(ask(max_tokens=1, temperature=1.0, top_p=1.0, top_k=0, seed=1000 + s))
            if not t1:   # the first token is structural (a channel header stripped from the text)
                if s >= 4:
                    break
                continue
            first[t1[0]] += 1
        if not first:
            print("SKIP first-token distribution: the first generated token is a structural channel header (not in the text)")
            sys.exit(1 if fails else 0)
        g = g0 if top_p1 is not None else llama_greedy(a.llama, P.ids(P.render_messages(msgs, None, thinking)), 1, P, n_probs=20)[0]
        probs = {t["id"]: math.exp(t["logprob"]) for t in g["completion_probabilities"][0]["top_logprobs"]}
        support = set(probs) | set(first)
        tv = 0.5 * sum(abs(probs.get(t, 0.0) - first[t] / a.draws) for t in support)
        # Sampling noise alone gives TV ~ sqrt(k / N); allow 3x that plus a small numerical margin.
        bound = 3 * math.sqrt(max(2, len([p for p in probs.values() if p > 0.01])) / a.draws) / 2 + 0.05
        ok = tv <= bound
        fails += not ok
        top = sorted(support, key=lambda t: -probs.get(t, 0))[:5]
        print(("OK  " if ok else "FAIL"), f"first-token distribution: total variation {tv:.3f} (bound {bound:.3f}); "
              + ", ".join(f"{P.tok.id_to_token(t)!r} llama {probs.get(t, 0):.2f} loom {first[t] / a.draws:.2f}" for t in top))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
