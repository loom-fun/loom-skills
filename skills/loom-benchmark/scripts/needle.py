"""Full-context gate: needle-in-a-haystack retrieval up to the package's context capacity.

An "access code" sentence (at a depth that varies with the size and probe) is hidden in a real-text haystack (the skill's docs, then the local
Python stdlib sources) and the model is asked to list them. Sizes default to 25/50/75/100% of the package's context_capacity (minus the answer
budget). It also checks that a prompt over the capacity is refused cleanly (HTTP 400)
and that the server is still healthy afterwards.

usage: needle.py PKG LOOM_URL [--sizes N ...] [--probes 2] [--llama URL] [--llama-max 65536] [--max-tokens 1024] [--out FILE]
  default sizes: 50 % (2 probes) and 100 % (1 probe, --top-probes) of context_capacity; a full-context
  prefill can take many minutes (a 261k prompt: ~11 min for a 0.8B hybrid on a busy Strix Halo)
  --llama also runs the identical prompt ids through llama.cpp /completion for comparison
  (start it with a single slot whose context covers the largest size).
Each size runs --probes haystacks (different needles and depths). Retrieval itself is NOT the
gate: a small model finds or misses a needle on a near-tie first token (Qwen3.5-0.8B at 130k:
llama.cpp's top two first tokens 0.013 nats apart), so found/missed is reported but only
decides a probe when it agrees. A probe passes when the request succeeds with a non-empty
answer and, for sizes up to --llama-max with --llama, Loom found the needle, or llama.cpp
missed it too, or Loom's first divergence from llama.cpp on the same prompt ids is a near-tie
(the compare rule, bench_common.near_tie). Beyond --llama-max
(a llama.cpp prefill there takes many minutes) a probe proves that the server serves the
size. Exit status 1 on a failed probe or a failed over-limit check. Answers go to --out.
"""
import argparse
import glob
import json
import os
import random
import sys
import sysconfig
import time

import requests

from bench_common import Package, doc, llama_greedy, near_tie, stream, top_logprobs

QUESTIONS = {
    1: ("\n\n----\nSomewhere in the material above there is exactly one sentence of the form "
        "'The access code for vault <NAME> is <CODE>.' What is the vault's name and code? Answer as NAME: CODE."),
}


def corpus():
    parts = [doc(n) for n in (32000, 16000, 8000)]
    stdlib = sysconfig.get_paths()["stdlib"]
    for f in sorted(glob.glob(os.path.join(stdlib, "*.py"))):
        parts.append(f"\n\n# ===== file: {os.path.basename(f)} =====\n" + open(f, encoding="utf-8", errors="replace").read())
    return "".join(parts)


def haystack(P, ids, n, seed, count=1):
    rng = random.Random(seed)
    body = P.tok.decode(ids[:n])
    names = rng.sample(["ORCHID", "TUNDRA", "MERIDIAN", "BASALT", "KESTREL", "QUARTZ", "LANTERN", "SAFFRON"], count)
    codes = [f"{rng.randint(100, 999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(10, 99)}" for _ in names]
    depths = (rng.choice((0.1, 0.3, 0.5, 0.7, 0.9)),)
    for depth, name, code in sorted(zip(depths, names, codes), reverse=True):
        cut = body.rfind("\n", 0, int(len(body) * depth)) + 1
        body = body[:cut] + f"The access code for vault {name} is {code}.\n" + body[cut:]
    return body + QUESTIONS[count], dict(zip(names, codes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("url")
    ap.add_argument("--sizes", type=int, nargs="*")
    ap.add_argument("--llama", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--probes", type=int, default=2)
    ap.add_argument("--top-probes", type=int, default=1, help="probes at the largest size (a full-context prefill is the slow part)")
    ap.add_argument("--llama-max", type=int, default=65536, help="compare with llama.cpp only up to this many tokens")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = Package(a.pkg)
    card = json.load(open(os.path.join(a.pkg, "model.json")))
    context = next(iter(card["presets"].values()))["execution"]["context_capacity"]
    # Thinking off (when the template allows it) keeps answers short and deterministic.
    thinking = False if P.chat.get("thinking") == "toggle" else None
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if thinking is False else {}
    overhead = len(P.ids(P.render_messages([{"role": "user", "content": QUESTIONS[1]}], None, thinking)))
    top = context - a.max_tokens - overhead - 64
    sizes = a.sizes or [top // 2, top]
    ids = P.ids(corpus())
    while len(ids) < top:
        ids = ids + ids  # small stdlib: repeat (needles stay unique)
    print(f"context {context}; haystack sizes {sizes}", flush=True)
    failures = 0
    results = []
    loom_total = probes_total = 0
    for n in sizes:
        for probe in range(a.top_probes if n == max(sizes) and len(sizes) > 1 else a.probes):
            prompt, needles = haystack(P, ids, n, n * 31 + probe)
            rec = {"n": n, "probe": probe}
            r = stream(a.url, P.model_id, [{"role": "user", "content": prompt}], a.max_tokens, extra, timeout=7200)
            text = r["text"]
            rec.update(prompt_tokens=r["prompt_tokens"] or None, status=r["status"], ttft_s=r["ttft"],
                       finish=r.get("finish"), found=sum(v in text for v in needles.values()), error=r.get("error"),
                       needles=needles, answer=text[-400:])
            ok = r["status"] == 200 and bool(text.strip())
            if ok and a.llama and n <= a.llama_max:
                pids = P.ids(P.render_messages([{"role": "user", "content": prompt}], None, thinking))
                t0 = time.time()
                g, gids, cp = llama_greedy(a.llama, pids, a.max_tokens, P)
                lids = P.ids(text)
                k = next((i for i, (x, y) in enumerate(zip(lids, gids)) if x != y), None)
                gap = None
                if k is not None:  # past the aligned top-5 lists (a byte-split token): a near-tie, as in compare.py
                    top = top_logprobs(cp, k)
                    gap = 0.0 if k >= len(cp) else round(top[gids[k]] - top[lids[k]], 3) if lids[k] in top else "not in top-5"
                rec.update(llama_found=sum(v in g.get("content", "") for v in needles.values()), llama_s=round(time.time() - t0, 1),
                           diverge_at=k, gap=gap, llama_answer=g.get("content", "")[-400:])
                near = k is None or k >= len(cp) or near_tie(top, gids[k], lids[k])
                ok = rec["found"] >= rec["llama_found"] or near
            rec["pass"] = ok
            failures += not ok
            loom_total += rec["found"]; probes_total += 1
            print(("OK  " if ok else "FAIL"), json.dumps({k: v for k, v in rec.items() if "answer" not in k}), flush=True)
            results.append(rec)
    print(f"needles found (informational): Loom {loom_total}/{probes_total}")
    # Over the capacity: a clean refusal, then the server still answers.
    prompt, _ = haystack(P, ids * 2, context + 1000, 1)
    r = requests.post(a.url + "/v1/chat/completions", timeout=600, json={
        "model": P.model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, **extra})
    healthy = requests.post(a.url + "/v1/chat/completions", timeout=600, json={
        "model": P.model_id, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 64, **extra}).status_code == 200
    over_ok = r.status_code == 400 and healthy
    failures += not over_ok
    print(("OK  " if over_ok else "FAIL"), f"over-capacity prompt -> HTTP {r.status_code} {r.text[:160]!r}; healthy afterwards: {healthy}")
    if a.out:
        json.dump({"context": context, "results": results, "over_capacity": {"status": r.status_code, "healthy": healthy}},
                  open(a.out, "w"), indent=1)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
