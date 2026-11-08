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
import json
import os
import sys

import requests

from bench_common import doc, kept_runs, loom_runs_key

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
    return {"text": (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""), "prompt_tokens": r["usage"]["prompt_tokens"],
            "cached": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0), "message": m,
            "stop": r["choices"][0]["finish_reason"] == "stop"}


# gates.sh defers llama.cpp references (LLAMA_DEFER: requested with Loom stopped, then the gate re-runs). Concurrent
# output depends on scheduling, so a re-run would diverge elsewhere and need new references every round: the Loom
# measurements of a run whose references are pending are kept (bench_common.loom_runs_key: the package, the Loom SDK
# and the prompts) and judged on the re-run, then dropped; gates.sh clears kept runs when it starts.
state = (os.environ.get("LLAMA_DEFER") or "") and os.environ["LLAMA_DEFER"] + ".loom.json"
key = loom_runs_key(a.pkg or "-", a.model, prompts)
kept = kept_runs(state, key)
if kept:
    alone, together, solo = kept["alone"], kept["together"], kept["solo"]
    print("judging the Loom runs kept from the previous round (their llama.cpp references were pending)")
else:
    kept = None
    alone = [run(p) for p in prompts]
    with cf.ThreadPoolExecutor(a.n) as ex:
        together = list(ex.map(run, prompts))
# Run-to-run determinism alone (a kernel race, e.g. a router reading back its own writes across a
# barrier, shows up here before it shows up anywhere else). Only runs that took the same path are
# compared: a prefix-cache hit (cached_tokens > 0, e.g. this prompt's history retained from the
# concurrent phase) recomputes just the uncached tail, through the decode-shaped kernels (int8 matvec,
# decode attention) instead of the prefill GEMM, so a hit and a miss round differently and may flip a
# near-tie, deterministically (Bolt-E4B, issue #39: alone miss vs again hit on 357 of 361 tokens, gap
# 0.141). A prompt whose two alone runs took different paths runs alone again (up to twice) until two
# runs share a path; a hit-vs-miss difference is then judged like a concurrent one (near-tie rule).
if not kept:
    again = [run(p) for p in prompts]
    solo = [[x, y] for x, y in zip(alone, again)]
    for i, p in enumerate(prompts):
        while len({r["cached"] for r in solo[i]}) == len(solo[i]) and len(solo[i]) < 4:
            solo[i].append(run(p))
paths = [{} for _ in prompts]
for i, rs in enumerate(solo):
    for r in rs:
        paths[i].setdefault(r["cached"], []).append(r)
paired = [any(len(g) > 1 for g in ps.values()) for ps in paths]
repeat = [all(len({r["text"] for r in g}) == 1 for g in ps.values()) for ps in paths]
concurrent_same = [x["text"] == y["text"] for x, y in zip(alone, together)]
# divergences that must be near-ties: concurrent vs alone, and an alone run on another path vs the first alone run
pairs = [(i, "concurrent", together[i]) for i in range(a.n) if not concurrent_same[i]]
pairs += [(i, f"alone with {r['cached']} cached tokens", r) for i in range(a.n) for r in solo[i][1:]
          if r["cached"] != alone[i]["cached"] and r["text"] != alone[i]["text"] and repeat[i]]
pairs = list({(i, r["text"]): (i, what, r) for i, what, r in pairs}.values())
same = [concurrent_same[i] and repeat[i] and not any(j == i for j, *_ in pairs) for i in range(a.n)]
print("prompt tokens", [x["prompt_tokens"] for x in alone], "identical when concurrent:", concurrent_same,
      "identical when repeated alone (same path):", repeat,
      "cached tokens (alone, concurrent, again...):", [[alone[i]["cached"], together[i]["cached"]] + [r["cached"] for r in solo[i][1:]] for i in range(a.n)])
if not all(repeat):
    print("NOT DETERMINISTIC: the same prompt alone, on the same path (equal cached tokens), gave different greedy output "
          "(a race in a kernel, not a near-tie)")
if not all(paired):
    print("note: prompts", [i for i, ok in enumerate(paired) if not ok], "never ran twice alone on the same path (determinism unchecked)")
gaps, bars, pending = [], [], False
if a.pkg and pairs:
    from bench_common import Package, exact_gaps, gap_noise, llama_greedy, llama_views, top_logprobs
    P = Package(a.pkg)
    for i, what, r in pairs:
        ref = alone[i]
        t1, t2 = P.ids(P.raw_from_chat(ref["message"], ref["stop"])), P.ids(P.raw_from_chat(r["message"], r["stop"]))
        k = next((j for j, (x, y) in enumerate(zip(t1, t2)) if x != y), min(len(t1), len(t2)))
        gap_val, bar = None, 1.0
        msg = (f"prompt {i}: first divergence at generated token {k}: alone ({ref['cached']} cached) {P.tok.decode(t1[k:k + 1])!r} "
               f"vs {what} {P.tok.decode(t2[k:k + 1])!r}")
        if a.llama and k < min(len(t1), len(t2)):
            ids = P.ids(P.render(prompts[i])) + t1[:k]
            g, _, cp = llama_greedy(a.llama, ids, 1, P, n_probs=10)
            top = top_logprobs(cp, 0)
            if g.get("_pending"):
                pending = True
                msg += "; llama.cpp reference pending"
            elif not top:
                # No reference to judge the divergence by (an error response, or no probabilities for a token that ends
                # inside a UTF-8 character): not a near-tie
                why = f"error {g.get('error') or g.get('_suspect')!r}" if g.get("error") or g.get("_suspect") or not g.get("tokens") \
                    else "no top-k list for a byte-fallback token"
                msg += f"; llama.cpp gave no reference at this position ({why}): not judged a near-tie"
            elif t1[k] in top and t2[k] in top:
                gap_val = abs(top[t1[k]] - top[t2[k]])
                msg += f"; llama.cpp log-prob gap between them {gap_val:.3f}"
                # The same bar as the text gates (near_tie): llama.cpp's own other numerics at this position (prefill batch,
                # decode-shaped micro-batches) may rank the pair closer; its spread across them widens the 1-nat bar, capped.
                hi, lo = (t1[k], t2[k]) if top[t1[k]] >= top[t2[k]] else (t2[k], t1[k])
                views = llama_views(a.llama, ids, P)
                noise = gap_noise((top, *views), hi, lo)
                gap_val = min(exact_gaps((top, *views), hi, lo))
                msg += f"; over llama.cpp's own views {gap_val:.3f} (noise {noise:.3f})"
                bar = 1.0 + noise
            else:
                msg += "; one of them is outside llama.cpp's top-10 (not a near-tie)"
        print(msg)
        gaps.append(gap_val)
        bars.append(bar)
# A concurrent-vs-alone (or prefix-hit-vs-miss) difference passes when every prompt is deterministic
# alone on its path and every divergence is a near-tie by the compare.py rule (gap <= 1 nat plus llama.cpp's view spread, both
# tokens in llama.cpp's top-k): decode tokens sharing an epoch with a prefill take the f16 GEMM path
# instead of the f32 matvec path, which flips near-ties (the identified cause; LFM2.5: gap 0.085).
if pending:
    json.dump({"key": key, "alone": alone, "together": together, "solo": solo}, open(state, "w"))
    print("PENDING: llama.cpp references deferred; the gate re-runs on these Loom runs once they are filled")
    sys.exit(0)
if state and os.path.exists(state):
    os.remove(state)
near = all(repeat) and bool(gaps) and all(g is not None and g <= b for g, b in zip(gaps, bars)) and len(gaps) == len(pairs)
if not all(same) and near:
    print("near-tie divergences only (mixed-epoch f16 path / prefix-cache tail), deterministic alone")
print("PASS" if all(same) or near else "FAIL")
from campaign import log_gate, run_audit
log_gate("consistency", "PASS" if all(same) or near else "FAIL", identical=same, near_tie_only=bool(near and not all(same)))
run_audit()
