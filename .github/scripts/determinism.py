"""Run-to-run determinism of greedy decoding on a running loom-serve: the consistency gate's prompts, each
sent REPS times (alone, one at a time, or CONC at once), every output compared with the first one.
Records prompt_tokens_details.cached_tokens per request, so a divergence can be told apart by path
(full prefill vs prefix-cache hit) from a race (same path, different output).

usage: python determinism.py URL --model ID [--n 4] [--reps 10] [--conc 1] [--max-tokens 64] [--log FILE.jsonl]
Prints one line per prompt and DETERMINISTIC / NOT DETERMINISTIC; exit 1 when any output differs.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "skills", "loom-benchmark", "scripts"))
from bench_common import doc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--conc", type=int, default=1, help="requests in flight (1: each prompt alone)")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--pairs", action="store_true", help="alone, each prompt twice in a row (the second finds the first's retained "
                "history: a prefix-cache hit); outputs are grouped by cached token count")
ap.add_argument("--log", default=None)
a = ap.parse_args()
body = doc(1000)
prompts = [f"Question {i}: summarize part {i} of this text.\n\n" + body[i * 700:(i * 700) + 700 + 157 * i] for i in range(a.n)]


def run(job):
    rep, i = job
    r = requests.post(a.url + "/v1/chat/completions", json={"model": a.model, "messages": [{"role": "user", "content": prompts[i]}],
                                                             "max_tokens": a.max_tokens, "temperature": 0}, timeout=600).json()
    m, u = r["choices"][0]["message"], r.get("usage", {})
    return {"rep": rep, "prompt": i, "text": (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "prompt_tokens": u.get("prompt_tokens")}


jobs = [(rep, i) for rep in range(a.reps) for i in range(a.n)]
if a.pairs:
    a.conc = 1
    jobs = [(2 * rep + k, i) for rep in range(a.reps) for i in range(a.n) for k in (0, 1)]
with cf.ThreadPoolExecutor(a.conc) as ex:
    res = list(ex.map(run, jobs))
if a.log:
    with open(a.log, "a") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
bad = 0
for i in range(a.n):
    rs = [r for r in res if r["prompt"] == i]
    ref = rs[0]["text"]
    diff = [r for r in rs if r["text"] != ref]
    bad += bool(diff)
    first = ""
    if diff:
        d = diff[0]["text"]
        k = next((j for j, (x, y) in enumerate(zip(ref, d)) if x != y), min(len(ref), len(d)))
        first = f" first differing rep {diff[0]['rep']} at char {k}: {ref[max(0, k - 20):k + 20]!r} vs {d[max(0, k - 20):k + 20]!r}"
    by = {}
    for r in rs:
        by.setdefault(r["cached"], set()).add(r["text"])
    paths = ", ".join(f"cached {c}: {len(rs_)} distinct" for c, rs_ in sorted(by.items()))
    same_path = all(len(x) == 1 for x in by.values())
    print(f"prompt {i} [{paths}; {'each path deterministic' if same_path else 'a path gave two outputs'}"
          f"{'; paths differ' if len({t for x in by.values() for t in x}) > 1 and same_path else ''}]")
    print(f"prompt {i} ({rs[0]['prompt_tokens']} tokens): {len(rs) - len(diff)}/{len(rs)} identical to rep 0, "
          f"{len({r['text'] for r in rs})} distinct outputs, cached tokens {[r['cached'] for r in rs]}{first}")
print(("NOT DETERMINISTIC" if bad else "DETERMINISTIC") + f": {bad}/{a.n} prompts gave more than one output over {a.reps} reps (conc {a.conc})")
sys.exit(1 if bad else 0)
