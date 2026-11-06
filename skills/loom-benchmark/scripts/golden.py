"""Golden reference: greedy llama.cpp generations (token ids + top-5 log-probs)
for the short prompt set and (optionally) long real-text prompts.

usage: python golden.py PACKAGE OUT.json [--llama http://127.0.0.1:8130] [--tokens 64] [--long 1000,2000,16000,32000] [--short N]

llama.cpp gets the *rendered prompt as raw token ids* (/completion), so it sees
exactly the tokens Loom's control program produces. Run llama-server with the
same GGUF the Loom package uses. llama.cpp ROCm is not bit-deterministic across
runs at near-ties: record a second golden (OUT2.json) and pass it to compare.py
with --alt so a prompt where llama.cpp disagrees with itself is not called a FAIL.
"""
import argparse
import json

from bench_common import PROMPTS, Package, doc, llama_greedy

ap = argparse.ArgumentParser()
ap.add_argument("package")
ap.add_argument("out")
ap.add_argument("--llama", default="http://127.0.0.1:8130")
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--long", default="", help="comma-separated doc sizes for long-context goldens (64 tokens each)")
ap.add_argument("--short", type=int, default=None, help="only the first N short prompts (gates.sh LEVEL=dev: 12)")
ap.add_argument("--salt", default="", help="cache salt: a 2nd recording (golden2) needs its own (llama.cpp is not run-to-run deterministic)")
a = ap.parse_args()
pkg = Package(a.package)
items = [(p, a.tokens) for p in PROMPTS[:a.short]]
# One question per document (alternating the two kinds across sizes): each long prefill is the
# expensive part on both engines, and a second question on the same document tested the same path.
QS = ("Summarize this documentation in one sentence.", "What is the main command shown in this documentation?")
for k, n in enumerate(int(x) for x in a.long.split(",") if x):
    items.append((QS[k % 2] + "\n\n" + doc(n), 64))
out = []
for prompt, n_predict in items:
    ids = pkg.ids(pkg.render(prompt))
    r, gen, cp = llama_greedy(a.llama, ids, n_predict, pkg, a.salt)
    if r.get("_suspect"):  # llama.cpp answered from a stale state: a wrong golden is worse than none
        print(f"SKIP {len(ids)} prompt tokens: {r['_suspect']}", flush=True)
        continue
    if "tokens" not in r:  # e.g. the prompt exceeds the server's context: skip, don't crash
        print(f"SKIP {len(ids)} prompt tokens: {str(r)[:200]}", flush=True)
        continue
    out.append({"prompt": prompt, "prompt_tokens": len(ids), "ids": gen, "text": r["content"],
                "top": [[(t["id"], t["logprob"]) for t in x["top_logprobs"]] for x in cp]})
    print(f"{len(ids):6d} prompt tokens -> {len(gen):3d} {r['content'][:60]!r}", flush=True)
json.dump(out, open(a.out, "w"))
print("wrote", a.out)
