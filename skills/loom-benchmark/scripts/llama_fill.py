#!/usr/bin/env python3
"""Answer deferred llama.cpp requests (one model at a time).

usage: llama_fill.py LLAMA_URL PENDING.jsonl [...]   (LLAMA_START, LLAMACPP_DIR, LLAMA_CACHE as for the gates)

gates.sh runs the gates with only Loom loaded and $LLAMA_DEFER set: every llama.cpp reference missing
from the cache is written to a pending file instead of being requested. With Loom stopped, this script
starts llama-server ($LLAMA_START, on the first request), fills the cache with every pending request
and returns; gates.sh then stops llama-server, restarts Loom and re-runs the gates from the cache.
"""
import json
import os
import sys

from bench_common import llama_completion

url, files = sys.argv[1], sys.argv[2:]
os.environ.pop("LLAMA_DEFER", None)
todo = {}
for path in files:
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            todo.setdefault(r["key"], r)
for i, r in enumerate(todo.values()):
    os.environ["LLAMA_GGUF"] = r["gguf"] or ""
    out = llama_completion(url, r["body"], None, r["salt"])
    n = len(out.get("tokens") or out.get("completion_probabilities") or [])
    print(f"[fill {i + 1}/{len(todo)}] {len(r['body'].get('prompt', []))} prompt tokens -> {n} tokens"
          + (f" ({out['_suspect']})" if out.get("_suspect") else ""), flush=True)
print(f"filled {len(todo)} llama.cpp references")
