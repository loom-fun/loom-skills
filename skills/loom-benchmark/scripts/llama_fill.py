#!/usr/bin/env python3
"""Answer deferred llama.cpp requests (one model at a time).

usage: llama_fill.py LLAMA_URL PENDING.jsonl [...]   (LLAMA_START, LLAMACPP_DIR, LLAMA_CACHE as for the gates)

gates.sh runs the gates with only Loom loaded and $LLAMA_DEFER set: every llama.cpp reference missing
from the cache is written to a pending file instead of being requested. With Loom stopped, this script
starts llama-server ($LLAMA_START, on the first request), fills the cache with every pending request
and returns; gates.sh then stops llama-server, restarts Loom and re-runs the gates from the cache.
A request with `"_server": NAME` (bench_common.SERVER_VARIANTS) is answered last, by a llama-server restarted
with those arguments ($LLAMA_STOP stops the first one; without it such requests are left unanswered).
"""
import json
import os
import subprocess
import sys

from bench_common import SERVER_VARIANTS, llama_completion
from shell_command import bash_command

url, files = sys.argv[1], sys.argv[2:]
os.environ.pop("LLAMA_DEFER", None)
todo = {}
for path in files:
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            todo.setdefault(r["key"], r)
queue = sorted(todo.values(), key=lambda r: r["body"].get("_server") or "")   # the default server's requests first
started = None
for i, r in enumerate(queue):
    variant = r["body"].get("_server")
    if variant != started and variant:
        if not os.environ.get("LLAMA_STOP"):
            print(f"[fill {i + 1}/{len(todo)}] left for a server started as {variant!r}: $LLAMA_STOP is not set", flush=True)
            continue
        subprocess.run(bash_command(os.environ["LLAMA_STOP"]), check=True)
        os.environ.update(LLAMA_FILL_VARIANT=variant, LLAMA_FILL_SPEC=SERVER_VARIANTS[variant])
        started = variant
    os.environ["LLAMA_GGUF"] = r["gguf"] or ""
    out = llama_completion(url, r["body"], None, r["salt"])
    n = len(out.get("tokens") or out.get("completion_probabilities") or [])
    print(f"[fill {i + 1}/{len(todo)}] {len(r['body'].get('prompt', []))} prompt tokens -> {n} tokens"
          + (f" ({out['_suspect']})" if out.get("_suspect") else ""), flush=True)
print(f"filled {len(todo)} llama.cpp references")
