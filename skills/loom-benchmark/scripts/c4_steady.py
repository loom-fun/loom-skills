"""Steady-state aggregate decode throughput with N concurrent streams: tokens
produced by all streams inside the window where every stream is decoding
(excludes ramp-up and tail). Tokens, not streamed events: with speculative decoding one event can carry several.

usage: python c4_steady.py URL [N=4] [--model MODEL_ID]   (omit --model for llama.cpp)
Loom needs execution.rows >= N in the package preset; llama-server needs -np N.
"""
import argparse
import json
import threading
import time

from bench_common import stream, thinking_extra

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("n", type=int, nargs="?", default=4)
ap.add_argument("--model", default=None)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--label", help="build label: Loom runs log c4_steady to targets.jsonl (needs roofline.json); "
                                 "llama.cpp runs (no --model) append a kind=llama row to --log for roofline.py --baseline")
ap.add_argument("--log", default="perf_log.jsonl")
ap.add_argument("--reverted", action="store_true")
ap.add_argument("--thinking", default="off", choices=["off", "on", "template"],
                help="chat_template_kwargs.enable_thinking sent to the server (perf.py's flag; template: send nothing)")
a = ap.parse_args()
events = [[] for _ in range(a.n)]
tokens = [0] * a.n


def run(i):
    r = stream(a.url, a.model, [{"role": "user", "content": f"[{time.time()} {i}] Write a very long, detailed essay about the history of lighthouses."}],
               a.max_tokens, extra=thinking_extra(a.thinking), on_token=events[i].append)
    tokens[i] = r.get("tokens") or len(events[i])


stream(a.url, a.model, [{"role": "user", "content": "Say hi."}], 8)  # warm-up
ts = [threading.Thread(target=run, args=(i,)) for i in range(a.n)]
[t.start() for t in ts]
[t.join() for t in ts]
start = max(e[0] for e in events)
end = min(e[-1] for e in events)
# A streamed event can carry several tokens (a drafting server commits up to depth + 1 per step): count tokens, each
# stream's events weighted by its tokens per event (usage.completion_tokens / events), not events.
count = sum(sum(1 for x in e if start <= x <= end) * (tokens[i] / max(1, len(e))) for i, e in enumerate(events))
print(json.dumps({"streams": a.n, "thinking": a.thinking, "window_s": round(end - start, 2), "agg_tps": round(count / (end - start), 1),
                  "per_stream_events": [len(e) for e in events], "per_stream_tokens": tokens}))
if a.label and a.model:  # Loom run: track it in the optimization campaign
    from campaign import log_metrics
    log_metrics({"c4_steady": round(count / (end - start), 1)}, a.label, a.reverted)
elif a.label:  # llama.cpp run: record it as baseline data next to perf.py's rows
    with open(a.log, "a") as f:
        f.write(json.dumps({"kind": "llama", "label": a.label, "c4_steady": round(count / (end - start), 1)}) + "\n")
