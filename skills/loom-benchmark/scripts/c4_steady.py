"""Steady-state aggregate decode throughput with N concurrent streams: tokens
produced by all streams inside the window where every stream is decoding
(excludes ramp-up and tail). Tokens, not streamed events: with speculative decoding one event can carry several.
With a drafting Loom server the JSON line also gives "acceptance": accepted / proposed drafted tokens over the N
streams (loom-serve's /metrics counters around them), the roofline's acceptance for the 4-stream row.

usage: python c4_steady.py URL [N=4] [--model MODEL_ID]   (omit --model for llama.cpp)
Loom needs execution.rows >= N in the package preset; llama-server needs -np N.
A run in which a stream failed (an exception, an HTTP error, no tokens) or the N streams never decoded together
measured nothing: it exits 1 with the reason on stderr, prints no JSON line and logs nothing.
"""
import argparse
import json
import sys
import threading
import time

from bench_common import acceptance_between, server_draft_counters, stream, thinking_extra

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
errors = [None] * a.n


def run(i):
    try:
        r = stream(a.url, a.model, [{"role": "user", "content": f"[{time.time()} {i}] Write a very long, detailed essay about the history of lighthouses."}],
                   a.max_tokens, extra=thinking_extra(a.thinking), on_token=events[i].append)
    except Exception as e:  # noqa: BLE001  (e.g. requests' ChunkedEncodingError: the stream broke)
        errors[i] = f"{type(e).__name__}: {e}"[:300]
        return
    if r.get("status") not in (None, 200):
        errors[i] = f"HTTP {r['status']}: {r.get('error')}"
    tokens[i] = r.get("tokens") or len(events[i])


stream(a.url, a.model, [{"role": "user", "content": "Say hi."}], 8)  # warm-up
before = server_draft_counters(a.url) if a.model else {}
ts = [threading.Thread(target=run, args=(i,)) for i in range(a.n)]
[t.start() for t in ts]
[t.join() for t in ts]
acc = acceptance_between(before, server_draft_counters(a.url)) if before else None
bad = [f"stream {i}: {err}" for i, err in enumerate(errors) if err]
bad += [f"stream {i}: no tokens" for i in range(a.n) if not errors[i] and not (tokens[i] and events[i])]
if bad:
    sys.exit("c4_steady.py: no measurement (a failed run is not data): " + "; ".join(bad))
start = max(e[0] for e in events)
end = min(e[-1] for e in events)
# A streamed event can carry several tokens (a drafting server commits up to depth + 1 per step): count tokens, each
# stream's events weighted by its tokens per event (usage.completion_tokens / events), not events.
count = sum(sum(1 for x in e if start <= x <= end) * (tokens[i] / max(1, len(e))) for i, e in enumerate(events))
if end <= start or count <= 0:
    sys.exit(f"c4_steady.py: no measurement: the {a.n} streams never decoded at the same time (window {end - start:.2f} s)")
print(json.dumps({"streams": a.n, "thinking": a.thinking, "window_s": round(end - start, 2), "agg_tps": round(count / (end - start), 1),
                  "per_stream_events": [len(e) for e in events], "per_stream_tokens": tokens,
                  **({"acceptance": acc} if acc is not None else {})}))
if a.label and a.model:  # Loom run: track it in the optimization campaign
    from campaign import log_metrics
    log_metrics({"c4_steady": round(count / (end - start), 1)}, a.label, a.reverted)
elif a.label:  # llama.cpp run: record it as baseline data next to perf.py's rows
    with open(a.log, "a") as f:
        f.write(json.dumps({"kind": "llama", "label": a.label, "c4_steady": round(count / (end - start), 1)}) + "\n")
