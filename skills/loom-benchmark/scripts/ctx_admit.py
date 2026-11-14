"""Native-context admission probe (a check, not a gate): does loom-serve ADMIT a request of ~the package's full context?

One streaming chat request whose prompt is ~context_capacity tokens (real text, minus the answer budget). Admitted = the
server accepts it and starts (status 200, or an active request on /v1/status); refused = an HTTP 4xx/503 (a KV pool or
memory budget below the context). The request is then cancelled (the connection is closed, max_tokens is 1) and the
script waits for the server to go idle again: no long prefill, no generation. Seconds, not the many minutes a full-context
prefill takes. A card must not claim a context its server refuses.

usage: ctx_admit.py PKG LOOM_URL [--out FILE] [--wait 120] [--tokens N]   (--tokens: a prompt of N tokens instead of the native context)
Prints "ADMITTED n tokens (...)" or "REFUSED n tokens: HTTP ...". Exit status 1 on a refusal or no answer.
"""
import argparse
import json
import os
import sys
import threading
import time

import requests

from bench_common import Package, doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("url")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokens", type=int, default=None, help="prompt tokens (default: the native context minus the answer)")
    ap.add_argument("--wait", type=float, default=120, help="seconds to wait for an admission or a refusal")
    a = ap.parse_args()
    P = Package(a.pkg)
    card = json.load(open(os.path.join(a.pkg, "model.json")))
    context = next(iter(card["presets"].values()))["execution"]["context_capacity"]
    thinking = False if P.chat.get("thinking") == "toggle" else None
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if thinking is False else {}
    overhead = len(P.ids(P.render_messages([{"role": "user", "content": "x"}], None, thinking)))
    n = a.tokens or context - 1 - overhead - 64   # one answer token
    ids = P.ids(doc(32000))
    while len(ids) < n:
        ids = ids + ids
    prompt = P.tok.decode(ids[:n])
    got = {}

    def post():
        try:
            r = requests.post(a.url + "/v1/chat/completions", stream=True, timeout=a.wait + 600, json={
                "model": P.model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
                "stream": True, **extra})
            got["resp"] = r
            got["status"] = r.status_code
            if r.status_code != 200:
                got["body"] = r.text[:200]
        except Exception as e:  # the connection closed below
            got.setdefault("error", str(e)[:200])

    def active():
        return requests.get(a.url + "/v1/status", timeout=10).json()["data"][0]["loom"]["status"]["active_requests"]

    threading.Thread(target=post, daemon=True).start()
    t0 = time.time()
    running = False
    while time.time() - t0 < a.wait and "status" not in got and "error" not in got:
        time.sleep(0.5)
        running = active() > 0
        if running:
            break
    admitted = got.get("status") == 200 or (running and "status" not in got)
    # Cancel: closing the connection makes the server drop the request; then it must go idle again (no long prefill).
    idle = True
    if "resp" in got:
        got["resp"].close()
    if admitted:
        t1 = time.time()
        while active() and time.time() - t1 < 60:
            time.sleep(1)
        idle = not active()
    if admitted and not idle:
        msg = f"ADMITTED {n} tokens of a {context}-token context, but the server did not cancel the request within 60s (WARN: later steps may stall)"
    elif admitted:
        msg = f"ADMITTED {n} tokens of a {context}-token context ({time.time() - t0:.1f}s, then cancelled)"
    elif "status" in got or "error" in got:
        msg = f"REFUSED {n} tokens of a {context}-token context: HTTP {got.get('status')} {got.get('body') or got.get('error')}"
    else:
        msg = f"NO ANSWER for {n} tokens of a {context}-token context within {a.wait:.0f}s"
    print(msg, flush=True)
    if a.out:
        json.dump({"admitted": admitted, "tokens": n, "context": context, "detail": msg}, open(a.out, "w"), indent=1)
    sys.exit(0 if admitted else 1)


if __name__ == "__main__":
    main()
