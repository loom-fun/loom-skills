"""Native-context admission probe (a check, not a gate): does loom-serve ADMIT a request of ~the package's full context?

One streaming chat request whose prompt is ~context_capacity tokens (real text, minus the answer budget). Admitted = the
server accepts it and starts (status 200, or an active request on /v1/status); refused = an HTTP 4xx/503 (a KV pool or
memory budget below the context). The request is then cancelled (the connection is closed, max_tokens is 1) and the
script waits for the server to go idle again: no long prefill, no generation. Seconds, not the many minutes a full-context
prefill takes. A card must not claim a context its server refuses.

usage: ctx_admit.py PKG LOOM_URL [--out FILE] [--wait 120] [--tokens N] [--complete [SECONDS]]
(--tokens: a prompt of N tokens instead of the native context; --complete: do not cancel, wait for the one-token answer of the full prefill
(default 1800 s) and print "COMPLETED ..." or the failure: a full-context request that really finishes, not only one that starts)
Prints "ADMITTED n tokens (...)" or "REFUSED n tokens: HTTP ...". Exit status 1 on a refusal or no answer.
"""
import argparse
from contextlib import suppress
from http.client import HTTPConnection
import json
import os
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

import requests

from bench_common import Package, doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("url")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", default=os.environ.get("LOOM_PRESET") or None, help="the preset the server serves (default: LOOM_PRESET, else the first)")
    ap.add_argument("--tokens", type=int, default=None, help="prompt tokens (default: the native context minus the answer)")
    ap.add_argument("--wait", type=float, default=120, help="seconds to wait for an admission or a refusal")
    ap.add_argument("--complete", type=float, nargs="?", const=1800, default=None, help="wait this long (default 1800 s) for the request to finish instead of cancelling it")
    a = ap.parse_args()
    P = Package(a.pkg)
    card = json.load(open(os.path.join(a.pkg, "model.json")))
    presets = card["presets"]   # the served preset (--preset, else LOOM_PRESET as _lib.sh exports it), else the first
    context = (presets.get(a.preset) or next(iter(presets.values())))["execution"]["context_capacity"]
    thinking = False if P.chat.get("thinking") == "toggle" else None
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if thinking is False else {}
    overhead = len(P.ids(P.render_messages([{"role": "user", "content": "x"}], None, thinking)))
    n = a.tokens or context - 1 - overhead - 64   # one answer token
    ids = P.ids(doc(32000))
    while len(ids) < n:
        ids = ids + ids
    prompt = P.tok.decode(ids[:n])
    url = a.url.rstrip("/") + "/v1/chat/completions"
    endpoint = urlsplit(url)
    loopback = endpoint.hostname in ("127.0.0.1", "::1", "localhost")
    proxy = requests.utils.select_proxy(url, requests.utils.get_environ_proxies(url))
    local_http = endpoint.scheme == "http" and loopback and endpoint.username is None and proxy is None
    stopped = threading.Event()
    got = {}

    def cancel():
        stopped.set()
        transport = got.get("transport")
        if transport is not None:
            with suppress(OSError):
                transport.shutdown(socket.SHUT_RDWR)
        response = got.get("response")
        if response is not None:
            response.close()

    def post():
        try:
            body = {
                "model": P.model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
                "stream": True, **extra}
            if stopped.is_set():
                return
            if not local_http:
                # Requests retains its configured proxies, credentials and certificate trust for remote endpoints.
                r = requests.post(url, stream=True, timeout=a.wait + 600, json=body)
                got["response"] = r
                if stopped.is_set():
                    r.close()
                    return
                if r.status_code != 200:
                    got["body"] = r.text[:200]
                got["status"] = r.status_code
                if r.status_code == 200 and a.complete is not None:
                    got["text"] = "".join(line.decode("utf-8", "replace") for line in r.iter_lines() if line)
                    got["finished"] = time.time()
                return
            conn = HTTPConnection(endpoint.hostname, endpoint.port, timeout=a.wait)
            try:
                conn.connect()
                # The service can register active prefill before sending any HTTP headers. Keep
                # the transport so the probe can close it even while getresponse() is blocked.
                got["transport"] = conn.sock
                if stopped.is_set():
                    return
                conn.sock.settimeout(a.wait + 600)
                conn.request("POST", endpoint.path, json.dumps(body).encode(), {"Content-Type": "application/json"})
                with conn.getresponse() as reply:
                    if reply.status != 200:
                        got["body"] = reply.read(200).decode("utf-8", "replace")
                    got["status"] = reply.status
                    if reply.status == 200 and a.complete is not None:
                        got["text"] = "".join(line.decode("utf-8", "replace") for line in reply if line.strip())
                        got["finished"] = time.time()
                    elif reply.status == 200:
                        # Keep the streaming connection open until the caller shuts it down.
                        reply.read()
            finally:
                conn.close()
        except Exception as e:  # the connection closed below
            got.setdefault("error", str(e)[:200])

    def active():
        return requests.get(a.url + "/v1/status", timeout=10).json()["data"][0]["loom"]["status"]["active_requests"]

    t0 = time.time()
    sender = threading.Thread(target=post, daemon=True)
    sender.start()
    running = False
    while time.time() - t0 < a.wait and "status" not in got and "error" not in got:
        time.sleep(0.5)
        running = active() > 0
        if running:
            break
    admitted = got.get("status") == 200 or (running and "status" not in got)
    if admitted and a.complete is not None:   # the full-context request must finish: a non-empty answer within the wait
        while "finished" not in got and "error" not in got and time.time() - t0 < a.complete:
            time.sleep(2)
        ok = "finished" in got and '"finish_reason"' in got.get("text", "") and "error" not in got
        if not ok and local_http:
            cancel()
        msg = (f"COMPLETED {n} tokens of a {context}-token context in {got['finished'] - t0:.0f}s" if ok else
               f"NOT COMPLETED {n} tokens of a {context}-token context after {time.time() - t0:.0f}s: {got.get('error') or got.get('text', '')[:200] or 'no answer'}")
        print(msg, flush=True)
        if a.out:
            json.dump({"admitted": True, "completed": ok, "tokens": n, "context": context, "detail": msg}, open(a.out, "w"), indent=1)
        sys.exit(0 if ok else 1)
    # Cancel: closing the connection makes the server drop the request; then it must go idle again (no long prefill).
    idle = True
    cancel()
    if local_http:
        sender.join(timeout=1)
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
