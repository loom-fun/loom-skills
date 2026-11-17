#!/usr/bin/env python3
"""Wait until a freshly started loom-serve answers a request: readiness is "registration ready" (the program compiled), but the weights load on
the FIRST request, and until that load has finished every request is refused (connection closed, or 503 "model is unavailable"). A step that
sends its first request too early then fails for a reason that has nothing to do with what it measures.

usage: wait_model.py URL [--model ID] [--wait 300]
Sends one one-token chat request and repeats it while the server refuses (connection error, HTTP 503 "unavailable"); exit 0 as soon as one is
answered (prints "model ready after N s"), exit 1 with the last refusal after --wait seconds, exit 2 on any other HTTP error (not a loading
problem). The model id defaults to the first of /v1/models. Stdlib only.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def first_model(url):
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url")
    ap.add_argument("--model")
    ap.add_argument("--wait", type=float, default=300)
    ap.add_argument("--retry", type=float, default=3)
    a = ap.parse_args(argv)
    t0 = time.time()
    last = "no answer"
    model = a.model
    while True:
        try:
            model = model or first_model(a.url)
            body = {"model": model, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 1, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=max(30, a.wait)) as r:
                r.read()
            print(f"model ready after {time.time() - t0:.0f} s", flush=True)
            return 0
        except urllib.error.HTTPError as e:
            msg = e.read()[:200].decode("utf-8", "replace")
            last = f"HTTP {e.code}: {msg}"
            if e.code != 503 or "unavailable" not in msg:
                print(f"first request failed: {last}", flush=True)
                return 2
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, KeyError, IndexError, ValueError) as e:   # the server closed the connection / is not listening yet
            last = f"{type(e).__name__}: {e}"
        if time.time() - t0 > a.wait:
            print(f"model never became available in {a.wait:.0f} s: {last}", flush=True)
            return 1
        time.sleep(a.retry)


if __name__ == "__main__":
    sys.exit(main())
