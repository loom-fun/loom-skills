#!/usr/bin/env python3
"""Does a package that serves on the Radeon AI PRO R9700 also serve on a 16 GB AMD Radeon RX 9070 XT, and at what context?

The RX 9070 XT is the R9700's chip (Navi 48, gfx1201) with half the memory: the same programs run, only the context that fits
differs. Run on a machine with a 9070 XT (not a DevLab job: there is no runner). The package is served elastic with a 16 GB
budget through a deployment document that sets its context_capacity: hrx-loom 0.2 refuses to register a model whose budget cannot
hold one full context ("shared pool byte limit cannot address one model context"), so the largest context_capacity that registers
(binary search, multiples of 1024 up to the native context) is found first, then confirmed on a FRESH server by one request of that
full length that must complete, followed by a correct answer to a sanity prompt (step down 5 % if not). max_context is the
context_capacity a 9070 XT preset would carry.

usage: test_9070xt.py PKG [--loom-root ~/loom] [--memory-bytes 15000000000] [--prefill-capacity N] [--port 8080] -o rx9070xt.json
       test_9070xt.py --untested REASON -o rx9070xt.json        (a package whose weights do not fit 16 GB: {tested: false})
PKG: a fetched package (loom-fetch MODEL.json --materialize=share --output=PKG, the default preset; not the -r9700 preset, whose
fixed pool is sized for 32 GB). Run with the tools venv's python ($LOOM_ROOT/.tools/bin/python: requests, tokenizers).

--prefill-capacity N: serve through a deployment document that sets the text definition's prefill_capacity (a serving setting, the package
is unchanged): for a package whose weights plus the default prefill chunk's workspace exceed the budget (registration fails with "model
parameters and invocation working set exceed the serving memory limit"; gpt-oss-20b: 4096 -> 1024). Recorded as "execution".

Serve flags (recorded in the result):
- --memory_bytes=15000000000: the board reports 17.1 GB; a desktop session holds about 1.5 GB of it.
- --state_slab_bytes=4mib: elastic state commits in slabs per KV plane; the default 64 MiB x 82 planes (Qwen3-8B with its drafter)
  is 5.5 GB committed before the first request, and the context that fits drops from 32k to 7k.
- --parameter_mode=preload: a stock Ubuntu desktop has an 8 MB RLIMIT_MEMLOCK; the default file mode pins io_uring buffers and
  the weights never load ("buffer pin preparation failed (12)"). Not needed where `ulimit -l` is unlimited.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = "What is the capital of France? Answer in one word."
DEVICE = "--device=amdgpu://?completion_spin_ns=4294967295"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def hrx_loom_version(root):
    try:
        j = json.load(open(os.path.join(root, "loom-install.json")))
        return j.get("hrx_loom") or j.get("version")
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkg", nargs="?")
    ap.add_argument("--loom-root", default=os.environ.get("LOOM_ROOT", os.path.expanduser("~/loom")))
    ap.add_argument("--memory-bytes", type=int, default=15_000_000_000)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--parameter-mode", default="preload", help="loom-serve --parameter_mode (preload: hosts with an 8 MB RLIMIT_MEMLOCK; file: the SDK default)")
    ap.add_argument("--prefill-capacity", type=int, help="serve through a deployment document with this prefill_capacity")
    ap.add_argument("--weight-slab-bytes", help="loom-serve --weight_slab_bytes (default 1 GiB): smaller slabs for a package whose weights leave "
                    "less than a 1 GiB workspace slab of the budget (LFM2.5-24B: 64mib with --prefill-capacity 256)")
    ap.add_argument("--untested", metavar="REASON", help="write {tested: false, reason} without serving")
    ap.add_argument("-o", required=True)
    a = ap.parse_args(argv)
    date = time.strftime("%Y-%m-%d")
    if a.untested:
        json.dump({"tested": False, "max_context": None, "reason": a.untested, "date": date}, open(a.o, "w"), indent=1)
        return 0
    if not a.pkg:
        ap.error("PKG is required unless --untested")
    pkg, url = os.path.abspath(a.pkg), f"http://127.0.0.1:{a.port}"
    card = json.load(open(os.path.join(pkg, "model.json")))
    cfg = json.load(open(os.path.join(pkg, "llm_config.json")))
    native = next(iter(card["presets"].values()))["execution"]["context_capacity"]
    flags = [f"--memory_bytes={a.memory_bytes}", "--state_slab_bytes=4mib"] + ([f"--parameter_mode={a.parameter_mode}"] if a.parameter_mode != "file" else [])
    if a.weight_slab_bytes:
        flags.append(f"--weight_slab_bytes={a.weight_slab_bytes}")
    # Served through a deployment document: the card's preset with context_capacity (the search variable) and the given execution
    # settings overridden. hrx-loom 0.2 refuses to register a model whose memory budget cannot hold one full context_capacity ("shared
    # pool byte limit cannot address one model context"), so the context the 9070 XT serves is a context_capacity, not a request length.
    model = os.path.abspath(a.o + ".deployment.json")
    execution = {"prefill_capacity": a.prefill_capacity} if a.prefill_capacity else None

    def ctx(k):
        return min(k * 1024, native)

    def deploy(k):
        json.dump({"name": cfg["model_id"], "model": {"path": f"{pkg}/model.json"},
                   "definition": {"kind": "text", "context_capacity": ctx(k), **(execution or {})}}, open(model, "w"), indent=1)
    serve = [os.path.join(a.loom_root, ".venv", "bin", "loom-serve"), f"--model={model}", DEVICE, *flags,
             f"--listen=127.0.0.1:{a.port}", "--heartbeat_ms=0"]
    res = {"tested": False, "max_context": None, "date": date, "memory_bytes": a.memory_bytes, "serve_args": flags, "search": "context_capacity",
           "hrx_loom": hrx_loom_version(a.loom_root), "native_context": native, "probes": [], "confirm": []}
    if execution:
        res["execution"] = execution
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if (cfg.get("chat") or {}).get("thinking") == "toggle" else {}
    slog = open(a.o + ".serve.log", "w")
    srv = None

    def start(k, load=True):   # a fresh server at context_capacity ctx(k); load=False: registration only (the pool fits one context)
        nonlocal srv
        if srv:
            srv.terminate()
            srv.wait(60)
            time.sleep(2)
        deploy(k)
        srv = subprocess.Popen(serve, stdout=slog, stderr=subprocess.STDOUT)
        for _ in range(600):   # registration: compiled (ready) or refused (failed, with the server's reason)
            try:
                with urllib.request.urlopen(url + "/v1/status", timeout=10) as r:
                    m = (json.load(r).get("data") or [{}])[0].get("loom") or {}
            except (OSError, ValueError):
                m = {}
            if m.get("registration") == "failed":
                res["load_error"] = ((m.get("error") or {}).get("message") or "registration failed")[:300]
                log("registration failed at", ctx(k), ":", res["load_error"])
                return False
            if m.get("registration") == "ready":
                break
            time.sleep(2)
        if not load:
            res["probes"].append([ctx(k), True, "registered"])
            log("registered at", ctx(k))
            return True
        t0, last = time.time(), ""   # the weights load on the first request: repeat a one-token request while the server refuses
        while time.time() - t0 < 900:
            try:
                body = {"model": cfg["model_id"], "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
                req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=600).read()
                log(f"model ready after {time.time() - t0:.0f} s")
                return True
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:200]!r}"
                if e.code != 503:
                    break
            except OSError as e:
                last = str(e)[:200]
            time.sleep(3)
        log("model never answered:", last)
        res["load_error"] = res.get("load_error") or last
        return False

    def sane():
        body = {"model": cfg["model_id"], "max_tokens": 1024, "temperature": 0, "messages": [{"role": "user", "content": PROMPT}], **extra}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            msg = json.load(r)["choices"][0]["message"]
        ok = "paris" in json.dumps(msg).lower()
        if not ok:   # the evidence of a wrong answer: what the model said
            res.setdefault("wrong_answers", []).append({k: (msg.get(k) or "")[:400] for k in ("content", "reasoning_content") if msg.get(k)})
        return ok

    def registers(k):
        ok = start(k, load=False)
        if not ok:
            res["probes"].append([ctx(k), False, res.get("load_error", "")])
        return ok

    def complete(k):   # the same prompt as ctx_admit.py's, prefilled to the end and answered (one token)
        sys.path.insert(0, HERE)
        from bench_common import Package, doc
        P = Package(pkg)
        ids = P.ids(doc(32000))
        n = ctx(k) - 128   # the prompt leaves 128 positions for the template and the one-token answer
        while len(ids) < n:
            ids = ids + ids
        body = {"model": cfg["model_id"], "max_tokens": 1, "messages": [{"role": "user", "content": P.tok.decode(ids[:n])}], **extra}
        t0 = time.time()
        try:
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2400) as r:
                ok = bool(json.load(r)["choices"][0].get("finish_reason"))
            msg = f"COMPLETED {n} tokens in {time.time() - t0:.0f}s"
        except urllib.error.HTTPError as e:
            ok, msg = False, f"HTTP {e.code} {e.read()[:200]!r}"
        except OSError as e:
            ok, msg = False, f"no answer: {str(e)[:200]}"
        res["confirm"].append([ctx(k), ok, msg[:300]])
        log("complete", ctx(k), ok, msg[:120])
        return ok

    try:
        hi = -(-native // 1024)
        if registers(hi):
            best = hi
        else:   # binary search on context_capacity: lo registers (0: none yet), hi is refused
            lo = 0
            while lo < hi - 1:
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if registers(mid) else (lo, mid)
            best = lo
        if best <= 0:
            res["reason"] = "the package did not register on the RX 9070 XT at any context" + (f": {res['load_error']}" if res.get("load_error") else "")
            return 1
        def serves(k):   # a fresh server at context_capacity ctx(k) completes a request of that full length, then answers correctly
            if not start(k) or not complete(k):
                return False
            try:
                ok = sane()
            except Exception as e:   # the server failed after the long request (it ran out of memory outside the budget)
                ok, res["after_error"] = False, str(e)[:200]
            if not ok:
                log("no correct answer after", ctx(k))
            return ok

        if serves(best):   # registration is a loose bound (it checks one context against the pool, not the request's completion credit)
            res.update(tested=True, max_context=ctx(best))
        else:   # binary search on the full request: lo serves (0: none yet), hi does not
            lo, hi = 0, best
            while lo < hi - 1:
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if serves(mid) else (lo, mid)
            if lo:
                res.update(tested=True, max_context=ctx(lo))
        if not res["tested"]:
            res.setdefault("reason", "no context completed with a correct answer after it")
        return 0 if res["tested"] else 1
    finally:
        if srv:
            srv.terminate()
            try:
                srv.wait(30)
            except subprocess.TimeoutExpired:
                srv.kill()
        json.dump(res, open(a.o, "w"), indent=1)
        log("RESULT", "tested" if res["tested"] else "not tested", res["max_context"], res.get("reason", ""))


if __name__ == "__main__":
    sys.exit(main())
