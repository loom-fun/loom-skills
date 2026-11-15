#!/usr/bin/env python3
"""Host logits readback check of one package on one GPU (devlab_perf.sh step `logits:V`; loom-skills#736).

The package is run through the Gufo bundle (loom-gufo-extension validation/devlab/build_bundle.sh: a gufo + libloom_text_embed.so
with their own loader; nothing from the host but the GPU driver and Python) and checked three ways:

  (a) embed_probe (no Gufo, the embed layer alone): with the span flagged for logits at T=0, every returned row's argmax is the
      device-selected token for every row and step of a packed epoch, padding is never the argmax, host-fed and device-fed tokens
      agree, and an unflagged request leaves no row (embed_probe exits 0 only if every check held);
  (b) llama.cpp: `gufo serve --loom-logits=on` answers a greedy request with logprobs (host-sampled from the returned row); its
      top-5 log-probs per generated token vs llama.cpp's own top-5 for the same prompt token ids (same GGUF, undrafted, same
      runner): top-1 exact (a divergence is judged by bench_common.near_tie, the golden gate's rule, and ends that prompt's
      comparison), the other tokens both list within a tolerance by probability (LOGPROB_TIERS, measured engine differences:
      tight where the mass is, loose in the tail, none below p=1e-4). llama.cpp's references are recorded FIRST and its server is
      stopped (--llama-stop) before any Gufo loads: a 24B+ package and the reference GGUF do not fit one GPU together;
  (c) default spans unchanged: the greedy text of the same server with and without logprobs (device path vs host path), and of a
      server loaded with `--loom-logits=off`, is identical token for token.

Writes OUT.json, a gates.json that make_info.py gates ingests (gate `logits`, a detail line) plus every check's numbers.

usage: logits_check.py PACKAGE OUT.json --bundle DIR [--llama http://127.0.0.1:8130 [--llama-stop CMD]] [--steps 32] [--prompts 6] [--port 8181]
The bundle (unpacked tarball with run.sh, run-embed-probe.sh, run-gufo.sh) comes from --bundle or $GUFO_BUNDLE. Without one the
check is "not supported" (exit 2, a JSON saying so), never replaced by another path. The comparison logic (check_*) takes plain
lists and is unit tested without a GPU (tests/test_logits_check.py).
"""
import argparse
import datetime
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "skills", "loom-benchmark", "scripts"))

# Log-prob tolerance (nats) of a token both engines list, by the larger of the two probabilities (the golden's near-tie gap, 1.0, applies to the
# confident tokens; the tail is graded by how much probability is there to be wrong about). Measured on the same GGUF through the two engines'
# quantized matmuls (DevLab fleet sweep, both GPUs): small dense models differ by <= 0.7 nats on tokens below p=0.1; mid-size and MoE models by
# 1.2 nats at p~0.1 (a routing flip changes the whole tail), 1.5-1.8 at p~1e-3 and up to 2.3 below 1e-4. A token below p=1e-4 on both sides
# carries no mass and is not compared. The gate still catches a wrong row (a shifted slot, a stale row): several nats on the top tokens.
LOGPROB_TIERS = ((0.5, 1.0), (1e-2, 1.5), (1e-4, 2.5))    # (probability >=, tolerance in nats), tried in order
LOGPROB_TOL = LOGPROB_TIERS[0][1]
TOPK = 5
MIN_COMPARED = 8       # greedy steps compared against llama.cpp on at least one prompt before the gate can pass
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


# ---------------------------------------------------------------- comparisons (pure)

SPECIAL = re.compile("^<[|\uff5c].+[|\uff5c]>$")   # <|im_end|>, <FULLWIDTH-BAR end of sentence FULLWIDTH-BAR>: the engines' spellings of special tokens


def special_key(key):
    """A token's text as both engines print it: llama.cpp prints '' for a special token (<|im_end|>, the end-of-sentence marker), Gufo
    the marker: both are one key. A sentencepiece-style vocabulary has Gufo print ' are' as 'are' (llama.cpp keeps the space): leading
    spaces do not count (a whitespace-only token keeps its text)."""
    if SPECIAL.match(key):
        return ""
    return key.lstrip(" ") or key


def tail_tol(a, b):
    """The tolerance (nats) of one token's log-probs a, b in the two engines, or None when it is below every tier (not compared)."""
    p = math.exp(max(a, b))
    return next((tol for floor, tol in LOGPROB_TIERS if p >= floor), None)


def best_by_key(pairs):
    """{key: best log-prob}: tokens that print as the same text (several special or empty-text tokens) count by their best entry."""
    out = {}
    for key, lp in pairs:
        key = special_key(key)
        if key not in out or lp > out[key]:
            out[key] = lp
    return out


def check_vs_llama(loom_tops, llama_keys, llama_tops, near_tie):
    """Compare Loom's per-step top lists with llama.cpp's greedy continuation of the same prompt.
    loom_tops: per generated step [(key, logprob)] best first (key: the token's text); llama_keys: llama.cpp's greedy token per
    step; llama_tops: per step {key: logprob}. near_tie(top, gold, loom) -> bool (bench_common.near_tie).
    Returns (compared_steps, problems): stops at the first greedy divergence (judged as a near-tie, or a problem)."""
    compared, problems = 0, []
    llama_keys = [special_key(k) for k in llama_keys]
    llama_tops = [best_by_key(t.items()) for t in llama_tops]
    for k, (top, gkey, ltop) in enumerate(zip(loom_tops, llama_keys, llama_tops)):
        if not top:
            problems.append(f"step {k}: no top-logprobs")
            break
        lkey = special_key(top[0][0])
        if lkey != gkey:
            if not near_tie(ltop, gkey, lkey):
                problems.append(f"step {k}: Loom top-1 {lkey!r} vs llama.cpp {gkey!r}, not a near-tie")
            break
        for key, lp in best_by_key(top).items():
            if key in ltop:
                allowed = tail_tol(lp, ltop[key])
                if allowed is not None and abs(lp - ltop[key]) > allowed:
                    problems.append(f"step {k}: token {key!r} logprob {lp:.3f} vs llama.cpp {ltop[key]:.3f} (tolerance {allowed})")
        compared += 1
    return compared, problems


def check_default_unchanged(off, on_device, on_host):
    """The greedy text is identical with the capability off, on (device path, no logprobs) and on (host path, logprobs)."""
    if off != on_device:
        return False, f"capability on changes the device path's text at character {first_diff(off, on_device)}"
    if off != on_host:
        return False, f"the host path (logprobs) changes the text at character {first_diff(off, on_host)}"
    return True, ""


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def probe_check(line):
    """(ok, why) of embed_probe's one JSON line: opened, no failures."""
    try:
        j = json.loads(line)
    except ValueError:
        return False, f"embed_probe printed no JSON: {line[:120]!r}"
    if not j.get("opened"):
        return False, f"embed_probe: open failed: {j.get('error')}"
    if j.get("failures", 1) != 0:
        return False, f"embed_probe: {j.get('failures')} failure(s)"
    return True, ""


def verdict(checks, compared_max, llama_used):
    """The gate: pass only when every check passed, and (with a llama.cpp reference) enough steps were compared."""
    bad = [c for c in checks if not c["ok"]]
    if llama_used and compared_max < MIN_COMPARED and not bad:
        bad = [{"name": "llama", "why": f"only {compared_max} steps compared (need {MIN_COMPARED})"}]
    return not bad, bad


# ---------------------------------------------------------------- the bundle, a gufo server, llama.cpp

def post(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Gufo:
    """`gufo serve llm --model PKG --loom-logits=on|off` from the bundle, stopped on close."""

    def __init__(self, bundle, package, port, logits, log, rows=4, extra=()):
        self.p = None
        self.url = f"http://127.0.0.1:{port}"
        self.log = open(log, "w")
        self.p = subprocess.Popen([os.path.join(bundle, "run-gufo.sh"), "serve", "llm", "--model", package, "--sessions", str(rows), "--port", str(port),
                                   "--host", "127.0.0.1", f"--loom-logits={'on' if logits else 'off'}", *extra],
                                  stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(600):   # up to 20 minutes: loading and the package's JIT
            try:
                urllib.request.urlopen(self.url + "/health", timeout=3).read()
                break
            except (urllib.error.URLError, OSError):
                if self.p.poll() is not None:
                    stop_group(self.p)
                    raise RuntimeError(f"gufo exited while loading (see {log})")
                time.sleep(2)
        else:
            stop_group(self.p)
            raise RuntimeError(f"gufo did not come up in 20 minutes (see {log})")
        self.model = json.loads(urllib.request.urlopen(self.url + "/v1/models").read())["data"][0]["id"]

    def chat(self, prompt, steps, logprobs):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": steps, **NOTHINK,
                **({"logprobs": True, "top_logprobs": TOPK} if logprobs else {})}
        c = post(self.url + "/v1/chat/completions", body)["choices"][0]
        m = c["message"]
        return (m.get("reasoning_content") or "") + (m.get("content") or ""), (c.get("logprobs") or {}).get("content") or []

    def close(self):
        """Stop the server and everything it started (the wrapper script's children hold the GPU), and wait until the last one is gone:
        the next load (or the next check's server) must find the GPU's memory free."""
        stop_group(self.p)
        self.log.close()


def stop_group(p, grace=60):
    """SIGTERM the process group of p (started with start_new_session), SIGKILL after `grace` seconds, wait for p and for the group."""
    if p is None:
        return
    if not hasattr(os, "killpg"):   # Windows: no process groups (the Gufo bundle runs on Linux runners only)
        p.terminate()
        try:
            p.wait(grace)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        return
    pg = p.pid
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 30)):
        try:
            os.killpg(pg, sig)
        except ProcessLookupError:
            break
        t = time.time() + wait
        while time.time() < t:
            if p.poll() is not None:
                try:
                    os.killpg(pg, 0)   # a child outliving the wrapper
                except ProcessLookupError:
                    return
            time.sleep(0.5)
    p.wait()


def llama_reference(pkg, llama, prompt, steps):
    """llama.cpp's greedy continuation of the package's rendered prompt ids: (generated token texts, per-step {text: best logprob}); tokens
    that print as the same text (several special tokens print '') count by their best entry."""
    from bench_common import llama_greedy
    ids = pkg.ids(pkg.render_messages([{"role": "user", "content": prompt}], enable_thinking=False))
    r, gen, cp = llama_greedy(llama, ids, steps, pkg)
    return [x["token"] for x in cp], [best_by_key((t["token"], t["logprob"]) for t in x["top_logprobs"]) for x in cp]


# ---------------------------------------------------------------- the run

def run(a):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out = {"verdict": "FAIL", "level": "dev", "ts": now, "gates": {}, "checks": []}
    bundle = a.bundle or os.environ.get("GUFO_BUNDLE")
    if not bundle or not os.path.isfile(os.path.join(bundle, "run.sh")):
        out.update(verdict="NOT_SUPPORTED", note=f"no Gufo bundle ({bundle or 'GUFO_BUNDLE unset'}): host logits cannot be checked on this runner")
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print("LOGITS NOT_SUPPORTED", out["note"])
        return 2
    from bench_common import PROMPTS, Package, near_tie
    pkg = Package(a.package)
    prompts = PROMPTS[:a.prompts]
    work = os.path.splitext(a.out)[0]
    checks, compared_max, stats = [], 0, {"steps": 0, "prompts": 0}

    # llama.cpp first: its references for every prompt, then its server is stopped (it holds the GPU the packages need)
    refs = {}
    if a.llama:
        for p in prompts:
            refs[p] = llama_reference(pkg, a.llama, p, a.steps)
    if a.llama_stop:
        subprocess.run(shlex.split(a.llama_stop), check=False, timeout=300)

    preset_args = ["--loom-preset", a.preset] if a.preset else []   # a card with several presets needs one named
    # (a) the embed layer alone
    r = subprocess.run([os.path.join(bundle, "run-embed-probe.sh"), a.package, str(a.steps), "3", a.preset], capture_output=True, text=True, timeout=1800)
    ok, why = probe_check((r.stdout.strip().splitlines() or [""])[-1])
    checks.append({"name": "embed_probe", "ok": ok and r.returncode == 0, "why": why or (r.stderr.strip().splitlines() or [""])[-1][:200]})
    with open(work + ".embed_probe.json", "w") as f:
        f.write(r.stdout)

    # (c) capability off: the device path's text
    off = {}
    g = Gufo(bundle, a.package, a.port, False, work + ".gufo_off.log", extra=preset_args)
    try:
        for p in prompts:
            off[p] = g.chat(p, a.steps, False)[0]
    finally:
        g.close()
    # (b) + (c): capability on
    g = Gufo(bundle, a.package, a.port, True, work + ".gufo_on.log", extra=preset_args)
    try:
        for p in prompts:
            dev, _ = g.chat(p, a.steps, False)
            host, lp = g.chat(p, a.steps, True)
            ok, why = check_default_unchanged(off[p], dev, host)
            checks.append({"name": "default_unchanged", "prompt": p[:40], "ok": ok, "why": why})
            stats["prompts"] += 1
            stats["steps"] += len(lp)
            if a.llama:
                lkeys, ltops = refs[p]
                tops = [[(t["token"], t["logprob"]) for t in e.get("top_logprobs") or []] for e in lp]
                n, problems = check_vs_llama(tops, lkeys, ltops, near_tie)
                compared_max = max(compared_max, n)
                checks.append({"name": "llama", "prompt": p[:40], "ok": not problems, "compared": n, "why": "; ".join(problems[:3]),
                               "sample": {"loom": tops[:4], "llama": [[k, sorted(t.items(), key=lambda kv: -kv[1])[:5]] for k, t in zip(lkeys[:4], ltops[:4])]}})
    finally:
        g.close()
    ok, bad = verdict(checks, compared_max, bool(a.llama))
    n_ok = sum(1 for c in checks if c["ok"])
    detail = (f"embed_probe argmax = device token; {stats['steps']} host-sampled steps on {stats['prompts']} prompts"
              + (f", top-1 = llama.cpp's and top-{TOPK} within {LOGPROB_TIERS[0][1]}-{LOGPROB_TIERS[-1][1]} nats by probability over {compared_max}+ steps" if a.llama else "") + ", greedy text unchanged")
    out.update({"verdict": "PASS" if ok else "FAIL", "checks": checks, "stats": stats,
                "gates": {"logits": {"pass": ok, "detail": detail if ok else "; ".join(f"{b.get('name')}: {b.get('why')}" for b in bad)[:200]}}})
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"LOGITS {out['verdict']} {n_ok}/{len(checks)} checks; {detail}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package")
    ap.add_argument("out")
    ap.add_argument("--bundle")
    ap.add_argument("--llama", help="llama.cpp server URL (same GGUF, same runner): enables check (b)")
    ap.add_argument("--llama-stop", default="", help="command that stops the llama.cpp server once the references are recorded (it must free the GPU before Gufo loads)")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--preset", default="", help="the preset to open (required by a card with several: Strix = the first, the R9700 = <first>-r9700)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
