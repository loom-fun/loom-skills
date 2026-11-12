"""Throughput on fixed workloads, for Loom or llama.cpp (same prompts both sides).

usage: python perf.py URL LABEL [--kind loom|llama] [--model MODEL_ID] [--long] [--log perf_log.jsonl] [--reverted]
                     [--no-campaign]
Every run appends one row to --log (also with --no-campaign, which only skips the optimization campaign's
requirements and tracking: make_info.py perf and devlab_perf.sh read those rows).

Metrics (all greedy, real-text documents, each prompt starts with a nonce so no
prefix cache can help):
  decode_tps        300-token essay from a short prompt
  prefill[]         ~1k and ~2k-token documents, max_tokens=4: prompt_tokens / TTFT,
                    median (and min/max) of --repeats runs; single samples vary up to 2x
  decode_tps_2k     256-token generation after a 2k document (decode with KV in use)
  --long: prefill 8k/16k/32k documents and decode_tps_32k (128 tokens after 32k). A long request the server
                    refuses with HTTP 400 (its answer to a prompt over its context: the needle gate checks it) is
                    skipped and logged in out["over_context"] (roofline.json metric names; out["nonce"] lets
                    make_info.py perf re-render and count it); the other sizes are kept. A tokenizer can make the
                    "32k" document longer than a 32768-token native context (34.9k tokens for Gemma 3).
  acceptance        Loom with speculative decoding: per decode metric, accepted / proposed drafted tokens of that
                    very request (loom-serve's /metrics counters before and after it), so the roofline of a drafted
                    row uses the acceptance of the text it decoded (make_info.py roofline), not the spec gate's
Completion tokens: every timed decode request logs the tokens it generated (out["completion_tokens"][metric]:
                    one entry per attempt, from the usage block on both engines). A request that stopped before half its
                    max_tokens is not a decode measurement: the rate is (n-1)/time over a few tokens, and with drafting
                    those arrive in bursts (a 32k decode that stopped early read 164 tok/s, 2.6x its KV-bandwidth
                    ceiling). It is re-requested once with another nonce; still short, the metric is logged as null
                    (missing), never as the inflated rate. Half max_tokens: every decode prompt asks for a long essay,
                    so an early stop is the anomaly, and the timed span stays >= 64 tokens even at max_tokens 128.
Warm-up (untimed, logged as warmup_decode_tps): the timed decode request itself with another nonce, so every path
the first timed request takes (prefill, decode, drafting, lazily loaded weights, first-use kernels) has run once.
A trivial "Say hi." left the short-prompt decode paying first-use costs (DevLab sweeps measured it 51-60 % below
another session while decode at 2k agreed). Both engines and every caller (qualify, finalize, devlab_perf)
get the same warm-up.
Loom numbers are client-side (streaming TTFT and token arrival). llama.cpp
numbers come from its server-side `timings` (excludes HTTP and tokenization),
which slightly favours llama.cpp.
--thinking off|on|template (default off): every request sends the same chat_template_kwargs.enable_thinking to
both engines, so they decode the same kind of text. llama-server turns thinking on by default while a Loom
package follows its chat template (off for Nemotron-3.5-Bolt): with nothing sent, the two engines decoded
different texts (a reasoning trace vs the answer) and drafting accepted very different amounts. `template` sends
nothing (each engine's default: the behaviour before this flag). The setting is recorded in every log row.
"""
import argparse
import json
import random
import time

import requests

from bench_common import acceptance_between, doc, long_prompt, perf_metrics, server_draft_counters, stream, thinking_extra

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("label")
ap.add_argument("--kind", default="loom", choices=["loom", "llama"])
ap.add_argument("--model", default=None, help="Loom model id (model.json model.id)")
ap.add_argument("--long", action="store_true")
ap.add_argument("--log", default="perf_log.jsonl")
ap.add_argument("--reverted", action="store_true", help="this build is not kept (still logged: its time counts)")
ap.add_argument("--no-campaign", action="store_true",
                help="ad-hoc/ablation measurement: skip the roofline/baseline requirement and the campaign's tracking "
                     "(targets.jsonl, rate-of-approach verdict); the row still goes to --log")
ap.add_argument("--repeats", type=int, default=5, help="samples per short-prefill size (median reported)")
ap.add_argument("--thinking", default="off", choices=["off", "on", "template"],
                help="chat_template_kwargs.enable_thinking sent to BOTH engines (template: send nothing)")
a = ap.parse_args()
TH = thinking_extra(a.thinking)
U, M = a.url, a.model
if not a.no_campaign:  # required steps fail loudly instead of warning (warnings get ignored)
    from campaign import require
    require(a.kind, a.log)
H = {"Content-Type": "application/json"}


class OverContext(Exception):
    """The server refused the prompt with HTTP 400: longer than its context."""


def run(content, max_tokens):
    """-> (prompt_tokens, prefill_tps, decode_tps, completion_tokens); raises OverContext on HTTP 400"""
    msgs = [{"role": "user", "content": content}]
    if a.kind == "loom":
        s = stream(U, M, msgs, max_tokens, extra=TH)
        status, error = s["status"], s.get("error")
    else:
        r = requests.post(U + "/v1/chat/completions", headers=H, json={"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": False, **TH})
        status, error = r.status_code, r.text[:300]
    if status == 400:
        raise OverContext(error)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {error}")
    if a.kind == "loom":
        return s["prompt_tokens"], s["prompt_tokens"] / s["ttft"], s["decode_tps"], s["tokens"]
    d = r.json()
    t = d["timings"]
    return t["prompt_n"], t["prompt_per_second"], t["predicted_per_second"], d["usage"]["completion_tokens"]


ESSAY = "Write a detailed 300-word essay about the history of the printing press."
out = {"label": a.label, "kind": a.kind, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "thinking": a.thinking}
nonce = str(time.time())
# warm-up (untimed): the first timed request's own shape, another nonce (no prefix-cache hit on the timed prompt)
out["warmup_decode_tps"] = round(run(f"[warm-up {nonce}] " + ESSAY, 300)[2], 2)


def decode(key, content, max_tokens):
    """A timed decode request -> its decode tok/s, or None when it generated fewer than max_tokens // 2 tokens twice
    (the second time with another nonce); out["completion_tokens"][key]: the tokens of each attempt;
    out["acceptance"][key]: the drafted-token acceptance of the request that counted (Loom)."""
    for attempt in range(2):
        prompt = content if attempt == 0 else f"[retry {time.time()}] " + content
        before = server_draft_counters(U) if a.kind == "loom" else {}
        _, _, dtps, n = run(prompt, max_tokens)
        after = server_draft_counters(U) if before else {}
        out.setdefault("completion_tokens", {}).setdefault(key, []).append(n)
        print(f"{key}: {n}/{max_tokens} completion tokens, {dtps if dtps is None else round(dtps, 2)} tok/s", flush=True)
        if dtps and n >= max_tokens // 2:
            acc = acceptance_between(before, after) if before else None
            if acc is not None:
                out.setdefault("acceptance", {})[key] = acc
            return round(dtps, 2)
    print(f"WARNING: {key} generated {out['completion_tokens'][key]} tokens (< {max_tokens // 2}) twice: "
          "logged as missing, not as the rate of a short request", flush=True)
    return None


def long_request(key, f, *args):
    """f(*args), or None with key logged in out["over_context"] when the server refuses the prompt as over its context."""
    try:
        return f(*args)
    except OverContext as e:
        out.setdefault("over_context", []).append(key)
        print(f"{key}: refused as over the server's context, skipped ({e})", flush=True)
        return None


if a.long:
    out["prefill_long"], out["nonce"] = [], nonce
    for n in (8000, 16000, 32000):
        r = long_request(f"prefill_{n}", run, long_prompt(f"prefill_{n}", nonce), 4)
        if r:
            out["prefill_long"].append({"prompt_tokens": r[0], "prefill_tps": round(r[1], 1)})
            print(out["prefill_long"][-1], flush=True)
    out["decode_tps_32k"] = long_request("decode_tps_32k", decode, "decode_tps_32k", long_prompt("decode_tps_32k", nonce), 128)
else:
    out["decode_tps"] = decode("decode_tps", ESSAY, 300)
    out["prefill"] = []
    for n in (1000, 2000):  # short prefills are noisy (scheduling, clocks): median of a.repeats fresh-nonce runs
        samples = [run(f"[{nonce} {i} {random.random()}] Summarize this documentation in one sentence.\n\n" + doc(n), 4) for i in range(a.repeats)]
        pps = sorted(x[1] for x in samples)
        out["prefill"].append({"prompt_tokens": samples[0][0], "prefill_tps": round(pps[len(pps) // 2], 1),
                               "min": round(pps[0], 1), "max": round(pps[-1], 1), "n": len(pps)})
    out["decode_tps_2k"] = decode("decode_tps_2k", f"[{nonce} ctx] Write a long, detailed essay discussing this documentation.\n\n" + doc(2000), 256)
for k, v in list(out.items()):
    if isinstance(v, float):
        out[k] = round(v, 2)
print(json.dumps(out))
with open(a.log, "a") as f:
    f.write(json.dumps(out) + "\n")
if a.kind == "loom" and not a.no_campaign:  # campaign tracking: targets.jsonl + rate-of-approach verdict (needs roofline.json)
    from campaign import log_metrics
    log_metrics(perf_metrics(out), a.label, a.reverted, a.log)
