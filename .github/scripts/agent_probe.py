#!/usr/bin/env python3
"""Diagnostic: why is a thinking-on turn with two parallel tool calls not reused (agent gate, Tokyo+London step 1)?
Replays the gate's conversation against one loom-serve and, per variant, prints the raw reply (reasoning, content, calls,
streamed deltas), the retained session (/v1/sessions), whether the finished session can be pinned (POST /v1/checkpoints:
it needs reusable history) and the cached_tokens of the next step.
usage: agent_probe.py PKG URL MODEL"""
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "skills", "loom-benchmark", "scripts"))
import agent_golden as G  # noqa: E402
from bench_common import Package  # noqa: E402

pkg, url, model = sys.argv[1:4]
LLAMA = sys.argv[sys.argv.index("--llama") + 1] if "--llama" in sys.argv else None   # phase 2: llama.cpp reads the saved replies
SAVE = sys.argv[sys.argv.index("--save") + 1]
P = Package(pkg)
SYS = {"role": "system", "content": "You are a helpful assistant. Use the tools when they help."}


def post(body, headers=None, stream=False):
    h = {"Content-Type": "application/json", **(headers or {})}
    return requests.post(url + "/v1/chat/completions", data=json.dumps(body, ensure_ascii=False).encode(), headers=h,
                         timeout=1800, stream=stream)


def chat(msgs, thinking, session=None):
    body = dict(model=model, messages=msgs, tools=G.TOOLS, max_tokens=1024, temperature=0)
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    r = post(body, {"X-Loom-Session": session} if session else None).json()
    if "error" in r:
        print("  ERROR", r["error"]); return None, None
    return r["choices"][0], r["usage"]


def show(tag, ch, u):
    m = ch["message"]
    print(f"  [{tag}] finish={ch['finish_reason']} usage={u}")
    print(f"    reasoning_content tail={(m.get('reasoning_content') or '')[-120:]!r} head={(m.get('reasoning_content') or '')[:60]!r}")
    print(f"    content={m.get('content')!r}")
    for c in m.get("tool_calls") or []:
        print(f"    call {c['function']['name']} {c['function']['arguments']!r}")


def sessions(tag):
    for p in ("/v1/sessions", "/v1/requests"):
        try:
            print(f"  [{tag}] GET {p}:", requests.get(url + p, headers={"X-Loom-Model": model}, timeout=30).text[:1500])
        except Exception as e:  # noqa: BLE001
            print("  sessions error", e)


def tool_msgs(ch):
    m = ch["message"]
    out = [{k: v for k, v in m.items() if v is not None}]
    for c in m.get("tool_calls") or []:
        out.append({"role": "tool", "tool_call_id": c["id"], "content": G.run_tool(c)})
    return out


def scenario(name, thinking, task_idx, session=None, pin=False, stream_first=False):
    print(f"=== {name} thinking={thinking} session={session}")
    msgs = [SYS, {"role": "user", "content": G.TASKS[task_idx]}]
    if task_idx == 1:   # the gate runs task 0 first; its retained history is what task 1 step 0 extends
        c0, u0 = chat([SYS, {"role": "user", "content": G.TASKS[0]}], thinking, session)
        if not c0: return
        m1 = tool_msgs(c0)
        c1, u1 = chat([SYS, {"role": "user", "content": G.TASKS[0]}] + m1, thinking, session)
        if not c1: return
        msgs = [SYS, {"role": "user", "content": G.TASKS[0]}] + m1 + tool_msgs(c1) + [{"role": "user", "content": G.TASKS[1]}]
    if stream_first:
        body = dict(model=model, messages=msgs, tools=G.TOOLS, max_tokens=1024, temperature=0, stream=True,
                    **({} if thinking is None else {"chat_template_kwargs": {"enable_thinking": thinking}}))
        r = post(body, {"X-Loom-Session": session} if session else None, stream=True)
        print("  STREAM:")
        for line in r.iter_lines():
            if line: print("   ", line.decode()[:400])
        return
    ch, u = chat(msgs, thinking, session)
    if not ch: return
    show("step0", ch, u)
    sessions("after step0")
    if pin and session:
        r = requests.post(f"{url}/v1/checkpoints/pin-{session}", headers={"X-Loom-Session": session}, timeout=60)
        print(f"  PIN {session}: {r.status_code} {r.text[:300]}")
    msgs2 = msgs + tool_msgs(ch)
    prompt = P.render_messages(msgs2, G.TOOLS, thinking)
    prev = P.render_messages(msgs, G.TOOLS, thinking)
    print(f"  client-rendered step1 prompt starts with prev prompt: {prompt.startswith(prev)}; tail of step1 prompt's assistant part: "
          f"{prompt[len(prev):len(prev) + 200]!r}")
    ch2, u2 = chat(msgs2, thinking, session)
    if ch2:
        print(f"  [step1] cached={u2.get('prompt_tokens_details')} prompt_tokens={u2['prompt_tokens']} completion={u2['completion_tokens']}")
    # an identical repeat
    ch3, u3 = chat(msgs2, thinking, session)
    if ch3:
        print(f"  [step1 repeat] cached={u3.get('prompt_tokens_details')} prompt_tokens={u3['prompt_tokens']}")


def llama_phase():
    from bench_common import llama_greedy
    def tops(ids):
        _, _, cp = llama_greedy(LLAMA, ids, 1, P, n_probs=8)
        return [(P.tok.decode([x["id"]], skip_special_tokens=False), round(x.get("logprob", 0), 2)) for x in (cp[0].get("top_logprobs") or [])][:5] if cp else None
    for i, d in enumerate(json.load(open(SAVE))):
        msgs, message = d["msgs"], d["message"]
        prev = P.render_messages(msgs, G.TOOLS, None)
        raw = P.raw_from_chat(message, True, None, prev)
        print(f"llama {i} {d['pin']} calls={len(message.get('tool_calls') or [])}")
        ends = [j + len("</tool_call>") for j in range(len(raw)) if raw.startswith("</tool_call>", j)]
        th = raw.find("</think>")
        print("   before </think>:", tops(P.ids(prev + raw[:th])) if th >= 0 else None)
        for k, e in enumerate(ends[:-1]):
            print(f"   after call {k}:", tops(P.ids(prev + raw[:e])))
        if ends:
            print("   after last call:", tops(P.ids(prev + raw[:ends[-1]])))


if LLAMA:
    llama_phase(); sys.exit(0)


QUERIES = ["Check the weather in Tokyo and London.", "What's the weather in Paris and in Berlin? Use the tool for both.",
           "Check the weather for Madrid, Rome and Oslo.", "Get the weather for Cairo and Lima, then tell me which is warmer.",
           "What is the weather in Sydney and in Toronto?", "Compare the weather in Boston and Denver.",
           "Look up the weather in Vienna, Prague and Warsaw.", "Weather in Dublin and Lisbon please.",
           "Tell me the weather in Seoul, Bangkok and Hanoi.", "Check Moscow and Kyiv weather and the sum 17 + 25."]


def sweep():
    """Several one-turn two/three-call requests: raw reply, whether the finished session is reusable (a pin is accepted or
    hits the capacity limit; 409 'reusable history' means not reusable) and the next turn's cached tokens."""
    saved = []
    for i, q in enumerate(QUERIES):
        msgs = [SYS, {"role": "user", "content": q}]
        sess = f"sw{i}"
        ch, u = chat(msgs, None, sess)
        if not ch: continue
        m = ch["message"]
        r = requests.post(f"{url}/v1/checkpoints/pin-{sess}", headers={"X-Loom-Session": sess}, timeout=60)
        pin = "REUSABLE" if r.status_code != 409 else "NOT-REUSABLE"
        m2 = msgs + tool_msgs(ch)
        ch2, u2 = chat(m2, None, sess)
        cached = u2["prompt_tokens_details"]["cached_tokens"] if ch2 else None
        print(f"sweep {i} q={q[:30]!r} calls={len(m.get('tool_calls') or [])} completion={u['completion_tokens']} {pin} ({r.status_code}) next cached={cached} of {u2['prompt_tokens'] if ch2 else None}")
        saved.append({"msgs": msgs, "message": m, "pin": pin})
    json.dump(saved, open(SAVE, "w"))


sweep()

scenario("paris-one-call-control", None, 0, session="ctl", pin=True)
scenario("tokyo-london plain", None, 1)
scenario("tokyo-london named+pin", None, 1, session="tl", pin=True)
scenario("tokyo-london stream", None, 1, stream_first=True)
scenario("tokyo-london thinking-off named+pin", False, 1, session="tloff", pin=True)
