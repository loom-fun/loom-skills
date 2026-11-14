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
    if name == "tokyo-london plain":
        json.dump({"msgs": msgs, "message": ch["message"]}, open(SAVE, "w"))
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
    d = json.load(open(SAVE))
    msgs, message = d["msgs"], d["message"]
    prev = P.render_messages(msgs, G.TOOLS, None)
    raw = P.raw_from_chat(message, True, None, prev)
    print("loom raw (rebuilt from channels):", repr(raw[-400:]))
    r, gen, _ = llama_greedy(LLAMA, P.ids(prev), 300, P)
    text = P.tok.decode(gen, skip_special_tokens=False)
    print("llama.cpp greedy from the same prompt:", repr(text))
    # continue llama.cpp after Loom's reasoning + the first call: what whitespace does the model write between calls?
    marker = P.chat.get("call_marker") or "<tool_call>"
    cut = raw.find("</tool_call>") + len("</tool_call>")
    for tag, tail in (("after first call", raw[:cut]),):
        _, g2, _ = llama_greedy(LLAMA, P.ids(prev + tail), 40, P)
        print(f"llama.cpp continuation {tag}:", repr(P.tok.decode(g2, skip_special_tokens=False)), g2[:12])
    # the whitespace tokens llama.cpp prefers right after </tool_call>
    _, g3, cp = llama_greedy(LLAMA, P.ids(prev + raw[:cut]), 1, P, n_probs=8)
    print("top after first </tool_call>:", [(P.tok.decode([x["id"]], skip_special_tokens=False), round(x.get("logprob", 0), 3)) for x in (cp[0].get("top_logprobs") or cp[0].get("top") or [])] if cp else cp)
    _, g4, cp4 = llama_greedy(LLAMA, P.ids(prev + raw[:raw.find("<tool_call>")]), 1, P, n_probs=8)
    print("top before first <tool_call>:", [(P.tok.decode([x["id"]], skip_special_tokens=False), round(x.get("logprob", 0), 3)) for x in (cp4[0].get("top_logprobs") or cp4[0].get("top") or [])] if cp4 else cp4)


if LLAMA:
    llama_phase(); sys.exit(0)


def rawtext():
    """The reply's own whitespace: the template's whole system block (tools included) is sent as the system message and no
    tools are declared, so the prompt tokens are the gate's and Loom returns the calls as plain content, bytes untouched."""
    full = P.render_messages([SYS, {"role": "user", "content": "x"}], G.TOOLS, None)
    head = full[:full.index("<|im_start|>user")]
    sysblock = head[len("<|im_start|>system\n"):-len("<|im_end|>\n")]
    print("=== rawtext: system block ends", repr(sysblock[-80:]))
    for q in ("Now check Tokyo and London too, then tell me which of the three cities is warmest.",
              "What's the weather in Paris and in Berlin? Use the tool for both.",
              "Check the weather for Madrid, Rome and Oslo.",
              "Get the weather for Cairo and Lima, then tell me which is warmer.",
              "What is the weather in Sydney and in Toronto?"):
        body = dict(model=model, messages=[{"role": "system", "content": sysblock}, {"role": "user", "content": q}],
                    max_tokens=1024, temperature=0)
        r = post(body).json()
        if "error" in r:
            print("  ERROR", r["error"]); continue
        m = r["choices"][0]["message"]
        print(f"  q={q[:40]!r} prompt_tokens={r['usage']['prompt_tokens']} reasoning_tail={(m.get('reasoning_content') or '')[-40:]!r}\n    content={m.get('content')!r}")


rawtext()

scenario("paris-one-call-control", None, 0, session="ctl", pin=True)
scenario("tokyo-london plain", None, 1)
scenario("tokyo-london named+pin", None, 1, session="tl", pin=True)
scenario("tokyo-london stream", None, 1, stream_first=True)
scenario("tokyo-london thinking-off named+pin", False, 1, session="tloff", pin=True)
