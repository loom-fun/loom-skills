"""TEMPORARY debug probe (not for merge): loom-serve input_tokens vs chat_probe.simulate on request variants."""
import json

import requests

TOOL_MIN = {"type": "function", "function": {"name": "f"}}
TOOL_DESC = {"type": "function", "function": {"name": "f", "description": "d"}}
TOOL_PARAMS = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def probe(P, url, tools_full):
    sysm = {"role": "system", "content": "You are a helpful assistant. Use the tools when they help."}
    u = {"role": "user", "content": "hi"}
    cases = [("user", [u], None), ("sys+user", [sysm, u], None), ("user+2tools", [u], tools_full),
             ("sys+user+2tools", [sysm, u], tools_full), ("user+tool0", [u], tools_full[:1]),
             ("user+tool1", [u], tools_full[1:]), ("user+min", [u], [TOOL_MIN]), ("user+min+min", [u], [TOOL_MIN, TOOL_MIN]),
             ("user+desc", [u], [TOOL_DESC]), ("user+params", [u], [TOOL_PARAMS]),
             ("sys+user+min", [sysm, u], [TOOL_MIN])]
    for th in (False, True):
        for name, msgs, tools in cases:
            body = dict(model=P.model_id, messages=msgs, chat_template_kwargs={"enable_thinking": th})
            if tools:
                body["tools"] = tools
            r = requests.post(url + "/v1/chat/completions/input_tokens", data=json.dumps(body, ensure_ascii=False).encode(),
                              headers={"Content-Type": "application/json"}).json()
            sim = P.render_messages(msgs, tools, th)
            print("PROBE", "on " if th else "off", f"{name:18s}", "loom", r.get("input_tokens", r), "sim", len(P.ids(sim)), flush=True)
