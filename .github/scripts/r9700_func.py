"""Functional checks of one served card (devlab_perf.sh step func:V): the Loom features a card claims, on the runner.

usage: python r9700_func.py URL MODEL OUT.json [--info info.json] [--vision] [--tools]

chat (always), tool calling (when the card's info.json claims it, or --tools), image input (the package ships a vision
tower, or --vision). Same checks as the Windows red-team harness (wincheck.py, branch wincheck/r9700). Exit 0 when every
check that applies passed; OUT.json = {"checks": {name: {"ok", "detail"}}, "verdict": "pass" | "fail: ..."}.
"""
import argparse
import base64
import json
import struct
import sys
import time
import urllib.request
import zlib

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--info")
ap.add_argument("--vision", action="store_true")
ap.add_argument("--tools", action="store_true")
a = ap.parse_args()

features = {}
if a.info:
    try:
        feats = (json.load(open(a.info)).get("card") or {}).get("features") or {}
        features = {k: v.get("status") for k, v in feats.items()}
    except (OSError, ValueError):
        pass
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
R = {"model": a.model, "claimed": features, "checks": {}}


def chat(messages, max_tokens=256, **extra):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, **extra}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    R["checks"][name] = {"ok": ok, "detail": detail}
    print(f"check {name}: {'OK' if ok else 'FAIL'} {str(detail)[:300]}", flush=True)


def c_chat():
    t = time.time()
    msg = chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}], 512, **NOTHINK)["choices"][0]["message"]
    text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    return "paris" in text.lower(), {"content": (msg.get("content") or "")[:200], "s": round(time.time() - t, 1)}


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def c_tools():
    msg = chat([{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], 1024,
               tools=TOOLS, **NOTHINK)["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and \
        "paris" in json.loads(tc[0]["function"]["arguments"] or "{}").get("city", "").lower()
    return ok, {"tool_calls": tc, "content": (msg.get("content") or "")[:200]}


def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def c_image():
    url = "data:image/png;base64," + base64.b64encode(png(224, 224, (220, 20, 20))).decode()
    msg = chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": "What color is this image? Answer in one word."}]}],
               512, **NOTHINK)["choices"][0]["message"]
    text = ((msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")).lower()
    return "red" in text, {"content": (msg.get("content") or "")[:200]}


check("chat", c_chat)
if features.get("tool_calling") == "yes" or a.tools:
    check("tool_calling", c_tools)
if features.get("image_input") == "yes" or a.vision:
    check("image_input", c_image)
bad = [k for k, v in R["checks"].items() if not v["ok"]]
R["verdict"] = "fail: " + ", ".join(bad) if bad else "pass"
json.dump(R, open(a.out, "w"), indent=1)
print(R["verdict"])
sys.exit(1 if bad else 0)
