"""Agent-loop gate: multi-turn tool calling on Loom, checked turn by turn.

For every assistant turn of a deterministic tool-using conversation it checks
  1. rendering: Loom's /v1/chat/completions/input_tokens == the template-exact rendering
     (chat_probe.simulate, which inspect_gguf verified against the model's Jinja template),
     tokenized with HF. (usage.prompt_tokens of a turn that reuses retained history is one
     lower: the previous turn's EOS is already in the row);
  2. reuse: prompt_tokens_details.cached_tokens (the previous turn's history is retained);
  3. golden: Loom's greedy output vs llama.cpp's on the *same prompt ids* (log-prob gap at
     the first divergence; a near-tie is <= 1 nat with Loom's token in the top-5, as in compare.py), when --llama is given;
  4. tool calls: parsed name/arguments are valid JSON for the declared function.

usage: agent_golden.py PKG LOOM_URL [--llama URL] [--thinking on|off|default] [--out FILE]
Exit status 1 if a rendering, reuse or tool-call check fails or a divergence is not a near-tie.
"""
import argparse
import json
import sys

import requests

from bench_common import Package, llama_greedy, top_logprobs

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
WEATHER = {"paris": {"temp_c": 18, "sky": "sunny"}, "tokyo": {"temp_c": 24, "sky": "cloudy"},
           "london": {"temp_c": 12, "sky": "rain"}}
TASKS = [
    "What's the weather in Paris right now? Use the tool.",
    "Now check Tokyo and London too, then tell me which of the three cities is warmest.",
    "What is (24 - 12) * 3.5? Use the calculator.",
]


def run_tool(call):
    f = call["function"]
    args = json.loads(f["arguments"])
    if f["name"] == "get_weather":
        return json.dumps(WEATHER.get(str(args.get("city", "")).lower(), {"error": "unknown city"}))
    if f["name"] == "calculator":
        expr = str(args.get("expression", ""))
        ok = set(expr) <= set("0123456789+-*/(). ")
        return json.dumps({"result": eval(expr)} if ok else {"error": "bad expression"})  # noqa: S307 (digits/operators only)
    return json.dumps({"error": "unknown tool"})


def post(url, body, path="/v1/chat/completions"):
    # Serialized like the template's `tojson` so tool definitions print identically.
    data = json.dumps(body, ensure_ascii=False).encode()
    return requests.post(url + path, data=data, timeout=1800, headers={"Content-Type": "application/json"}).json()


def gap_at(ids, tops, loom_ids, k):
    """logp(golden) - logp(loom) at divergence k; 0.0 past the aligned top-5 lists (a byte-split token,
    counted as a near-tie as in compare.py); None when Loom's token is outside the top-5."""
    if k >= len(tops):
        return 0.0
    top = top_logprobs(tops, k)
    return top[ids[k]] - top[loom_ids[k]] if loom_ids[k] in top else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("url")
    ap.add_argument("--llama", default=None)
    ap.add_argument("--thinking", default="default", choices=["on", "off", "default"])
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = Package(a.pkg)
    names = {t["function"]["name"] for t in TOOLS}
    thinking = None if a.thinking == "default" else a.thinking == "on"
    extra = {} if thinking is None else {"chat_template_kwargs": {"enable_thinking": thinking}}
    msgs = [{"role": "system", "content": "You are a helpful assistant. Use the tools when they help."}]
    turns, failures, prev_total = [], 0, 0
    for task in TASKS:
        msgs.append({"role": "user", "content": task})
        for step in range(6):
            r = post(a.url, dict(model=P.model_id, messages=msgs, tools=TOOLS, max_tokens=a.max_tokens, temperature=0, **extra))
            if "error" in r:
                print("LOOM ERROR", r["error"]); failures += 1; break
            ch = r["choices"][0]; m = ch["message"]; u = r["usage"]
            prompt = P.render_messages(msgs, TOOLS, thinking)
            ids = P.ids(prompt)
            counted = post(a.url, dict(model=P.model_id, messages=msgs, tools=TOOLS, **extra), "/v1/chat/completions/input_tokens")
            rec = {"task": task[:40], "step": step, "prompt_tokens": counted.get("input_tokens"), "rendered_tokens": len(ids),
                   "cached": u.get("prompt_tokens_details", {}).get("cached_tokens", 0), "completion": u["completion_tokens"],
                   "finish": ch["finish_reason"], "calls": [c["function"]["name"] for c in m.get("tool_calls") or []]}
            rec["render_ok"] = rec["prompt_tokens"] == rec["rendered_tokens"]
            # Inside a tool loop the previous turn's prompt + output is retained and must be reused.
            # (A new user query may legitimately re-render earlier turns: Qwen templates drop old reasoning.)
            rec["reuse_ok"] = step == 0 or rec["cached"] >= 0.8 * prev_total
            calls_ok = True
            for c in m.get("tool_calls") or []:
                try:
                    json.loads(c["function"]["arguments"])
                    calls_ok &= c["function"]["name"] in names
                except ValueError:
                    calls_ok = False
            rec["calls_ok"] = calls_ok
            if a.llama:
                raw = P.raw_from_chat(m, ch["finish_reason"] in ("stop", "tool_calls"), thinking, prompt)
                loom_ids = P.ids(raw)
                g, gids, gtops = llama_greedy(a.llama, ids, max(1, u["completion_tokens"]), P)
                k = next((i for i, (x, y) in enumerate(zip(loom_ids, gids)) if x != y), None)
                import chat_probe as _cp
                if k is not None and hasattr(_cp, "raw_alternatives"):
                    # A kit whose channel format is ambiguous in the parsed message (Glimmer: an optional
                    # <|eom|> before the answer header) lists every raw text consistent with it: use the
                    # one that agrees longest with the golden.
                    for alt in _cp.raw_alternatives(P.chat, m, ch["finish_reason"] in ("stop", "tool_calls"), thinking, prompt)[1:]:
                        aids = P.ids(alt)
                        ka = next((i for i, (x, y) in enumerate(zip(aids, gids)) if x != y), None)
                        if ka is None or ka > k:
                            loom_ids, k = aids, ka
                rec["golden_match"] = k is None
                if k is not None:
                    gap = gap_at(gids, gtops, loom_ids, k)
                    top = {t["id"]: t["logprob"] for t in gtops[k]["top_logprobs"]} if k < len(gtops) else {}
                    # llama.cpp ROCm is not run-to-run deterministic on some models (flint: identical
                    # temperature-0 requests gave different greedy text): as compare.py does, re-record a
                    # turn that would fail and score against the recording Loom follows further.
                    for salt in ("rec2", "rec3"):
                        if gap is not None and gap <= 1.0:
                            break
                        _, gids2, cp2 = llama_greedy(a.llama, ids, max(1, u["completion_tokens"]), P, salt)
                        k2 = next((i for i, (x, y) in enumerate(zip(loom_ids, gids2)) if x != y), None)
                        rec.setdefault("rerecorded", []).append(k2)
                        if k2 is None:
                            k, gap, gids = None, None, gids2
                            break
                        if k2 > k:
                            k, gids, gtops = k2, gids2, cp2
                            gap = gap_at(gids, gtops, loom_ids, k)
                            top = {t["id"]: t["logprob"] for t in gtops[k]["top_logprobs"]} if k < len(gtops) else {}
                    if k is None:
                        rec["golden_match"] = True
                    rec.update(diverge_at=k, of=len(gids), gap=gap, near_tie=k is None or (gap is not None and gap <= 1.0))
                    if not rec["near_tie"]:
                        print("     at the divergence: loom", repr(P.tok.decode(loom_ids[max(0, k - 8):k + 4], skip_special_tokens=False)),
                              "| llama", repr(P.tok.decode(gids[max(0, k - 8):k + 4], skip_special_tokens=False)),
                              "| top", [(P.tok.id_to_token(t), round(v, 2)) for t, v in top.items()], flush=True)
            golden_ok = rec.get("golden_match", True) or rec.get("near_tie", False)
            ok = rec["render_ok"] and rec["reuse_ok"] and rec["calls_ok"] and golden_ok
            failures += not ok
            print(("OK  " if ok else "FAIL"), json.dumps(rec), flush=True)
            if ch["finish_reason"] == "stop" and not (m.get("content") or m.get("tool_calls")):
                print("     note: the model ended inside its reasoning (EOS before the think close); with greedy decoding"
                      " small thinking models do this - compare llama.cpp's choice at that point, not a Loom bug by itself")
            turns.append(rec)
            prev_total = u["prompt_tokens"] + u["completion_tokens"]
            # loom-serve rejects "reasoning_content": null, so only send it when present.
            msgs.append({"role": "assistant", "content": m.get("content") or "",
                         **({"reasoning_content": m["reasoning_content"]} if m.get("reasoning_content") else {}),
                         **({"tool_calls": m["tool_calls"]} if m.get("tool_calls") else {})})
            if not m.get("tool_calls"):
                break
            for c in m["tool_calls"]:
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": run_tool(c)})
    print(f"{len(turns)} turns, {failures} failing; final answer: {(msgs[-1].get('content') or '')[:200]!r}")
    if a.out:
        json.dump({"turns": turns, "messages": msgs}, open(a.out, "w"), indent=1)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
