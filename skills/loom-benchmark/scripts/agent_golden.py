"""Agent-loop gate: multi-turn tool calling on Loom, checked turn by turn.

For every assistant turn of a deterministic tool-using conversation it checks
  1. rendering: Loom's /v1/chat/completions/input_tokens == the template-exact rendering
     (chat_probe.simulate, which inspect_gguf verified against the model's Jinja template),
     tokenized with HF. (usage.prompt_tokens of a turn that reuses retained history is one
     lower: the previous turn's EOS is already in the row);
  2. reuse: prompt_tokens_details.cached_tokens (the previous turn's history is retained);
  3. golden: Loom's greedy output vs llama.cpp's on the *same prompt ids* (log-prob gap at
     the first divergence; a near-tie by the compare rule, bench_common.near_tie), when --llama is given;
  4. tool calls: parsed name/arguments are valid JSON for the declared function.

usage: agent_golden.py PKG LOOM_URL [--llama URL] [--thinking on|off|default] [--out FILE]
Exit status 1 if a rendering, reuse or tool-call check fails or a divergence is not a near-tie.
"""
import argparse
import json
import sys

import requests

from bench_common import Package, llama_greedy, near_tie, text_divergence, top_logprobs

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


def near_at(ids, tops, loom_id, k):
    """The compare rule (bench_common.near_tie) at divergence k (Loom's token there: loom_id); past the
    aligned top-5 lists (a byte-split token) it counts as a near-tie, as in compare.py."""
    return k >= len(tops) or near_tie(top_logprobs(tops, k), ids[k], loom_id)


def gap_at(ids, tops, loom_id, k):
    """logp(golden) - logp(loom) at divergence k; 0.0 past the aligned top-5 lists (a byte-split token,
    counted as a near-tie as in compare.py); None when Loom's token is outside the top-5."""
    if k >= len(tops):
        return 0.0
    top = top_logprobs(tops, k)
    return top[ids[k]] - top[loom_id] if loom_id in top else None


def call_whitespace(P, raw, gids, d):
    """Parsed tool calls do not keep the whitespace the model wrote between or after them (the rebuild
    prints call_separator), so any whitespace there is consistent with Loom's reply. At a divergence inside
    the calls that is whitespace on either side, take the golden's whitespace and compare on (only while
    that moves the divergence later). -> (raw, divergence)"""
    marker = P.chat.get("call_marker") or ""
    start = raw.find(marker) if marker else -1
    for _ in range(16):
        if d is None or start < 0 or not raw.endswith(d[2]):
            break
        k, _, rest = d
        pos = len(raw) - len(rest)
        gold = P.tok.decode(gids[k:], skip_special_tokens=False)
        gws, lws = gold[:len(gold) - len(gold.lstrip())], rest[:len(rest) - len(rest.lstrip())]
        if pos < start or not (gws or lws):
            break
        alt = raw[:pos] + gws + rest[len(lws):]
        da = text_divergence(P.tok, P.ids(alt), gids)
        if da is not None and da[0] <= k:
            break
        raw, d = alt, da
    return raw, d


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
    turns, failures, prev_total, prev_text = [], 0, 0, ""
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
            # A template that rewrites the retained history (e.g. drops the previous turn's reasoning when
            # thinking is on) leaves nothing verbatim to reuse: require reuse only when the previous prompt +
            # output is a prefix of this prompt. loom-serve resumes a retained row only when this prompt
            # extends its history exactly (it cannot cut a row back to a common prefix), so when the template
            # re-renders bytes the model verifiably generated (the previous turn matched the golden), nothing
            # is reusable however late the change (Gemma 4 with thinking: a "\n" before <channel|> 15 tokens
            # from the end). Unverified output (no golden, or it diverged) keeps the 80 % rule: a wrong
            # rebuild of the output must not hide a reuse bug.
            if step:
                pids = P.ids(prev_text)
                common = next((i for i, (x, y) in enumerate(zip(pids, ids)) if x != y), min(len(pids), len(ids)))
                exact = prompt.startswith(prev_text)
                rewritten = not exact and turns[-1].get("golden_match") is True   # previous output verified
                rec["history_verbatim"] = exact or (common >= 0.8 * len(pids) and not rewritten)
                if exact and turns[-1].get("output_dropped"):
                    # The previous reply's parse dropped a second call (output_dropped below): the retained history
                    # holds output this prompt does not, so it is no exact extension, whatever the rebuilt text says.
                    rec["history_verbatim"], rec["history_rewritten_at"] = False, turns[-1]["output_dropped"]
                    print(f"     note: the previous reply generated {turns[-1]['output_dropped'][1]} tokens, its parsed channels "
                          f"account for {turns[-1]['output_dropped'][0]}: the rest is not in this prompt, so nothing is reusable", flush=True)
                if rewritten and common >= 0.8 * len(pids):
                    rec["history_rewritten_at"] = [common, len(pids)]
                    print(f"     note: the template re-renders the previous turn from token {common} of {len(pids)}; loom-serve reuses"
                          " only exact extensions of a retained history (llama.cpp would reuse the common prefix)", flush=True)
            else:
                rec["history_verbatim"] = True
            rec["reuse_ok"] = step == 0 or not rec["history_verbatim"] or rec["cached"] >= 0.8 * prev_total
            calls_ok = True
            for c in m.get("tool_calls") or []:
                try:
                    json.loads(c["function"]["arguments"])
                    calls_ok &= c["function"]["name"] in names
                except ValueError:
                    calls_ok = False
            rec["calls_ok"] = calls_ok
            raw = None   # the rebuilt output, completed from the golden where the reply can't tell (below)
            if a.llama:
                raw = P.raw_from_chat(m, ch["finish_reason"] in ("stop", "tool_calls"), thinking, prompt)
                loom_ids = P.ids(raw)
                g, gids, gtops = llama_greedy(a.llama, ids, max(1, u["completion_tokens"]), P)
                # The same text split into other tokens is no divergence (bench_common.text_divergence).
                d = text_divergence(P.tok, loom_ids, gids)
                import chat_probe as _cp
                finished = ch["finish_reason"] in ("stop", "tool_calls")
                alts = []
                if hasattr(_cp, "raw_alternatives"):
                    # A kit whose channel format is ambiguous in the parsed message (Glimmer: an optional
                    # <|eom|> before the answer header) lists every raw text consistent with it.
                    alts = _cp.raw_alternatives(P.chat, m, finished, thinking, prompt)[1:]
                if finished and not (m.get("content") or m.get("tool_calls")):
                    # Nothing after the reasoning: the model either closed it and stopped, or stopped inside it
                    # (EOS before the think close; small thinking models do under greedy). The parsed message
                    # can't tell, so the unclosed rebuild is an alternative too.
                    alts.append(P.raw_from_chat(m, False, thinking, prompt))
                if d is not None:
                    # Use the rebuild that agrees longest with the golden.
                    for alt in alts:
                        aids = P.ids(alt)
                        da = text_divergence(P.tok, aids, gids)
                        if da is None or da[0] > d[0]:
                            raw, loom_ids, d = alt, aids, da
                            if d is None:
                                break
                raw, d = call_whitespace(P, raw, gids, d)
                loom_ids = P.ids(raw)
                rec["golden_match"] = d is None
                if d is not None:
                    k, loom_id, loom_rest = d
                    gap = gap_at(gids, gtops, loom_id, k)
                    top = {t["id"]: t["logprob"] for t in gtops[k]["top_logprobs"]} if k < len(gtops) else {}
                    # llama.cpp ROCm is not run-to-run deterministic on some models (flint: identical
                    # temperature-0 requests gave different greedy text): as compare.py does, re-record a
                    # turn that would fail and score against the recording Loom follows further.
                    for salt in ("rec2", "rec3"):
                        if near_at(gids, gtops, loom_id, k):
                            break
                        _, gids2, cp2 = llama_greedy(a.llama, ids, max(1, u["completion_tokens"]), P, salt)
                        d2 = text_divergence(P.tok, loom_ids, gids2)
                        rec.setdefault("rerecorded", []).append(d2 and d2[0])
                        if d2 is None:
                            k, gap, gids = None, None, gids2
                            break
                        if d2[0] > k:
                            (k, loom_id, loom_rest), gids, gtops = d2, gids2, cp2
                            gap = gap_at(gids, gtops, loom_id, k)
                            top = {t["id"]: t["logprob"] for t in gtops[k]["top_logprobs"]} if k < len(gtops) else {}
                    if k is None:
                        rec["golden_match"] = True
                    rec.update(diverge_at=k, of=len(gids), gap=gap, near_tie=k is None or near_at(gids, gtops, loom_id, k))
                    if not rec["near_tie"]:
                        agreed = P.tok.decode(gids[max(0, k - 8):k], skip_special_tokens=False)
                        print("     at the divergence: loom", repr(agreed + loom_rest[:24]),
                              "| llama", repr(agreed + P.tok.decode(gids[k:k + 4], skip_special_tokens=False)),
                              "| top", [(P.tok.id_to_token(t), round(v, 2)) for t, v in top.items()], flush=True)
                        # TEMPORARY debug: the same request again (no retained row is a prefix of it: a fresh prefill)
                        r2 = post(a.url, dict(model=P.model_id, messages=msgs, tools=TOOLS, max_tokens=a.max_tokens, temperature=0, **extra))
                        print("     DEBUG same request again: cached", r2.get("usage", {}).get("prompt_tokens_details"),
                              repr(str(r2["choices"][0]["message"].get("content"))[:60]), flush=True)
                        # TEMPORARY debug: llama.cpp on the same ids when it reuses its cache for the first
                        # rec["cached"] tokens (an incremental prefill, like Loom's reused turn), and with ubatch-sized steps
                        import bench_common as _bc
                        for cut in (rec["cached"], len(ids) - 1, len(ids) // 2):
                            _bc.llama_completion(a.llama, dict(_bc.GREEDY, prompt=ids[:cut], n_predict=1, cache_prompt=True), P, f"dbg{cut}")
                            rr = _bc.llama_completion(a.llama, dict(_bc.GREEDY, prompt=ids, n_predict=4, n_probs=5, cache_prompt=True), P, f"dbg{cut}")
                            cp_ = rr.get("completion_probabilities", [])
                            print("     DEBUG llama after a cached prefix of", cut, "tokens:", repr(rr.get("content", "")[:30]),
                                  [(t.get("token"), round(t["logprob"], 2)) for t in (cp_[0]["top_logprobs"] if cp_ else [])], flush=True)
                # Output the parser dropped (a second call where the template takes one): the reply generated
                # more tokens than its parsed channels hold, the golden agrees with all of those, and the golden's
                # continuation past them opens another call. Only then is the next turn's reuse not required.
                marker = P.chat.get("call_marker")
                if rec["golden_match"] and marker and u["completion_tokens"] > len(loom_ids) + 1 \
                        and marker in P.tok.decode(gids[len(loom_ids):], skip_special_tokens=False):
                    rec["output_dropped"] = [len(loom_ids), u["completion_tokens"]]
            golden_ok = rec.get("golden_match", True) or rec.get("near_tie", False)
            ok = rec["render_ok"] and rec["reuse_ok"] and rec["calls_ok"] and golden_ok
            failures += not ok
            print(("OK  " if ok else "FAIL"), json.dumps(rec), flush=True)
            if ch["finish_reason"] == "stop" and not (m.get("content") or m.get("tool_calls")):
                print("     note: the model ended inside its reasoning (EOS before the think close); with greedy decoding"
                      " small thinking models do this - compare llama.cpp's choice at that point, not a Loom bug by itself")
            # The history this turn leaves: its output as the golden comparison settled it (e.g. the model's own
            # whitespace between calls, which the template may re-render), else the plain rebuild.
            out_text = raw if raw is not None else P.raw_from_chat(m, ch["finish_reason"] in ("stop", "tool_calls"), thinking, prompt)
            turns.append(rec)
            prev_total = u["prompt_tokens"] + u["completion_tokens"]
            prev_text = prompt + out_text
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
