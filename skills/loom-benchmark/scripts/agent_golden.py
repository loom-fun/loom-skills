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
import difflib
import json
import sys

import requests

from bench_common import Package, llama_greedy, near_tie_at, text_divergence, top_logprobs

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


def near_at(ids, tops, loom_id, k, llama=None, prompt=(), pkg=None):
    """The compare rule (bench_common.near_tie) at divergence k (Loom's token there: loom_id); past the
    aligned top-5 lists (a byte-split token) it counts as a near-tie, as in compare.py. With llama and the
    turn's prompt ids, with llama.cpp's own rounding noise there (bench_common.near_tie_at)."""
    return k >= len(tops) or near_tie_at(llama, list(prompt) + ids[:k], top_logprobs(tops, k), ids[k], loom_id, pkg)


def gap_at(ids, tops, loom_id, k):
    """logp(golden) - logp(loom) at divergence k; 0.0 past the aligned top-5 lists (a byte-split token,
    counted as a near-tie as in compare.py); None when Loom's token is outside the top-5."""
    if k >= len(tops):
        return 0.0
    top = top_logprobs(tops, k)
    return top[ids[k]] - top[loom_id] if loom_id in top else None


def history_edits(hist, prompt):
    """Edits that turn the retained history HIST (the previous prompt + output) into the start of PROMPT, as character
    offsets (i1, i2, j1, j2): HIST[i1:i2] became PROMPT[j1:j2]. What PROMPT adds after the history is not an edit."""
    sm = difflib.SequenceMatcher(None, hist, prompt[:len(hist) + 256], autojunk=False)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag != "equal" and not (tag == "insert" and i1 == len(hist))]


MERGE_GAP = 64   # characters


def template_rewrite(prev_prompt, prev_out, verified, prompt, continuation=None, first_token=lambda s: s[:1]):
    """Is PROMPT's change of the retained history (PREV_PROMPT + PREV_OUT) the template's? Each edit's replaced span must lie
    inside the previous prompt (e.g. thinking off: a generation prompt's empty think block, which history drops) or
    inside the first VERIFIED characters of the previous output, the part that matched the golden. Any other edit is
    anchored on the reference instead: CONTINUATION(k, n) is llama.cpp's greedy text (about n characters) after the
    previous prompt + PREV_OUT[:k]; the edit is the template's only when that text writes what the history holds
    there (the replaced text and the token after it, FIRST_TOKEN) and not the replacement. Otherwise the gate can't
    tell a template rewrite from a wrong rebuild of what Loom generated."""
    hist, n = prev_prompt + prev_out, len(prev_prompt)
    def ok(i1, i2, j1, j2):
        if i1 < n + verified and i2 <= n + verified:   # the edit lies in the prompt or the verified output
            return True
        if continuation is None:
            return False
        ctx = first_token(hist[i2:i2 + 64])
        mine, tmpl = hist[i1:i2] + ctx, prompt[j1:j2] + ctx
        cont = continuation(i1 - n, max(len(mine), len(tmpl)))
        template_rewrite.trace.append((i1 - n, mine, tmpl, cont))
        return cont.startswith(mine) and not cont.startswith(tmpl)
    edits = history_edits(hist, prompt)
    # difflib splits one rewrite into character-level fragments around text both sides share (harmony: "<|channel|>commentary
    # to=functions.f <|constrain|>json" -> " to=functions.f<|channel|>commentary json" is 5 edits, cut inside words where
    # llama.cpp's continuation has nothing to say). Edits less than MERGE_GAP characters apart are also judged as the one
    # span they make up; a span passes when the reference writes the history's text over all of it, or every fragment passes.
    spans, group = [], []
    for e in edits:
        if group and e[0] - group[-1][1] >= MERGE_GAP:
            spans.append(group); group = []
        group.append(e)
    spans.append(group) if group else None
    return bool(edits) and all(len(g) > 1 and ok(g[0][0], g[-1][1], g[0][2], g[-1][3]) or all(ok(*e) for e in g) for g in spans)


template_rewrite.trace = []   # (offset in the output, history text, template text, llama.cpp's continuation) of every anchored edit


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
    prev_prompt, prev_verified = "", 0   # the previous turn's prompt, and how much of its output the golden verified
    prev_ids = []
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
            # from the end). An edit past the verified output (after a near-tie divergence) is the template's only
            # where llama.cpp's own continuation at that point writes what the rebuilt history holds; otherwise (or
            # without a golden) the 80 % rule stays: a wrong rebuild of the output must not hide a reuse bug.
            if step:
                pids = P.ids(prev_text)
                common = next((i for i, (x, y) in enumerate(zip(pids, ids)) if x != y), min(len(pids), len(ids)))
                exact = prompt.startswith(prev_text)
                po = prev_text[len(prev_prompt):]
                def continuation(k, nchars):   # llama.cpp greedy after the previous prompt + Loom's output up to k
                    _, gen, _ = llama_greedy(a.llama, prev_ids + P.ids(po[:k]), min(nchars + 2, 64), P)
                    return P.tok.decode(gen, skip_special_tokens=False)
                first = lambda t: P.tok.decode(P.ids(t)[:1], skip_special_tokens=False)
                template_rewrite.trace.clear()
                rewritten = not exact and template_rewrite(prev_prompt, po, prev_verified, prompt,
                                                           continuation if a.llama else None, first)
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
            if step and not rec["reuse_ok"]:   # what the retained history (previous prompt + rebuilt output) lacks of this prompt
                hist = prev_text
                print(f"     reuse miss: cached {rec['cached']} of {prev_total}, previous output verified to char {prev_verified}; "
                      f"history edits (hist[i1:i2] -> prompt[j1:j2]):", flush=True)
                for t in template_rewrite.trace[-6:]:
                    print(f"       anchored on llama.cpp at output offset {t[0]}: history {t[1]!r} template {t[2]!r} llama.cpp wrote {t[3]!r}", flush=True)
                for i1, i2, j1, j2 in history_edits(hist, prompt)[:6]:
                    print(f"       @{i1} (prompt {len(prev_prompt)}) {hist[max(0, i1 - 30):i2 + 30]!r} -> {prompt[max(0, j1 - 30):j2 + 30]!r}", flush=True)
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
                gp, plain = P.chat.get("generation_prompt", ""), P.chat.get("generation_prompt_plain", "")
                on = P.chat.get("thinking_default") if thinking is None else thinking
                block = plain[len(gp):] if plain.startswith(gp) else ""
                if block and not on and not prompt.endswith(plain) and not m.get("reasoning_content") \
                        and u["completion_tokens"] >= len(P.ids(block + raw)):
                    # Thinking off with a generation prompt that adds an empty reasoning block (Gemma 4 12B/26B/31B):
                    # where the prompt does not end with it (the turn continues after tool results), the model may
                    # write that block itself, which the parsed reply can't show; only when Loom generated enough
                    # tokens to hold it (as output_dropped counts), so it can't hide a divergence at token 0.
                    alts += [block + r for r in [raw] + alts]
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
                loom_rest = ""   # set below on a divergence; the unpublished-output rule can fail a matching turn
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
                    rec.update(diverge_at=k, of=len(gids), gap=gap, near_tie=k is None or near_at(gids, gtops, loom_id, k, a.llama, ids, P))
                    if not rec["near_tie"]:
                        agreed = P.tok.decode(gids[max(0, k - 8):k], skip_special_tokens=False)
                        print("     at the divergence: loom", repr(agreed + loom_rest[:24]),
                              "| llama", repr(agreed + P.tok.decode(gids[k:k + 4], skip_special_tokens=False)),
                              "| top", [(P.tok.id_to_token(t), round(v, 2)) for t, v in top.items()], flush=True)
                # Output the parser dropped (a second call where the template takes one): the reply generated
                # more tokens than its parsed channels hold, the golden agrees with all of those, and the golden's
                # continuation past them opens another call. Only then is the next turn's reuse not required.
                marker = P.chat.get("call_marker")
                if rec["golden_match"] and marker and m.get("tool_calls") and u["completion_tokens"] > len(loom_ids) + 1 \
                        and marker in P.tok.decode(gids[len(loom_ids):], skip_special_tokens=False):
                    rec["output_dropped"] = [len(loom_ids), u["completion_tokens"]]
                elif rec["golden_match"] and u["completion_tokens"] > len(loom_ids) + 4:
                    # Generated tokens the parsed channels do not hold (and not a dropped second call): an empty or
                    # short reply is a prefix of any golden, so it is not a match.
                    rec["golden_match"], rec["near_tie"], rec["unpublished"] = False, False, [len(loom_ids), u["completion_tokens"]]
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
            prev_prompt, prev_ids = prompt, ids
            prev_verified = len(out_text) if rec.get("golden_match") else len(out_text) - len(loom_rest) if a.llama else 0
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
