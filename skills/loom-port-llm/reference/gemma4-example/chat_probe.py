"""Derives the control program's "chat" block from a model's Jinja chat template.

The template is rendered on probe conversations and the pieces the control
program needs are cut out of the output: role prefixes/suffix, generation
prompts with thinking on/off, the default system prompt, the tools block and
its order relative to the system message, the tool-call format and separators,
tool-response framing, how reasoning appears in history, and content trimming.
`verify()` then re-renders a set of conversations the way control.loom will
(`simulate()`) and compares them with the template, so mismatches show up
before anything runs on the GPU.

usage: python chat_probe.py TEMPLATE_SOURCE [OUT.json]
  TEMPLATE_SOURCE: a .gguf (tokenizer.chat_template), a tokenizer_config.json, or a .jinja file
  (inspect_gguf.py calls derive() + verify() for you)
"""
import datetime
import json
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.ext import Extension


class GenerationTag(Extension):
    """HF's `{% generation %}...{% endgeneration %}` (marks assistant tokens for training masks, e.g.
    LFM2.5): renders its body unchanged, as transformers' apply_chat_template does."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.Scope(body, lineno=lineno)

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
CALL1 = {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}
CALL2 = {"name": "search", "arguments": {"query": "louvre hours"}}


class ProbeError(Exception):
    pass


def environment():
    def raise_exception(message):
        raise ProbeError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols", GenerationTag])
    # Same as transformers' chat-template environment.
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env


def load_template(source):
    if source.endswith(".gguf"):
        import gguf
        f = gguf.GGUFReader(source).fields.get("tokenizer.chat_template")
        return f.contents() if f is not None else ""
    if source.endswith(".json"):
        t = json.load(open(source, encoding="utf-8")).get("chat_template", "")
        if isinstance(t, list):  # named templates: take "default"
            t = {x["name"]: x["template"] for x in t}.get("default", "")
        return t
    return open(source, encoding="utf-8").read()


class Template:
    def __init__(self, text, bos="", eos=""):
        self.t = environment().from_string(text)
        self.bos, self.eos = bos, eos

    def render(self, messages, tools=None, gen=False, **kw):
        return self.t.render(messages=messages, tools=tools, add_generation_prompt=gen,
                             bos_token=self.bos, eos_token=self.eos, **kw)


def calls_msg(content, calls, reasoning=None):
    m = {"role": "assistant", "content": content,
         "tool_calls": [{"id": f"call_{i}", "type": "function", "function": c} for i, c in enumerate(calls)]}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


def between(text, a, b, start=0):
    i = text.index(a, start) + len(a)
    return text[i:text.index(b, i)]


def derive(template_text, bos="", eos=""):
    T = Template(template_text, bos, eos)
    if "<|turn>" in template_text and "<|tool_call>call:" in template_text:
        return derive_gemma4(T, bos)
    notes = []
    chat = {}
    # BOS: emitted by the control program at the very start when the template prints it.
    chat["bos"] = bos if bos and T.render([{"role": "user", "content": "USERPROBE1"}]).startswith(bos) else ""
    # Prefixes and suffix: an assistant turn before the last user query renders plainly.
    r = T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"},
                  {"role": "assistant", "content": "ASSTPROBE1"}, {"role": "user", "content": "USERPROBE2"}])
    if r.startswith(bos) and bos:
        r = r[len(bos):]
    chat["system_prefix"] = r[:r.index("SYSPROBE")]
    single = T.render([{"role": "user", "content": "USERPROBE1"}])
    chat["suffix"] = single[single.index("USERPROBE1") + len("USERPROBE1"):]
    gap = between(r, "SYSPROBE", "USERPROBE1")
    if not gap.startswith(chat["suffix"]):
        raise ProbeError(f"system message is not closed by the user suffix {chat['suffix']!r}: {gap!r}")
    chat["user_prefix"] = gap[len(chat["suffix"]):]
    gap = between(r, "USERPROBE1", "ASSTPROBE1")
    chat["assistant_prefix"] = gap[len(chat["suffix"]):]
    head = single[:single.index("USERPROBE1")]
    if head.startswith(bos) and bos:
        head = head[len(bos):]
    default = head[:len(head) - len(chat["user_prefix"])] if head.endswith(chat["user_prefix"]) else ""
    if default:
        if not (default.startswith(chat["system_prefix"]) and default.endswith(chat["suffix"])):
            raise ProbeError(f"default system block has an unexpected shape: {default!r}")
        chat["default_system"] = default[len(chat["system_prefix"]):len(default) - len(chat["suffix"])]
        notes.append(f"template inserts a default system prompt: {chat['default_system'][:60]!r}...")
    else:
        chat["default_system"] = ""
    trimmed = T.render([{"role": "user", "content": "  USERPROBE1 \n"}])
    chat["trim_content"] = "  USERPROBE1" not in trimmed
    # Generation prompts and thinking.
    base = T.render([{"role": "user", "content": "USERPROBE1"}])
    gen = lambda **kw: T.render([{"role": "user", "content": "USERPROBE1"}], gen=True, **kw)[len(base):]
    g_default, g_on, g_off = gen(), gen(enable_thinking=True), gen(enable_thinking=False)
    # Reasoning in history (agent loop: assistant after the last user query, followed by a tool result).
    agent = lambda m: T.render([{"role": "user", "content": "USERPROBE1"}, m,
                                {"role": "tool", "tool_call_id": "call_0", "content": "TOOLPROBE1"}], tools=TOOLS)
    with_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning="REASONPROBE"))
    without_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning=""))
    mid_r = between(with_r, "USERPROBE1", "ASSTPROBE1")
    if "REASONPROBE" in mid_r:
        pre = mid_r[:mid_r.index("REASONPROBE")]
        cut = pre.index(chat["assistant_prefix"]) + len(chat["assistant_prefix"]) if chat["assistant_prefix"] in pre else None
        if cut is None:
            raise ProbeError(f"assistant prefix {chat['assistant_prefix']!r} not found before reasoning: {mid_r!r}")
        chat["think_open"] = pre[cut:]
        chat["think_close"] = mid_r[mid_r.index("REASONPROBE") + len("REASONPROBE"):]
        mid_n = between(without_r, "USERPROBE1", "ASSTPROBE1")
        empty_block = mid_n[mid_n.index(chat["assistant_prefix"]) + len(chat["assistant_prefix"]):]
        chat["history_think"] = "always" if empty_block == chat["think_open"] + chat["think_close"] else "nonempty"
        if empty_block not in ("", chat["think_open"] + chat["think_close"]):
            notes.append(f"empty reasoning renders as {empty_block!r} (not modeled)")
    else:
        chat["think_open"], chat["think_close"], chat["history_think"] = "", "", "nonempty"
    early = T.render([{"role": "user", "content": "USERPROBE1"},
                      {"role": "assistant", "content": "ASSTPROBE1", "reasoning_content": "REASONPROBE"},
                      {"role": "user", "content": "USERPROBE2"}])
    # Qwen-style templates drop reasoning from turns before the last user query.
    chat["history_think_scope"] = "all" if "REASONPROBE" in early else "after_last_query"
    think_marker = chat["think_open"].strip() or "<think>"
    thinks = any(think_marker in g for g in (g_default, g_on)) or bool(chat["think_open"])
    plain = g_off if g_off != g_on else g_default
    think = g_on if g_on != g_off else g_default
    if thinks and chat["think_open"] and not think.endswith(chat["think_open"]):
        # The model emits <think> itself (Qwen3 hybrid); open the block in the prompt so
        # output starts in the reasoning channel. Deliberate deviation: +2 prompt tokens.
        think = think + chat["think_open"]
        notes.append("generation_prompt appends think_open (the model would emit it): +tokens vs the template")
    toggle = g_on != g_off
    chat["generation_prompt"] = think if thinks else g_default
    chat["generation_prompt_plain"] = plain if toggle else chat["generation_prompt"]
    chat["thinking"] = "toggle" if (thinks and toggle) else ("always" if thinks else "never")
    chat["thinking_default"] = bool(thinks and (not toggle or g_default == g_on))
    chat["starts_in_reasoning"] = bool(thinks)
    chat["think_end"] = chat["think_close"].strip() or "</think>"
    # Tools block.
    t_plain = T.render([{"role": "user", "content": "USERPROBE1"}], tools=TOOLS)
    t_sys = T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"}], tools=TOOLS)
    printed = None
    for only in (False, True):
        for seps in ((", ", ": "), (",", ":")):
            cand = [json.dumps(t["function"] if only else t, ensure_ascii=False, separators=seps) for t in TOOLS]
            if all(c in t_plain for c in cand):
                printed, chat["tools_function_only"] = cand, only
                if seps != (", ", ": "):
                    notes.append("template prints tools compactly; Loom prints the request's tool JSON as sent")
                break
        if printed:
            break
    if not printed:
        raise ProbeError("tool definitions are not printed as JSON objects one per line; fill tools_* by hand")
    block = t_plain[:t_plain.index("USERPROBE1")]
    if block.startswith(bos) and bos:
        block = block[len(bos):]
    block = block[:len(block) - len(chat["user_prefix"])]
    if not (block.startswith(chat["system_prefix"]) and block.endswith(chat["suffix"])):
        raise ProbeError(f"tools are not rendered as a system block: {block[:200]!r}")
    body = block[len(chat["system_prefix"]):len(block) - len(chat["suffix"])]
    i0, i1 = body.index(printed[0]), body.index(printed[1])
    # Each tool is preceded by tool_line_prefix (Qwen: "\n"; GLM: "").
    gap_tools = body[i0 + len(printed[0]):i1]
    if gap_tools.strip() or not body[:i0].endswith(gap_tools):
        # A separator between tools (LFM2.5: a JSON list "[t0, t1]"), not a prefix before each.
        chat["tool_line_prefix"], chat["tool_separator"] = "", gap_tools
    else:
        chat["tool_line_prefix"], chat["tool_separator"] = gap_tools, ""
    line = chat["tool_line_prefix"]
    tools_prefix, tools_suffix = body[:i0 - len(line)] if line else body[:i0], body[i1 + len(printed[1]):]
    s_body = t_sys[:t_sys.index("USERPROBE1")]
    s_body = s_body[len(bos) if bos and s_body.startswith(bos) else 0:]
    s_body = s_body[len(chat["system_prefix"]):len(s_body) - len(chat["user_prefix"]) - len(chat["suffix"])]
    if s_body.startswith("SYSPROBE"):
        chat["tools_after_system"] = True
        rest = s_body[len("SYSPROBE"):]
        lead = rest[:rest.index(printed[0]) - len(line)]
        body_start = len(lead) - len(lead.lstrip())
        chat["tools_system_separator"], chat["tools_prefix"] = lead[:body_start], lead[body_start:]
    elif s_body.endswith("SYSPROBE"):
        chat["tools_after_system"] = False
        rest = s_body[:-len("SYSPROBE")]
        chat["tools_prefix"] = rest[:rest.index(printed[0]) - len(line)]
        after = rest[rest.index(printed[1]) + len(printed[1]):]
        if not after.startswith(tools_suffix):
            raise ProbeError("tools suffix differs when a system message follows")
        chat["tools_system_separator"] = after[len(tools_suffix):]
    else:
        raise ProbeError(f"system message is not merged with the tools block: {s_body[:200]!r}")
    chat["tools_suffix"] = tools_suffix
    # Tool calls.
    two = T.render([{"role": "user", "content": "USERPROBE1"}, calls_msg("ASSTPROBE1", [CALL1, CALL2]),
                    {"role": "tool", "tool_call_id": "call_0", "content": "TOOLPROBE1"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "TOOLPROBE2"},
                    {"role": "user", "content": "USERPROBE2"}], tools=TOOLS)
    turn = two[two.index("ASSTPROBE1"):]
    if '{"name": "get_weather"' in turn:
        chat["call_format"] = "hermes"
    elif "<function=get_weather>" in turn:
        chat["call_format"] = "qwen_xml"
    elif "get_weather<arg_key>" in turn:
        chat["call_format"] = "glm"
    else:
        raise ProbeError("unknown tool-call format: " + turn[:300])
    start1 = two.index("<tool_call>", two.index("ASSTPROBE1"))
    chat["call_marker"] = "<tool_call>"
    chat["content_call_separator"] = two[two.index("ASSTPROBE1") + len("ASSTPROBE1"):start1]
    end1 = two.index("</tool_call>", start1) + len("</tool_call>")
    start2 = two.index("<tool_call>", end1)
    chat["call_separator"] = two[end1:start2]
    end2 = two.index("</tool_call>", start2) + len("</tool_call>")
    after_calls = two[end2:]
    if not after_calls.startswith(chat["suffix"]):
        raise ProbeError(f"assistant turn with calls is not closed by the suffix: {after_calls[:80]!r}")
    after_calls = after_calls[len(chat["suffix"]):]
    pre = after_calls[:after_calls.index("TOOLPROBE1")]
    mid = between(after_calls, "TOOLPROBE1", "TOOLPROBE2")
    post = after_calls[after_calls.index("TOOLPROBE2") + len("TOOLPROBE2"):after_calls.index("USERPROBE2")]
    post = post[:len(post) - len(chat["user_prefix"])]
    for k in range(len(mid) + 1):
        item_suffix, item_prefix = mid[:k], mid[k:]
        if pre.endswith(item_prefix) and post.startswith(item_suffix):
            chat["tool_turn_prefix"] = pre[:len(pre) - len(item_prefix)]
            chat["tool_item_prefix"], chat["tool_item_suffix"] = item_prefix, item_suffix
            chat["tool_turn_suffix"] = post[len(item_suffix):]
            break
    else:
        raise ProbeError(f"cannot split tool-response framing: pre={pre!r} mid={mid!r} post={post!r}")
    chat["terminal"] = chat["suffix"].strip() or chat["suffix"]
    return chat, notes


def simulate(chat, messages, tools=None, gen=False, enable_thinking=None):
    """Python model of control.loom's rendering (chat_begin + chat_message + chat_end)."""
    if chat.get("format") == "gemma4":
        return simulate_gemma4(chat, messages, tools, gen, enable_thinking)
    trim = (lambda s: s.strip()) if chat["trim_content"] else (lambda s: s)
    out = chat.get("bos", "")
    first_system = messages and messages[0]["role"] == "system" and isinstance(messages[0]["content"], str)
    sys_text = trim(messages[0]["content"]) if first_system else ""
    use_default = not first_system and chat["default_system"]
    body = chat["default_system"] if use_default else sys_text
    if tools:
        printed = chat.get("tool_separator", "").join(
            chat["tool_line_prefix"] + json.dumps(t["function"] if chat["tools_function_only"] else t, ensure_ascii=False)
            for t in tools)
        tb = chat["tools_prefix"] + printed + chat["tools_suffix"]
        sep = chat["tools_system_separator"] if body else ""
        merged = body + sep + tb if chat["tools_after_system"] else tb + sep + body
        out += chat["system_prefix"] + merged + chat["suffix"]
    elif use_default:
        out += chat["system_prefix"] + chat["default_system"] + chat["suffix"]
    prior = None
    last_query = max([i for i, m in enumerate(messages) if m["role"] == "user"], default=-1)
    for i, m in enumerate(messages):
        role, text = m["role"], trim(m.get("content") or "")
        if prior == "tool" and role != "tool":
            out += chat["tool_turn_suffix"]
        if role == "system" and i == 0 and tools and first_system:
            pass
        elif role == "tool":
            out += ("" if prior == "tool" else chat["tool_turn_prefix"]) + chat["tool_item_prefix"] + text + chat["tool_item_suffix"]
        elif role == "assistant":
            thought = (m.get("reasoning_content") or "").strip()
            shown = chat["history_think_scope"] == "all" or i > last_query
            think = chat["think_open"] + thought + chat["think_close"] if shown and (thought or chat["history_think"] == "always") and chat["think_open"] else ""
            calls = "".join((chat["call_separator"] if j else "") + render_call(chat, c["function"])
                            for j, c in enumerate(m.get("tool_calls") or []))
            sep = chat["content_call_separator"] if text and calls else ""
            out += chat["assistant_prefix"] + think + text + sep + calls + chat["suffix"]
        else:
            out += (chat["user_prefix"] if role == "user" else chat["system_prefix"]) + text + chat["suffix"]
        prior = role
    if gen:
        if prior == "tool":
            out += chat["tool_turn_suffix"]
        on = chat["thinking_default"] if enable_thinking is None or chat["thinking"] != "toggle" else enable_thinking
        out += chat["generation_prompt"] if on else chat["generation_prompt_plain"]
    elif prior == "tool":
        out += chat["tool_turn_suffix"]
    return out


def render_call(chat, call):
    if chat["call_format"] == "gemma4":
        return g4_call(call)
    args = call["arguments"]
    if chat["call_format"] == "hermes":
        a = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        return '<tool_call>\n{"name": "' + call["name"] + '", "arguments": ' + a + "}\n</tool_call>"
    args = json.loads(args) if isinstance(args, str) else args
    val = lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if chat["call_format"] == "qwen_xml":
        return ("<tool_call>\n<function=" + call["name"] + ">\n"
                + "".join(f"<parameter={k}>\n{val(v)}\n</parameter>\n" for k, v in args.items())
                + "</function>\n</tool_call>")
    return ("<tool_call>" + call["name"]
            + "".join(f"<arg_key>{k}</arg_key><arg_value>{val(v)}</arg_value>" for k, v in args.items()) + "</tool_call>")


CONVERSATIONS = [
    ("chat", [{"role": "user", "content": "Hi there"}], None),
    ("system + multi-turn", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "One"},
                             {"role": "assistant", "content": "Two"}, {"role": "user", "content": "Three"}], None),
    ("tools", [{"role": "user", "content": "Weather in Paris?"}], TOOLS),
    ("system + tools", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather?"}], TOOLS),
    ("agent loop", [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "Weather in Paris?"},
                    calls_msg("", [{"name": "get_weather", "arguments": {"city": "Paris"}}], reasoning="Need the tool."),
                    {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": 18}"}], TOOLS),
    ("parallel calls", [{"role": "user", "content": "Weather and search"},
                        calls_msg("Checking.", [CALL1, CALL2], reasoning="Two tools."),
                        {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
                        {"role": "tool", "tool_call_id": "call_1", "content": "9-18"}], TOOLS),
]


# Gemma 4 look-ahead cases: a multi-step tool loop inside one model turn, a tool result followed by a
# user turn, calls with no result yet, content parts, nested/typed tool schemas and arguments.
RICH_TOOLS = [
    {"type": "function", "function": {"name": "book", "description": "Book a table.", "parameters": {
        "type": "object", "required": ["where", "when"], "properties": {
            "where": {"type": "string", "description": "Restaurant", "enum": ["Chez A", "B's"]},
            "when": {"type": "object", "description": "Time slot", "properties": {
                "day": {"type": "string"}, "hour": {"type": "integer", "nullable": True}}, "required": ["day"]},
            "Party": {"type": "number"}, "vegan": {"type": "boolean"},
            "notes": {"type": "array", "items": {"type": "string"}},
            "guests": {"type": "array", "description": "People", "items": {"type": "object", "properties": {
                "name": {"type": "string"}}, "required": ["name"]}}}}}},
    {"type": "function", "function": {"name": "now", "description": "Current time.", "parameters": {"type": "object", "properties": {}}}},
]
GEMMA4_CONVERSATIONS = [
    ("tool loop, two steps", [{"role": "user", "content": "Weather in Paris then search"},
                              calls_msg("", [CALL1], reasoning="First weather."),
                              {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": 18}"},
                              calls_msg("Now searching.", [CALL2], reasoning="Then search."),
                              {"role": "tool", "tool_call_id": "call_0", "content": "9-18"}], TOOLS),
    ("tool loop, answered, next query", [{"role": "system", "content": " Sys. "}, {"role": "user", "content": "Weather?"},
                                         calls_msg("", [CALL1], reasoning="r1"),
                                         {"role": "tool", "tool_call_id": "call_0", "content": " sunny\n"},
                                         {"role": "assistant", "content": "It is sunny.", "reasoning_content": "r2"},
                                         {"role": "user", "content": "Thanks"}], TOOLS),
    ("tool result then user", [{"role": "user", "content": "A"}, calls_msg("x", [CALL2]),
                               {"role": "tool", "tool_call_id": "zzz", "name": "search", "content": "r"},
                               {"role": "user", "content": "B"}], TOOLS),
    ("call without result", [{"role": "user", "content": "A"}, calls_msg("", [CALL1])], TOOLS),
    ("unnamed tool result", [{"role": "user", "content": "A"}, calls_msg("", [CALL1]),
                             {"role": "tool", "tool_call_id": "nope", "content": "r"}], TOOLS),
    ("content parts", [{"role": "user", "content": [{"type": "text", "text": " Hi "}, {"type": "text", "text": "there "}]},
                       {"role": "assistant", "content": "Yo"}, {"role": "user", "content": "Q"}], None),
    ("rich schema + nested args", [{"role": "user", "content": "Book"},
                                   calls_msg("", [{"name": "book", "arguments": {"where": "B's", "when": {"hour": 19, "day": "Fri"},
                                                   "Party": 2.5, "vegan": True, "notes": ["a \"q\"", None],
                                                   "guests": [{"name": "Al"}]}}, {"name": "now", "arguments": {}}]),
                                   {"role": "tool", "tool_call_id": "call_0", "content": "ok"},
                                   {"role": "tool", "tool_call_id": "call_1", "content": "12:00"}], RICH_TOOLS),
]


# ---- Gemma 4 (`<|turn>role\n...<turn|>\n`, model role, `<|channel>thought` reasoning, `<|tool_call>call:NAME{...}`) ----
# The template has look-ahead (an assistant turn's close depends on what follows; tool responses are
# rendered inside the model turn, named by the matching call id) and a custom, non-JSON tool
# declaration syntax. simulate_gemma4() renders the way control.loom does: one message at a time, the
# previous assistant turn's close deferred to the next non-tool message or the generation prompt.
G4_Q = '<|"|>'


def g4_items(d):
    """Jinja dictsort (case-insensitive by key)."""
    return sorted(d.items(), key=lambda kv: kv[0].lower() if isinstance(kv[0], str) else kv[0])


def g4_arg(a, escape_keys=True):
    """format_argument macro."""
    if a is None:
        return "null"
    if isinstance(a, str):
        return G4_Q + a + G4_Q
    if isinstance(a, bool):
        return "true" if a else "false"
    if isinstance(a, dict):
        return "{" + ",".join(((G4_Q + k + G4_Q) if escape_keys else k) + ":" + g4_arg(v, escape_keys)
                              for k, v in g4_items(a)) + "}"
    if isinstance(a, (list, tuple)):
        return "[" + ",".join(g4_arg(v, escape_keys) for v in a) + "]"
    return str(a)


def g4_upper(v):
    return v.upper() if isinstance(v, str) else ""


def g4_params(properties, required, filter_keys=False):
    """format_parameters macro."""
    standard = ["description", "type", "properties", "required", "nullable"]
    out, first = "", True
    for key, value in g4_items(properties):
        if filter_keys and key in standard:
            continue
        if not first:
            out += ","
        first = False
        value = value if isinstance(value, dict) else {}
        out += key + ":{"
        comma = False

        def sep():
            nonlocal comma
            s = "," if comma else ""
            comma = True
            return s
        if value.get("description"):
            out += "description:" + G4_Q + str(value["description"]) + G4_Q
            comma = True
        t = g4_upper(value.get("type"))
        if t == "STRING":
            if value.get("enum"):
                out += sep() + "enum:" + g4_arg(value["enum"])
        elif t == "ARRAY":
            items = value.get("items")
            if isinstance(items, dict) and items:
                out += sep() + "items:{"
                f2 = True
                for ik, iv in g4_items(items):
                    if iv is None:
                        continue
                    if not f2:
                        out += ","
                    f2 = False
                    if ik == "properties":
                        out += "properties:{" + (g4_params(iv, items.get("required") or []) if isinstance(iv, dict) else "") + "}"
                    elif ik == "required":
                        out += "required:[" + ",".join(G4_Q + str(r) + G4_Q for r in iv) + "]"
                    elif ik == "type":
                        out += "type:" + (g4_arg(iv.upper()) if isinstance(iv, str) else g4_arg([str(x).upper() for x in iv]))
                    else:
                        out += ik + ":" + g4_arg(iv)
                out += "}"
        if value.get("nullable"):
            out += sep() + "nullable:true"
        if t == "OBJECT":
            if isinstance(value.get("properties"), dict):
                out += sep() + "properties:{" + g4_params(value["properties"], value.get("required") or []) + "}"
            else:
                out += sep() + "properties:{" + g4_params(value, value.get("required") or [], filter_keys=True) + "}"
            if value.get("required"):
                out += sep() + "required:[" + ",".join(G4_Q + str(r) + G4_Q for r in value["required"]) + "]"
        out += sep() + "type:" + G4_Q + t + G4_Q + "}"
    return out


def g4_declaration(tool):
    """format_function_declaration macro (| trim)."""
    f = tool["function"]
    s = "declaration:" + f["name"] + "{description:" + G4_Q + str(f.get("description") or "") + G4_Q
    p = f.get("parameters")
    if p:
        s += ",parameters:{"
        if p.get("properties"):
            s += "properties:{" + g4_params(p["properties"], p.get("required")) + "},"
        if p.get("required"):
            s += "required:[" + ",".join(G4_Q + str(r) + G4_Q for r in p["required"]) + "],"
        if p.get("type"):
            s += "type:" + G4_Q + g4_upper(p["type"]) + G4_Q + "}"
    if "response" in f:
        r = f["response"]
        s += ",response:{"
        if r.get("description"):
            s += "description:" + G4_Q + r["description"] + G4_Q + ","
        if g4_upper(r.get("type")) == "OBJECT":
            s += "type:" + G4_Q + "OBJECT" + G4_Q + "}"
    return (s + "}").strip()


def g4_call(call):
    args = call.get("arguments")
    args = json.loads(args) if isinstance(args, str) else (args or {})
    return ("<|tool_call>call:" + call["name"] + "{"
            + ",".join(k + ":" + g4_arg(v, escape_keys=False) for k, v in g4_items(args)) + "}<tool_call|>")


def g4_text(content, trim=False):
    """Message text; content parts are trimmed one by one (user/model) or joined as is (tool results)."""
    if isinstance(content, list):
        return "".join((p.get("text", "").strip() if trim else p.get("text", "")) for p in content if p.get("type") == "text")
    return (content or "").strip() if trim else (content or "")


def g4_close(c, pending, next_role):
    """How the template closes an assistant turn, given the next non-tool role (None: end)."""
    if pending["calls"] and not pending["responses"]:
        return c["tool_response_open"]
    if next_role == "assistant":
        return ""
    if pending["responses"] and not pending["content"] and next_role is None:
        return ""
    return c["suffix"]


def simulate_gemma4(c, messages, tools=None, gen=False, enable_thinking=None):
    on = c["thinking_default"] if enable_thinking is None or c["thinking"] != "toggle" else enable_thinking
    out = c.get("bos", "")
    first_system = bool(messages) and messages[0]["role"] == "system"
    if on or tools or first_system:
        out += c["system_prefix"] + (c["think_flag"] if on else "")
        if first_system:
            sys_content = messages[0]["content"]
            out += sys_content.strip() if isinstance(sys_content, str) else "".join(
                p.get("text", "").strip() + " " for p in sys_content)
        out += "".join(c["tool_decl_open"] + g4_declaration(t) + c["tool_decl_close"] for t in tools or [])
        out += c["suffix"]
    msgs = messages[1:] if first_system else messages
    last_user = max([i for i, m in enumerate(msgs) if m["role"] == "user"], default=-1)
    pending, prior, calls = None, None, []
    for i, m in enumerate(msgs):
        role = m["role"]
        if role == "tool":
            if pending and pending["calls"]:   # tool messages outside a call chain are not rendered
                name = next((x["function"]["name"] for x in calls if x.get("id") == m.get("tool_call_id")), None) \
                    or m.get("name") or "unknown"
                out += c["tool_item_prefix"] + name + c["tool_value_open"] + g4_arg(g4_text(m.get("content")), False) \
                    + c["tool_item_suffix"]
                pending["responses"] = True
            continue
        if pending:
            out += g4_close(c, pending, role)
        pending = None
        if role == "assistant":
            if prior != "assistant":
                out += c["assistant_prefix"]
            thought = m.get("reasoning_content") or m.get("reasoning")
            if thought and i > last_user:
                out += c["think_open"] + thought + c["think_close"]
            text = g4_text(m.get("content"), trim=True)
            calls = m.get("tool_calls") or []
            out += text + "".join(g4_call(x["function"]) for x in calls)
            pending = {"calls": bool(calls), "responses": False, "content": bool(text)}
        else:
            out += (c["user_prefix"] if role == "user" else c["system_prefix"]) + g4_text(m.get("content"), trim=True) + c["suffix"]
        prior = role
    after = None
    if pending:
        out += g4_close(c, pending, None)
        if pending["calls"]:
            after = "response" if pending["responses"] else "call"
    if gen:
        if after is None:
            out += c["generation_prompt"]
        elif after == "response" and on:
            out += c["think_open"]
    return out


def derive_gemma4(T, bos):
    """Cuts the Gemma 4 strings out of the template and checks the fixed syntax control.loom implements."""
    c = {"format": "gemma4", "call_format": "gemma4"}
    c["bos"] = bos if bos and T.render([{"role": "user", "content": "USERPROBE1"}]).startswith(bos) else ""
    strip = lambda s: s[len(c["bos"]):] if c["bos"] and s.startswith(c["bos"]) else s
    r = strip(T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"},
                        {"role": "assistant", "content": "ASSTPROBE1"}, {"role": "user", "content": "USERPROBE2"}]))
    c["system_prefix"] = r[:r.index("SYSPROBE")]
    single = strip(T.render([{"role": "user", "content": "USERPROBE1"}]))
    c["user_prefix"] = single[:single.index("USERPROBE1")]
    c["suffix"] = single[single.index("USERPROBE1") + len("USERPROBE1"):]
    gap = between(r, "USERPROBE1", "ASSTPROBE1")
    c["assistant_prefix"] = gap[len(c["suffix"]):]
    c["default_system"] = ""
    c["trim_content"] = True
    base = T.render([{"role": "user", "content": "USERPROBE1"}])
    c["generation_prompt"] = T.render([{"role": "user", "content": "USERPROBE1"}], gen=True)[len(base):]
    c["generation_prompt_plain"] = c["generation_prompt"]
    on = strip(T.render([{"role": "user", "content": "USERPROBE1"}], enable_thinking=True))
    if not on.endswith(single) or not on.startswith(c["system_prefix"]):
        raise ProbeError("enable_thinking does not add a system turn")
    c["think_flag"] = on[len(c["system_prefix"]):len(on) - len(single) - len(c["suffix"])]
    c["thinking"], c["thinking_default"] = "toggle", "<|think|>" in strip(T.render([{"role": "user", "content": "U"}], gen=True))
    agent = T.render([{"role": "user", "content": "USERPROBE1"}, calls_msg("ASSTPROBE1", [CALL1], reasoning="REASONPROBE"),
                      {"role": "tool", "tool_call_id": "call_0", "content": "TOOLPROBE1"}], tools=TOOLS)
    mid = between(agent, "USERPROBE1", "ASSTPROBE1")
    pre = mid[:mid.index("REASONPROBE")]
    c["think_open"] = pre[pre.index(c["assistant_prefix"]) + len(c["assistant_prefix"]):]
    c["think_close"] = mid[mid.index("REASONPROBE") + len("REASONPROBE"):]
    c["think_end"] = c["think_close"].strip()
    c["history_think"], c["history_think_scope"] = "nonempty", "after_last_query"
    c["starts_in_reasoning"] = False  # the model opens the thought channel itself (output state 3 detects it)
    c["tool_decl_open"], c["tool_decl_close"] = "<|tool>", "<tool|>"
    c["call_marker"] = "<|tool_call>"
    c["tool_response_open"] = "<|tool_response>"
    c["tool_item_prefix"], c["tool_value_open"], c["tool_item_suffix"] = "<|tool_response>response:", "{value:", "}<tool_response|>"
    # Unused generic fields (build.py substitutes them).
    for k in ("tools_prefix", "tools_suffix", "tool_line_prefix", "tool_separator", "tools_system_separator",
              "tool_turn_prefix", "tool_turn_suffix", "content_call_separator", "call_separator"):
        c[k] = ""
    c["tools_after_system"], c["tools_function_only"] = True, True
    c["terminal"] = c["suffix"].strip()
    # Structural tokens control.loom must see in the decoded output (loom-serve strips special tokens).
    c["nonspecial_tokens"] = ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", '<|"|>']
    expect = {"tool declaration": T.render([{"role": "user", "content": "U"}], tools=TOOLS[:1]),
              "tool call": agent}
    want = {"tool declaration": "<|tool>" + g4_declaration(TOOLS[0]) + "<tool|>",
            "tool call": g4_call(CALL1) + c["tool_item_prefix"] + "get_weather{value:" + G4_Q + "TOOLPROBE1" + G4_Q + c["tool_item_suffix"]}
    for k, v in want.items():
        if v not in expect[k]:
            raise ProbeError(f"gemma4 {k} syntax differs from the template: {v!r} not in {expect[k]!r}")
    return c, ["gemma4 format: look-ahead turn closes, tool responses inside the model turn, custom tool syntax"]


def verify(template_text, chat, bos="", eos=""):
    """Returns [(name, ok, detail)]: template render vs simulate() for each conversation."""
    T = Template(template_text, bos, eos)
    results = []
    for name, msgs, tools in CONVERSATIONS + (GEMMA4_CONVERSATIONS if chat.get("format") == "gemma4" else []):
        for kw in ({},) if chat["thinking"] != "toggle" else ({"enable_thinking": True}, {"enable_thinking": False}):
            label = name + (f" (enable_thinking={kw['enable_thinking']})" if kw else "")
            try:
                ref = T.render(msgs, tools=tools, gen=True, **kw)
            except ProbeError as e:
                results.append((label, None, f"template refuses: {e}"))
                continue
            got = simulate(chat, msgs, tools, gen=True, enable_thinking=kw.get("enable_thinking"))
            # The thinking prompt may deliberately append think_open.
            if got != ref and chat["thinking"] != "never" and got.endswith(chat["think_open"]) and got[:-len(chat["think_open"])] == ref:
                results.append((label, True, "exact except the deliberate think_open in the generation prompt"))
                continue
            if got == ref:
                results.append((label, True, "exact"))
            else:
                k = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
                results.append((label, False, f"differs at byte {k}: loom {got[max(0, k - 40):k + 60]!r} vs template {ref[max(0, k - 40):k + 60]!r}"))
    return results


def main():
    text = load_template(sys.argv[1])
    bos = ""
    if sys.argv[1].endswith(".gguf"):
        import gguf
        r = gguf.GGUFReader(sys.argv[1])
        f, tokens = r.fields.get("tokenizer.ggml.bos_token_id"), r.fields.get("tokenizer.ggml.tokens")
        if f is not None and tokens is not None:
            bos = tokens.contents()[f.contents()]
    chat, notes = derive(text, bos)
    for n in notes:
        print("note:", n)
    print(json.dumps(chat, indent=1, ensure_ascii=False))
    for name, ok, detail in verify(text, chat, bos):
        print(("OK  " if ok else "SKIP" if ok is None else "FAIL"), name, "-", detail)
    if len(sys.argv) > 2:
        json.dump(chat, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()


def raw_from_chat(chat, message, finished, thinking=None, prompt=None):
    """The benchmark scripts' hook (LOOM_KIT=this dir): rebuild Gemma 4's raw output from Loom's channels.
    The model opens <|channel>thought\\n itself, except after tool results with thinking on, where the
    prompt ends with it. It closes reasoning with a bare <channel|> (the template re-renders it as
    "\\n<channel|>"); Loom's reasoning keeps every byte before it."""
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    calls = "".join(render_call(chat, t["function"]) for t in message.get("tool_calls") or [])
    answer = content + calls
    resumed = bool(prompt) and prompt.endswith(chat["think_open"])
    if not reasoning and not resumed:
        return answer
    text = ("" if resumed else chat["think_open"]) + reasoning
    if answer or finished:
        text += chat["think_end"] + answer
    return text
