"""Derives the control program's "chat" block from a model's Jinja chat template.

The template is rendered on probe conversations and the pieces the control
program needs are cut out of the output: role prefixes/suffix, generation
prompts with thinking on/off, the default system prompt, per-reasoning_effort
instructions that open the system block with thinking on (Qwen3.8), the tools block and
its order relative to the system message, the tool-call format and separators,
tool-response framing, how reasoning appears in history, and content trimming.
`verify()` then re-renders a set of conversations the way control.loom will
(`simulate()`) and compares them with the template, so mismatches show up
before anything runs on the GPU.

usage: python chat_probe.py TEMPLATE_SOURCE [OUT.json]
       python chat_probe.py TEMPLATE_SOURCE --verify cfg.json   (a hand-filled chat block: verify only;
                         with a kit's own chat_probe.py for formats this one can't simulate)
  TEMPLATE_SOURCE: a .gguf (tokenizer.chat_template), a tokenizer_config.json, or a .jinja file
  (inspect_gguf.py calls derive() + verify() for you)
"""
import datetime
import json
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
CALL1 = {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}
CALL2 = {"name": "search", "arguments": {"query": "louvre hours"}}


from jinja2 import nodes as _nodes
from jinja2.ext import Extension as _Extension


class GenerationTag(_Extension):
    """transformers' {% generation %}...{% endgeneration %} (assistant mask marker): renders its body."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return _nodes.Scope(body).set_lineno(lineno)


class ProbeError(Exception):
    pass


def environment():
    def raise_exception(message):
        raise ProbeError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", GenerationTag])
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


# loom-serve's reasoning effort levels (chat_reasoning's efforts bitmap bit / default value).
EFFORT_LEVELS = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5, "max": 6}


def reasoning_heads(T):
    """The text before the first user message for each reasoning_effort the template accepts (thinking on),
    plus the default ("") and thinking off; None when the template ignores reasoning_effort."""
    user1 = [{"role": "user", "content": "USERPROBE1"}]
    head = lambda **kw: (lambda r: r[:r.index("USERPROBE1")])(T.render(user1, **kw))
    heads = {}
    for e in EFFORT_LEVELS:
        try:
            heads[e] = head(enable_thinking=True, reasoning_effort=e)
        except ProbeError:   # Qwen3.8 raises on efforts it does not support
            pass
    if len(set(heads.values())) < 2:
        return None
    heads[""], heads["off"] = head(enable_thinking=True), head(enable_thinking=False)
    return heads


def reasoning_instructions(T, chat, heads, notes):
    """Cuts the per-effort instruction prefix out of the system block. Model (control.loom @chat_begin):
    with thinking on, the system block's body (tools text and/or system message, or the default system
    prompt) becomes INSTRUCTIONS + reasoning_separator + body; INSTRUCTIONS alone when the body is empty;
    no instructions (and no change) with thinking off or an effort whose text is empty."""
    bos, prefix, suffix, user_prefix = chat["bos"], chat["system_prefix"], chat["suffix"], chat["user_prefix"]

    def body(h):
        h = h[len(bos):] if bos and h.startswith(bos) else h
        if not h.endswith(user_prefix):
            raise ProbeError(f"reasoning instructions: unexpected head {h!r}")
        h = h[:len(h) - len(user_prefix)]
        if not h:
            return ""
        if not (h.startswith(prefix) and h.endswith(suffix)):
            raise ProbeError(f"reasoning instructions are not rendered in a system block: {h!r}")
        return h[len(prefix):len(h) - len(suffix)]

    default = chat["default_system"]
    if body(heads["off"]) != default:
        raise ProbeError(f"thinking off leaves the system block as {body(heads['off'])!r}, not {default!r}")
    bodies = {e: body(h) for e, h in heads.items() if e not in ("", "off", "none")}
    if "none" in heads and body(heads["none"]) != default:
        notes.append("template accepts reasoning_effort=none with thinking on; Loom turns thinking off for it")
    # Separator: from a system message after the instructions of an effort that has some.
    effort = next((e for e, b in bodies.items() if b != default), None)
    if effort is None:
        raise ProbeError("reasoning_effort changes the prompt outside the system block (not modeled)")
    r = T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"}],
                 enable_thinking=True, reasoning_effort=effort)
    lead = r[:r.index("SYSPROBE")]
    lead = lead[len(bos):] if bos and lead.startswith(bos) else lead
    if not lead.startswith(prefix):
        raise ProbeError(f"reasoning instructions and the system message are not one system block: {lead!r}")
    lead = lead[len(prefix):]
    instr = bodies[effort] if not default else lead.rstrip()
    if not lead.startswith(instr):
        raise ProbeError(f"reasoning instructions differ when a system message follows: {lead!r}")
    separator = lead[len(instr):]
    join = lambda i, b: i + separator + b if i and b else i or b
    table = {}
    for e, b in bodies.items():
        if b == default:
            table[e] = ""
        elif default and b.endswith(separator + default):
            table[e] = b[:len(b) - len(separator + default)]
        elif not default:
            table[e] = b
        else:
            raise ProbeError(f"reasoning instructions for {e!r} do not prefix the default system prompt: {b!r}")
        if join(table[e], default) != b:
            raise ProbeError(f"reasoning instructions for {e!r} do not compose as modeled: {b!r}")
    matches = [e for e in table if heads[e] == heads[""]]
    if not matches:
        raise ProbeError("the default reasoning effort is none of " + ", ".join(table))
    chat["reasoning_instructions"] = table
    chat["reasoning_effort_default"] = max(matches, key=EFFORT_LEVELS.get)
    chat["reasoning_separator"] = separator
    notes.append(f"template prepends reasoning instructions per reasoning_effort ({', '.join(table)}; default "
                 f"{chat['reasoning_effort_default']}) to the system block with thinking on")


# A template that ignores tools: no tool declarations, calls or results (call_format "none"; build.py makes
# control.loom ignore a request's tools and never parse calls). call_marker keeps build.py's default: output_next
# ends content at the marker, and an empty one would end it at once.
CHAT_ONLY = {"tools_prefix": "", "tools_suffix": "", "tool_line_prefix": "", "tool_separator": "",
             "tools_system_separator": "", "tools_after_system": True, "tools_function_only": True,
             "call_format": "none", "calls_open": "", "calls_close": "", "call_end": "", "call_marker": "<tool_call>",
             "content_call_separator": "", "call_separator": "", "tool_turn_prefix": "", "tool_item_prefix": "",
             "tool_item_suffix": "", "tool_turn_suffix": ""}


def derive(template_text, bos="", eos=""):
    T = Template(template_text, bos, eos)
    notes = []
    chat = {}
    # BOS: emitted by the control program at the very start when the template prints it.
    # Decided with a sentinel: a GGUF whose bos id is the template's own role tag (MiniCPM-V: <|im_start|>) starts every
    # render with that text although the template never prints bos_token.
    printed = Template(template_text, "BOSPROBE", eos).render([{"role": "user", "content": "USERPROBE1"}]).startswith("BOSPROBE")
    chat["bos"] = bos = bos if bos and printed else ""
    # Reasoning instructions (Qwen3.8: "Reasoning effort is set to xhigh. ..." per reasoning_effort, only
    # with thinking on): the system-block probes below render with thinking off (kw0), where they vanish.
    efforts = reasoning_heads(T)
    kw0 = {"enable_thinking": False} if efforts else {}
    # Prefixes and suffix: an assistant turn before the last user query renders plainly.
    r = T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"},
                  {"role": "assistant", "content": "ASSTPROBE1"}, {"role": "user", "content": "USERPROBE2"}], **kw0)
    if r.startswith(bos) and bos:
        r = r[len(bos):]
    chat["system_prefix"] = r[:r.index("SYSPROBE")]
    single = T.render([{"role": "user", "content": "USERPROBE1"}], **kw0)
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
    if efforts:
        reasoning_instructions(T, chat, efforts, notes)
    trimmed = T.render([{"role": "user", "content": "  USERPROBE1 \n"}], **kw0)
    chat["trim_content"] = "  USERPROBE1" not in trimmed
    # Generation prompts and thinking (each relative to the same render without the prompt: with
    # reasoning instructions the system block depends on enable_thinking too).
    user1 = [{"role": "user", "content": "USERPROBE1"}]
    gen = lambda **kw: T.render(user1, gen=True, **kw)[len(T.render(user1, **kw)):]
    g_default, g_on, g_off = gen(), gen(enable_thinking=True), gen(enable_thinking=False)
    # Reasoning in history (agent loop: assistant after the last user query, followed by a tool result).
    agent = lambda m: T.render([{"role": "user", "content": "USERPROBE1"}, m,
                                {"role": "tool", "tool_call_id": "call_0", "content": "TOOLPROBE1"}], tools=TOOLS)
    with_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning="REASONPROBE"))
    without_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning=""))
    # Templates that keep reasoning in every assistant turn (preserve_thinking: Qwen3.6, Qwen3.8) render
    # an empty think block in the plain turn above, which then is not part of the assistant prefix.
    guess = chat["user_prefix"].replace("user", "assistant")
    if guess != chat["user_prefix"] and chat["assistant_prefix"].startswith(guess) and chat["assistant_prefix"] != guess:
        rest, mid_r, lead = chat["assistant_prefix"][len(guess):], between(with_r, "USERPROBE1", "ASSTPROBE1"), chat["suffix"] + guess
        if "REASONPROBE" in mid_r and mid_r.startswith(lead):
            opened, closed = mid_r[len(lead):mid_r.index("REASONPROBE")], mid_r[mid_r.index("REASONPROBE") + len("REASONPROBE"):]
            if opened + closed == rest:
                chat["assistant_prefix"] = guess
                notes.append(f"every assistant turn renders its (possibly empty) think block: assistant prefix {guess!r}")
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
    # ... but some keep an empty think block there (MiniCPM-V 4.6/4.7: <think>\n\n</think>\n\n). Present only then.
    if chat["history_think_scope"] == "after_last_query" and chat["think_open"]:
        empty = chat["assistant_prefix"] + chat["think_open"] + chat["think_close"]
        if between(early, "USERPROBE1", "ASSTPROBE1").endswith(empty):
            chat["history_think_before"] = "empty"
            notes.append("turns before the last user query keep an empty think block")
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
    t_plain = T.render([{"role": "user", "content": "USERPROBE1"}], tools=TOOLS, **kw0)
    if t_plain == single and T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"}], tools=TOOLS, **kw0) \
            == T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"}], **kw0):
        # The template ignores tools (a plain chat model, e.g. bare ChatML): a chat-only block. Requests' tools are
        # ignored as the template does, no call is parsed, and the card says tool calling "no" (agent gate n/a).
        chat.update(CHAT_ONLY)
        chat["terminal"] = chat["suffix"].strip() or chat["suffix"]
        notes.append("template ignores tools: chat-only block, tool calling unsupported (call_format none)")
        return chat, notes
    t_sys = T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"}], tools=TOOLS, **kw0)
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
    if not printed and all(xml_tool(t) in t_plain for t in TOOLS):
        # Tool declarations as XML elements (Qwen3-Coder: <function><name>..</name><parameters>..): tools_format "xml".
        printed, chat["tools_function_only"], chat["tools_format"] = [xml_tool(t) for t in TOOLS], True, "xml"
        notes.append("template declares tools as XML elements (tools_format xml)")
    if not printed:
        raise ProbeError("tool definitions are not printed as JSON objects or XML elements one per line; fill tools_* by hand")
    block = t_plain[:t_plain.index("USERPROBE1")]
    if block.startswith(bos) and bos:
        block = block[len(bos):]
    block = block[:len(block) - len(chat["user_prefix"])]
    if not (block.startswith(chat["system_prefix"]) and block.endswith(chat["suffix"])):
        raise ProbeError(f"tools are not rendered as a system block: {block[:200]!r}")
    body = block[len(chat["system_prefix"]):len(block) - len(chat["suffix"])]
    i0, i1 = body.index(printed[0]), body.index(printed[1])
    # Each tool is preceded by tool_line_prefix (Qwen: "\n"; GLM: "").
    line = body[i0 + len(printed[0]):i1]
    if line and body[:i0].endswith(line) and not line.strip():
        # Every tool is preceded by tool_line_prefix (Qwen: "\n"; GLM: "").
        chat["tool_line_prefix"], chat["tool_separator"] = line, ""
    else:
        # Tools joined by a separator printed between them only (LFM2: ", " inside "[...]").
        chat["tool_line_prefix"], chat["tool_separator"] = "", line
        line = ""
    tools_prefix, tools_suffix = body[:i0 - len(line)] if line else body[:i0], body[i1 + len(printed[1]):]
    s_body = t_sys[:t_sys.index("USERPROBE1")]
    s_body = s_body[len(bos) if bos and s_body.startswith(bos) else 0:]
    s_body = s_body[len(chat["system_prefix"]):len(s_body) - len(chat["user_prefix"]) - len(chat["suffix"])]
    if s_body.startswith("SYSPROBE"):
        chat["tools_after_system"] = True
        rest = s_body[len("SYSPROBE"):]
        lead = rest[:rest.index(printed[0]) - len(line)]
        body_start = len(lead) - len(lead.lstrip())
        # Tools in their own system block after the system message (Flint: "<|system|>SYS<|system|>\n# Tools"):
        # the second role tag belongs to the separator, not to the tools prefix.
        if chat["system_prefix"] and lead.startswith(chat["system_prefix"]):
            body_start = len(chat["system_prefix"])
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
    # Without a system message the tools block follows the default system prompt and its separator, unless the
    # template swaps in another one when tools are present (Nanbeige: a "tool-calling expert" prompt, with no
    # separator): default_system_tools is that text, separator included. Present only when it differs.
    if chat["tools_after_system"]:
        plain_lead = tools_prefix
        default = chat["default_system"]
        if plain_lead != (default + chat["tools_system_separator"] if default else "") + chat["tools_prefix"]:
            if not plain_lead.endswith(chat["tools_prefix"]):
                raise ProbeError(f"tools block without a system message has an unexpected lead: {plain_lead[:200]!r}")
            chat["default_system_tools"] = plain_lead[:len(plain_lead) - len(chat["tools_prefix"])]
            notes.append(f"template swaps the default system prompt when tools are present: "
                         f"{chat['default_system_tools'][:60]!r}...")
    # Tool calls.
    two = T.render([{"role": "user", "content": "USERPROBE1"}, calls_msg("ASSTPROBE1", [CALL1, CALL2]),
                    {"role": "tool", "tool_call_id": "call_0", "content": "TOOLPROBE1"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "TOOLPROBE2"},
                    {"role": "user", "content": "USERPROBE2"}], tools=TOOLS)
    turn = two[two.index("ASSTPROBE1"):]
    if "get_weather(city=" in turn:
        # Pythonic calls (LFM2): <|tool_call_start|>[f(a='x', b=3), g()]<|tool_call_end|>
        chat["call_format"] = "pythonic"
        name0 = turn.index("get_weather(")
        chat["calls_open"] = opened = turn[len("ASSTPROBE1"):name0]
        end1 = turn.index(")", name0) + 1
        chat["call_separator"] = turn[end1:turn.index("search(", end1)]
        close_at = turn.index(")", turn.index("search(")) + 1
        chat["calls_close"] = turn[close_at:turn.index(chat["suffix"], close_at)]
        chat["call_marker"] = opened.rstrip("[(")  # output_next stops content here
        chat["call_end"] = chat["calls_close"].lstrip("])")
        chat["content_call_separator"] = ""
        if not opened.startswith(chat["call_marker"]) or not chat["call_marker"]:
            raise ProbeError(f"unexpected pythonic call framing: {opened!r}")
        after_calls = turn[close_at + len(chat["calls_close"]):]
        two = "ASSTPROBE1" + after_calls  # tool-response framing below
        end2 = len("ASSTPROBE1")
    elif '{"name": "get_weather"' in turn:
        chat["call_format"] = "hermes"
    elif "<function=get_weather>" in turn:
        chat["call_format"] = "qwen_xml"
    elif "get_weather<arg_key>" in turn:
        chat["call_format"] = "glm"
    else:
        raise ProbeError("unknown tool-call format: " + turn[:300])
    if chat["call_format"] != "pythonic":
        chat["calls_open"] = chat["calls_close"] = chat["call_end"] = ""
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


def simulate(chat, messages, tools=None, gen=False, enable_thinking=None, reasoning_effort=None):
    """Python model of control.loom's rendering (chat_begin + chat_message + chat_end)."""
    tools = None if chat.get("call_format") == "none" else tools   # a chat-only block ignores tools, as its template does
    trim = (lambda s: s.strip()) if chat["trim_content"] else (lambda s: s)
    out = chat.get("bos", "")
    first_system = messages and messages[0]["role"] == "system" and isinstance(messages[0]["content"], str)
    sys_text = trim(messages[0]["content"]) if first_system else ""
    use_default = not first_system and chat["default_system"]
    body = chat["default_system"] if use_default else sys_text
    # Thinking: the default, then enable_thinking, then reasoning_effort ("none" turns it off).
    toggle = chat["thinking"] == "toggle"
    on = chat["thinking_default"] if enable_thinking is None or not toggle else enable_thinking
    on = reasoning_effort != "none" if toggle and reasoning_effort is not None else on
    # Reasoning instructions (Qwen3.8) open the system block with thinking on; the block then also takes
    # the first system message when there are no tools.
    table = chat.get("reasoning_instructions") or {}
    instr = table.get(reasoning_effort, table.get(chat.get("reasoning_effort_default"), "")) if on else ""
    instructed = lambda b: instr + chat["reasoning_separator"] + b if instr and b else instr or b
    if tools:
        declare = xml_tool if chat.get("tools_format") == "xml" else \
            lambda t: json.dumps(t["function"] if chat["tools_function_only"] else t, ensure_ascii=False)
        printed = chat.get("tool_separator", "").join(chat["tool_line_prefix"] + declare(t) for t in tools)
        tb = chat["tools_prefix"] + printed + chat["tools_suffix"]
        sep = chat["tools_system_separator"] if body else ""
        if not first_system and "default_system_tools" in chat:   # the template's tools-time default, separator included
            body, sep = chat["default_system_tools"], ""
        merged = body + sep + tb if chat["tools_after_system"] else tb + sep + body
        out += chat["system_prefix"] + instructed(merged) + chat["suffix"]
    elif instr:
        out += chat["system_prefix"] + instructed(body) + chat["suffix"]
    elif use_default:
        out += chat["system_prefix"] + chat["default_system"] + chat["suffix"]
    prior = None
    last_query = max([i for i, m in enumerate(messages) if m["role"] == "user"], default=-1)
    for i, m in enumerate(messages):
        role, text = m["role"], trim(m.get("content") or "")
        if prior == "tool" and role != "tool":
            out += chat["tool_turn_suffix"]
        if role == "system" and i == 0 and (tools or instr) and first_system:
            pass
        elif role == "tool":
            out += ("" if prior == "tool" else chat["tool_turn_prefix"]) + chat["tool_item_prefix"] + text + chat["tool_item_suffix"]
        elif role == "assistant":
            thought = (m.get("reasoning_content") or "").strip()
            shown = chat["history_think_scope"] == "all" or i > last_query
            think = chat["think_open"] + thought + chat["think_close"] if shown and (thought or chat["history_think"] == "always") and chat["think_open"] else ""
            if not shown and chat.get("history_think_before") == "empty":
                think = chat["think_open"] + chat["think_close"]
            calls = render_calls(chat, m.get("tool_calls") or [])
            sep = chat["content_call_separator"] if text and calls else ""
            out += chat["assistant_prefix"] + think + text + sep + calls + chat["suffix"]
        else:
            out += (chat["user_prefix"] if role == "user" else chat["system_prefix"]) + text + chat["suffix"]
        prior = role
    if gen:
        if prior == "tool":
            out += chat["tool_turn_suffix"]
        out += chat["generation_prompt"] if on else chat["generation_prompt_plain"]
    elif prior == "tool":
        out += chat["tool_turn_suffix"]
    return out


def render_calls(chat, tool_calls):
    """All of an assistant turn's calls as control.loom prints them (render_tool per call + CALLS_CLOSE)."""
    calls = "".join((chat["call_separator"] if j else "") + render_call(chat, c["function"]) for j, c in enumerate(tool_calls))
    if calls and chat.get("calls_open"):
        calls = chat["calls_open"] + calls + chat["calls_close"]
    return calls


def render_call(chat, call):
    args = call["arguments"]
    if chat["call_format"] == "hermes":
        a = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        return '<tool_call>\n{"name": "' + call["name"] + '", "arguments": ' + a + "}\n</tool_call>"
    args = json.loads(args) if isinstance(args, str) else args
    if chat["call_format"] == "pythonic":
        return call["name"] + "(" + ", ".join(f"{k}={py_value(v)}" for k, v in args.items()) + ")"
    val = lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if chat["call_format"] == "qwen_xml":
        return ("<tool_call>\n<function=" + call["name"] + ">\n"
                + "".join(f"<parameter={k}>\n{val(v)}\n</parameter>\n" for k, v in args.items())
                + "</function>\n</tool_call>")
    return ("<tool_call>" + call["name"]
            + "".join(f"<arg_key>{k}</arg_key><arg_value>{val(v)}</arg_value>" for k, v in args.items()) + "</tool_call>")


def jinja_string(v):
    """Jinja's `| string` of a JSON value as Loom prints it: strings as they are, true/false/null as Python's
    True/False/None, numbers and containers as the request's JSON text (json.dumps defaults: Jinja's tojson)."""
    if isinstance(v, str):
        return v
    if isinstance(v, bool) or v is None:
        return str(v)
    return json.dumps(v, ensure_ascii=False)


def xml_tool(tool):
    """One tool declaration as Qwen3-Coder's template prints it (control.loom @xml_tool, chat.tools_format "xml"):
    the function's name, trimmed description, each parameter's name, type, trimmed description and other keys,
    then the parameters' and the function's other keys, each as <key>value</key>. A key's value is printed with
    jinja_string, so a list-valued parameter type prints as JSON where the template prints Python's repr."""
    t = tool.get("function", tool)
    extra = lambda d, handled: "".join(f"\n<{k}>{jinja_string(v)}</{k}>" for k, v in d.items() if k not in handled) \
        if isinstance(d, dict) else ""
    out = "<function>\n<name>" + jinja_string(t["name"]) + "</name>"
    if "description" in t:
        out += "\n<description>" + jinja_string(t["description"]).strip() + "</description>"
    out += "\n<parameters>"
    params = t.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        for name, fields in params["properties"].items():
            out += "\n<parameter>\n<name>" + name + "</name>"
            if isinstance(fields, dict) and "type" in fields:
                out += "\n<type>" + jinja_string(fields["type"]) + "</type>"
            if isinstance(fields, dict) and "description" in fields:
                out += "\n<description>" + jinja_string(fields["description"]).strip() + "</description>"
            out += extra(fields, ("name", "type", "description")) + "\n</parameter>"
    out += extra(params, ("type", "properties")) + "\n</parameters>"
    return out + extra(t, ("type", "name", "description", "parameters")) + "\n</function>"


# Declarations with every key kind xml_tool models: no description, nested and list values, numbers, booleans, null,
# a parameter without a type, parameters without properties, and keys beside the function's own.
TYPED_TOOLS = [
    {"type": "function", "function": {"name": "edit_file", "description": "  Edit a file.\n", "strict": True,
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": " Absolute path. "},
         "mode": {"type": "string", "enum": ["append", "replace"], "default": "replace"},
         "lines": {"type": "array", "items": {"type": "integer", "minimum": 0}, "maxItems": 100},
         "ratio": {"type": "number", "minimum": 0.5, "exclusiveMaximum": False, "nullable": None},
         "anything": {"description": "No type."}},
         "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}},
]


def py_value(v):
    """LFM2 template's format_arg_value: quoted escaped strings, tojson containers, str() otherwise."""
    if isinstance(v, str):
        return "'" + v.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r") + "'"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


TYPED_CALL = {"name": "get_weather", "arguments": {"city": "O'Hare \\ \"x\"\nline", "days": 3, "ratio": 0.5, "flag": True,
                                                   "off": False, "none": None, "tags": ["a", 1], "opts": {"k": "v", "n": [1, 2]}}}

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


def verify(template_text, chat, bos="", eos=""):
    """Returns [(name, ok, detail)]: template render vs simulate() for each conversation."""
    T = Template(template_text, bos, eos)
    results = []
    extra = [("typed arguments", [{"role": "user", "content": "Typed"}, calls_msg("", [TYPED_CALL, CALL2], reasoning="R"),
                                   {"role": "tool", "tool_call_id": "call_0", "content": "ok"}], TOOLS)] \
        if chat["call_format"] == "pythonic" else []
    if chat.get("tools_format") == "xml":
        extra.append(("typed tool declarations", [{"role": "user", "content": "Edit it"}], TYPED_TOOLS))
    for name, msgs, tools in CONVERSATIONS + extra:
        for kw in ({},) if chat["thinking"] != "toggle" else ({"enable_thinking": True}, {"enable_thinking": False}):
            label = name + (f" (enable_thinking={kw['enable_thinking']})" if kw else "")
            try:
                results.append(compare(T, chat, label, msgs, tools, kw))
            except ProbeError as e:
                results.append((label, None, f"template refuses: {e}"))
        # Every other reasoning effort the template's instructions support (the default is covered above).
        table = chat.get("reasoning_instructions") or {}
        for effort in [e for e in table if e != chat.get("reasoning_effort_default")]:
            label = f"{name} (reasoning_effort={effort})"
            try:
                results.append(compare(T, chat, label, msgs, tools, {"reasoning_effort": effort}))
            except ProbeError as e:
                results.append((label, None, f"template refuses: {e}"))
    return results


def compare(T, chat, label, msgs, tools, kw):
    """(label, ok, detail): the template's render with kw vs simulate()."""
    if chat.get("call_format") == "none" and any(m["role"] == "tool" or m.get("tool_calls") for m in msgs):
        return label, None, "tool calling unsupported (the template ignores tools): not modeled"
    ref = T.render(msgs, tools=tools, gen=True, **kw)
    effort = {"reasoning_effort": kw["reasoning_effort"]} if "reasoning_effort" in kw else {}   # kits may wrap simulate()
    got = simulate(chat, msgs, tools, gen=True, enable_thinking=kw.get("enable_thinking"), **effort)
    # The thinking prompt may deliberately append think_open.
    if got != ref and chat["thinking"] != "never" and got.endswith(chat["think_open"]) and got[:-len(chat["think_open"])] == ref:
        return label, True, "exact except the deliberate think_open in the generation prompt"
    if got == ref:
        return label, True, "exact"
    k = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
    return label, False, f"differs at byte {k}: loom {got[max(0, k - 40):k + 60]!r} vs template {ref[max(0, k - 40):k + 60]!r}"


def thinking_never(chat):
    """The block with thinking off, for a model whose template renders reasoning in history (so derive() reads it as
    thinking "always") but which never reasons (Qwen3-*-Instruct-2507): its replies would otherwise land in the
    reasoning channel. The template's own generation prompt, without the deliberate think_open. port.sh's smoke test
    decides this from the model's behavior (a reply that ends without think_close); a template can't tell it from
    models that do always think (Flint, Jan-v1)."""
    chat = dict(chat)
    prompt, think_open = chat["generation_prompt"], chat.get("think_open", "")
    if think_open and prompt.endswith(think_open):
        prompt = prompt[:-len(think_open)]
    chat.update(thinking="never", thinking_default=False, starts_in_reasoning=False,
                generation_prompt=prompt, generation_prompt_plain=prompt)
    return chat


def main():
    if len(sys.argv) > 3 and sys.argv[2] == "--never":
        # cfg.json's chat block with thinking off (thinking_never), kept only if it still verifies against the template.
        cfg = json.load(open(sys.argv[3]))
        chat = thinking_never(cfg["chat"])
        results = verify(load_template(sys.argv[1]), chat, chat.get("bos", ""))
        for name, ok, detail in results:
            print(("OK  " if ok else "SKIP" if ok is None else "FAIL"), name, "-", detail)
        if any(ok is False for _, ok, _ in results):
            sys.exit(1)
        cfg["chat"] = chat
        json.dump(cfg, open(sys.argv[3], "w"), indent=1, ensure_ascii=False)
        return
    text = load_template(sys.argv[1])
    bos = ""
    if sys.argv[1].endswith(".gguf"):
        import gguf
        r = gguf.GGUFReader(sys.argv[1])
        f, tokens = r.fields.get("tokenizer.ggml.bos_token_id"), r.fields.get("tokenizer.ggml.tokens")
        if f is not None and tokens is not None:
            bos = tokens.contents()[f.contents()]
    if len(sys.argv) > 3 and sys.argv[2] == "--verify":
        # A hand-filled block (a format derive() can't model): verify cfg.json's chat block only.
        chat = json.load(open(sys.argv[3]))
        chat = chat.get("chat", chat)
        bad = 0
        for name, ok, detail in verify(text, chat, bos):
            bad += ok is False
            print(("OK  " if ok else "SKIP" if ok is None else "FAIL"), name, "-", detail)
        sys.exit(1 if bad else 0)
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
