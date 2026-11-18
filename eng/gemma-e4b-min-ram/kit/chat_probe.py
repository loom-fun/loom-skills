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
    env.globals["strftime_now"] = lambda fmt: CLOCK.strftime(fmt)
    return env


# strftime_now's clock. llama.cpp's server renders the request's local date (Llama 3.2: "Today Date: <today>" in
# every system block), which a control program, rendering without a clock, cannot: Loom renders this fixed date,
# the Llama 3.x templates' own date when no clock is given ("26 Jul 2024"), so the chat block is deterministic and
# every package prints the same prompt on every day.
CLOCK = datetime.datetime(2024, 7, 26)


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
    bos, prefix, suffix, user_prefix = chat["bos"], chat["system_prefix"], sys_suffix(chat), chat["user_prefix"]

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


def system_lead(T, chat, default, bos, kw0, notes):
    """Templates whose first system block always opens with the default system prompt and append the request's first
    system message (Spark-X2.5: '<|System|>\nyou are a helpful assistant.' + '\n\n' + SYS, the tools block in between).
    Sets system_lead (that prompt), system_lead_separator (before the system message without tools), and
    system_prefix from a later system message. A template that prints its BOS token before every message (the same
    one-turn tag) gets bos "" and the token kept in the prefixes."""
    suffix = chat["suffix"]
    r2 = T.render([{"role": "user", "content": "USERPROBE1"}, {"role": "system", "content": "SYSPROBE2"}], **kw0)
    later = between(r2, "USERPROBE1" + suffix, "SYSPROBE2") if "USERPROBE1" + suffix in r2 else None
    b = bos if bos and later is not None and later.startswith(bos) and chat["user_prefix"].startswith(bos) else ""
    blk, probe = b + default[:len(default) - len(suffix)], b + chat["system_prefix"]
    if not (later and default.endswith(suffix) and blk.startswith(later) and probe.startswith(blk)):
        raise ProbeError(f"default system block has an unexpected shape: {default!r}")
    if b:
        chat["bos"] = ""
        notes.append(f"the template prints {bos!r} before every message: part of each role prefix (bos '')")
    chat["system_prefix"], chat["default_system"] = later, ""
    chat["system_lead"], chat["system_lead_separator"] = blk[len(later):], probe[len(blk):]
    notes.append(f"every first system block opens with the default system prompt {chat['system_lead'][:60]!r}; a system "
                 f"message follows it after {chat['system_lead_separator']!r} (system_lead)")


def lead_instructions(chat, heads, notes):
    """Per-effort reasoning instructions appended to the system lead with thinking on (Spark-X2.6:
    default_system + '\n`thinking_effort` ... `thinking_effort=high`.'); chat.reasoning_after_lead."""
    prefix, suffix, lead = chat["system_prefix"], sys_suffix(chat), chat["system_lead"]
    def body(h):
        h = h[len(chat["bos"]):] if chat["bos"] and h.startswith(chat["bos"]) else h
        cut = suffix + chat["user_prefix"]
        if not (h.startswith(prefix + lead) and h.endswith(cut)):
            raise ProbeError(f"reasoning instructions: unexpected head {h!r}")
        return h[len(prefix) + len(lead):len(h) - len(cut)]
    if body(heads["off"]):
        raise ProbeError(f"thinking off changes the default system prompt: {body(heads['off'])!r}")
    table = {e: body(h) for e, h in heads.items() if e not in ("", "off", "none")}
    matches = [e for e in table if heads[e] == heads[""]]
    if not matches:
        raise ProbeError("the default reasoning effort is none of " + ", ".join(table))
    chat["reasoning_instructions"], chat["reasoning_separator"], chat["reasoning_after_lead"] = table, "", True
    chat["reasoning_effort_default"] = max(matches, key=EFFORT_LEVELS.get)
    notes.append(f"template appends reasoning instructions per reasoning_effort ({', '.join(table)}; default "
                 f"{chat['reasoning_effort_default']}) to the default system prompt with thinking on")


# A template that ignores tools: no tool declarations, calls or results (call_format "none"; build.py makes
# control.loom ignore a request's tools and never parse calls). call_marker keeps build.py's default: output_next
# ends content at the marker, and an empty one would end it at once.
CHAT_ONLY = {"tools_prefix": "", "tools_suffix": "", "tool_line_prefix": "", "tool_separator": "",
             "tools_system_separator": "", "tools_after_system": True, "tools_function_only": True,
             "call_format": "none", "calls_open": "", "calls_close": "", "call_end": "", "call_marker": "<tool_call>",
             "content_call_separator": "", "call_separator": "", "tool_turn_prefix": "", "tool_item_prefix": "",
             "tool_item_suffix": "", "tool_turn_suffix": ""}


def sys_suffix(chat):
    """What closes a system block (chat.system_suffix when the template closes each role with its own tag)."""
    return chat.get("system_suffix", chat["suffix"])


def asst_suffix(chat):
    """What closes an assistant turn (chat.assistant_suffix when the template closes each role with its own tag)."""
    return chat.get("assistant_suffix", chat["suffix"])


def system_in_user(T, chat, r, gap, bos, kw0):
    """Templates with no system turn that print the first system message inside the first user turn, before its
    text (Gemma 3: '<start_of_turn>user\\n' + system + '\\n\\n' + user text): chat["system_in_user"] is the separator
    after the system text, chat["system_in_user_trim"] whether that text is trimmed like other content. Returns the
    plain conversation's rendering (no system message, BOS cut), from which derive() reads the other prefixes."""
    plain = T.render([{"role": "user", "content": "USERPROBE1"}, {"role": "assistant", "content": "ASSTPROBE1"},
                      {"role": "user", "content": "USERPROBE2"}], **kw0)
    plain = plain[len(bos):] if bos and plain.startswith(bos) else plain
    user_prefix = plain[:plain.index("USERPROBE1")]
    if not gap or r != user_prefix + "SYSPROBE" + gap + plain[len(user_prefix):]:
        raise ProbeError(f"system message is not closed by the user suffix {chat['suffix']!r}: {gap!r}")
    spaced = T.render([{"role": "system", "content": " SYSPROBE \n"}, {"role": "user", "content": "USERPROBE1"}], **kw0)
    chat["user_prefix"], chat["system_in_user"] = user_prefix, gap
    chat["system_in_user_trim"] = " SYSPROBE \n" + gap not in spaced
    # a later system message (no system turn to model it on): rendered as a turn of that role
    chat["system_prefix"] = user_prefix.replace("user", "system", 1)
    return plain


def derive(template_text, bos="", eos=""):
    T = Template(template_text, bos, eos)
    if "<|turn>" in template_text and "<|tool_call>call:" in template_text:
        return derive_gemma4(T, bos)
    notes = []
    chat = {}
    # BOS: emitted by the control program at the very start when the template prints it.
    # Decided with a sentinel: a GGUF whose bos id is the template's own role tag (MiniCPM-V: <|im_start|>) starts every
    # render with that text although the template never prints bos_token.
    sentinel = Template(template_text, "BOSPROBE", eos)
    first = sentinel.render([{"role": "user", "content": "USERPROBE1"}])
    # ... but a template that prints the BOS text itself, once, at the start (Laguna: '〈|EOS|〉') does print it.
    turns = sentinel.render([{"role": "user", "content": "USERPROBE1"}, {"role": "assistant", "content": "ASSTPROBE1"},
                             {"role": "user", "content": "USERPROBE2"}])
    printed = first.startswith("BOSPROBE") or bool(bos) and first.startswith(bos) and turns.count(bos) == 1
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
        # Per-role closing tags (Laguna: '<system>..</system>\n', '<user>..</user>\n', '<assistant>..</assistant>\n'):
        # the user suffix with the role's name (system_suffix / assistant_suffix, present only when they differ).
        cand = chat["suffix"].replace("user", "system")
        if "user" in chat["suffix"] and gap.startswith(cand):
            chat["system_suffix"] = cand
            notes.append(f"system messages close with {cand!r} (system_suffix)")
            chat["user_prefix"] = gap[len(cand):]
        else:
            r = system_in_user(T, chat, r, gap, bos, kw0)   # raises when it is not that shape either
    else:
        chat["user_prefix"] = gap[len(chat["suffix"]):]
    gap = between(r, "USERPROBE1", "ASSTPROBE1")
    chat["assistant_prefix"] = gap[len(chat["suffix"]):]
    gap = between(r, "ASSTPROBE1", "USERPROBE2")
    if not gap.startswith(chat["suffix"]):
        cand = chat["suffix"].replace("user", "assistant")
        if "user" in chat["suffix"] and gap == cand + chat["user_prefix"]:
            chat["assistant_suffix"] = cand
            notes.append(f"assistant turns close with {cand!r} (assistant_suffix)")
    head = single[:single.index("USERPROBE1")]
    if head.startswith(bos) and bos:
        head = head[len(bos):]
    default = head[:len(head) - len(chat["user_prefix"])] if head.endswith(chat["user_prefix"]) else ""
    # A system block the template always prints, opening with fixed text before the system message (Llama 3:
    # "Cutting Knowledge Date: ...\nToday Date: ...\n\n"): the role prefix is the user prefix's spelling, the rest
    # system_preamble. Present only then.
    role_prefix = chat["user_prefix"].replace("user", "system")
    if role_prefix != chat["user_prefix"] and chat["system_prefix"] != role_prefix and chat["system_prefix"].startswith(role_prefix) \
            and default.startswith(chat["system_prefix"]):
        chat["system_preamble"] = chat["system_prefix"][len(role_prefix):]
        chat["system_prefix"] = role_prefix
        notes.append(f"the system block is always printed and opens with {chat['system_preamble'][:60]!r}...")
    if default and "system_preamble" not in chat and not (default.startswith(chat["system_prefix"]) and default.endswith(sys_suffix(chat))):
        system_lead(T, chat, default, bos, kw0, notes)   # raises when it is not that shape either
        bos = chat["bos"]
    elif default:
        lead = chat["system_prefix"] + chat.get("system_preamble", "")
        if not (default.startswith(lead) and default.endswith(sys_suffix(chat))):
            if "system_preamble" in chat:
                raise ProbeError(f"default system block has an unexpected shape: {default!r}")
            system_lead(T, chat, default, bos, kw0, notes)   # raises when it is not that shape either
            bos = chat["bos"]
        else:
            chat["default_system"] = default[len(lead):len(default) - len(sys_suffix(chat))]
            if chat["default_system"]:
                notes.append(f"template inserts a default system prompt: {chat['default_system'][:60]!r}...")
    else:
        chat["default_system"] = ""
    if efforts and chat.get("system_lead"):
        lead_instructions(chat, efforts, notes)
    elif efforts:
        reasoning_instructions(T, chat, efforts, notes)
    trimmed = T.render([{"role": "user", "content": "  USERPROBE1 \n"}], **kw0)
    chat["trim_content"] = "  USERPROBE1" not in trimmed
    # Generation prompts and thinking (each relative to the same render without the prompt: with
    # reasoning instructions the system block depends on enable_thinking too).
    user1 = [{"role": "user", "content": "USERPROBE1"}]
    gen = lambda **kw: T.render(user1, gen=True, **kw)[len(T.render(user1, **kw)):]
    g_default, g_on, g_off = gen(), gen(enable_thinking=True), gen(enable_thinking=False)
    # Reasoning in history (agent loop: assistant after the last user query, followed by a tool result).
    agent = lambda m, **kw: T.render([{"role": "user", "content": "USERPROBE1"}, m,
                                      {"role": "tool", "tool_call_id": "call_0", "content": "TOOLPROBE1"}], tools=TOOLS, **kw)
    try:
        with_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning="REASONPROBE"))
    except ProbeError:   # a template that refuses tool turns (Gemma 3: roles must alternate): the assistant turn alone
        agent = lambda m, **kw: T.render([{"role": "user", "content": "USERPROBE1"}, m], tools=TOOLS, **kw)
        with_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning="REASONPROBE"))
    # Templates that show reasoning in history only with thinking on (Laguna: '<think>R</think>' in every assistant
    # turn with enable_thinking, a bare '</think>' without it, and thinking off by default): read the reasoning
    # framing from the thinking-on render; history_think_off below says what a turn renders with thinking off.
    hist_kw = {}
    if "REASONPROBE" not in with_r and g_on != g_off and "REASONPROBE" in agent(
            calls_msg("ASSTPROBE1", [CALL1], reasoning="REASONPROBE"), enable_thinking=True):
        hist_kw = {"enable_thinking": True}
        with_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning="REASONPROBE"), **hist_kw)
    without_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning=""), **hist_kw)
    # Templates that keep reasoning in every assistant turn (preserve_thinking: Qwen3.6, Qwen3.8) render
    # an empty think block in the plain turn above, which then is not part of the assistant prefix.
    guess = chat["user_prefix"].replace("user", "assistant")
    # A template that drops an assistant turn's content when it has tool calls (Llama 3) shows no reasoning here.
    shown = "ASSTPROBE1" in with_r
    if shown and guess != chat["user_prefix"] and chat["assistant_prefix"].startswith(guess) and chat["assistant_prefix"] != guess:
        rest, mid_r, lead = chat["assistant_prefix"][len(guess):], between(with_r, "USERPROBE1", "ASSTPROBE1"), chat["suffix"] + guess
        if "REASONPROBE" in mid_r and mid_r.startswith(lead):
            opened, closed = mid_r[len(lead):mid_r.index("REASONPROBE")], mid_r[mid_r.index("REASONPROBE") + len("REASONPROBE"):]
            if opened + closed == rest:
                chat["assistant_prefix"] = guess
                notes.append(f"every assistant turn renders its (possibly empty) think block: assistant prefix {guess!r}")
    mid_r = between(with_r, "USERPROBE1", "ASSTPROBE1") if shown else ""
    if "REASONPROBE" in mid_r and chat["assistant_prefix"] not in mid_r[:mid_r.index("REASONPROBE")]:
        # A turn without reasoning renders a bare think close (Spark-X2.5: "<|Bot|></think>"), so the plain turn above put
        # it in the assistant prefix: cut it back out (history_think "close" below).
        closed = mid_r[mid_r.index("REASONPROBE") + len("REASONPROBE"):]
        cand = chat["assistant_prefix"][:len(chat["assistant_prefix"]) - len(closed)]
        if closed and chat["assistant_prefix"].endswith(closed) and mid_r.startswith(chat["suffix"] + cand):
            chat["assistant_prefix"] = cand
            notes.append(f"an assistant turn without reasoning renders only {closed!r}: assistant prefix {cand!r}")
    if "REASONPROBE" in mid_r:
        pre = mid_r[:mid_r.index("REASONPROBE")]
        cut = pre.index(chat["assistant_prefix"]) + len(chat["assistant_prefix"]) if chat["assistant_prefix"] in pre else None
        if cut is None:
            raise ProbeError(f"assistant prefix {chat['assistant_prefix']!r} not found before reasoning: {mid_r!r}")
        chat["think_open"] = pre[cut:]
        chat["think_close"] = mid_r[mid_r.index("REASONPROBE") + len("REASONPROBE"):]
        mid_n = between(without_r, "USERPROBE1", "ASSTPROBE1")
        empty_block = mid_n[mid_n.index(chat["assistant_prefix"]) + len(chat["assistant_prefix"]):]
        chat["history_think"] = ("always" if empty_block == chat["think_open"] + chat["think_close"] else
                                 "close" if empty_block and empty_block == chat["think_close"] else "nonempty")
        if empty_block not in ("", chat["think_open"] + chat["think_close"], chat["think_close"]):
            notes.append(f"empty reasoning renders as {empty_block!r} (not modeled)")
    else:
        chat["think_open"], chat["think_close"], chat["history_think"] = "", "", "nonempty"
    early = T.render([{"role": "user", "content": "USERPROBE1"},
                      {"role": "assistant", "content": "ASSTPROBE1", "reasoning_content": "REASONPROBE"},
                      {"role": "user", "content": "USERPROBE2"}], **hist_kw)
    # Qwen-style templates drop reasoning from turns before the last user query.
    chat["history_think_scope"] = "all" if "REASONPROBE" in early else "after_last_query"
    if hist_kw and chat["think_open"]:
        # Thinking off: every assistant turn renders think_close alone, its reasoning dropped (history_think_off "close").
        off = between(agent(calls_msg("ASSTPROBE1", [CALL1], reasoning="REASONPROBE"), enable_thinking=False), "USERPROBE1", "ASSTPROBE1")
        if off != chat["suffix"] + chat["assistant_prefix"] + chat["think_close"]:
            raise ProbeError(f"assistant turn with thinking off renders {off!r} (not modeled)")
        chat["history_think_off"] = "close"
        notes.append(f"reasoning shows in history only with thinking on; with it off every assistant turn renders "
                     f"{chat['think_close']!r} alone (history_think_off)")
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
        chat["terminal"] = asst_suffix(chat).strip() or asst_suffix(chat)
        notes.append("template ignores tools: chat-only block, tool calling unsupported (call_format none)")
        return chat, notes
    t_sys = T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"}], tools=TOOLS, **kw0)
    printed = None
    for indent in (None, 4, 2):   # tojson(indent=N) (Llama 3: 4): tools_indent, present only then
        for only in (False, True):
            for seps in ((", ", ": "), (",", ":")) if indent is None else ((",", ": "),):
                cand = [json.dumps(t["function"] if only else t, ensure_ascii=False, separators=seps, indent=indent) for t in TOOLS]
                if all(c in t_plain for c in cand):
                    printed, chat["tools_function_only"] = cand, only
                    if indent:
                        chat["tools_indent"] = indent
                        notes.append(f"template prints tools with indent {indent} (Loom re-indents the request's tool JSON)")
                    elif seps != (", ", ": "):   # control.loom compacts each tool's JSON (host json codec)
                        chat["tools_compact"] = True
                        notes.append("template prints tools compactly (tools_compact)")
                    break
            if printed:
                break
        if printed:
            break
    if not printed and all(xml_tool(t) in t_plain for t in TOOLS):
        # Tool declarations as XML elements (Qwen3-Coder: <function><name>..</name><parameters>..): tools_format "xml".
        printed, chat["tools_function_only"], chat["tools_format"] = [xml_tool(t) for t in TOOLS], True, "xml"
        notes.append("template declares tools as XML elements (tools_format xml)")
    if not printed:
        raise ProbeError("tool definitions are not printed as JSON objects or XML elements one per line; fill tools_* by hand")
    first_tool = t_plain.index(printed[0])
    in_user = t_plain.rfind(chat["user_prefix"], 0, first_tool)
    if in_user >= 0 and chat["suffix"] not in t_plain[in_user + len(chat["user_prefix"]):first_tool]:
        # Tools open the first user message (Llama 3); the always-printed system block takes another preamble
        # (system_preamble_tools: "Environment: ipython\n" + the preamble). tools_in_user, present only then.
        if "system_preamble" not in chat:
            raise ProbeError("tools in the first user message without an always-printed system block (not modeled)")
        chat["tools_in_user"] = True
        sys_block = t_plain[len(bos) if bos and t_plain.startswith(bos) else 0:in_user]
        body = sys_block[len(chat["system_prefix"]):len(sys_block) - len(chat["suffix"])]
        if not (sys_block.startswith(chat["system_prefix"]) and sys_block.endswith(chat["suffix"]) and body.endswith(chat["default_system"])):
            raise ProbeError(f"system block with tools has an unexpected shape: {sys_block[:200]!r}")
        chat["system_preamble_tools"] = body[:len(body) - len(chat["default_system"])]
        user_text = t_plain[in_user + len(chat["user_prefix"]):t_plain.index("USERPROBE1")]
        i0, i1 = user_text.index(printed[0]), user_text.index(printed[1])
        chat["tools_prefix"], chat["tools_suffix"] = user_text[:i0], user_text[i1 + len(printed[1]):]
        chat["tool_line_prefix"], chat["tool_separator"] = "", user_text[i0 + len(printed[0]):i1]
        chat["tools_after_system"], chat["tools_system_separator"] = True, ""
        notes.append("tools are printed in the first user message")
    else:
        block = t_plain[:t_plain.index("USERPROBE1")]
        if block.startswith(bos) and bos:
            block = block[len(bos):]
        block = block[:len(block) - len(chat["user_prefix"])]
        if not (block.startswith(chat["system_prefix"]) and block.endswith(sys_suffix(chat))):
            raise ProbeError(f"tools are not rendered as a system block: {block[:200]!r}")
        body = block[len(chat["system_prefix"]):len(block) - len(sys_suffix(chat))]
        lead = chat.get("system_lead", "")
        if lead:
            if not body.startswith(lead):
                raise ProbeError(f"tools block does not follow the default system prompt: {body[:200]!r}")
            body = body[len(lead):]
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
        s_body = s_body[len(chat["system_prefix"]):len(s_body) - len(chat["user_prefix"]) - len(sys_suffix(chat))]
        if lead:
            s_body = s_body[len(lead):] if s_body.startswith(lead) else s_body
            if s_body.startswith("SYSPROBE"):
                raise ProbeError("system message before the tools block after the default system prompt (not modeled)")
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
    calls_probe = lambda calls: T.render([{"role": "user", "content": "USERPROBE1"}, calls_msg("ASSTPROBE1", calls),
                                          {"role": "tool", "tool_call_id": "call_0", "content": "TOOLPROBE1"},
                                          {"role": "tool", "tool_call_id": "call_1", "content": "TOOLPROBE2"},
                                          {"role": "user", "content": "USERPROBE2"}], tools=TOOLS)
    try:
        two, single_call = calls_probe([CALL1, CALL2]), False
    except ProbeError:   # one call per turn (Llama 3 raises on parallel calls; verify's parallel probe then SKIPs)
        two, single_call = calls_probe([CALL1]), True
        notes.append("template accepts one tool call per assistant turn")
    if "ASSTPROBE1" not in two:
        # The template drops the content of an assistant turn that has calls (Llama 3): calls_drop_content, present
        # only then. The probe's content marker is put back where the content would be, for the framing below.
        chat["calls_drop_content"] = True
        at = two.index(chat["assistant_prefix"], two.index("USERPROBE1")) + len(chat["assistant_prefix"])
        two = two[:at] + "ASSTPROBE1" + two[at:]
        notes.append("an assistant turn with tool calls renders no content")
    turn = two[two.index("ASSTPROBE1"):]
    if "get_weather(city=" in turn:
        # Pythonic calls (LFM2): <|tool_call_start|>[f(a='x', b=3), g()]<|tool_call_end|>
        chat["call_format"] = "pythonic"
        name0 = turn.index("get_weather(")
        chat["calls_open"] = opened = turn[len("ASSTPROBE1"):name0]
        end1 = turn.index(")", name0) + 1
        chat["call_separator"] = turn[end1:turn.index("search(", end1)]
        close_at = turn.index(")", turn.index("search(")) + 1
        chat["calls_close"] = turn[close_at:turn.index(asst_suffix(chat), close_at)]
        chat["call_marker"] = opened.rstrip("[(")  # output_next stops content here
        chat["call_end"] = chat["calls_close"].lstrip("])")
        chat["content_call_separator"] = ""
        if not opened.startswith(chat["call_marker"]) or not chat["call_marker"]:
            raise ProbeError(f"unexpected pythonic call framing: {opened!r}")
        after_calls = turn[close_at + len(chat["calls_close"]):]
        two = "ASSTPROBE1" + after_calls  # tool-response framing below
        end2 = len("ASSTPROBE1")
    elif '{"name": "get_weather", "' in turn and "<tool_call>" not in turn:
        # Bare JSON calls (Llama 3: {"name": NAME, "parameters": ARGS}): call_arguments_key names the arguments.
        chat["call_format"] = "json"
        key_at = turn.index('{"name": "get_weather", "') + len('{"name": "get_weather", "')
        chat["call_arguments_key"] = turn[key_at:turn.index('"', key_at)]
    elif '{"name": "get_weather"' in turn:
        chat["call_format"] = "hermes"
    elif "<function=get_weather>" in turn:
        chat["call_format"] = "qwen_xml"
    elif "get_weather<arg_key>" in turn:
        chat["call_format"] = "glm"
    else:
        raise ProbeError("unknown tool-call format: " + turn[:300])
    if chat["call_format"] == "json":
        chat["calls_open"] = chat["calls_close"] = chat["call_end"] = ""
        chat["call_marker"] = '{"name"'   # output_next ends content here
        start1 = two.index('{"name": "get_weather"', two.index("ASSTPROBE1"))
        chat["content_call_separator"] = two[two.index("ASSTPROBE1") + len("ASSTPROBE1"):start1]
        end1 = start1 + len(render_call(chat, CALL1))
        if two[start1:end1] != render_call(chat, CALL1):
            raise ProbeError(f"JSON call printed differently: {two[start1:end1 + 20]!r}")
        if single_call:
            chat["call_separator"], end2 = "", end1
        else:
            start2 = two.index('{"name": "search"', end1)
            chat["call_separator"], end2 = two[end1:start2], start2 + len(render_call(chat, CALL2))
    elif chat["call_format"] != "pythonic":
        chat["calls_open"] = chat["calls_close"] = chat["call_end"] = ""
        start1 = two.index("<tool_call>", two.index("ASSTPROBE1"))
        chat["call_marker"] = "<tool_call>"
        chat["content_call_separator"] = two[two.index("ASSTPROBE1") + len("ASSTPROBE1"):start1]
        end1 = two.index("</tool_call>", start1) + len("</tool_call>")
        start2 = two.index("<tool_call>", end1)
        chat["call_separator"] = two[end1:start2]
        end2 = two.index("</tool_call>", start2) + len("</tool_call>")
    after_calls = two[end2:]
    if not after_calls.startswith(asst_suffix(chat)):
        raise ProbeError(f"assistant turn with calls is not closed by the suffix: {after_calls[:80]!r}")
    after_calls = after_calls[len(asst_suffix(chat)):]
    if '"TOOLPROBE1"' in after_calls and '"TOOLPROBE2"' in after_calls:
        # Tool results printed as JSON strings (Llama 3: content | tojson): tool_content_json, present only then.
        chat["tool_content_json"] = True
        after_calls = after_calls.replace('"TOOLPROBE1"', "TOOLPROBE1").replace('"TOOLPROBE2"', "TOOLPROBE2")
        notes.append("tool results are printed as JSON strings")
    pre = after_calls[:after_calls.index("TOOLPROBE1")]
    mid = between(after_calls, "TOOLPROBE1", "TOOLPROBE2")
    post = after_calls[after_calls.index("TOOLPROBE2") + len("TOOLPROBE2"):after_calls.index("USERPROBE2")]
    post = post[:len(post) - len(chat["user_prefix"])]
    splits = [k for k in range(len(mid) + 1) if pre.endswith(mid[k:]) and post.startswith(mid[:k])]
    if not splits:
        raise ProbeError(f"cannot split tool-response framing: pre={pre!r} mid={mid!r} post={post!r}")
    # The first split that cuts no <...> tag in two (Spark-X2.5: '<|Tool|>' and '</tool_response>' both end in '>'),
    # else the first one.
    open_tag = lambda t: t.rfind("<") > t.rfind(">")
    k = next((k for k in splits if not open_tag(pre[:len(pre) - len(mid) + k]) and not open_tag(mid[:k])), splits[0])
    item_suffix, item_prefix = mid[:k], mid[k:]
    chat["tool_turn_prefix"] = pre[:len(pre) - len(item_prefix)]
    chat["tool_item_prefix"], chat["tool_item_suffix"] = item_prefix, item_suffix
    chat["tool_turn_suffix"] = post[len(item_suffix):]
    chat["terminal"] = asst_suffix(chat).strip() or asst_suffix(chat)
    return chat, notes


def simulate(chat, messages, tools=None, gen=False, enable_thinking=None, reasoning_effort=None):
    """Python model of control.loom's rendering (chat_begin + chat_message + chat_end)."""
    if chat.get("format") == "gemma4":
        return simulate_gemma4(chat, messages, tools, gen, enable_thinking)
    tools = None if chat.get("call_format") == "none" else tools   # a chat-only block ignores tools, as its template does
    trim = (lambda s: s.strip()) if chat["trim_content"] else (lambda s: s)
    out = chat.get("bos", "")
    first_system = messages and messages[0]["role"] == "system" and isinstance(messages[0]["content"], str)
    sys_text = trim(messages[0]["content"]) if first_system else ""
    folded = first_system and bool(chat.get("system_in_user"))   # printed at the start of the first user turn
    if folded:
        sys_text = (trim if chat.get("system_in_user_trim") else (lambda s: s))(messages[0]["content"])
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
    preamble = "system_preamble" in chat   # Llama 3: the system block is always printed, opening with fixed text
    in_user = bool(tools) and chat.get("tools_in_user", False)   # Llama 3: tools open the first user message
    system_lead = chat.get("system_lead", "")   # the default prompt that always opens the first system block
    if tools:
        seps = (",", ":") if chat.get("tools_compact") else None
        declare = xml_tool if chat.get("tools_format") == "xml" else lambda t: json.dumps(
            t["function"] if chat["tools_function_only"] else t, ensure_ascii=False, separators=seps, indent=chat.get("tools_indent"))
        printed = chat.get("tool_separator", "").join(chat["tool_line_prefix"] + declare(t) for t in tools)
        tb = chat["tools_prefix"] + printed + chat["tools_suffix"]
        sep = chat["tools_system_separator"] if body else ""
    if preamble:
        lead = chat["system_preamble_tools"] if tools else chat["system_preamble"]
        out += chat["system_prefix"] + lead + body + chat["suffix"]
    elif system_lead:   # the default prompt (+ effort instructions), the tools block, then the first system message
        rest = (tb if tools else "") + ((chat["tools_system_separator"] if tools else chat["system_lead_separator"]) + body if body else "")
        out += chat["system_prefix"] + system_lead + instr + rest + sys_suffix(chat)
    elif tools:
        sep = chat["tools_system_separator"] if body else ""
        if not first_system and "default_system_tools" in chat:   # the template's tools-time default, separator included
            body, sep = chat["default_system_tools"], ""
        merged = body + sep + tb if chat["tools_after_system"] else tb + sep + body
        out += chat["system_prefix"] + instructed(merged) + sys_suffix(chat)
    elif instr:
        out += chat["system_prefix"] + instructed(body) + sys_suffix(chat)
    elif use_default:
        out += chat["system_prefix"] + chat["default_system"] + sys_suffix(chat)
    prior = None
    last_query = max([i for i, m in enumerate(messages) if m["role"] == "user"], default=-1)
    for i, m in enumerate(messages):
        role, text = m["role"], trim(m.get("content") or "")
        if prior == "tool" and role != "tool":
            out += chat["tool_turn_suffix"]
        if role == "system" and i == 0 and (tools or instr or preamble or system_lead or folded) and first_system:
            pass
        elif folded and i == 1:
            out += chat["user_prefix"] + sys_text + chat["system_in_user"] + text + chat["suffix"]
        elif in_user:   # the first message after the system block takes the tools, whatever its role (as the template)
            out += chat["user_prefix"] + tb + text + chat["suffix"]
            in_user = False
        elif role == "tool":
            if chat.get("tool_content_json"):
                text = json.dumps(m.get("content") or "", ensure_ascii=False)
            out += ("" if prior == "tool" else chat["tool_turn_prefix"]) + chat["tool_item_prefix"] + text + chat["tool_item_suffix"]
        elif role == "assistant":
            thought = (m.get("reasoning_content") or "").strip()
            shown = chat["history_think_scope"] == "all" or i > last_query
            think = chat["think_open"] + thought + chat["think_close"] if shown and (thought or chat["history_think"] == "always") and chat["think_open"] else ""
            think = chat["think_close"] if shown and not thought and chat["history_think"] == "close" else think
            think = chat["think_close"] if not on and chat.get("history_think_off") == "close" else think
            if not shown and chat.get("history_think_before") == "empty":
                think = chat["think_open"] + chat["think_close"]
            calls = render_calls(chat, m.get("tool_calls") or [])
            text = "" if calls and chat.get("calls_drop_content") else text
            sep = chat["content_call_separator"] if text and calls else ""
            out += chat["assistant_prefix"] + think + text + sep + calls + asst_suffix(chat)
        elif role == "user":
            out += chat["user_prefix"] + text + chat["suffix"]
        else:
            out += chat["system_prefix"] + text + sys_suffix(chat)
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
    if chat["call_format"] == "gemma4":
        return g4_call(call)
    args = call["arguments"]
    if chat["call_format"] == "hermes":
        a = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        return '<tool_call>\n{"name": "' + call["name"] + '", "arguments": ' + a + "}\n</tool_call>"
    if chat["call_format"] == "json":   # the arguments as an object (llama.cpp parses a string before rendering)
        a = json.dumps(json.loads(args) if isinstance(args, str) else args, ensure_ascii=False)
        return '{"name": "' + call["name"] + '", "' + chat["call_arguments_key"] + '": ' + a + "}"
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


def xml_tool(tool):
    """One tool declaration as Qwen3-Coder's template prints it (control.loom @xml_tool, chat.tools_format "xml"):
    the function's name, trimmed description, each parameter's name, type, trimmed description and other keys,
    then the parameters' and the function's other keys, each as <key>value</key>. A value is printed as Jinja's
    `| string` (Loom prints the request's JSON text for containers and numbers: py_value), so a list-valued
    parameter type prints as JSON where the template prints Python's repr."""
    string = lambda v: v if isinstance(v, str) else py_value(v)
    t = tool.get("function", tool)
    extra = lambda d, handled: "".join(f"\n<{k}>{string(v)}</{k}>" for k, v in d.items() if k not in handled) \
        if isinstance(d, dict) else ""
    out = "<function>\n<name>" + string(t["name"]) + "</name>"
    if "description" in t:
        out += "\n<description>" + string(t["description"]).strip() + "</description>"
    out += "\n<parameters>"
    params = t.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        for name, fields in params["properties"].items():
            out += "\n<parameter>\n<name>" + name + "</name>"
            if isinstance(fields, dict) and "type" in fields:
                out += "\n<type>" + string(fields["type"]) + "</type>"
            if isinstance(fields, dict) and "description" in fields:
                out += "\n<description>" + string(fields["description"]).strip() + "</description>"
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
    ("content and call without result", [{"role": "user", "content": "A"}, calls_msg("Checking.", [CALL1], reasoning="R")], TOOLS),
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
            out += pending["later"] + g4_close(c, pending, role)
        pending = None
        if role == "assistant":
            if prior != "assistant":
                out += c["assistant_prefix"]
            thought = m.get("reasoning_content") or m.get("reasoning")
            if thought and i > last_user:
                out += c["think_open"] + thought + c["think_close"]
            text = g4_text(m.get("content"), trim=True)
            calls = m.get("tool_calls") or []
            # The content before the calls, or (content_after_calls) after the calls and their results.
            later = text if c.get("content_after_calls") else ""
            out += text[len(later):] + "".join(g4_call(x["function"]) for x in calls)
            pending = {"calls": bool(calls), "responses": False, "content": bool(text), "later": later}
        else:
            out += (c["user_prefix"] if role == "user" else c["system_prefix"]) + g4_text(m.get("content"), trim=True) + c["suffix"]
        prior = role
    after = None
    if pending:
        out += pending["later"] + g4_close(c, pending, None)
        if pending["calls"]:
            after = "response" if pending["responses"] else "call"
    if gen:
        if after is None:
            out += c["generation_prompt"] if on else c["generation_prompt_plain"]
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
    # Generation prompts with thinking on and off (Gemma 4 12B/26B: an empty thought channel only with thinking off).
    user1 = [{"role": "user", "content": "USERPROBE1"}]
    gen = lambda on: T.render(user1, gen=True, enable_thinking=on)[len(T.render(user1, enable_thinking=on)):]
    c["generation_prompt"], c["generation_prompt_plain"] = gen(True), gen(False)
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
    # The thought channel's close ends where the turn's content or calls start: Gemma 4's own templates render the
    # calls and their results first and the content after them (content_after_calls), some fine-tunes' the content first.
    c["think_close"] = mid[mid.index("REASONPROBE") + len("REASONPROBE"):].split("<|tool_call>")[0]
    content_after = agent.index("ASSTPROBE1") > agent.index("<|tool_call>")
    if content_after:
        c["content_after_calls"] = True
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
    answered = g4_call(CALL1) + c["tool_item_prefix"] + "get_weather{value:" + G4_Q + "TOOLPROBE1" + G4_Q + c["tool_item_suffix"]
    want = {"tool declaration": "<|tool>" + g4_declaration(TOOLS[0]) + "<tool|>",
            "tool call": answered + "ASSTPROBE1" if content_after else "ASSTPROBE1" + answered}
    for k, v in want.items():
        if v not in expect[k]:
            raise ProbeError(f"gemma4 {k} syntax differs from the template: {v!r} not in {expect[k]!r}")
    return c, ["gemma4 format: look-ahead turn closes, tool responses inside the model turn, custom tool syntax"]


def verify(template_text, chat, bos="", eos=""):
    """Returns [(name, ok, detail)]: template render vs simulate() for each conversation."""
    T = Template(template_text, bos, eos)
    results = []
    extra = [("typed arguments", [{"role": "user", "content": "Typed"}, calls_msg("", [TYPED_CALL, CALL2], reasoning="R"),
                                   {"role": "tool", "tool_call_id": "call_0", "content": "ok"}], TOOLS)] \
        if chat["call_format"] == "pythonic" else \
        [("typed arguments", [{"role": "user", "content": "Typed"}, calls_msg("", [TYPED_CALL]),
                              {"role": "tool", "tool_call_id": "call_0", "content": 'say "hi"\n\tback\\slash é'}], TOOLS)] \
        if chat["call_format"] == "json" else []
    if chat.get("tools_format") == "xml":
        extra.append(("typed tool declarations", [{"role": "user", "content": "Edit it"}], TYPED_TOOLS))
    for name, msgs, tools in CONVERSATIONS + extra + (GEMMA4_CONVERSATIONS if chat.get("format") == "gemma4" else []):
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


def raw_alternatives(chat, message, finished, thinking=None, prompt=None):
    """Benchmark hook (agent_golden.py): every raw text consistent with a parsed reply, the default first. A model
    may write chat.call_prefix_token (Llama 3.1: <|python_tag|>) before its JSON call; the reply's channels don't
    say whether it did, so both spellings are offered. [] when there is no such ambiguity."""
    calls = render_calls(chat, message.get("tool_calls") or [])
    if not (calls and chat.get("call_prefix_token")):
        return []
    content = message.get("content") or ""
    base = content + (chat["content_call_separator"] if content else "") + calls
    return [base, content + chat["call_prefix_token"] + calls]


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
