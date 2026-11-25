"""Gate renderer for the inkling2 channel chat format (llm_config.json chat.call_format "inkling2").

The starter's chat_probe.py models the chat programs the starter generates; a package whose control.loom renders a
channel format the starter cannot model gets its renderer here, picked by _lib.sh from the package's call_format
(LOOM_KIT). The gates check it against Loom on every request: the agent gate's render_ok (rendered prompt tokens ==
Loom's prompt_tokens) and compare's token-exact prompts. Facts: skills/loom-kb/arch/llm/moe/inkling2/porting.md.

Format (control.loom of the Strix Halo package): every message is <|message_ROLE|>[name]<|content_KIND|>text<|end_message|>;
a system message "Thinking effort level: N" before the first non-system message (N from chat.reasoning_instructions,
"none" when thinking is off); tool declarations first, as compact JSON with sorted keys; an assistant tool call is
<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":..}<|end_message|> and every assistant turn in the
history ends with <|content_model_end_sampling|>. A tool result's header carries the tool name (the message's own
name, else the name of the call its tool_call_id answers). The generation prompt is empty: the model opens its own
channel header, so the raw reply starts with <|message_model|><|content_thinking|> or <|message_model|><|content_text|>.
"""

import json

MODEL = "<|message_model|>"
SYSTEM = "<|message_system|>"
TOOL = "<|message_tool|>"
TEXT = "<|content_text|>"
THINK = "<|content_thinking|>"
INVOKE = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
TERMINAL = "<|content_model_end_sampling|>"
ROLE = {"system": SYSTEM, "user": "<|message_user|>", "assistant": MODEL, "tool": TOOL}


def compact(value, sort=False):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort)


def call_text(call, sort_args):
    function = call.get("function", call)
    name = function["name"]
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return MODEL + name + INVOKE + '{"name":' + compact(name) + ',"args":' + compact(arguments, sort_args) + "}" + END


def render_calls(chat, calls):
    """Calls as the model emits them (argument order as generated)."""
    return "".join(call_text(call, False) for call in calls)


def tool_declaration(tools):
    specs = []
    for tool in tools:
        function = tool.get("function", tool)
        specs.append({"description": function.get("description") or "", "name": function["name"],
                      "parameters": function.get("parameters") or {}, "type": tool.get("type") or "function"})
    return SYSTEM + "tool_declare<|content_xml|>" + compact(specs, True) + END


def simulate(chat, messages, tools=None, gen=False, enable_thinking=None, reasoning_effort=None):
    """control.loom's prompt for messages (+ tools); the generation prompt is empty (gen changes nothing)."""
    effort = reasoning_effort
    if effort is None:
        effort = "none" if enable_thinking is False else chat["reasoning_effort_default"]
    if isinstance(effort, str):
        effort = chat["reasoning_instructions"][effort]
    effort_text = SYSTEM + TEXT + f"Thinking effort level: {effort}" + END
    parts = [tool_declaration(tools)] if tools else []
    call_names = {call["id"]: call["function"]["name"] for m in messages if m["role"] == "assistant"
                  for call in m.get("tool_calls") or [] if call.get("id")}
    emitted = False
    for message in messages:
        role = message["role"]
        if not emitted and role != "system":
            parts.append(effort_text)
            emitted = True
        suffix = message.get("name") or ""
        if role == "tool" and not suffix:
            suffix = call_names.get(message.get("tool_call_id"), "")
        header = ROLE[role] + suffix
        reasoning = message.get("reasoning_content") if role == "assistant" else None
        calls = (message.get("tool_calls") or []) if role == "assistant" else []
        if reasoning:
            parts.append(header + THINK + reasoning + END)
        content = message.get("content")
        if isinstance(content, str) and (content or not (reasoning or calls)):
            parts.append(header + TEXT + content + END)
        elif isinstance(content, list):
            for part in content:
                text = part if isinstance(part, str) else part.get("text", "")
                parts.append(header + TEXT + text + END)
        parts.extend(call_text(call, True) for call in calls)
        if role == "assistant":
            parts.append(TERMINAL)
    if not emitted:
        parts.append(effort_text)
    return "".join(parts)


def raw_from_chat(chat, message, finished, thinking=None, prompt=None):
    """The generated token text behind a parsed reply: channel headers included, the stop token excluded."""
    reasoning = message.get("reasoning_content")
    content = message.get("content") or ""
    calls = message.get("tool_calls") or []
    raw = ""
    if reasoning:
        raw += MODEL + THINK + reasoning + (END if content or calls or finished else "")
    if content:
        raw += MODEL + TEXT + content + (END if calls or finished else "")
    return raw + render_calls(chat, calls)


def raw_alternatives(chat, message, finished, thinking=None, prompt=None):
    """An empty finished reply is either nothing or an empty text message."""
    raw = raw_from_chat(chat, message, finished, thinking, prompt)
    return [raw, MODEL + TEXT + END] if not raw and finished else [raw]
