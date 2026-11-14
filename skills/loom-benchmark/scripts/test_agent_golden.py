"""Unit tests of the agent gate's history-rewrite rule (agent_golden.template_rewrite): python test_agent_golden.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_golden import template_rewrite  # noqa: E402

P = "<|turn>user\nweather?<turn|>\n<|turn>model\n"
CALL = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
RESP = "<|tool_response>response:get_weather{temp:18}<tool_response|>"
OUT = "<|channel>thought\nplan2<channel|>" + CALL
QP = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # a thinking-off ChatML prompt
at = lambda s: OUT.index(s)   # an output offset

cases = [
    # Gemma 4, thinking on: the template re-renders the reasoning close as "\n<channel|>". The model's own bare
    # <channel|> is verified (the output matched the golden, or diverged only after it): the template's edit.
    ("gemma close, output verified", P, OUT, len(OUT), P + OUT.replace("<channel|>", "\n<channel|>") + RESP, True),
    ("gemma close, verified past it", P, OUT, at("<|tool_call>"), P + OUT.replace("<channel|>", "\n<channel|>") + RESP, True),
    # The same edit after a divergence: the gate can't tell it from a rebuild that lost Loom's "\n", so no excuse.
    ("gemma close, diverged before it", P, OUT, at("plan2"), P + OUT.replace("<channel|>", "\n<channel|>") + RESP, False),
    # Reuse bugs a rebuild could hide: Loom wrote "plan2\n<channel|>" and the rebuild stripped the "\n"; Loom wrote
    # a space before <|tool_call> and the rebuild dropped it. Both edits lie past the verified output.
    ("rebuild stripped a newline", P, OUT, at("plan2") + 2, P + OUT.replace("plan2<", "plan2\n<") + RESP, False),
    ("rebuild dropped a space", P, OUT, at("plan2"), P + OUT.replace("<|tool_call>", " <|tool_call>") + RESP, False),
    # An insertion elsewhere: verified up to the call, the edit inside the call's tail is not excused, while
    # one inside the verified reasoning is.
    ("insertion past the verified part", P, OUT, at("<|tool_call>"), P + OUT.replace("<tool_call|>", "\n<tool_call|>") + RESP, False),
    ("insertion inside the verified part", P, OUT, at("<|tool_call>"), P + OUT.replace("plan2", "plan 2") + RESP, True),
    # Thinking off: the generation prompt's empty think block, which the history drops, is in the previous prompt
    # (before anything the model generated): the template's, verified or not.
    ("dropped empty think block", QP, "Sunny.", 0,
     "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nSunny.<|im_end|>\n",
     True),
    # An exact extension has no edit to excuse.
    ("exact extension", P, OUT, len(OUT), P + OUT + RESP, False),
    # An edit exactly at the divergence offset is past the verified output.
    ("gemma close, at the divergence offset", P, OUT, at("<channel|>"), P + OUT.replace("<channel|>", "\n<channel|>") + RESP, False),
    # Edits that start in the verified part (or the prompt) but whose replaced span runs into unverified output:
    # "plan2" -> "pQRS9" with the output verified up to "pl" (difflib: "lan2" -> "QRS9" from the last verified char),
    # and a dropped think block that difflib merges with Loom's first character into one replace from the prompt.
    ("span runs past the verified part", P, OUT, at("plan2") + 2, P + OUT.replace("plan2", "pQRS9") + RESP, False),
    ("span from the prompt into the output", QP, "Sunny.", 0,
     "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nFunny.<|im_end|>\n", False),
]
bad = 0
for name, prev_prompt, out, verified, prompt, want in cases:
    got = template_rewrite(prev_prompt, out, verified, prompt)
    bad += got != want
    print(("ok  " if got == want else "FAIL"), name, "->", got)

# Edits past the verified output, anchored on llama.cpp's own continuation at the edit (mocked here: what llama.cpp
# would write after the previous prompt + the output up to offset k). The first token of the text after an edit is
# its context (FIRST: "<channel|>" or "<tool_call|>" as one token).
FIRST = lambda t: next((t[:len(s)] for s in ("<channel|>", "<tool_call|>", "<|tool_call>") if t.startswith(s)), t[:1])
reference = lambda text: (lambda k, n: text[k:k + n])   # llama.cpp writes TEXT (the same output) from offset k on
GEMMA = P + OUT.replace("<channel|>", "\n<channel|>") + RESP
anchored = [
    # Diverged before the close; llama.cpp, continuing Loom's reasoning, closes it with a bare <channel|> as Loom did:
    # the "\n" is the template's.
    ("anchored accept: bare close", OUT, at("plan2"), GEMMA, reference(OUT), True),
    # Loom's rebuild says bare <channel|>, but llama.cpp at that point writes "\n<channel|>" (as Loom presumably did and
    # the rebuild stripped): not excused.
    ("anchored reject: reference writes the replacement", OUT, at("plan2"), GEMMA,
     reference(OUT.replace("<channel|>", "\n<channel|>")), False),
    # An unrelated insertion (a "\n" before the call's close) where llama.cpp writes something else there entirely.
    # The reference's text starts with both spellings (it writes "\n\n<channel|>": the template's single "\n" is
    # a prefix of it, and so is Loom's): it doesn't tell them apart, so no excuse.
    ("anchored reject: both spellings are prefixes", OUT.replace("<channel|>", "\n\n<channel|>"), at("plan2"),
     P + OUT.replace("<channel|>", "\n<channel|>") + RESP,
     reference(OUT.replace("<channel|>", "\n\n<channel|>")), False),
    ("anchored reject: unrelated insertion", OUT, at("plan2"), P + OUT.replace("<tool_call|>", "\n<tool_call|>") + RESP,
     lambda k, n: "Let me think again."[:n], False),
]
# gpt-oss (harmony): the model writes a call as "<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>", the template
# re-renders it as "<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>": difflib splits that one rewrite into
# several small edits around the unchanged "to=functions.NAME"; each is anchored on llama.cpp's continuation as ONE edit.
HP = "<|start|>user<|message|>weather?<|end|><|start|>assistant"
HCALL = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>"
HOUT = "<|channel|>analysis<|message|>Need the tool.<|end|><|start|>assistant" + HCALL
HNEXT = (HP + "<|channel|>analysis<|message|>Need the tool.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>"
         "{\"city\":\"Paris\"}<|call|><|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>{}<|end|>")
hat = HOUT.index("<|channel|>commentary")
atoken = lambda text: (lambda k, n: text[k:k + n] if k == len(text) or text[k:k + 2] == "<|" else "?" * n)   # a mid-word cut: no usable continuation
for name, verified, cont, want in [
    ("harmony call spelling, edits cut inside words (merged into one span)", 10, atoken(HOUT), True),
    ("harmony call spelling, diverged before it, llama.cpp writes the model's spelling", 10, reference(HOUT), True),
    ("harmony call spelling, llama.cpp writes the template's spelling", 10, reference(HOUT.replace(HCALL, HNEXT[HNEXT.index(" to=functions.get_weather<"):HNEXT.index("<|call|>")] + "<|call|>")), False),
    ("harmony call spelling, verified past it", len(HOUT), None, True),
]:
    got = template_rewrite(HP, HOUT, verified, HNEXT, cont, lambda t: t[:1])
    bad += got != want
    print(("ok  " if got == want else "FAIL"), name, "->", got)
for name, out, verified, prompt, cont, want in anchored:
    got = template_rewrite(P, out, verified, prompt, cont, FIRST)
    bad += got != want
    print(("ok  " if got == want else "FAIL"), name, "->", got)

# whitespace_ties: the whitespace the parsed reply cannot show (between calls, before the stop token), anchored on llama.cpp's
# own top list there (mocked: TOPS maps the text before the position to {token text: logprob}).
from agent_golden import whitespace_ties  # noqa: E402

VOCAB = ["\n\n", "\n", "<tool_call>", "</tool_call>", "<|im_end|>"]


class FakeP:
    chat = {"call_marker": "<tool_call>", "terminal": "<|im_end|>"}

    class tok:
        decode = staticmethod(lambda ids, skip_special_tokens=False: "".join(VOCAB[i] if i < len(VOCAB) else chr(i) for i in ids))

    @staticmethod
    def ids(text):
        out, i = [], 0
        while i < len(text):
            for k, t in enumerate(VOCAB):
                if text.startswith(t, i):
                    out.append(k); i += len(t); break
            else:
                out.append(ord(text[i])); i += 1
        return out


RAW = "think\n</think>\n\n<tool_call>A</tool_call>\n<tool_call>B</tool_call>"
tie_cases = [
    # the model is torn between "\n" (the template's) and "\n\n" after the first call: a tie, so the gate can excuse a reuse miss
    ("tie between calls", {"think\n</think>\n\n<tool_call>A</tool_call>": {1: -0.7, 0: -0.93}, RAW: {4: 0.0}}, True),
    # the model is sure of "\n" (the other whitespace 5 nats down): no excuse, a reuse miss is a bug
    ("sure of the template's whitespace", {"think\n</think>\n\n<tool_call>A</tool_call>": {1: -0.01, 0: -5.0}, RAW: {4: 0.0}}, False),
    # after the last call: "\n" before the stop within a nat of the stop token
    ("tie before the stop", {"think\n</think>\n\n<tool_call>A</tool_call>": {1: 0.0}, RAW: {4: -0.3, 1: -0.8}}, True),
    ("sure of the stop", {"think\n</think>\n\n<tool_call>A</tool_call>": {1: 0.0}, RAW: {4: 0.0, 1: -6.0}}, False),
    # the template's token is not in llama.cpp's list at all: nothing to compare
    ("template token unlisted", {"think\n</think>\n\n<tool_call>A</tool_call>": {0: -0.1, 2: -0.2}, RAW: {4: 0.0}}, False),
]
for name, tops, want in tie_cases:
    fn = lambda ids, t=tops: {FakeP.ids(k)[0] if False else None: 0 for k in ()} or t.get(FakeP.tok.decode(ids), {})
    got = bool(whitespace_ties(FakeP, "x", [], RAW, True, tops_at=fn))
    bad += got != want
    print(("ok  " if got == want else "FAIL"), name, "->", got)
sys.exit(1 if bad else 0)
