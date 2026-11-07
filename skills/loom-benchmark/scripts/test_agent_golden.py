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
    ("anchored reject: unrelated insertion", OUT, at("plan2"), P + OUT.replace("<tool_call|>", "\n<tool_call|>") + RESP,
     lambda k, n: "Let me think again."[:n], False),
]
for name, out, verified, prompt, cont, want in anchored:
    got = template_rewrite(P, out, verified, prompt, cont, FIRST)
    bad += got != want
    print(("ok  " if got == want else "FAIL"), name, "->", got)
sys.exit(1 if bad else 0)
