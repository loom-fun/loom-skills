"""Shared definitions for the loom-fun card generator (schema loom-card/1).

make_info.py writes info.json, render_card.py renders README.md from it alone. Both are pure functions of
their inputs: no clock, no randomness, sorted JSON keys. The fixed row sets below are the card template:
every card shows the same performance rows, feature rows and gate rows, in this order.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_ID = "loom-card/1"
EVIDENCE_ID = "loom-card-evidence/1"
SCHEMA_PATH = os.path.join(HERE, "info_schema.json")

# (key, label): the performance table's rows.
METRICS = [
    ("decode", "decode tok/s, short prompt"),
    ("decode_2k", "decode tok/s, 2k context"),
    ("decode_32k", "decode tok/s, 32k context"),
    ("prefill_1k", "prefill tok/s, 1k document"),
    ("prefill_2k", "prefill tok/s, 2k document"),
    ("prefill_8k", "prefill tok/s, 8k document"),
    ("prefill_16k", "prefill tok/s, 16k document"),
    ("prefill_32k", "prefill tok/s, 32k document"),
    ("streams4", "4 streams, aggregate decode tok/s"),
]
METRIC_KEYS = [k for k, _ in METRICS]

# (key, label): the features table's rows.
FEATURES = [
    ("api", "API"),
    ("tool_calling", "tool calling"),
    ("thinking", "reasoning / thinking"),
    ("reasoning_effort", "reasoning effort"),
    ("speculative_decoding", "speculative decoding"),
    ("image_input", "image input"),
    ("audio", "audio input / output"),
    ("sampling", "sampling"),
    ("context", "context size"),
    ("concurrency", "concurrency"),
    ("max_output_tokens", "max output tokens"),
]
FEATURE_KEYS = [k for k, _ in FEATURES]

# (name, description): the correctness table's rows (loom-benchmark gates.sh gate names).
GATES = [
    ("kernel_tests", "Unit tests of the generated kernels against reference implementations"),
    ("golden", "llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents"),
    ("compare", "Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie"),
    ("vision", "Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties"),
    ("consistency", "Concurrent requests give the same output as the same requests run alone"),
    ("spec", "Speculative decoding on gives exactly the output of speculative decoding off"),
    ("agent", "Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp"),
    ("agent_think_off", "The tool-calling loop with thinking off"),
    ("sampling", "temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp"),
    ("needle", "Needle-in-a-haystack prompt at up to half the context, compared with llama.cpp; over-capacity prompt refused"),
    ("needle_full", "Needle-in-a-haystack prompt at the full context is served"),
    ("decision", "Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts"),
]
GATE_NAMES = [n for n, _ in GATES]
# speculative decoding methods (llm_config.json draft.kind) as the card names them
METHOD = {"dflash": "DFlash", "mtp": "MTP", "ngram": "n-gram prompt lookup", "assist": "assistant drafter"}

HF = "https://huggingface.co"
LOOM_URL = "https://github.com/ROCm/hrx-system"
SKILLS_URL = "https://github.com/loom-fun/loom-skills"


def dumps(obj):
    """The one JSON encoding of info.json and evidence files: sorted keys, 1-space indent, trailing newline."""
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(dumps(obj))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def quant_of(filename):
    """The quantization in a GGUF file name (Q4_K_M, UD-Q5_K_M, Q8_0, BF16, ...), or None."""
    stem = os.path.basename(filename)
    stem = stem[:-5] if stem.lower().endswith(".gguf") else stem
    found = re.findall(r"(?<![A-Za-z0-9])((?:UD-)?I?Q[0-9](?:_[0-9A-Z]+)*|BF16|F16|F32)(?![a-z0-9])", stem)
    return found[-1] if found else None


def build_tag(text):
    """A lemonade-sdk/llamacpp-rocm release tag (b1355) found in free text, or None."""
    m = re.search(r"\bb(1[0-9]{3,})\b", text or "")
    return f"b{m.group(1)}" if m else None


# ---- validation: the subset of JSON Schema info_schema.json uses (no third-party dependency) ----

def _types(t):
    return t if isinstance(t, list) else [t]


def _is(v, t):
    return {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str),
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": isinstance(v, bool), "null": v is None}[t]


def _check(v, s, root, path, errs):
    if "$ref" in s:
        s = root["$defs"][s["$ref"].split("/")[-1]]
    if "oneOf" in s:
        ok = [sub for sub in s["oneOf"] if not _check(v, sub, root, path, [])]
        if len(ok) != 1:
            errs.append(f"{path}: matches {len(ok)} of the oneOf alternatives")
        return errs
    if "const" in s and v != s["const"]:
        errs.append(f"{path}: must be {s['const']!r}")
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} not in {s['enum']}")
    if "type" in s and not any(_is(v, t) for t in _types(s["type"])):
        errs.append(f"{path}: {type(v).__name__} is not {s['type']}")
        return errs
    if isinstance(v, str) and "pattern" in s and not re.search(s["pattern"], v):
        errs.append(f"{path}: {v!r} does not match {s['pattern']}")
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                errs.append(f"{path}: missing {k}")
        props = s.get("properties", {})
        for k, x in v.items():
            if k in props:
                _check(x, props[k], root, f"{path}.{k}", errs)
            elif s.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {k}")
    if isinstance(v, list) and "items" in s:
        for i, x in enumerate(v):
            _check(x, s["items"], root, f"{path}[{i}]", errs)
    return errs


def validate(info):
    """Errors of info against info_schema.json, plus the template's own invariants (empty: valid)."""
    schema = load_json(SCHEMA_PATH)
    errs = _check(info, schema, schema, "info", [])
    if not errs:
        names = [g["name"] for g in info["gates"]]
        if names != GATE_NAMES:
            errs.append(f"info.gates: names must be {GATE_NAMES} in this order, got {names}")
        for k in info["performance"]["not_applicable"]:
            if k not in METRIC_KEYS:
                errs.append(f"info.performance.not_applicable: unknown metric {k}")
    return errs


def require_valid(info, what="info.json"):
    errs = validate(info)
    if errs:
        raise SystemExit(f"{what} is not valid {SCHEMA_ID}:\n  " + "\n  ".join(errs))
