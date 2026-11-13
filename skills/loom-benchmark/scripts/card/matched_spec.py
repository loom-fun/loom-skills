#!/usr/bin/env python3
"""The llama.cpp speculative-decoding arguments that match a Loom package's own drafting (apples to apples).

A card's llama.cpp baselines (ROCm and Vulkan) run in the same drafting configuration as the published Loom package, for
every row: the package drafts (llm_config.json "draft": kind + depth) -> llama.cpp drafts with the equivalent method, the
same drafter GGUF where it has one, n-max = Loom's depth; the package does not draft -> llama.cpp does not either.
No best-of-several-configurations per metric.

  matched_spec.py LLM_CONFIG.json     prints the arguments ("{drafter}" = the package's drafter / assistant GGUF; empty: no drafting)

kind mapping: dflash -> draft-dflash + the drafter GGUF; mtp -> draft-mtp (the MTP block inside the GGUF);
assist -> draft-mtp + the assistant GGUF (gemma's assistant drafter); ngram -> ngram-simple.
"""
import json
import sys

SPEC_TYPE = {"dflash": "draft-dflash", "mtp": "draft-mtp", "assist": "draft-mtp", "ngram": "ngram-simple"}
NEEDS_DRAFTER = ("dflash", "assist")   # llama.cpp loads a separate draft GGUF for these


def matched_spec(cfg):
    """llama.cpp server arguments (a string, "{drafter}" a placeholder) for the package's drafting, "" when it does not draft."""
    d = cfg.get("draft") or {}
    kind, depth = d.get("kind"), int(d.get("depth") or 0)
    if not kind or kind == "none" or depth < 1:
        return ""
    if kind not in SPEC_TYPE:
        raise ValueError(f"unknown draft kind {kind!r}")
    a = f"--spec-type {SPEC_TYPE[kind]}"
    if kind in NEEDS_DRAFTER:
        a += " --spec-draft-model {drafter}"
    a += f" --spec-draft-n-max {depth}"
    if kind in NEEDS_DRAFTER:
        a += " -ngld 99"
    return a


# THE per-row drafting table (benchmark rule, Nov 13), ONE value per row applied identically to Loom, llama.cpp ROCm and
# llama.cpp Vulkan: every 1-stream row (decode, decode_2k, decode_32k, prefill_*) drafts; streams4 does not.
ROWS = {"decode": 1, "decode_2k": 1, "decode_32k": 1, "prefill_1k": 1, "prefill_2k": 1, "prefill_4k": 1, "prefill_8k": 1,
        "prefill_16k": 1, "prefill_32k": 1, "streams4": 4}   # row -> concurrent streams


def row_states(cfg):
    """{row: True|False}: whether drafting is on for that row, for BOTH engines. A card with no draft is a rule violation
    (rule_violations), not a table of False."""
    on = bool(matched_spec(cfg))
    return {row: on and streams == 1 for row, streams in ROWS.items()}


def rule_violations(cfg):
    """Why the package as shipped does not follow the rule (empty: it does): it must draft and gate drafting off at
    4 streams (drafter.occ_min between 1 and 4, default 2). The context gate (ctx_max) is runtime-adaptive and allowed."""
    out = []
    if not matched_spec(cfg):
        out.append("no drafting")
    g = cfg.get("drafter") or {}
    if "occ_min" in g and not 1 <= int(g.get("occ_min") or 0) <= 4:
        out.append(f"occ_min {g.get('occ_min')}: drafting is not gated off at 4 streams")
    return out


def drafter_file(cfg):
    """The drafter / assistant GGUF file name the package ships (None: none, e.g. MTP inside the main GGUF or n-gram)."""
    for k in ("drafter", "assistant"):
        if isinstance(cfg.get(k), dict) and cfg[k].get("gguf"):
            return cfg[k]["gguf"]
    return (cfg.get("draft") or {}).get("gguf")


if __name__ == "__main__":
    c = json.load(open(sys.argv[1]))
    if "--rows" in sys.argv:
        print(json.dumps(row_states(c)))
    elif "--violations" in sys.argv:
        print("; ".join(rule_violations(c)))
    else:
        print(matched_spec(c))
