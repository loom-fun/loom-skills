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


# THE per-row drafting table: (context tokens, concurrent streams) of every timed row. A row's drafting state is ONE value
# applied identically to Loom and to llama.cpp (the final rule on which gates count is the coordinator's; today it mirrors
# Loom's shipped gating: llm_config.json drafter.ctx_max (drafting off above that context, 0 = no limit) and
# drafter.occ_min (off at >= that many streams, 0 = no limit)).
ROWS = {"decode": (0, 1), "decode_2k": (2000, 1), "decode_32k": (32000, 1), "streams4": (600, 4),
        "prefill_1k": (1000, 1), "prefill_2k": (2000, 1), "prefill_4k": (4000, 1), "prefill_8k": (8000, 1),
        "prefill_16k": (16000, 1), "prefill_32k": (32000, 1)}


def row_states(cfg, honor_gates=True):
    """{row: True|False}: whether drafting is on for that row, for BOTH engines. No draft: all False."""
    on = bool(matched_spec(cfg))
    g = cfg.get("drafter") or {}
    cmax, omin = int(g.get("ctx_max") or 0), int(g.get("occ_min") or 0)
    out = {}
    for row, (ctx, streams) in ROWS.items():
        ok = on
        if on and honor_gates:
            ok = not ((cmax and ctx > cmax) or (omin and streams >= omin))
        out[row] = ok
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
    else:
        print(matched_spec(c))
