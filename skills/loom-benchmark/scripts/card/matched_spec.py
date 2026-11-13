#!/usr/bin/env python3
"""The drafting choice of a card, one per model and identical on Loom and llama.cpp (final benchmark rule).

The choice comes from skills/loom-benchmark/data/drafting_classification.json (work/baselines/CLASSIFICATION.md has the provenance):
MTP head in the card's GGUF > DFlash/assistant drafter published for this exact model (maker or Z-Lab) > no drafting. Never n-gram,
never a drafter trained for another model, never "whatever is faster". Loom keeps its tuned depth; llama.cpp uses the drafter's
upstream-recommended n-max, else its own default. streams4 is loaded without drafting on both engines.

  matched_spec.py LOOM-FUN/REPO       the llama.cpp server arguments ("{drafter}" = the card's drafter GGUF; empty: no drafting)
  matched_spec.py LOOM-FUN/REPO --choice    mtp | dflash | assist | none
  matched_spec.py LLM_CONFIG.json --violations   what the package's own drafting differs in from the classification (needs --repo)
"""
import json
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "drafting_classification.json")


def classification(repo):
    """The card's classification entry (choice, llama_spec, provenance...); KeyError when the repo is not classified."""
    return json.load(open(DATA))[repo]


def matched_spec(repo):
    """llama.cpp server arguments for the card's drafting choice ("{drafter}" a placeholder), "" for no drafting."""
    return classification(repo)["llama_spec"]


# THE per-row drafting table (final rule): the 1-stream rows (decode, decode_2k, decode_32k, prefill_*) load each engine once with
# the card's drafting choice; streams4 is loaded without drafting on every engine (the one allowed load-time exception).
ROWS = {"decode": 1, "decode_2k": 1, "decode_32k": 1, "prefill_1k": 1, "prefill_2k": 1, "prefill_4k": 1, "prefill_8k": 1,
        "prefill_16k": 1, "prefill_32k": 1, "streams4": 4}   # row -> concurrent streams


def row_states(repo):
    """{row: True|False}: whether drafting is loaded for that row, for BOTH engines."""
    on = classification(repo)["choice"] != "none"
    return {row: on and streams == 1 for row, streams in ROWS.items()}


def violations(repo, cfg):
    """How the package's own drafting (llm_config.json draft) differs from the classification (a rebuild is needed)."""
    want = classification(repo)["choice"]
    have = (cfg.get("draft") or {}).get("kind") or "none"
    return [] if want == have else [f"drafts with {have}, the classification says {want}"]


def drafter_file(cfg):
    """The drafter / assistant GGUF file name the package ships (None: none, e.g. MTP inside the main GGUF or n-gram)."""
    for k in ("drafter", "assistant"):
        if isinstance(cfg.get(k), dict) and cfg[k].get("gguf"):
            return cfg[k]["gguf"]
    return (cfg.get("draft") or {}).get("gguf")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if "--choice" in sys.argv:
        print(classification(a[0])["choice"])
    elif "--violations" in sys.argv:
        print("; ".join(violations(sys.argv[sys.argv.index("--repo") + 1], json.load(open(a[0])))))
    elif "--rows" in sys.argv:
        print(json.dumps(row_states(a[0])))
    else:
        print(matched_spec(a[0]))
