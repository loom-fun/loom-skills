#!/usr/bin/env python3
"""The drafting choice of a card, one per model and identical on Loom and llama.cpp (final benchmark rule).

The choice comes from skills/loom-benchmark/data/drafting_classification.json (work/baselines/CLASSIFICATION.md has the provenance):
MTP head in the card's GGUF > DFlash/assistant drafter published for this exact model (maker or Z-Lab) > no drafting. Never n-gram,
never a drafter trained for another model, never "whatever is faster". Loom keeps its tuned depth; llama.cpp uses the drafter's
upstream-recommended n-max, else its own default. streams4 is loaded without drafting on both engines.

  matched_spec.py LOOM-FUN/REPO       the llama.cpp server arguments ("{drafter}" = the card's drafter GGUF; empty: no drafting)
  matched_spec.py LOOM-FUN/REPO --choice    mtp | dflash | assist | none
  matched_spec.py LLM_CONFIG.json --violations   what the package's own drafting differs in from the classification (needs --repo)

The normative text is skills/loom-benchmark/reference/methodology.md; this module holds its constants and checks (TIE_BAND, outcome,
tally, rule_violations) so the renderer, the audit and the tests share one copy.
"""
import json
import os
import re
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


def rule_violations(repo, cfg=None, llama_args=None, row_loads=None):
    """Every way a card's setup breaks the drafting rule (methodology.md rules 1, 3, 4); [] = compliant.
    cfg: the package's llm_config.json; llama_args: {"rocm": "...", "vulkan": "..."} the 1-stream server arguments;
    row_loads: {row: {"loom": bool, "llama": bool}} whether each engine was loaded drafting for that row."""
    out = violations(repo, cfg) if cfg is not None else []
    want = classification(repo)["choice"]
    spec = classification(repo)["llama_spec"]
    stype = (re.search(r"--spec-type (\S+)", spec) or [None, None])[1]
    for bk, args in (llama_args or {}).items():
        have = (re.search(r"--spec-type (\S+)", args or "") or [None, None])[1]
        if have and "ngram" in have:
            out.append(f"llama.cpp {bk} drafts with n-gram (never on a card)")
        elif have != stype:
            out.append(f"llama.cpp {bk} spec type {have or 'none'}, the classification says {stype or 'none'}")
    for row, load in (row_loads or {}).items():
        for eng, on in load.items():
            if on != row_states(repo)[row]:
                out.append(f"{eng} loaded {'with' if on else 'without'} drafting for {row} (the rule says {'with' if row_states(repo)[row] else 'without'}: no per-row toggles)")
    return out


# Outcome of a row against the better llama.cpp backend (methodology.md rule 7); TIE_BAND is the measured same-runner noise.
TIE_BAND = 0.03


def outcome(loom, llama):
    """'win' (> +TIE_BAND), 'tie' (within +-TIE_BAND) or 'loss' (< -TIE_BAND) of Loom against the better of the llama.cpp
    backends (`llama`: a number or an iterable of numbers); None when either side is missing (not supported / unmeasured: neutral)."""
    vals = [v for v in ([llama] if isinstance(llama, (int, float)) else llama or []) if v]
    if not loom or not vals:
        return None
    r = round(loom / max(vals) - 1, 9)
    return "win" if r > TIE_BAND else "loss" if r < -TIE_BAND else "tie"


def tally(outcomes):
    """The card/report counts over the rows' outcomes (None rows are neutral): wins every row, wins-or-ties every row, loses >= 1 row."""
    o = [x for x in outcomes if x]
    return {"wins_every_row": bool(o) and all(x == "win" for x in o), "wins_or_ties_every_row": bool(o) and "loss" not in o,
            "loses_a_row": "loss" in o, **{k: o.count(k) for k in ("win", "tie", "loss")}}


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
