"""Step-list presets of a DevLab job (ab_jobs.py, device_jobs.py --preset): what an experiment job runs at each stage.

  full     (the default of the generators: a card regeneration needs every step; regen.py always uses it)
           func, ktest, [gates], perf rows, c4, [baselines]: the lists the generators always wrote.
  iterate  the loop of an optimization: build + ktest + perf + c4 (+ the A/B text match), NO gates, and no func beyond `smoke:V`
           (one served chat request). Measured on a typical 95-minute job (skills/loom-fleet/reference/campaign.md, DevLab): gates 72 %,
           func 16 %, perf 10 %, setup + builds 3 %, ktest 1 %.
  final    the winner only: func + ktest + gates:LEVEL (default dev) once, on the new variant; no perf (iterate measured it), and the
           unchanged base variant is not even built (a base that is already gated is never gated again).
"""
PRESETS = ("full", "iterate", "final")


def apply(preset, steps, new, gates="dev"):
    """The step list of PRESET from the FULL list STEPS; NEW is the variant the change is in ("staging" for an A/B, "card" for a
    per-model job); GATES the level of the final preset's gates step."""
    if preset == "full":
        return list(steps)
    if preset == "iterate":
        out = []
        for s in steps:
            if s == f"func:{new}":
                out.append(f"smoke:{new}")
            elif s.startswith("gates:") or s.startswith("vllm") or s.startswith("llama"):
                continue   # gates and the baseline engines are not part of an iteration (a baseline is measured once, on the final job)
            else:
                out.append(s)
        return out
    if preset == "final":
        return [f"func:{new}", f"ktest:{new}", f"gates:{new}:{gates or 'dev'}"]
    raise ValueError(f"unknown preset {preset!r}: {', '.join(PRESETS)}")


def trim_variants(job):
    """Drop the variants no step uses (the final preset serves only the new variant: the unchanged base is not built)."""
    used = {s.split(":")[1] for s in job["steps"] if s.count(":") >= 1}
    job["variants"] = [v for v in job["variants"] if v["name"] in used]
    return job
