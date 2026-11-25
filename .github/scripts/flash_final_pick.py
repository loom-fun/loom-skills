"""Select the measured configuration for this one Strix qualification job."""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skills/loom-benchmark/scripts/card"))
from make_info import loom_sessions, perf_rows

VARIANTS = ("base", "p8k", "a8k", "b8k", "ab8k")
PREFILL = ("prefill_1k", "prefill_2k", "prefill_8k", "prefill_16k", "prefill_32k")
PRODUCTION = {"decode": 21.0, "decode_2k": 34.8, "decode_32k": 31.7, "streams4": 66.6}

def select(rows):
    base = rows["base"]
    eligible = [
        name for name in VARIANTS if all(
            rows[name][key] >= 0.97 * max(base[key], floor) for key, floor in PRODUCTION.items()
        ) and all(rows[name][key] >= 0.97 * base[key] for key in PREFILL)
    ]
    if not eligible:
        return None
    leaders = {key: max(rows[name][key] for name in eligible) for key in PREFILL}
    near_leaders = [name for name in eligible if all(rows[name][key] >= 0.97 * leaders[key] for key in PREFILL)]
    if near_leaders:
        # Keep the simplest measured configuration within noise of the best at every size.
        return near_leaders[0]
    # No variant wins every size. Keep the measured chunk improvement or the control.
    return next((name for name in ("p8k", "base") if name in eligible), None)

if __name__ == "__main__":
    out = Path(sys.argv[1])
    rows = {}
    for name in VARIANTS:
        vals, _, _ = loom_sessions(perf_rows(out / f"perf_{name}.jsonl"), out / f"c4_{name}nd.txt")
        for key in (*PREFILL, *PRODUCTION):
            if len(vals.get(key, [])) < 2:
                sys.exit(f"{name} has fewer than two measured sessions for {key}")
        rows[name] = {key: statistics.median(vals[key]) for key in (*PREFILL, *PRODUCTION)}
    winner = select(rows)
    (out / "selection.json").write_text(json.dumps({"rows": rows, "winner": winner}, indent=2) + "\n")
    if winner is None:
        sys.exit("No measured configuration clears the regression limits")
    print(winner)
