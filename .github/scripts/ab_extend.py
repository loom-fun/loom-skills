#!/usr/bin/env python3
"""The sequential extension of an A/B job (methodology.json "sequential": start at 2+2 sessions, extend to 6+6 near a reject threshold).

usage: ab_extend.py JOB_OUT_DIR --to 6 --have 2 --rot 0

Called by devlab_perf.sh at the step `extend:TO:HAVE:ROT` after the first HAVE sessions per side have run. Reads the sessions measured so far
(perf_fun.jsonl / perf_staging.jsonl, c4_funnd.txt / c4_stagingnd.txt), computes each row's new/old ratio exactly as strix_ab_compare.py does
(median of the per-session medians), and prints on stdout the steps to run next, one per line, in the ABBA order the first sessions used
(sessions HAVE..TO-1, the same rotation ROT as ab_jobs.py):
  perf:V:1      for short rows (decode, decode_2k, prefill_1k, prefill_2k)      when one of them is within `extend_within` (0.02) of its reject threshold
  perf:V:1:long for long rows (decode_32k, prefill_8k/16k/32k)                   the same
  c4:Vnd        for streams4
A row whose ratio is clear of its threshold (either side) adds nothing: no third run is queued for it. stderr: the ratios and the reason.
"""
import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
sys.path.insert(0, os.path.join(ROOT, "fleet"))
from make_info import loom_sessions, read_rows  # noqa: E402
import sample_lib as L  # noqa: E402
from card_lib import registry  # noqa: E402


def abba(k, start=0):
    out = []
    for i in range(k):
        out += ["fun", "staging"] if (i + start) % 2 == 0 else ["staging", "fun"]
    return out


def ratios(out_dir):
    """{row: new/old} from the sessions in OUT_DIR (a row missing on a side is absent)"""
    def side(v, nd):
        p, c4 = os.path.join(out_dir, f"perf_{v}.jsonl"), os.path.join(out_dir, f"c4_{nd}.txt")
        return loom_sessions(read_rows(p) if os.path.exists(p) else [], c4 if os.path.exists(c4) else None)[0]
    old, new = side("fun", "funnd"), side("staging", "stagingnd")
    out = {}
    for k in L.ROWS:
        if old.get(k) and new.get(k):
            out[k] = statistics.median(new[k]) / statistics.median(old[k])
    return out


def groups_to_extend(rat, meth=None, within=None):
    """-> {'short'|'long'|'c4': [rows near their reject threshold]}"""
    meth = meth or L.load_methodology()
    within = within if within is not None else registry.methodology()["sequential"]["extend_within"]
    group = {**{r: "short" for r in L.SHORT_ROWS}, **{r: "long" for r in L.LONG_ROWS}, "streams4": "c4"}
    near = {}
    for r, ratio in rat.items():
        band, reject = meth["bands"][r]
        if abs(ratio - reject) <= within + 1e-9:
            near.setdefault(group[r], []).append(r)
    return near


def steps(near, to, have, rot):
    out = []
    for g, start in (("short", rot), ("long", 1 + rot), ("c4", rot)):
        if g not in near:
            continue
        for v in abba(to, start)[2 * have:]:
            out.append({"short": f"perf:{v}:1", "long": f"perf:{v}:1:long", "c4": f"c4:{v}nd"}[g])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--to", type=int, required=True)
    ap.add_argument("--have", type=int, required=True)
    ap.add_argument("--rot", type=int, default=0)
    a = ap.parse_args(argv)
    rat = ratios(a.out)
    near = groups_to_extend(rat)
    print("ratios: " + ", ".join(f"{r} {v:.3f}" for r, v in rat.items()) + f"; within 0.02 of the reject threshold: {near or 'none: no extension'}", file=sys.stderr)
    for s in steps(near, a.to, a.have, a.rot):
        print(s)


if __name__ == "__main__":
    main()
