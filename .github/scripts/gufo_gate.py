#!/usr/bin/env python3
"""The `gufo` gate of one package from run_conformance.sh's files (devlab_perf.sh step gufo:V; loom-skills#736).

usage: gufo_gate.py OUT_DIR GATE.json RC     OUT_DIR holds conformance.json (conformance.py --logits) and logits_check.json (gufo's own).
Writes GATE.json, a gates.json make_info.py gates ingests: gate `gufo` passes when run_conformance.sh exited 0, no conformance check
failed or is a gap (with --logits a gap is a miss), and every logits check passed. Prints one result line."""
import datetime
import json
import os
import sys


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def gate(conf, lc, rc):
    """(pass, detail) from the two result files (None = missing) and run_conformance.sh's exit status."""
    if conf is None or lc is None:
        return False, "conformance run produced no result files (the server did not load the package with logits, or a check script died)"
    res = conf["results"]
    n = {k: sum(r["status"] == k for r in res) for k in ("pass", "fail", "gap")}
    lres = lc["results"]
    lpass = sum(r["status"] == "pass" for r in lres)
    ok = rc == 0 and n["fail"] == 0 and n["gap"] == 0 and lpass == len(lres) and bool(res) and bool(lres)
    if ok:
        return True, f"{n['pass']}/{len(res)} conformance checks, {lpass}/{len(lres)} logits checks"
    bad = [r["check"] for r in res if r["status"] != "pass"] + [r["check"] for r in lres if r["status"] != "pass"]
    return False, f"{n['pass']}/{len(res)} conformance ({n['fail']} fail, {n['gap']} gap), {lpass}/{len(lres)} logits; first: {', '.join(bad[:3])}"[:200]


def main(argv):
    out_dir, dst, rc = argv[1], argv[2], int(argv[3])
    ok, detail = gate(load(os.path.join(out_dir, "conformance.json")), load(os.path.join(out_dir, "logits_check.json")), rc)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    with open(dst, "w") as f:
        json.dump({"verdict": "PASS" if ok else "FAIL", "level": "dev", "ts": now, "gates": {"gufo": {"pass": ok, "detail": detail}}}, f, indent=1)
    print(f"GUFO {'PASS' if ok else 'FAIL'} {detail}")


if __name__ == "__main__":
    main(sys.argv)
