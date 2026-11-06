#!/usr/bin/env python3
"""DevLab Dispatch nightly-gates results for one card -> a make_info.py evidence file.

usage: pull_devlab.py loom-fun/CARD_REPO [--run ID] [--download-dir DIR] -o devlab.json

Finds the latest nightly-gates.yml run (or --run ID) of loom-fun/loom-skills with a finished job for the card,
downloads that run's artifacts (gh run download; nightly_gates.sh writes summary.json and gates.json per card
and level) and writes gates: every gate of every finished level job (dev, quick, full) with its verdict, level
and time, the job's URL as the evidence.
Gates only: the full job's report-only perf.jsonl is not card data. DevLab runners measure differently from a
local same-session run (Flint, run 42259576436: 1k prefill 515 vs 848 tok/s on the same build), so their
numbers are not comparable with the llama.cpp column.
A job that was cancelled or did not write gates.json is skipped. Needs `gh` authenticated for loom-fun/loom-skills.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_lib import EVIDENCE_ID, write_json  # noqa: E402

GH_REPO = "loom-fun/loom-skills"
WORKFLOW = "nightly-gates.yml"


def gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def runs(limit=30):
    return json.loads(gh("run", "list", "-R", GH_REPO, "--workflow", WORKFLOW, "-L", str(limit),
                         "--json", "databaseId,createdAt,status,conclusion,url"))


def jobs(run_id):
    return json.loads(gh("run", "view", str(run_id), "-R", GH_REPO, "--json", "jobs"))["jobs"]


def finished_artifacts(run_id, dest):
    """Download the run's nightly-* artifacts (only those not downloaded yet) and return their directories."""
    names = [a["name"] for a in json.loads(gh("api", f"repos/{GH_REPO}/actions/runs/{run_id}/artifacts?per_page=100"))["artifacts"]
             if a["name"].startswith("nightly-") and not a.get("expired")]
    for n in names:
        if not os.path.isdir(os.path.join(dest, n)):
            gh("run", "download", str(run_id), "-R", GH_REPO, "-n", n, "-D", os.path.join(dest, n))
    return [os.path.join(dest, n) for n in sorted(names)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--run", help="this nightly-gates run instead of the latest one with a finished job for the card")
    ap.add_argument("--download-dir")
    ap.add_argument("-o", required=True)
    a = ap.parse_args(argv)
    candidates = [{"databaseId": int(a.run)}] if a.run else runs()
    dl_root = a.download_dir or tempfile.mkdtemp(prefix="devlab-")
    for r in candidates:
        rid = r["databaseId"]
        job_url = {}
        for j in jobs(rid):
            if j["name"].startswith("gates (") and j.get("status") == "completed" and j.get("conclusion") in ("success", "failure"):
                slug, level = j["name"][len("gates ("):-1].split(", ")
                job_url[(slug, level)] = j["url"]
        if not job_url:
            continue
        found = []
        for d in finished_artifacts(rid, os.path.join(dl_root, str(rid))):
            sp, gp = os.path.join(d, "summary.json"), os.path.join(d, "gates.json")
            if not (os.path.exists(sp) and os.path.exists(gp)):
                continue
            s = json.load(open(sp))
            if s.get("repo") != a.repo or (s.get("slug"), s.get("level")) not in job_url:
                continue
            found.append((s, json.load(open(gp)), d, job_url[(s["slug"], s["level"])]))
        if found:
            break
    else:
        sys.exit(f"no finished nightly-gates job for {a.repo}" + (f" in run {a.run}" if a.run else ""))

    gates = []
    for s, g, d, url in sorted(found, key=lambda x: x[1].get("ts") or ""):
        src = {"kind": "devlab", "run": str(rid), "url": url}
        for name, x in sorted(g.get("gates", {}).items()):
            gates.append({"name": name, "status": "pass" if x.get("pass") else "fail", "level": g.get("level"),
                          "time": g.get("ts"), "source": src, "detail": None})
    write_json(a.o, {"evidence": EVIDENCE_ID, "gates": gates})
    print(f"wrote {a.o}: run {rid}, " + ", ".join(f"{x['level']}:{x['name']}={x['status']}" for x in gates),
          file=sys.stderr)


if __name__ == "__main__":
    main()
