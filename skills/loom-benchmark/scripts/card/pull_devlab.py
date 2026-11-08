#!/usr/bin/env python3
"""DevLab Dispatch nightly-gates results for one card -> a make_info.py evidence file.

usage: pull_devlab.py loom-fun/CARD_REPO [--run ID] [--package DIR] [--download-dir DIR] [--hrx-loom FILE.json]
                      -o devlab.json

Finds the latest nightly-gates.yml run (or --run ID) of loom-fun/loom-skills with a finished job for the card,
downloads that run's artifacts (gh run download; nightly_gates.sh writes summary.json and gates.json per card
and level) and writes the gates: every gate of every finished level job (dev, quick, full) with its verdict, level and
time, the job's URL as the evidence. A job's performance files (an older PERF=1 run's Loom-only perf.jsonl) are not
read: a card's DevLab perf session measures both engines (devlab-perf, loom-org SKILL.md).
The spec gate's drafted-token acceptance ("acceptance N% of M proposals" in the job's gates_spec.log or gates.out)
is not a timing: it depends on the model, drafter and text, not the machine, so it is carried as the card's
spec_acceptance (a session holding only that metric, stamped with the job's programs_sha256), as make_info.py gates
records it; the latest job that reports one wins.
Each source records the runner's device and Loom SDK when summary.json carries them ("device" from
machine_info.py detect_device(), "hrx_loom" from its hrx_loom_identity()); otherwise they stay null (not
recorded). --hrx-loom FILE.json supplies an SDK identity established otherwise (e.g. from the job logs).
--package DIR (the card repo's current files): a job whose fetched model.json or programs (summary.json
programs_sha256) differ from DIR's measured another revision of the package and is skipped. Programs can change with
model.json unchanged (a regenerated package), and a job started right after a publish has fetched the old files. A job that was cancelled or did not write gates.json is skipped. Needs `gh` authenticated for loom-fun/loom-skills.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_lib import EVIDENCE_ID, load_json, programs_sha256, write_json  # noqa: E402
from make_info import acceptance  # noqa: E402

GH_REPO = "loom-fun/loom-skills"
WORKFLOW = "nightly-gates.yml"


def gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def runs(limit=30):
    return json.loads(gh("run", "list", "-R", GH_REPO, "--workflow", WORKFLOW, "-L", str(limit),
                         "--json", "databaseId,createdAt,status,conclusion,url"))


def jobs(run_id):
    """The run's jobs: name, status, conclusion and url."""
    out = json.loads(gh("api", f"repos/{GH_REPO}/actions/runs/{run_id}/jobs?per_page=100"))["jobs"]
    return [{"name": j["name"], "status": j["status"], "conclusion": j["conclusion"], "url": j["html_url"]}
            for j in out]


def finished_artifacts(run_id, dest):
    """Download the run's nightly-* artifacts (only those not downloaded yet) and return their directories."""
    names = [a["name"] for a in json.loads(gh("api", f"repos/{GH_REPO}/actions/runs/{run_id}/artifacts?per_page=100"))["artifacts"]
             if a["name"].startswith("nightly-") and not a.get("expired")]
    for n in names:
        if not os.path.isdir(os.path.join(dest, n)):
            gh("run", "download", str(run_id), "-R", GH_REPO, "-n", n, "-D", os.path.join(dest, n))
    return [os.path.join(dest, n) for n in sorted(names)]


def evidence(found, rid, hrx_loom=None):
    """The evidence file of a run's finished jobs for one card: found = [(summary.json, gates.json, job dir, job URL)].
    Every gate of every job; the spec gate's acceptance (gates_spec.log, else gates.out) as a spec_acceptance-only
    session stamped with the job's programs_sha256, the latest job's that reports one."""
    gates, session = [], None
    for s, g, d, url in sorted(found, key=lambda x: x[1].get("ts") or ""):
        src = {"kind": "devlab", "run": str(rid), "url": url, "device": s.get("device"),
               "hrx_loom": s.get("hrx_loom") or hrx_loom}
        for name, x in sorted(g.get("gates", {}).items()):
            gates.append({"name": name, "status": "pass" if x.get("pass") else "fail", "level": g.get("level"),
                          "time": g.get("ts"), "source": src, "detail": None, "programs_sha256": s.get("programs_sha256")})
        acc = acceptance([os.path.join(d, "gates_spec.log"), os.path.join(d, "gates.out")]) if "spec" in g.get("gates", {}) else None
        if acc is not None:   # the latest job's (sorted by time)
            session = {"date": (g.get("ts") or "")[:10] or None, "source": src,
                       "runs": {"decode_prefill": None, "streams4": None}, "commands": [],
                       "programs_sha256": s.get("programs_sha256"), "metrics": {"spec_acceptance": {"loom": acc}}}
    ev = {"evidence": EVIDENCE_ID, "gates": gates}
    if session:
        ev["performance"] = {"session": session}
    return ev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--run", help="this nightly-gates run instead of the latest one with a finished job for the card")
    ap.add_argument("--download-dir")
    ap.add_argument("--hrx-loom", help="SDK identity JSON for jobs whose summary.json records none")
    ap.add_argument("--package", help="the card's current files: skip jobs that fetched another model.json")
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
            mj = os.path.join(d, "model.json")
            if a.package and ((os.path.exists(mj) and load_json(mj) != load_json(os.path.join(a.package, "model.json")))
                              or s.get("programs_sha256") not in (None, programs_sha256(a.package))):
                print(f"skipped {os.path.basename(d)}: it measured another revision of the package", file=sys.stderr)
                continue
            found.append((s, json.load(open(gp)), d, job_url[(s["slug"], s["level"])]))
        if found:
            break
    else:
        sys.exit(f"no finished nightly-gates job for {a.repo}" + (f" in run {a.run}" if a.run else ""))

    ev = evidence(found, rid, load_json(a.hrx_loom) if a.hrx_loom else None)
    gates, session = ev["gates"], (ev.get("performance") or {}).get("session")
    write_json(a.o, ev)
    print(f"wrote {a.o}: run {rid}, " + ", ".join(f"{x['level']}:{x['name']}={x['status']}" for x in gates)
          + (f", spec_acceptance={session['metrics']['spec_acceptance']['loom']}" if session else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
