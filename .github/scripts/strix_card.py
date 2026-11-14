#!/usr/bin/env python3
"""Put a passing Strix A/B (strix_ab_jobs.py + strix_ab_compare.py) on a loom-fun-staging card: the loom-fun card's published Strix table
(Loom + llama.cpp, as measured), one line saying the table was re-measured on this package within the band (make_info.py remeasure),
and the dev gates + kernel tests the job ran on this package's Strix programs. No llama.cpp re-measure (user decision, phase 3).

usage: strix_card.py SLUG --job JOB_ARTIFACT_DIR --cmp CMP.json --run RUN --work DIR [--upload]

JOB_ARTIFACT_DIR / CMP.json / RUN: the passing run (the confirming re-run when there was one). Without --upload it builds, renders and
checks the card in DIR/SLUG/out and runs the upload dry run; --upload publishes the card files (card-only, --parent-commit) to
loom-fun-staging/SLUG.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
C = os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card")
PY = sys.executable
GH = "loom-fun/loom-skills"


def run(*cmd, capture=False):
    r = subprocess.run([str(x) for x in cmd], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))[:300]}\n{(r.stdout + r.stderr)[-2000:]}")
    return r.stdout


ap = argparse.ArgumentParser()
ap.add_argument("slug")
ap.add_argument("--job", required=True)
ap.add_argument("--cmp", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--upload", action="store_true")
a = ap.parse_args()
from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

slug, J = a.slug, os.path.abspath(a.job)
W = os.path.join(a.work, slug)
out = os.path.join(W, "out")
shutil.rmtree(W, ignore_errors=True)
os.makedirs(W)
api = HfApi()
stg = f"loom-fun-staging/{slug}"
sha = api.model_info(stg).sha
snapshot_download(stg, revision=sha, local_dir=out)
shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)
shutil.copy(os.path.join(out, "info.json"), os.path.join(W, "info_prev.json"))
shutil.copy(hf_hub_download(f"loom-fun/{slug}", "info.json"), os.path.join(W, "fun_info.json"))
job_name = json.load(open(os.path.join(J, "summary.json")))["job"]
jobs = json.loads(run("gh", "run", "view", a.run, "-R", GH, "--json", "jobs", capture=True))["jobs"]
job = next(j for j in jobs if j["name"] == job_name)
url, end = job["url"], job["completedAt"]
rev = re.search(r"resolve/([0-9a-f]{40})", json.dumps(json.load(open(os.path.join(J, "model_fun.json"))))).group(1)
fetched = open(os.path.join(J, "programs_sha256.txt")).read().strip()
dev, hrx = os.path.join(W, "dev.json"), os.path.join(W, "hrx.json")
run(PY, f"{C}/pull_devlab.py", "--identity", J, "--gpu", "strix-halo", "--device-out", dev, "--hrx-out", hrx)
src = ["--run", a.run, "--url", url, "--device", dev, "--hrx-loom", hrx, "--source-kind", "devlab"]
run(PY, f"{C}/make_info.py", "remeasure", "--cmp", a.cmp, "--package", out, "--compared-repo", f"loom-fun/{slug}", "--compared-revision", rev,
    "--fetched-programs", fetched, "--date", end[:10], *src, "-o", f"{W}/ev_rm.json")
gates = [os.path.join(J, "gates_staging", "gates.json"), os.path.join(J, "ktest_staging.log")]
gates = [g for g in gates if os.path.exists(g)]
if gates:
    run(PY, f"{C}/make_info.py", "gates", "--gpu", "strix-halo", "--time", end[:19], *gates, "--package", out, *src, "-o", f"{W}/ev_gates.json")
    g = json.load(open(f"{W}/ev_gates.json"))
    g.pop("performance", None)   # the spec gate's acceptance session: the published Strix table stays as it is (spec_acceptance included)
    json.dump(g, open(f"{W}/ev_gates.json", "w"), indent=1)
prev = json.load(open(os.path.join(W, "info_prev.json")))
mem = (prev.get("card") or {}).get("memory_bytes") or json.load(open(os.path.join(out, "model.json"))).get("memory_bytes")
if not mem:
    mem = next(m["memory_bytes"] for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]
               if m["repo"].split("/")[-1] == slug)
ev = [f"{W}/ev_rm.json"] + ([f"{W}/ev_gates.json"] if gates else [])
run(PY, f"{C}/make_info.py", "build", out, "--gpus", "strix-halo,radeon-r9700", "--fit", os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"),
    "--memory-bytes", mem, "--from-info", f"{W}/info_prev.json", "--perf-from", f"strix-halo={W}/fun_info.json", "--evidence", *ev,
    "-o", f"{W}/info_new.json")
shutil.copy(f"{W}/info_new.json", os.path.join(out, "info.json"))
run(PY, f"{C}/render_card.py", out)
run(PY, f"{C}/render_card.py", "--check", out, os.path.join(out, "README.md"))
for f in (".loom-fetch.lock",):
    if os.path.exists(os.path.join(out, f)):
        os.remove(os.path.join(out, f))
txt = open(os.path.join(out, "README.md")).read()
line = next((ln for ln in txt.splitlines() if "re-measured on this package" in ln), None)
if not line:
    sys.exit(f"{slug}: no re-measure line in the README")
if "not re-measured" in txt.split("### Strix Halo", 1)[-1].split("###", 1)[0]:
    sys.exit(f"{slug}: the Strix table is empty")
run(PY, f"{C}/publish_package.py", "upload", out, "--repo", slug, "--org", "loom-fun-staging", "--card-only", "--dry-run")
print(f"{slug}: card built ({line.strip()[:160]})")
if a.upload:
    print(run(PY, f"{C}/publish_package.py", "upload", out, "--repo", slug, "--org", "loom-fun-staging", "--card-only", "--parent-commit", sha,
              "--message", f"Strix re-measured on this package (run {a.run}): within the band; published Strix table kept"))
