#!/usr/bin/env python3
"""One command per campaign step: regenerate the cards of many models on one or more devices, resumable, state in one JSON file.

  regen.py plan    --device D [--device D2] [--mode ab|card] [--slugs S... | --all] [--old ORG|refs.json] [--batch N] --state S.json
  regen.py submit  S.json [--dry-run] [--max-batches N]    experiment branch + perf.json + PR + label per batch (REST), R9700-class devices first
  regen.py poll    S.json                                  ONE poll of the runs (exit 0: nothing pending, 10: jobs pending); loop it every >= 120 s
  regen.py collect S.json                                  download each finished job into a fresh dir, compare, per-row verdicts
  regen.py rerun   S.json [--dry-run]                      confirming -r2 jobs for models whose verdict needs one, new batch (then: submit)
  regen.py ingest  S.json                                  card from the job (perf / gates / roofline / build + render --check), per model
  regen.py publish S.json [--dry-run]                      card-only upload to loom-fun-staging (never loom-fun/*)
  regen.py status  S.json [--status-json status.json]      table; with --status-json: models.<slug>.<gpu> = stage / verdict / run for the dashboard
  regen.py accept  S.json SLUG... --note TEXT              a person's call: greedy text differs only at a near-tie, decide again with that accepted
  regen.py adopt   S.json --run RUN                        attach an already running / finished run to the state's jobs (by job name)

Modes. ab: regeneration of a device that already has a package to compare with (ab_jobs.py: old vs new in ABBA sessions, text match, func, ktest, gates,
  llama.cpp baselines where the card's table comes from the job). card: a NEW device, no old package: one card measurement per model (device_jobs.py: card +
  nodraft variants, func, ktest, gates, llama.cpp ROCm + Vulkan baselines), no A/B: about half the runner time of ab.
A job per (model, device), one runner. A model's state (state["jobs"][NAME]["stage"]):
  planned -> submitted -> queued -> running -> done | failed -> collected -> ingested -> published
  or needs_decision (a confirmed regression / failed gate / text difference: that model stops, the rest go on), skipped (does not fit), unsupported.
A verdict that wants a confirming run (rerun, improved-pending) gets ONE -r2 job (`rerun`); the second run decides (promote_check.py, --prev).
Everything the hand-run recipe did is a step here; only judgement stops a model with a reason in state["jobs"][NAME]["reason"].
Run with a python that has huggingface_hub and gguf (the tools venv); GPU work only through the devlab-perf workflow (submit); no step uploads
anywhere but loom-fun-staging, and `publish` without --dry-run is the only uploading step.
"""
import argparse
import base64
import contextlib
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CARD = os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card")
SCRIPTS = os.path.join(ROOT, ".github", "scripts")
for p in (HERE, CARD, os.path.join(ROOT, "skills", "loom-kb")):
    if p not in sys.path:
        sys.path.insert(0, p)
import card_lib  # noqa: E402
import promote_check  # noqa: E402
import sample_lib  # noqa: E402

try:
    import fcntl
except ImportError:   # Windows: the tests import this module; a campaign runs on Linux
    fcntl = None
REPO = "loom-fun/loom-skills"
BASELINES = os.path.join(HERE, "baselines.json")   # the ratchet store: what each card says today
PY = sys.executable
STAGING_ORG = "loom-fun-staging"
LOOMBIN = os.environ.get("REGEN_LOOMBIN", "/home/jfowers/lt/loom010/.venv/bin")   # loom-fetch: the publish tools need it on PATH
TERMINAL = ("collected", "ingested", "published", "needs_decision", "skipped", "unsupported", "failed")
STATUS_STAGE = {"planned": "queued", "submitted": "queued", "queued": "queued", "running": "running", "done": "running", "failed": "measured",
                "collected": "measured", "ingested": "measured", "needs_decision": "measured", "published": "published",
                "skipped": "unsupported", "unsupported": "unsupported"}
GPU_KEY = {"strix-halo": "strix", "radeon-r9700": "r9700"}   # status.json's per-GPU keys
# the files a variant leaves in a job's output: ab = old "fun" and new "staging"; card = the single "card" (+ "nodraft")
VARIANT = {"ab": ("staging", "stagingnd"), "card": ("card", "nodraft")}


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- shell, GitHub, Hugging Face (the only places that touch the network)

def sh(cmd, check=True, cwd=None, env=None):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=cwd, env=env)
    if check and r.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))[:300]}\n{(r.stdout + r.stderr)[-1500:]}")
    return r


def gh_json(args):
    return json.loads(sh(["gh", *args]).stdout or "null")


def gh_api(method, path, fields=None, jq=None):
    cmd = ["gh", "api", "-X", method, path]
    for k, v in (fields or {}).items():
        cmd += ["-f", f"{k}={v}"]
    if jq:
        cmd += ["--jq", jq]
    out = sh(cmd).stdout
    return out.strip() if jq else (json.loads(out) if out.strip() else None)


def hf_head(repo):
    from huggingface_hub import HfApi
    return HfApi().model_info(repo).sha


def hf_snapshot(repo, rev, out):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=rev, local_dir=out)
    shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)


def hf_file(repo, name, rev=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, revision=rev)


# ---------------------------------------------------------------- state

class State:
    """The campaign state: one JSON file, rewritten atomically under a lock after every step."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.d = json.load(open(self.path))
        self.work = self.d["work"]

    @classmethod
    def create(cls, path, d):
        s = cls.__new__(cls)
        s.path, s.d, s.work = os.path.abspath(path), d, d["work"]
        return s

    def save(self):
        with locked(self.path):
            tmp = self.path + ".tmp"
            json.dump(self.d, open(tmp, "w"), indent=1, sort_keys=True)
            os.replace(tmp, self.path)

    @property
    def jobs(self):
        return self.d["jobs"]

    def log(self, msg):
        self.d.setdefault("log", []).append(f"{now()} {msg}")
        print(msg)


@contextlib.contextmanager
def locked(path):
    """an exclusive lock on PATH.lock (several steps and the other engineers' status writers share status.json)"""
    with open(path + ".lock", "w") as lk:
        if fcntl:
            fcntl.flock(lk, fcntl.LOCK_EX)
        yield


def die(msg):
    sys.exit(f"regen: {msg}")


# ---------------------------------------------------------------- plan

def device_order(device):
    """Scarce runners first: a device with a fixed small runner pool (device.json runner.concurrent_runners) before the elastic ones."""
    r = card_lib.registry.device(device)["runner"]
    return (r.get("concurrent_runners") or 10_000, device)


def slug_of(job):
    return job["card"].split("/")[-1].split("@")[0]


def generator_cmd(st, device, slugs, suffix=""):
    d = st["opts"]
    if st["mode"] == "ab":
        cmd = [PY, os.path.join(SCRIPTS, "ab_jobs.py"), "--device", device, "--gates", d["gates"], "--new-org", d["new_org"], "--old-org", d["old_org"]]
        if d.get("refs"):
            cmd += ["--refs", d["refs"]]
        if device != "strix-halo":
            cmd += ["--baselines"]
        if suffix:
            cmd += [f"--suffix={suffix}"]
    else:
        script = os.path.join(SCRIPTS, "device_jobs.py")
        if not os.path.exists(script):
            script = os.path.join(SCRIPTS, "r9700_jobs.py")   # same bytes for the R9700 (the device-generic generator is PR #734)
            if device != "radeon-r9700" or not os.path.exists(script):
                die("--mode card needs .github/scripts/device_jobs.py (device-generic job generator, PR #734)")
            cmd = [PY, script, "--card-org", d["new_org"]]
        else:
            cmd = [PY, script, "--device", device, "--card-org", d["new_org"]]
        if d["gates"]:
            cmd += ["--gates", d["gates"]]
    return cmd + slugs


def gen_jobs(st, device, slugs, suffix=""):
    """(perf.json document, [job], [skipped slug]) for the slugs on the device: the repository's own generators, unchanged."""
    r = sh(generator_cmd(st, device, slugs, suffix), check=False)
    if r.returncode:
        if "no job" in r.stderr or "do not fit" in r.stderr:
            return None, [], list(slugs)
        die(f"job generator failed for {device}: {r.stderr[-800:]}")
    doc = json.loads(r.stdout)
    key = card_lib.registry.device(device)["runner"]["key"]
    for j in doc["jobs"]:
        j["runner"] = key
        if st["mode"] == "card":
            j["name"] = f"{key[0]}c-{j['name']}"[:40 - len(suffix)] + suffix   # a card job per device: names differ across devices
        elif suffix and not j["name"].endswith(suffix):
            j["name"] = j["name"][:40 - len(suffix)] + suffix
    got = {slug_of(j) for j in doc["jobs"]}
    return doc, doc["jobs"], [s for s in slugs if s not in got]


def fleet_slugs():
    return sorted(json.load(open(os.path.join(HERE, "manifest.json")))["models"])


def cmd_plan(a):
    slugs = fleet_slugs() if a.all else a.slugs
    if not slugs:
        die("no models: give SLUG... or --all")
    state_path = os.path.abspath(a.state)
    if os.path.exists(state_path) and not a.force:
        die(f"{a.state} exists (--force replaces it)")
    work = os.path.abspath(a.work or (state_path[:-5] if state_path.endswith(".json") else state_path) + ".d")
    os.makedirs(os.path.join(work, "batches"), exist_ok=True)
    old_org, refs = "loom-fun", None
    if a.old and a.old.endswith(".json"):
        refs = json.load(open(a.old))
    elif a.old:
        old_org = a.old
    devices = sorted(dict.fromkeys(a.device), key=device_order)
    unknown = [d for d in devices if d not in card_lib.registry.devices()]
    if unknown:
        die(f"unknown device {unknown}: {', '.join(card_lib.registry.devices())}")
    if a.new_org != STAGING_ORG:
        die(f"--new-org {a.new_org}: cards are measured and published on {STAGING_ORG} only")
    if a.mode == "ab" and not a.no_pin:
        # pin the new package of every slug to the staging revision it is measured at (the card the run measures is the card ingested)
        refs = {s: dict(e) for s, e in (refs or {}).items()}
        for s in slugs:
            ent = refs.setdefault(s, {})
            if "new" not in ent:
                ent["new"] = f"{a.new_org}/{s}@{hf_head(f'{a.new_org}/{s}')}"
    st = {"schema": "regen/1", "created": now(), "mode": a.mode, "work": work, "state": state_path,
          "opts": {"gates": a.gates, "new_org": a.new_org, "old_org": old_org, "batch": a.batch, "refs": None},
          "devices": devices, "batches": [], "jobs": {}, "log": []}
    for dev in devices:
        # per-device refs file. Where the device has no published table to compare with (anything but Strix Halo, whose old package is the
        # published loom-fun card) and no old package was named, the old package is the new revision itself: an A/A, the runner's own noise.
        dref = {s: dict(e) for s, e in refs.items()} if refs else None
        if dref and dev != "strix-halo" and a.old is None:
            for e in dref.values():
                e.setdefault("old", e["new"])
        if dref:
            json.dump(dref, open(os.path.join(work, f"refs_{dev}.json"), "w"), indent=1)
        st["opts"]["refs"] = os.path.join(work, f"refs_{dev}.json") if dref else None
        for i in range(0, len(slugs), a.batch):
            chunk = slugs[i:i + a.batch]
            doc, jobs, skipped = gen_jobs(st, dev, chunk)
            for s in skipped:
                st["jobs"][f"skipped:{s}:{dev}"] = {"slug": s, "device": dev, "stage": "skipped", "reason": "the weights do not fit the device / no job",
                                                    "round": 1}
            if not jobs:
                continue
            bid = len(st["batches"])
            pj = os.path.join(work, "batches", f"{bid}.perf.json")
            json.dump(doc, open(pj, "w"), indent=1)
            st["batches"].append({"id": bid, "device": dev, "perf": pj, "jobs": [j["name"] for j in jobs], "round": 1, "stage": "planned"})
            for j in jobs:
                e = (dref or {}).get(slug_of(j)) or {}
                st["jobs"][j["name"]] = {"slug": slug_of(j), "device": dev, "batch": bid, "round": 1, "stage": "planned",
                                         "card": j["card"], "steps": len(j["steps"]),
                                         "aa": bool(e.get("old") and e.get("old") == e.get("new"))}   # old == new: the A/B shows noise only
    st["opts"]["refs"] = None
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    State.create(state_path, st).save()
    n = sum(1 for j in st["jobs"].values() if j["stage"] == "planned")
    print(f"planned {n} jobs in {len(st['batches'])} batches ({', '.join(devices)}; mode {a.mode}); "
          f"skipped {sum(1 for j in st['jobs'].values() if j['stage'] == 'skipped')}: {state_path}")


# ---------------------------------------------------------------- submit

def batch_body(st, b):
    jobs = b["jobs"]
    return (f"Experiment PR (never merged): DevLab measurement of {len(jobs)} models on {b['device']} (mode {st.d['mode']}, round {b['round']}), generated by "
            f"fleet/regen.py from {os.path.basename(st.path)}. Only .github/devlab/perf.json differs from the integration branch.\n\n"
            "Jobs: " + ", ".join(f"`{j}`" for j in jobs) + "\n\n```loom-scope\nkind: tooling\nscopes: none\n```\n\n"
            "🤖 Generated with [Claude Code](https://claude.com/claude-code)\n")


def submit_batch(st, b, dry):
    tag = re.sub(r"[^a-z0-9]+", "-", os.path.basename(st.path)[:-5].lower()).strip("-")
    branch = f"regen-{tag}-b{b['id']}" + (f"-r{b['round']}" if b["round"] > 1 else "")
    perf = open(b["perf"], "rb").read()
    title = f"Experiment: regen {tag} {b['device']} batch {b['id']}" + (f" confirming run {b['round']}" if b["round"] > 1 else "")
    if dry:
        print(f"[dry-run] branch {branch} from refactor, perf.json ({len(perf)} bytes, {len(b['jobs'])} jobs), PR '{title}', label devlab-perf")
        return
    sha = gh_api("GET", f"repos/{REPO}/git/ref/heads/refactor", jq=".object.sha")
    gh_api("POST", f"repos/{REPO}/git/refs", {"ref": f"refs/heads/{branch}", "sha": sha})
    f = "repos/%s/contents/.github/devlab/perf.json" % REPO
    fsha = gh_api("GET", f"{f}?ref={branch}", jq=".sha")
    gh_api("PUT", f, {"message": f"devlab perf: {title}\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>", "branch": branch,
                      "content": base64.b64encode(perf).decode(), "sha": fsha})
    pr = gh_api("POST", f"repos/{REPO}/pulls", {"title": title, "head": branch, "base": "refactor", "body": batch_body(st, b)}, jq=".number")
    gh_api("POST", f"repos/{REPO}/issues/{pr}/labels", {"labels[]": "devlab-perf"})   # the label starts the run
    b.update(branch=branch, pr=int(pr), stage="submitted", submitted=now())
    for n in b["jobs"]:
        st.jobs[n].update(stage="submitted", pr=int(pr))
    st.log(f"batch {b['id']} ({b['device']}, {len(b['jobs'])} jobs) submitted: PR #{pr}")
    st.save()


def cmd_submit(a):
    st = State(a.state)
    todo = [b for b in st.d["batches"] if b["stage"] == "planned"]
    todo.sort(key=lambda b: (device_order(b["device"]), b["id"]))   # R9700-class (few runners) first: their queue is the long one
    if a.max_batches:
        todo = todo[:a.max_batches]
    if not todo:
        print("nothing to submit")
    for b in todo:
        submit_batch(st, b, a.dry_run)


# ---------------------------------------------------------------- poll

def find_run(b):
    runs = gh_json(["run", "list", "-R", REPO, "-w", "devlab-perf", "--branch", b["branch"], "-L", "10", "--json", "databaseId,status,createdAt,event"]) or []
    for r in sorted(runs, key=lambda r: r["createdAt"], reverse=True):
        jobs = gh_json(["run", "view", str(r["databaseId"]), "-R", REPO, "--json", "jobs"])["jobs"]
        if any(j["name"] in b["jobs"] for j in jobs):
            return r["databaseId"]
    return None


def apply_run_jobs(st, b, run, jobs):
    names = set(b["jobs"])
    for j in jobs:
        if j["name"] not in names:
            continue
        e = st.jobs[j["name"]]
        if e["stage"] in ("collected", "ingested", "published", "needs_decision"):
            continue
        e.update(run=str(run), job_url=j["url"], started=j.get("startedAt"), completed=j.get("completedAt"))
        if j["status"] == "completed":
            e["stage"] = "done" if j["conclusion"] == "success" else "failed"
            e["conclusion"] = j["conclusion"]
        elif j["status"] == "in_progress":
            e["stage"] = "running"
        else:
            e["stage"] = "queued"


def cmd_poll(a):
    st = State(a.state)
    pending = 0
    for b in st.d["batches"]:
        if b["stage"] == "planned":
            pending += len(b["jobs"])
            continue
        if not b.get("run"):
            b["run"] = find_run(b)
        if b.get("run"):
            jobs = gh_json(["run", "view", str(b["run"]), "-R", REPO, "--json", "status,conclusion,jobs"])
            apply_run_jobs(st, b, b["run"], jobs["jobs"])
    for e in st.jobs.values():
        if e["stage"] in ("planned", "submitted", "queued", "running"):
            pending += 1
    st.save()
    print_table(st)
    sys.exit(10 if pending else 0)


def print_table(st):
    by = {}
    for n, e in st.jobs.items():
        by.setdefault((e["device"], e["stage"]), []).append(n)
    print(f"{now()} {os.path.basename(st.path)}")
    for (dev, stage), names in sorted(by.items()):
        print(f"  {dev:14s} {stage:15s} {len(names):3d}")
    runs = sorted({e["run"] for e in st.jobs.values() if e.get("run")})
    print("  runs:", " ".join(runs) or "-")


# ---------------------------------------------------------------- collect

def job_dir(st, name):
    return os.path.join(st.work, "jobs", name)


def download(st, name, e):
    d = job_dir(st, name)
    shutil.rmtree(d, ignore_errors=True)   # a fresh directory per job (a stale file would be read as this job's)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    sh(["gh", "run", "download", e["run"], "-R", REPO, "-D", d, "-n", f"devlab-perf-{name}"])
    return d


def read(path, default=None):
    try:
        return open(path).read().strip()
    except OSError:
        return default


def compare(st, name, e, d):
    """strix_ab_compare.py of an A/B job -> cmp.json; the round-2 compare reads --prev (the first run's)."""
    cmp = os.path.join(st.work, "cmp", f"{name}.json")
    os.makedirs(os.path.dirname(cmp), exist_ok=True)
    cmd = [PY, os.path.join(SCRIPTS, "strix_ab_compare.py"), d, "--json", cmp]
    if e["device"] == "strix-halo" and st.d["opts"]["old_org"] == "loom-fun":
        cmd += ["--fun-info", hf_file(f"{st.d['opts']['old_org']}/{e['slug']}", "info.json")]   # new vs the published card value too
    if e.get("prev"):
        cmd += ["--prev", st.jobs[e["prev"]]["cmp"]]
    sh(cmd)
    return cmp


def baseline_low(st, e, cmp):
    """Rows whose new value is below the stored card value (fleet/baselines.json) by more than the row's band: when the old package of
    the A/B is not the published card (an A/A of the staging revision) the A/B ratio cannot show a regression against the card."""
    base = json.load(open(BASELINES)) if os.path.exists(BASELINES) else {}
    b = ((base.get("models") or {}).get(e["slug"]) or {}).get(e["device"]) or {}
    meth = sample_lib.load_methodology()
    low = {}
    for r, m in (cmp.get("metrics") or {}).items():
        v = (b.get(r) or {}).get("value")
        if v and m.get("new") and r in meth["bands"] and m["new"] / v < 1 - meth["bands"][r][0]:
            low[r] = round(m["new"] / v, 3)
    return low


def decide(st, name, e, cmp_paths):
    """The promotion ratchet on the compare(s) of this model/device (oldest first): promote | improved-pending | rerun | hold, plus baseline checks."""
    compares = [json.load(open(p)) for p in cmp_paths]
    base = json.load(open(BASELINES)) if os.path.exists(BASELINES) else None
    res = promote_check.check(e["slug"], e["device"], compares, base, sample_lib.load_methodology(), near_tie=e["slug"] in st.d["opts"].get("near_tie", []))
    decision, reasons = res["decision"], list(res["reasons"])
    if e.get("aa"):
        # old == new (an A/A, the device has no published table to compare with): the ratios are the runner's noise, so they decide nothing;
        # a gate failure or a text difference still holds, a row below the card's published value (fleet/baselines.json) is what regresses
        decision = decision if reasons and reasons[0].startswith(("gate failed", "greedy text")) else "promote"
        reasons = reasons if decision == "hold" else []
        lows = [baseline_low(st, e, c) for c in compares]
        if decision == "promote" and lows[-1]:
            both = len(lows) > 1 and lows[-2]
            decision = "hold" if both else "rerun"
            reasons.append(f"below the published card value by more than the row band: {lows[-1]}" + (" (in both runs)" if both else ""))
    return decision, reasons, res


def card_mode_verdict(d, e):
    """Card mode (a new device): no old package; every step must have run and the gates / kernel tests pass."""
    s = json.load(open(os.path.join(d, "summary.json"))).get("results", {})
    bad = {k: v for k, v in s.items() if str(v).lower().startswith(("fail", "error")) or "FAIL" in str(v)[:40]}
    need = [k for k in ("perf:card:1", "c4:nodraft", "func:card", "ktest:card") if k not in s]
    if bad:
        return "hold", [f"{k}: {str(v)[:120]}" for k, v in bad.items()]
    if need:
        return "rerun", ["missing steps: " + ", ".join(need)]
    return "promote", []


def collect_one(st, name, e):
    d = download(st, name, e)
    e["dir"] = d
    mode = st.d["mode"]
    if e["stage"] == "failed" and not os.path.exists(os.path.join(d, "summary.json")):
        e.update(stage="needs_decision" if e["round"] > 1 else "collected", decision="rerun" if e["round"] == 1 else "hold",
                 reason=f"the job {e.get('conclusion')} before writing summary.json (see {e.get('job_url')})")
        return
    if mode == "ab":
        e["cmp"] = compare(st, name, e, d)
        paths = ([st.jobs[e["prev"]]["cmp"]] if e.get("prev") else []) + [e["cmp"]]
        decision, reasons, res = decide(st, name, e, paths)
        e["rows"] = {r: x["state"] for r, x in res["rows"].items()}
        e["improved"] = [r for r, x in res["rows"].items() if "improved" in x["verdicts"] and x["state"] != "rerun"]
    else:
        decision, reasons = card_mode_verdict(d, e)
    e["decision"], e["reason"] = decision, "; ".join(reasons)
    e["card_revision"] = read(os.path.join(d, "card_revision.txt"), "").split("\n")[0]
    if decision == "promote":
        e["stage"] = "collected"
    elif decision in ("rerun", "improved-pending"):
        e["stage"] = "needs_decision" if e["round"] > 1 else "collected"   # round 1: `rerun` queues the confirming job; round 2 undecided: a person decides
        if e["round"] > 1:
            e["reason"] = "still undecided after the confirming run: " + e["reason"]
    else:
        e["stage"] = "needs_decision"
        e["reason"] = "confirmed regression / failed gate / text difference: " + e["reason"]


def cmd_collect(a):
    st = State(a.state)
    n = 0
    for name, e in list(st.jobs.items()):
        if e["stage"] not in ("done", "failed"):
            continue
        collect_one(st, name, e)
        st.log(f"{e['slug']} {e['device']} r{e['round']}: {e['stage']} ({e.get('decision')}) {e.get('reason', '')[:140]}")
        st.save()
        n += 1
    if not n:
        print("no finished jobs to collect")


# ---------------------------------------------------------------- rerun (confirming -r2 jobs)

def cmd_rerun(a):
    st = State(a.state)
    want = {}
    for name, e in st.jobs.items():
        if e["stage"] == "collected" and e.get("decision") in ("rerun", "improved-pending") and e["round"] == 1 and not e.get("rerun_of_me"):
            want.setdefault(e["device"], []).append((name, e))
    if not want:
        print("no confirming runs needed")
        return
    if a.dry_run:
        for dev, items in sorted(want.items(), key=lambda kv: device_order(kv[0])):
            print(f"would queue a confirming batch ({dev}): {', '.join(e['slug'] for _, e in items)}")
        return
    for dev, items in sorted(want.items(), key=lambda kv: device_order(kv[0])):
        slugs = [e["slug"] for _, e in items]
        # the same card revisions as the first run: refs of the first plan (pinned)
        st.d["opts"]["refs"] = os.path.join(st.work, f"refs_{dev}.json") if os.path.exists(os.path.join(st.work, f"refs_{dev}.json")) else None
        doc, jobs, skipped = gen_jobs(st.d, dev, slugs, suffix="-r2")
        if not jobs:
            die(f"cannot generate the confirming jobs for {slugs}")
        bid = len(st.d["batches"])
        pj = os.path.join(st.work, "batches", f"{bid}.perf.json")
        json.dump(doc, open(pj, "w"), indent=1)
        st.d["batches"].append({"id": bid, "device": dev, "perf": pj, "jobs": [j["name"] for j in jobs], "round": 2, "stage": "planned"})
        first = {e["slug"]: n for n, e in items}
        for j in jobs:
            s = slug_of(j)
            st.jobs[j["name"]] = {"slug": s, "device": dev, "batch": bid, "round": 2, "stage": "planned", "card": j["card"], "steps": len(j["steps"]),
                                  "prev": first[s], "aa": st.jobs[first[s]].get("aa", False)}
            st.jobs[first[s]]["rerun_of_me"] = j["name"]
        st.log(f"confirming batch {bid} ({dev}): {', '.join(slugs)}")
    st.save()
    print("now: regen.py submit " + a.state)


# ---------------------------------------------------------------- ingest

def run_tool(args, cwd):
    r = sh([PY, *args], cwd=cwd, check=False)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, args))[:200]}\n{(r.stdout + r.stderr)[-1200:]}")
    return r.stdout


def job_facts(st, e):
    """(run id, job URL, job end UTC) of a collected job."""
    return e["run"], e["job_url"], (e.get("completed") or now())[:19]


def ingest_full(st, name, e, W):
    """The card's performance, gates and roofline of this device from the job (REGEN_RECIPE: ingest the NEW package)."""
    d, dev, slug = e["dir"], e["device"], e["slug"]
    V, ND = VARIANT[st.d["mode"]]
    S = CARD
    rev = e["card_revision"].split("@")[-1]
    out = os.path.join(W, "out")
    hf_snapshot(f"{st.d['opts']['new_org']}/{slug}", rev, out)
    shutil.copy(os.path.join(out, "info.json"), os.path.join(W, "info_prev.json"))
    prev = json.load(open(os.path.join(W, "info_prev.json")))
    mem = prev["card"]["memory_bytes"]
    run, url, end = job_facts(st, e)
    src = ["--source-kind", "devlab", "--run", run, "--url", url]
    rd = lambda f: read(os.path.join(d, f))
    g = lambda pat: sorted(glob.glob(os.path.join(d, pat)))
    run_tool([f"{S}/pull_devlab.py", "--identity", d, "--gpu", dev, "--device-out", "dev.json", "--hrx-out", "hrx.json"], W)
    llm, vk = g("perf_llamacpp-rocm.np1_n*.jsonl"), g("perf_llamacpp-vulkan.np1_n*.jsonl")
    c4l, c4v = g("c4_llamacpp-rocm.nodraft_n*.txt"), g("c4_llamacpp-vulkan.perslot.nodraft_n*.txt")
    if not (llm and c4l):
        raise RuntimeError("the job has no llama.cpp ROCm files (a job without --baselines cannot make this device's table)")
    baseline = card_lib.registry.methodology()["baselines"][dev]
    run_tool([f"{S}/make_info.py", "perf", "--gpu", dev, "--loom-jsonl", f"{d}/perf_{V}.jsonl", "--llama-jsonl", *llm, "--c4-loom", f"{d}/c4_{ND}.txt",
              "--c4-llama", c4l[0], "--package", out, *src, "--device", "dev.json", "--hrx-loom", "hrx.json",
              "--baseline-version", "llamacpp-rocm/" + baseline["rocm"]["build"], "--baseline-args", rd("llamacpp_args-rocm.np1.txt"),
              "--fetched-programs", rd("programs_sha256.txt"), "-o", "ev_perf.json"], W)
    ev = ["ev_perf.json"]
    if vk and c4v:
        run_tool([f"{S}/make_info.py", "perf", "--gpu", dev, "--backend", "vulkan", "--llama-jsonl", *vk, "--c4-llama", c4v[0],
                  "--baseline-version", baseline["vulkan"]["build"], "--baseline-args", rd("llamacpp_args-vulkan.np1.txt"),
                  "--baseline-sha256", rd("llamacpp_release-vulkan.np1.sha256"), "--fetched-programs", rd("programs_sha256.txt"), "--package", out,
                  *src, "--device", "none", "--hrx-loom", "none", "-o", "ev_vulkan.json"], W)
        ev.append("ev_vulkan.json")
    gfiles = [f for f in (f"{d}/gates_{V}/gates.json", f"{d}/ktest_{V}.log") if os.path.exists(f)]
    if gfiles:
        run_tool([f"{S}/make_info.py", "gates", "--gpu", dev, "--time", end, *gfiles, "--package", out, *src, "--device", "dev.json", "--hrx-loom", "hrx.json",
                  "-o", "ev_gates.json"], W)
        ev.append("ev_gates.json")
    run_tool([f"{S}/make_info.py", "roofline", "--gpu", dev, "--device", card_lib.registry.device(dev)["roofline"]["default"], "--package", out,
              "--evidence", "ev_perf.json", "-o", "ev_roof.json"], W)
    ev.append("ev_roof.json")
    gpus = [x for x in card_lib.registry.devices() if x in (prev.get("devices") or {}) or x == dev]
    fit = card_lib.registry.device(dev).get("fit_table")
    build = [f"{S}/make_info.py", "build", out, "--org", st.d["opts"]["new_org"], "--gpus", ",".join(gpus)]
    if fit:
        build += ["--fit", os.path.join(ROOT, fit)]
    run_tool(build + ["--memory-bytes", str(mem), "--from-info", "info_prev.json", "--evidence", *ev, "-o", "info_new.json"], W)
    shutil.copy(os.path.join(W, "info_new.json"), os.path.join(out, "info.json"))
    return out, hf_head(f"{st.d['opts']['new_org']}/{slug}"), (f"{card_lib.GPUS[dev]['runner']} results measured on this package (run {run})")


def ingest_strix(st, name, e, W):
    """Strix Halo, A/B against the published loom-fun card, within the band: the published table stays, one line says it was re-measured
    on this package (make_info.py remeasure; strix_card.py is the recipe)."""
    run = e["run"]
    cmd = [PY, os.path.join(SCRIPTS, "strix_card.py"), e["slug"], "--job", e["dir"], "--cmp", e["cmp"], "--run", run, "--work", os.path.dirname(W)]
    if e.get("prev"):   # the confirming run: strix_card.py takes rows faster in both runs onto the card (the ratchet)
        first = st.jobs[e["prev"]]
        cmd += ["--first-cmp", first["cmp"], "--first-run", first["run"]]
    r = sh(cmd, check=False)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr)[-1500:])
    out = os.path.join(os.path.dirname(W), e["slug"], "out")   # strix_card.py writes WORK/SLUG/out
    return out, hf_head(f"{st.d['opts']['new_org']}/{e['slug']}"), f"Strix re-measured on this package (run {run}): within the band; published Strix table kept"


def cmd_ingest(a):
    st = State(a.state)
    todo = [(n, e) for n, e in st.jobs.items() if e["stage"] == "collected" and e.get("decision") == "promote"]
    # a model with a round-2 job: the second run ingests, the first is superseded
    todo = [(n, e) for n, e in todo if not e.get("rerun_of_me")]
    if a.slug:
        todo = [(n, e) for n, e in todo if e["slug"] in a.slug]
    for name, e in todo:
        W = os.path.join(st.work, "cards", f"{e['slug']}.{e['device']}")
        shutil.rmtree(W, ignore_errors=True)
        os.makedirs(W)
        try:
            full = st.d["mode"] == "card" or bool(glob.glob(os.path.join(e["dir"], "perf_llamacpp-rocm.np1_n*.jsonl")))
            out, parent, msg = (ingest_full if full else ingest_strix)(st, name, e, W)
            run_tool([f"{CARD}/render_card.py", out], W) if full else None
            for f in (".loom-fetch.lock",):
                if os.path.exists(os.path.join(out, f)):
                    os.remove(os.path.join(out, f))
            run_tool([f"{CARD}/render_card.py", "--check", out, os.path.join(out, "README.md")], W)
        except (RuntimeError, SystemExit) as ex:
            e.update(stage="needs_decision", reason=f"ingest failed: {str(ex)[-400:]}")
            st.log(f"{e['slug']} {e['device']}: ingest failed")
            st.save()
            continue
        e.update(stage="ingested", card_dir=out, parent=parent, message=msg)
        if str(e.get("reason", "")).startswith("ingest failed"):
            e.pop("reason")   # a retried ingest that works leaves no stale failure text
        st.log(f"{e['slug']} {e['device']}: card built ({'full table' if full else 'Strix re-measure line'}) at {out}")
        st.save()
    if not todo:
        print("nothing to ingest")


# ---------------------------------------------------------------- publish (card only, staging only)

def cmd_publish(a):
    st = State(a.state)
    org = st.d["opts"]["new_org"]
    if org != STAGING_ORG or "/" in org.strip("/") or org == "loom-fun":
        die(f"publish writes {STAGING_ORG} only")
    env = dict(os.environ, PATH=LOOMBIN + os.pathsep + os.environ["PATH"])
    for name, e in st.jobs.items():
        if e["stage"] != "ingested":
            continue
        cmd = [PY, f"{CARD}/publish_package.py", "upload", e["card_dir"], "--repo", e["slug"], "--org", org, "--card-only",
               "--parent-commit", e["parent"], "--message", e["message"]]
        r = sh(cmd + (["--dry-run"] if a.dry_run else []), check=False, env=env)
        if r.returncode:
            e.update(stage="needs_decision", reason=f"publish failed: {(r.stdout + r.stderr)[-300:]}")
            st.log(f"{e['slug']} {e['device']}: publish failed")
        elif a.dry_run:
            e["dry_run_ok"] = True
            print(f"{e['slug']} {e['device']}: dry run OK\n" + "\n".join("   " + ln for ln in r.stdout.strip().splitlines()[-6:]))
        else:
            e.update(stage="published", published=now())
            st.log(f"{e['slug']} {e['device']}: published to {org}")
        st.save()


# ---------------------------------------------------------------- status

def cmd_status(a):
    st = State(a.state)
    print_table(st)
    for n, e in sorted(st.jobs.items(), key=lambda kv: (kv[1]["device"], kv[1]["slug"])):
        print(f"  {e['slug'][:46]:46s} {e['device']:13s} r{e['round']} {e['stage']:14s} {e.get('decision') or '':16s} {(e.get('reason') or '')[:70]}")
    if a.status_json:
        write_status(st, a.status_json)


def write_status(st, path):
    """models.<slug>.<gpu> = {stage, verdict, run}: what the dashboard reads (a round-2 job's entry wins over its first run's)."""
    with locked(path):
        d = json.load(open(path))
        for n, e in sorted(st.jobs.items(), key=lambda kv: kv[1]["round"]):
            m = d.setdefault("models", {}).setdefault(e["slug"], {})
            verdict = {"promote": "pass", "improved-pending": "rerun", "hold": "regressed"}.get(e.get("decision") or "", e.get("decision"))
            if e["stage"] == "skipped":
                verdict = "unsupported"
            m[GPU_KEY.get(e["device"], e["device"])] = {"stage": STATUS_STAGE.get(e["stage"], "queued"), "verdict": verdict, "run": e.get("run")}
            if e["stage"] == "needs_decision":
                m["note"] = f"{e['device']}: needs decision: {e.get('reason', '')[:200]}"
        json.dump(d, open(path, "w"), indent=1)
    print(f"status written: {path}")


def cmd_accept(a):
    """A person's judgement on a model that stopped with a text difference: the greedy texts were read and differ only at a near-tie
    (log-prob band), so the verdict is decided again with that accepted. Gate failures and row regressions are not accepted here."""
    st = State(a.state)
    st.d["opts"].setdefault("near_tie", [])
    for name, e in st.jobs.items():
        if e["slug"] in a.slug and e["stage"] == "needs_decision" and e.get("cmp") and "greedy text" in (e.get("reason") or ""):
            st.d["opts"]["near_tie"].append(e["slug"])
            paths = ([st.jobs[e["prev"]]["cmp"]] if e.get("prev") else []) + [e["cmp"]]
            decision, reasons, res = decide(st, name, e, paths)
            e["decision"], e["reason"] = decision, "; ".join(reasons)
            if decision == "promote" or (decision in ("rerun", "improved-pending") and e["round"] == 1):
                e["stage"] = "collected"     # round 1: `rerun` queues the confirming job, as collect does
            else:
                e["stage"] = "needs_decision"
            e["accepted"] = f"near-tie text accepted: {a.note}"
            st.log(f"{e['slug']} {e['device']}: near-tie accepted -> {e['stage']} ({decision})")
    st.save()


def cmd_adopt(a):
    st = State(a.state)
    jobs = gh_json(["run", "view", a.run, "-R", REPO, "--json", "jobs"])["jobs"]
    hit = 0
    for b in st.d["batches"]:
        if any(j["name"] in b["jobs"] for j in jobs):
            b["run"] = a.run
            b["stage"] = "submitted" if b["stage"] == "planned" else b["stage"]
            apply_run_jobs(st, b, a.run, jobs)
            hit += 1
    st.save()
    print(f"adopted run {a.run}: {hit} batches")
    print_table(st)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("slugs", nargs="*")
    p.add_argument("--device", action="append", required=True)
    p.add_argument("--mode", choices=("ab", "card"), default="ab")
    p.add_argument("--all", action="store_true", help="every model of the fleet manifest (a device skips the ones that do not fit)")
    p.add_argument("--old", help="ORG of the old packages (default loom-fun) or a refs.json {slug: {old, new}} (card revisions pinned: org/repo@rev)")
    p.add_argument("--new-org", default=STAGING_ORG)
    p.add_argument("--gates", default="dev")
    p.add_argument("--batch", type=int, default=8, help="jobs per experiment PR (smaller batches finish, and can be collected, sooner)")
    p.add_argument("--no-pin", action="store_true", help="do not pin the new package to the staging head revision at plan time")
    p.add_argument("--work", help="work directory (default: STATE.d next to the state file)")
    p.add_argument("--state", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_plan)
    for name, fn in (("submit", cmd_submit), ("poll", cmd_poll), ("collect", cmd_collect), ("rerun", cmd_rerun), ("ingest", cmd_ingest),
                     ("publish", cmd_publish), ("status", cmd_status), ("adopt", cmd_adopt), ("accept", cmd_accept)):
        q = sub.add_parser(name)
        q.add_argument("state")
        q.set_defaults(fn=fn)
        if name in ("submit", "rerun", "publish"):
            q.add_argument("--dry-run", action="store_true")
        if name == "submit":
            q.add_argument("--max-batches", type=int)
        if name == "status":
            q.add_argument("--status-json")
        if name == "accept":
            q.add_argument("slug", nargs="+")
            q.add_argument("--note", required=True, help="what was read: e.g. 'hyphen vs space, log-prob gap < band'")
        if name == "adopt":
            q.add_argument("--run", required=True)
        if name == "ingest":
            q.add_argument("--slug", action="append")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
