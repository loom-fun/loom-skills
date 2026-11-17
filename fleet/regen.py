#!/usr/bin/env python3
"""One command per campaign step: regenerate the cards of many models on one or more devices, resumable, state in one JSON file.

  regen.py plan    --base BRANCH --device D [--device D2] [--mode ab|card|sdk-ab|sdk-loom] [--slugs S... | --all] [--old ORG|refs.json] [--batch N] --state S.json
                   [--replicas 2] [--sequential] [--no-baselines] [--priority]   a narrow change: confirming run as a parallel replica, 2+2 -> 6+6, no llama.cpp
  regen.py submit  S.json [--dry-run] [--max-batches N] [--max-inflight N|auto]   experiment branch + perf.json + PR + label per batch (REST), priority
                   batches and R9700-class devices first; --max-inflight sends a sweep in waves (default: all at once)
  regen.py poll    S.json                                  ONE poll of the runs (exit 0: nothing pending, 10: jobs pending); loop it every >= 120 s
  regen.py collect S.json                                  download each finished job into a fresh dir, compare, per-row verdicts
  regen.py rerun   S.json [--dry-run]                      confirming -r2 jobs for models whose verdict needs one, new batch (then: submit)
  regen.py package S.json [--dry-run] [--slugs S...] [--ref REF]   sdk-ab: publish the v0.2 package (programs) to loom-fun-staging before `ingest`
  regen.py ingest  S.json                                  card from the job (perf / gates / roofline / build + render --check), per model
  regen.py publish S.json [--dry-run]                      card-only upload to loom-fun-staging (never loom-fun/*)
  regen.py rerender [--slugs S... | --all] [--work DIR] [--dry-run]   re-render the PUBLISHED staging cards from their own info.json + package files (no state, no
                   measurement: a template or serve-rule change such as --parameter_mode=preload); uploads README.md / REPORT.md only (card-only), staging org only
  regen.py status  S.json [--status-json status.json]      table; with --status-json: models.<slug>.<gpu> = stage / verdict / run for the dashboard
  regen.py accept  S.json SLUG... --note TEXT              a person's call: greedy text differs only at a near-tie, decide again with that accepted
  regen.py accept-regression S.json SLUG... --note TEXT   SDK modes: publish a CONFIRMED slowdown as measured (the note names its explanation)
  regen.py adopt   S.json --run RUN                        attach an already running / finished run to the state's jobs (by job name)

Modes. sdk-ab: the SDK migration's A/B (ab_jobs.py --sdk-ab): per model and device ONE job, ABBA on one runner: variant "fun" = the published staging card
  (the staging repo's head at plan time) on its own SDK pin (hrx-loom 0.1.0), variant "staging" = the package built in the job from the job's branch on
  hrx-loom 0.2.0 (devlab_perf.sh per-variant SDK roots), plus the card job's gates and llama.cpp baselines. `collect` writes per model and device the
  % delta new vs old per metric (state: perf = {decode, prefill, longctx, 4stream}, positive = v0.2 faster); a metric more than 3 % slower
  triggers ONE confirming -r2 job, confirmed in both runs = needs_decision with the numbers; `ingest` builds the v0.2 card from the job (its own
  measurement; the delta goes into the publish message); `status --status-json` also writes models.<slug>.perf.<gpu>.
  sdk-loom: the SDK migration's Loom-only sweep: per model and device ONE job that serves the package built in the job on the new SDK (variants staging +
  stagingnd; Loom perf short and long at the card job's session count, c4, func, ktest, gates; NO llama.cpp steps, NO old-card variant). `collect` compares the
  job's decode / prefill / longctx / 4stream rows with the Loom rows of the OLD card's info.json for that device (another job, the same runner type); a
  metric outside +-3 % either way (or missing) = decision "confirm": `rerun` queues ONE targeted ABBA job (the sdk-ab job without baselines: old card on its
  own pin vs the new package, same job) whose verdict decides (slower than 3 % there = needs_decision with the numbers, else promote). State per job:
  perf = the cross-job deltas, the ABBA ones on the confirmed rows, confirmed = {row: ABBA %}. `ingest` builds the device's table from the job's Loom rows
  plus the previous card's llama.cpp rows (copied with their own run and date, never re-measured); the card's methodology line says "different jobs".
  ab: regeneration of a device that already has a package to compare with (ab_jobs.py: old vs new in ABBA sessions, text match, func, ktest, gates,
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
import difflib
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
import sdk_package  # noqa: E402

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
VARIANT = {"ab": ("staging", "stagingnd"), "card": ("card", "nodraft"), "logits": ("staging", "stagingnd"), "sdk-ab": ("staging", "stagingnd"),
           "sdk-loom": ("staging", "stagingnd")}
SDK_MODES = ("sdk-ab", "sdk-loom")   # sdk-loom: Loom only on the new SDK, deltas against the old card's Loom rows (cross-job), ABBA only where a row moves
# sdk-ab: the metric of each reported delta (card_lib.METRIC_KEYS rows) and the ratchet: slower than this fraction on a metric = a confirming run
SDK_METRICS = {"decode": "decode", "prefill": "prefill_8k", "longctx": "decode_32k", "4stream": "streams4"}
SDK_REGRESSION = 0.03
# old side = "card": the pin of the published card (pin-less = 0.1.0; Flash-Next's card is pinned to a 0.2.0 dev build)
SDK_OLD, SDK_NEW = "card", "0.2.0"


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


def new_sdk_pin(pin):
    """True for a package pinned to the SDK this migration moves TO (SDK_NEW or one of its release candidates): every other pin, or none, is an
    old card (0.1.0; Flash-Next's 0.2.0.dev0 CI build is its "previous" SDK)"""
    base = SDK_NEW.split("rc")[0]   # works for SDK_NEW = "0.2.0rc1" and, after the final release, "0.2.0" alike
    return bool(pin) and (pin == base or str(pin).startswith(base + "rc"))


def old_sdk_revision(repo):
    """the newest commit of REPO whose llm_config.json is not pinned to the new SDK (None if there is none)"""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    for c in HfApi().list_repo_commits(repo):
        try:
            pin = (json.load(open(hf_hub_download(repo, "llm_config.json", revision=c.commit_id))).get("sdk") or {}).get("hrx_loom")
        except EntryNotFoundError:   # a commit without llm_config.json (the repo's first commits) is not a package; any other error reaches plan's die
            continue
        if not new_sdk_pin(pin):
            return c.commit_id
    return None


def hf_snapshot(repo, rev, out):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=rev, local_dir=out)
    shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)


def hf_snapshot_small(repo, rev, out):
    """the repo at REV without weights (the package step's snapshot: programs, tokenizer, card; never *.gguf, parameter/)"""
    from huggingface_hub import snapshot_download
    shutil.rmtree(out, ignore_errors=True)
    snapshot_download(repo, revision=rev, local_dir=out, ignore_patterns=["*.gguf", "*.safetensors", "*.bin", "*.mmproj", "parameter/*"])
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
    if st["mode"] in ("ab", "logits", *SDK_MODES):   # logits: ab_jobs.py's job of the pinned staging package, steps cut down in gen_jobs
        cmd = [PY, os.path.join(SCRIPTS, "ab_jobs.py"), "--device", device, "--gates", d["gates"], "--new-org", d["new_org"], "--old-org", d["old_org"]]
        if st["mode"] == "sdk-ab":
            cmd += ["--sdk-ab", "--old-sdk", SDK_OLD, "--new-sdk", SDK_NEW, "--baselines"]   # the v0.2 card's own table needs the llama.cpp rows of the same job
        elif st["mode"] == "sdk-loom":   # the new package alone; a confirming job (suffix -r2) is the sdk-ab ABBA of that model without baselines
            cmd += ["--sdk-ab", "--old-sdk", SDK_OLD, "--new-sdk", SDK_NEW] + ([] if suffix else ["--loom-only"])
        if d.get("refs"):
            cmd += ["--refs", d["refs"]]
        if device != "strix-halo" and not d.get("no_baselines") and st["mode"] == "ab":
            cmd += ["--baselines"]
        if d.get("replicas", 1) > 1 and not suffix:
            cmd += ["--replicas", str(d["replicas"])]   # independent replicas of one submission: the ratchet's confirming run in parallel
        if d.get("sequential"):
            cmd += ["--sequential"]   # 2+2 sessions, the job extends itself to 6+6 near a reject threshold
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
        elif suffix and not j["name"].endswith(suffix) and not j.get("replica"):
            j["name"] = j["name"][:40 - len(suffix)] + suffix
    if st["mode"] == "logits":
        logits_steps(doc, st["opts"].get("bundle"))
    got = {slug_of(j) for j in doc["jobs"]}
    return doc, doc["jobs"], [s for s in slugs if s not in got]


LOGITS_KEEP = ("func", "ktest")


def logits_steps(doc, bundle):
    """The final pass's job of a package: serve it, check it (func, ktest) and run the host-logits step (logits:staging) of devlab_perf.sh.
    Functional checks only: no performance step (the card's rows carry over by stage equivalence, methodology rule 14). The Gufo
    conformance step (gufo:staging) is not part of it: a card claims only the logits readback."""
    for j in doc["jobs"]:
        j["steps"] = [x for x in j["steps"] if x.split(":")[0] in LOGITS_KEEP and x.endswith(":staging")] + ["logits:staging"]
        j["variants"] = [v for v in j["variants"] if v["name"] == "staging"]
        if bundle:   # a prebuilt tarball (URL, SHA256), or {"build": ...}: the devlab-perf run builds it (workflow job `bundle`), nothing is hosted
            j["bundle"] = bundle if isinstance(bundle, dict) else {"url": bundle[0], "sha256": bundle[1]}


def logits_verdict(d, e):
    """Logits mode: func, ktest and logits ran on the package and none failed (a missing bundle is `not supported`, never a pass)."""
    s = json.load(open(os.path.join(d, "summary.json"))).get("results", {})
    need = [k for k in ("func:staging", "ktest:staging", "logits:staging") if k not in s]
    bad = {k: v for k, v in s.items() if str(v).lower().startswith(("fail", "error", "not supported")) or "FAIL" in str(v)[:40]}
    if bad:
        return "hold", [f"{k}: {str(v)[:120]}" for k, v in bad.items()]
    if need:
        return "rerun", ["missing steps: " + ", ".join(need)]
    return "promote", []


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
    if a.mode == "logits" and a.old is None:
        old_org = a.new_org   # one package, checked: old == new in the pinned refs
    if a.mode in SDK_MODES:
        # old = the newest staging revision whose package is still on the OLD SDK (no pin or a 0.1.x pin in llm_config.json: the card as published
        # before the migration; fleet/manifest.json staging.revision can lag it, and once a v0.2 package is uploaded the head is not it any more);
        # new = the head at plan time (the job's card files and the revision the v0.2 package is published on top of)
        if a.old:
            die(f"--mode {a.mode} finds its old package in the staging repo's history (no --old)")
        refs = {}
        for s in slugs:
            repo = f"{a.new_org}/{s}"
            try:
                head, old = hf_head(repo), old_sdk_revision(repo)
            except Exception as x:   # one transient Hub failure: a clear message, not a traceback
                die(f"{repo}: cannot resolve its revisions on the Hub ({x}); rerun plan")
            if not old:
                die(f"{repo}: no revision with an old-SDK package (pin other than {SDK_NEW}) in its history")
            refs[s] = {"old": f"{repo}@{old}", "new": f"{repo}@{head}"}
    elif a.mode in ("ab", "logits") and not a.no_pin:
        # pin the new package of every slug to the staging revision it is measured at (the card the run measures is the card ingested)
        refs = {s: dict(e) for s, e in (refs or {}).items()}
        for s in slugs:
            ent = refs.setdefault(s, {})
            if "new" not in ent:
                ent["new"] = f"{a.new_org}/{s}@{hf_head(f'{a.new_org}/{s}')}"
    if a.mode == "logits" and not (a.bundle or a.bundle_build or a.bundle_run):   # the pinned bundle (.github/devlab/gufo_bundle.json), built in the run
        pins = json.load(open(os.path.join(ROOT, ".github", "devlab", "gufo_bundle.json")))
        a.bundle_build = [pins["extension"].split("@")[1], pins["gufo"].split("@")[1]]
    st = {"schema": "regen/1", "created": now(), "mode": a.mode, "work": work, "state": state_path,
          "opts": {"gates": a.gates, "new_org": a.new_org, "base": a.base, "old_org": old_org, "batch": a.batch, "refs": None,
                   **({"replicas": a.replicas} if a.replicas > 1 else {}), **({"sequential": True} if a.sequential else {}),
                   **({"no_baselines": True} if a.no_baselines else {}), **({"bundle": a.bundle} if a.bundle else {}),
                   **({"bundle": {"build": {"extension": f"loom-fun/loom-gufo-extension@{a.bundle_build[0]}", "gufo": f"loom-fun/gufo@{a.bundle_build[1]}"}}}
                      if a.bundle_build else {}),
                   **({"bundle": {"run": a.bundle_run}} if a.bundle_run else {})},
          "devices": devices, "batches": [], "jobs": {}, "log": []}
    for dev in devices:
        # per-device refs file. Where the device has no published table to compare with (anything but Strix Halo, whose old package is the
        # published loom-fun card) and no old package was named, the old package is the new revision itself: an A/A, the runner's own noise.
        dref = {s: dict(e) for s, e in refs.items()} if refs else None
        if dref and (dev != "strix-halo" or a.mode == "logits") and a.old is None and a.mode not in SDK_MODES:
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
            st["batches"].append({"id": bid, "device": dev, "perf": pj, "jobs": [j["name"] for j in jobs], "round": 1, "stage": "planned",
                                  **({"priority": True} if a.priority else {})})
            first = {}
            for j in jobs:
                e = (dref or {}).get(slug_of(j)) or {}
                rep_no = j.get("replica", 1)
                st["jobs"][j["name"]] = {"slug": slug_of(j), "device": dev, "batch": bid, "round": rep_no, "stage": "planned",
                                         "card": j["card"], "steps": len(j["steps"]),
                                         "aa": a.mode not in SDK_MODES and bool(e.get("old") and e.get("old") == e.get("new")),   # old == new: the A/B shows noise only
                                         **({"old": e["old"]} if a.mode == "sdk-loom" and e.get("old") else {})}   # the old card the cross-job deltas are taken against
                if rep_no == 1:
                    first[slug_of(j)] = j["name"]
            for j in jobs:   # a replica is a confirming run submitted with the first: the same links `rerun` makes (prev, rerun_of_me)
                if j.get("replica", 1) > 1:
                    f = first[slug_of(j)]
                    st["jobs"][j["name"]].update(prev=f, replica=j["replica"])
                    st["jobs"][f]["rerun_of_me"] = j["name"]
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
    base = st.d["opts"].get("base") or "refactor"   # states planned before --base existed: the branch they were submitted from
    tag = re.sub(r"[^a-z0-9]+", "-", os.path.basename(st.path)[:-5].lower()).strip("-")
    branch = f"regen-{tag}-b{b['id']}" + (f"-r{b['round']}" if b["round"] > 1 else "")
    with open(b["perf"], "rb") as fh:
        perf = fh.read()
    title = f"Experiment: regen {tag} {b['device']} batch {b['id']}" + (f" confirming run {b['round']}" if b["round"] > 1 else "")
    if dry:
        print(f"[dry-run] branch {branch} from {base}, perf.json ({len(perf)} bytes, {len(b['jobs'])} jobs), PR '{title}', label devlab-perf")
        return
    sha = gh_api("GET", f"repos/{REPO}/git/ref/heads/{base}", jq=".object.sha")
    gh_api("POST", f"repos/{REPO}/git/refs", {"ref": f"refs/heads/{branch}", "sha": sha})
    f = "repos/%s/contents/.github/devlab/perf.json" % REPO
    fsha = gh_api("GET", f"{f}?ref={branch}", jq=".sha")
    gh_api("PUT", f, {"message": f"devlab perf: {title}\n\nCo-Authored-By: Claude <noreply@anthropic.com>", "branch": branch,
                      "content": base64.b64encode(perf).decode(), "sha": fsha})
    pr = gh_api("POST", f"repos/{REPO}/pulls", {"title": title, "head": branch, "base": base, "body": batch_body(st, b)}, jq=".number")
    gh_api("POST", f"repos/{REPO}/issues/{pr}/labels", {"labels[]": "devlab-perf"})   # the label starts the run
    b.update(branch=branch, pr=int(pr), stage="submitted", submitted=now())
    for n in b["jobs"]:
        st.jobs[n].update(stage="submitted", pr=int(pr))
    st.log(f"batch {b['id']} ({b['device']}, {len(b['jobs'])} jobs) submitted: PR #{pr}")
    st.save()


def max_inflight(opt, device):
    """The wave size for submit --max-inflight: None/0 = no cap (the default: all planned batches at once, as before), N, or 'auto' = the device's
    runner count (device.json runner.concurrent_runners; a device without one is elastic: no cap)."""
    if opt in (None, "0", 0):
        return 0
    if opt == "auto":
        return int(card_lib.registry.device(device)["runner"].get("concurrent_runners") or 0)
    return int(opt)


def cmd_submit(a):
    st = State(a.state)
    todo = [b for b in st.d["batches"] if b["stage"] == "planned"]
    todo.sort(key=lambda b: (not b.get("priority"), device_order(b["device"]), b["id"]))   # priority batches, then R9700-class (few runners): their queue is the long one
    if a.max_batches:
        todo = todo[:a.max_batches]
    if not todo:
        print("nothing to submit")
    flying = {}   # device -> jobs of this state submitted and not yet finished (the wave in the queue)
    for e in st.jobs.values():
        if e["stage"] in ("submitted", "queued", "running"):
            flying[e["device"]] = flying.get(e["device"], 0) + 1
    for b in todo:
        cap = max_inflight(a.max_inflight, b["device"])
        if cap and not b.get("priority") and flying.get(b["device"], 0) and flying.get(b["device"], 0) + len(b["jobs"]) > cap:
            print(f"batch {b['id']} ({b['device']}, {len(b['jobs'])} jobs) waits for the next wave: {flying[b['device']]} in flight, cap {cap}")
            continue
        submit_batch(st, b, a.dry_run)
        flying[b["device"]] = flying.get(b["device"], 0) + len(b["jobs"])


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


def prev_cmp(st, e):
    """the compare of the first run (a round-2 job, or a replica) when it has one"""
    return e.get("prev") and st.jobs[e["prev"]].get("cmp")


def compare(st, name, e, d):
    """strix_ab_compare.py of an A/B job -> cmp.json; the round-2 compare reads --prev (the first run's)."""
    cmp = os.path.join(st.work, "cmp", f"{name}.json")
    os.makedirs(os.path.dirname(cmp), exist_ok=True)
    cmd = [PY, os.path.join(SCRIPTS, "strix_ab_compare.py"), d, "--json", cmp]
    if st.d["mode"] in SDK_MODES:   # the old side IS the staging card at its pinned (old-SDK) revision: the runner guard compares it with that card's value on this device
        ref = e["card"] if st.d["mode"] == "sdk-ab" else (e.get("old") or e["card"])   # sdk-loom: the card's Loom rows are the old side of the cross-job delta
        rev = ref.split("@")[-1] if "@" in ref else None
        cmd += ["--fun-info", hf_file(ref.split("@")[0], "info.json", rev), "--gpu", e["device"]]
    elif e["device"] == "strix-halo" and st.d["opts"]["old_org"] == "loom-fun":
        cmd += ["--fun-info", hf_file(f"{st.d['opts']['old_org']}/{e['slug']}", "info.json")]   # new vs the published card value too
    elif e["device"] != "strix-halo":   # R9700: the old side IS the staging card; the runner guard compares it with that card's value
        cmd += ["--fun-info", hf_file(f"{st.d['opts']['old_org']}/{e['slug']}", "info.json"), "--gpu", e["device"]]
    if prev_cmp(st, e) and st.d["mode"] != "sdk-loom":   # sdk-loom: the confirming job is judged alone (its verdict decides), not against the first run
        cmd += ["--prev", prev_cmp(st, e)]
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


def sdk_deltas(cmp, cross=False):
    """{decode, prefill, longctx, 4stream: % new vs old (positive = v0.2 faster)} of a compare, None where a row is missing; plus every row's %.
    cross: the old side is the old card's published Loom value (another job), the compare's new_vs_card, not an ABBA ratio"""
    m = cmp.get("metrics") or {}
    key = "new_vs_card" if cross else "ratio"
    pct = lambda r: round((r - 1) * 100, 1) if r else None
    return ({k: pct((m.get(row) or {}).get(key)) for k, row in SDK_METRICS.items()},
            {row: pct(x.get(key)) for row, x in m.items()})


def sdk_decide(st, name, e, cmp_path, use_prev=True):
    """The SDK A/B ratchet. Round 1: a gate failure holds; a metric more than 3 % slower (or missing) or a degraded runner asks for ONE confirming -r2 job;
    else promote. Round 2 (e["prev"]): a metric slower than 3 % in BOTH runs is a confirmed regression = hold with the numbers (needs_decision);
    otherwise promote. The reported perf of a confirmed pair is the mean of the two runs. Greedy text differences are reported, not held (v0.2 is another SDK).
    Returns (decision, reasons, perf, rows)."""
    cmp = json.load(open(cmp_path))
    perf, rows = sdk_deltas(cmp)
    reasons = []
    if cmp.get("gate_failures"):
        return "hold", ["gate failed: " + ", ".join(cmp["gate_failures"])], perf, rows
    if (cmp.get("text") or {}).get("status") == "DIFF":
        t = cmp["text"]
        reasons.append(f"greedy text differs from v0.1 on {t['prompts'] - t['identical']}/{t['prompts']} prompts (reported)")
    if cmp.get("degraded_runner"):
        return ("hold" if e["round"] > 1 else "rerun"), [f"degraded runner (old side off the card value): {cmp['degraded_runner']}"] + reasons, perf, rows
    # decode, prefill and 4stream must be measured on both sides; only the long-context row may be absent on BOTH (beyond the context both serve
    # on this device, e.g. a 32k row on a 32k card): that is "not measured", not missing. A failed perf step on both sides still reruns.
    m = cmp.get("metrics") or {}
    gone = [k for k, v in perf.items() if v is None and (k != "longctx" or (m.get(SDK_METRICS[k]) or {}).get("old") is not None)]
    newonly = [k for k, v in perf.items() if v is None and (m.get(SDK_METRICS[k]) or {}).get("old") is None and (m.get(SDK_METRICS[k]) or {}).get("new") is not None]
    if newonly:
        reasons.append(f"served only on the new SDK (no old value to compare): {', '.join(newonly)}")
    prev = st.jobs[e["prev"]] if e.get("prev") and use_prev else None
    if prev and prev.get("perf"):
        p1 = prev["perf"]
        both = {k: (p1[k], perf[k]) for k in perf if p1.get(k) is not None and perf[k] is not None and p1[k] < -SDK_REGRESSION * 100 and perf[k] < -SDK_REGRESSION * 100}
        perf = {k: (round((p1[k] + perf[k]) / 2, 1) if p1.get(k) is not None and perf[k] is not None else perf[k]) for k in perf}
        if gone:   # the confirming run measured too little to decide: hold with the reason (never a promote on missing rows)
            return "hold", [f"missing in the confirming run: {', '.join(gone)}"] + reasons, perf, rows
        if both:
            return "hold", [f"confirmed regression > {SDK_REGRESSION:.0%} in both runs: " + ", ".join(f"{k} {a:+.1f} % / {b:+.1f} %" for k, (a, b) in both.items())] + reasons, perf, rows
        return "promote", reasons, perf, rows
    slow = {k: v for k, v in perf.items() if v is not None and v < -SDK_REGRESSION * 100}
    if slow or gone:
        return "rerun", ([f"slower than -{SDK_REGRESSION:.0%}: " + ", ".join(f"{k} {v:+.1f} %" for k, v in slow.items())] if slow else []) + \
            ([f"missing: {', '.join(gone)}"] if gone else []) + reasons, perf, rows
    return "promote", reasons, perf, rows


def sdk_loom_decide(st, name, e, cmp_path):
    """Round 1 of --mode sdk-loom: the job measured the new package alone; each metric (SDK_METRICS rows) is compared with the Loom value of the OLD card's
    info.json for this device (another job, the same runner type). A gate / func / ktest failure holds. A metric outside +-SDK_REGRESSION either way, or
    one the old card has and the job did not measure, asks for ONE targeted ABBA job (decision "confirm", e["confirm_rows"]); else promote. A metric the old
    card has no value for is reported, not confirmed (nothing to compare with). Returns (decision, reasons, perf, rows)."""
    cmp = json.load(open(cmp_path))
    perf, rows = sdk_deltas(cmp, cross=True)
    if cmp.get("gate_failures"):
        return "hold", ["gate failed: " + ", ".join(cmp["gate_failures"])], perf, rows
    m = cmp.get("metrics") or {}
    limit = SDK_REGRESSION * 100
    gone = [k for k, v in perf.items() if v is None and (m.get(SDK_METRICS[k]) or {}).get("card") and not (m.get(SDK_METRICS[k]) or {}).get("new")]
    nocard = [k for k, v in perf.items() if v is None and (m.get(SDK_METRICS[k]) or {}).get("new") and not (m.get(SDK_METRICS[k]) or {}).get("card")]
    moved = {k: v for k, v in perf.items() if v is not None and abs(v) > limit}
    if not any((m.get(r) or {}).get("new") for r in SDK_METRICS.values()):
        return "hold", ["the job measured none of decode, prefill, longctx, 4stream"], perf, rows
    reasons = []
    if nocard:
        reasons.append(f"no value on the old card to compare with: {', '.join(nocard)}")
    e["confirm_rows"] = sorted(set(moved) | set(gone))
    if e["confirm_rows"]:
        return "confirm", [f"outside +-{SDK_REGRESSION:.0%} of the old card's Loom row (another job): "
                           + ", ".join(f"{k} {moved[k]:+.1f} %" if k in moved else f"{k} missing" for k in e["confirm_rows"])] + reasons, perf, rows
    return "promote", reasons, perf, rows


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
    if mode == "sdk-ab":
        e["cmp"] = compare(st, name, e, d)
        decision, reasons, e["perf"], e["rows_pct"] = sdk_decide(st, name, e, e["cmp"])
    elif mode == "sdk-loom" and not e.get("prev"):   # the sweep job: cross-job deltas against the old card's Loom rows
        e["cmp"] = compare(st, name, e, d)
        decision, reasons, e["perf"], e["rows_pct"] = sdk_loom_decide(st, name, e, e["cmp"])
        e["perf_cross"] = dict(e["perf"])
    elif mode == "sdk-loom":   # the targeted ABBA job: old card on its pin vs the new package, same job; its verdict decides
        e["cmp"] = compare(st, name, e, d)
        decision, reasons, e["perf_abba"], e["rows_pct"] = sdk_decide(st, name, e, e["cmp"], use_prev=False)
        if decision == "rerun":   # a confirming job is never asked to repeat: slower than the band, missing rows or a degraded runner stop the model
            decision = "hold"
        sdk_loom_settle(st, e, decision, reasons)
    elif mode == "ab":
        e["cmp"] = compare(st, name, e, d)
        paths = ([st.jobs[e["prev"]]["cmp"]] if prev_cmp(st, e) else []) + [e["cmp"]]
        decision, reasons, res = decide(st, name, e, paths)
        e["rows"] = {r: x["state"] for r, x in res["rows"].items()}
        e["improved"] = [r for r, x in res["rows"].items() if "improved" in x["verdicts"] and x["state"] != "rerun"]
    elif mode == "logits":
        decision, reasons = logits_verdict(d, e)
    else:
        decision, reasons = card_mode_verdict(d, e)
    e["decision"], e["reason"] = decision, "; ".join(reasons)
    if mode in SDK_MODES and e.get("perf"):
        e["reason"] = e["reason"] or "perf " + " ".join(f"{k} {v:+.1f}%" for k, v in e["perf"].items() if v is not None)
    e["card_revision"] = read(os.path.join(d, "card_revision.txt"), "").split("\n")[0]
    if decision == "promote":
        e["stage"] = "collected"
    elif decision in ("rerun", "improved-pending", "confirm"):
        e["stage"] = "needs_decision" if e["round"] > 1 else "collected"   # round 1: `rerun` queues the confirming job; round 2 undecided: a person decides
        if e["round"] > 1:
            e["reason"] = "still undecided after the confirming run: " + e["reason"]
    else:
        e["stage"] = "needs_decision"
        e["reason"] = "confirmed regression / failed gate / text difference: " + e["reason"]


def sdk_loom_settle(st, e, decision, reasons):
    """sdk-loom: the ABBA job e answers its first job's question. promote: the first job's rows that moved take the ABBA deltas (state: confirmed =
    {row: ABBA %}, perf = the cross-job deltas with those rows replaced) and the first job promotes; hold: the first job stops with the ABBA numbers."""
    f = st.jobs[e["prev"]]
    abba = e.get("perf_abba") or {}
    rows = f.get("confirm_rows") or []
    if decision == "promote":
        f["confirmed"] = {k: abba[k] for k in rows if abba.get(k) is not None}
        f["perf"] = {k: (f["confirmed"][k] if k in f["confirmed"] else v) for k, v in (f.get("perf_cross") or f.get("perf") or {}).items()}
        f.update(decision="promote", reason="confirmed by ABBA job " + e["run"] + ": " + ", ".join(f"{k} {v:+.1f} %" for k, v in f["confirmed"].items()))
    else:
        f.update(stage="needs_decision", reason="ABBA confirmation (job " + str(e.get("run")) + "): " + "; ".join(reasons)
                 + " (ABBA " + ", ".join(f"{k} {abba[k]:+.1f} %" for k in rows if abba.get(k) is not None) + ")")
    e.pop("perf", None)


def cmd_collect(a):
    st = State(a.state)
    n = 0
    for name, e in list(st.jobs.items()):
        if e["stage"] not in ("done", "failed"):
            continue
        if e.get("prev") and st.jobs[e["prev"]]["stage"] in ("planned", "submitted", "queued", "running", "done", "failed"):
            continue   # a replica is judged with its first run: that one is collected first
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
        if e["stage"] == "collected" and e.get("decision") in ("rerun", "improved-pending", "confirm") and e["round"] == 1 and not e.get("rerun_of_me"):
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
                                  "prev": first[s], "aa": st.jobs[first[s]].get("aa", False),
                                  **({"old": st.jobs[first[s]]["old"]} if st.jobs[first[s]].get("old") else {})}
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


def sdk_line(e):
    """the delta line of a sdk-ab job: 'decode +1.2 %, prefill -0.4 %, ...' (positive = v0.2 faster)"""
    return ", ".join(f"{k} {v:+.1f} %" if v is not None else f"{k} n/a" for k, v in (e.get("perf") or {}).items())


def card_message(e, dev, run, loom_only, sdk):
    """the publish message of a card made from a job: where the numbers come from, the SDK delta line, an accepted slowdown's explanation"""
    msg = f"{card_lib.GPUS[dev]['runner']} results measured on this package (run {run})"
    if loom_only:
        msg += (f"; hrx-loom {SDK_NEW} Loom rows vs the previous card's Loom rows (another job, same runner type): " + sdk_line(e)
                + ("; confirmed by an ABBA job (same job): " + ", ".join(f"{k} {v:+.1f} %" for k, v in e["confirmed"].items()) if e.get("confirmed") else "")
                + "; llama.cpp rows are the previous card's")
    elif sdk:
        msg += f"; hrx-loom {SDK_NEW} vs the previous card on its own SDK pin (same job, ABBA): " + sdk_line(e)
    if e.get("accepted_regression"):   # a confirmed slowdown published as measured: the message names its explanation
        msg += "; confirmed slowdown published as measured: " + e["accepted_regression"].split(" (was:")[0]
    if e.get("accepted_gate_failure"):   # kernel tests that failed: the card's gate table shows them failed; the message says why it was published
        msg += "; " + e["accepted_gate_failure"]
    return msg


def ingest_full(st, name, e, W):
    """The card's performance, gates and roofline of this device from the job (REGEN_RECIPE: ingest the NEW package)."""
    d, dev, slug = e["dir"], e["device"], e["slug"]
    V, ND = VARIANT[st.d["mode"]]
    S = CARD
    sdk = st.d["mode"] in SDK_MODES
    loom_only = st.d["mode"] == "sdk-loom"
    # sdk-ab: the job measured a package BUILT from the branch; the card is made on the staging head, which must hold exactly those programs (the
    # v0.2 package published before this card-only step): make_info.py refuses a different programs hash
    rev = hf_head(f"{st.d['opts']['new_org']}/{slug}") if sdk else e["card_revision"].split("@")[-1]
    out = os.path.join(W, "out")
    hf_snapshot(f"{st.d['opts']['new_org']}/{slug}", rev, out)
    if sdk:   # the package step's programs, whatever the head's revision is now (another device's card may have been published since)
        got = card_lib.programs_sha256(out)
        if got != e["package"]["programs_sha256"]:
            raise RuntimeError(f"the staging head {rev[:10]} holds programs {str(got)[:12]}, not the published v0.2 package's {e['package']['programs_sha256'][:12]}: "
                               f"another publish replaced them; run `regen.py package` again")
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
    vl, c4vl = g("perf_vllm_n*.jsonl"), g("c4_vllm_n*.txt")   # a safetensors card: vLLM ROCm is the baseline engine (devlab_perf.sh vllm: steps)
    st_card = (json.load(open(os.path.join(ROOT, "fleet", slug, "llm_config.json"))).get("source") or {}).get("kind") == "safetensors"
    if loom_only:   # the job measured Loom alone: the baseline rows (llama.cpp ROCm and Vulkan, or vLLM) are the previous card's, with their own run and date
        run_tool([f"{S}/make_info.py", "perf", "--gpu", dev, "--loom-jsonl", f"{d}/perf_{V}.jsonl", "--c4-loom", f"{d}/c4_{ND}.txt",
                  "--baseline-from-info", "info_prev.json", "--package", out, *src, "--device", "dev.json", "--hrx-loom", "hrx.json",
                  "--fetched-programs", rd("programs_sha256_staging.txt"), "-o", "ev_perf.json"], W)
    else:
        if not st_card and not (llm and c4l):
            raise RuntimeError("the job has no llama.cpp ROCm files (a job without --baselines cannot make this device's table)")
        baseline = card_lib.registry.methodology()["baselines"][dev]
        perf = [f"{S}/make_info.py", "perf", "--gpu", dev, "--loom-jsonl", f"{d}/perf_{V}.jsonl", "--c4-loom", f"{d}/c4_{ND}.txt", "--package", out, *src,
                "--device", "dev.json", "--hrx-loom", "hrx.json", "--fetched-programs", rd("programs_sha256_staging.txt" if sdk else "programs_sha256.txt"),
                "-o", "ev_perf.json"]
        if st_card:   # no llama.cpp: the baseline column is vLLM ROCm on the same runner when the job measured it, else none (the card says so)
            c4vl = [f for f in c4vl if "agg_tps" in open(f).read()]   # a c4 file without a rate (the engine served too few streams) is not a baseline
            if vl:
                perf += ["--llama-jsonl", *vl, *(["--c4-llama", c4vl[0]] if c4vl else []), "--baseline-name", "vLLM",
                         "--baseline-version", "vllm/" + rd("vllm_version.txt").strip(), "--baseline-args", rd("vllm_args_on.txt")]
                if os.path.exists(os.path.join(d, "vllm_note.txt")):   # rows vLLM could not serve on this GPU (never silent)
                    kv = (" (vLLM reports: " + "; ".join(x for x in rd("vllm_kv_on.txt").split("\n") if x.strip()) + ")") if os.path.exists(os.path.join(d, "vllm_kv_on.txt")) else ""
                    perf += ["--differs", rd("vllm_note.txt").strip().rstrip(".") + kv]
                if os.path.exists(os.path.join(d, "vllm_note_c4.txt")):
                    perf += ["--differs", rd("vllm_note_c4.txt").strip()]
            else:
                why = ("the job's vLLM step failed: " + rd("vllm_error.txt").strip()[:300]) if os.path.exists(os.path.join(d, "vllm_error.txt")) \
                    else "not measured on this card yet (llama.cpp cannot read the safetensors checkpoint)"
                perf += ["--baseline-name", "vLLM", "--differs", why]
        else:
            perf += ["--llama-jsonl", *llm, "--c4-llama", c4l[0], "--baseline-version", "llamacpp-rocm/" + baseline["rocm"]["build"],
                     "--baseline-args", rd("llamacpp_args-rocm.np1.txt")]
        run_tool(perf, W)
    ev = ["ev_perf.json"]
    if vk and c4v and not loom_only and not st_card:
        run_tool([f"{S}/make_info.py", "perf", "--gpu", dev, "--backend", "vulkan", "--llama-jsonl", *vk, "--c4-llama", c4v[0],
                  "--baseline-version", baseline["vulkan"]["build"], "--baseline-args", rd("llamacpp_args-vulkan.np1.txt"),
                  "--baseline-sha256", rd("llamacpp_release-vulkan.np1.sha256"), "--fetched-programs", rd("programs_sha256_staging.txt" if sdk else "programs_sha256.txt"), "--package", out,
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
    # --from-info keeps only DevLab-measured sessions: another GPU's published table (Strix Halo's loom-fun numbers) would be dropped, so every
    # other device's performance is taken from the head card as published
    keep = [f"{g}=info_prev.json" for g, x in (prev.get("devices") or {}).items() if g != dev and (x or {}).get("performance")]
    for spec in keep:
        build += ["--perf-from", spec]
    # sdk-loom: the previous card's Loom sessions are the v0.1 rows this table replaces (its llama.cpp rows were copied into the new session above;
    # the Vulkan baselines stay as published)
    run_tool(build + ["--memory-bytes", str(mem), "--from-info", "info_prev.json", *(["--reset-performance"] if loom_only else []), "--evidence", *ev,
                      "-o", "info_new.json"], W)
    shutil.copy(os.path.join(W, "info_new.json"), os.path.join(out, "info.json"))
    msg = card_message(e, dev, run, loom_only, sdk)
    return out, hf_head(f"{st.d['opts']['new_org']}/{slug}"), msg


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


def job_programs(e):
    """the programs_sha256 the job BUILT (devlab_perf.sh programs_sha256_staging.txt), or None when the job directory has none"""
    p = os.path.join(e.get("dir") or "", "programs_sha256_staging.txt")
    words = open(p).read().split() if os.path.isfile(p) else []
    return words[0] if words else None


def sdk_ingest_ready(st, name, e):
    """sdk-ab: the v0.2 package must be published (`package`) with the programs the job measured; else this model stops with the reason"""
    pkg, built = e.get("package"), job_programs(e)
    why = None
    if not pkg:
        why = "the v0.2 package is not published yet: run `regen.py package` first"
    elif built and built != pkg["programs_sha256"]:
        why = f"the published v0.2 package's programs ({pkg['programs_sha256'][:12]}) are not the ones the job built and measured ({built[:12]})"
    if why:
        e.update(stage="needs_decision", reason=f"ingest: {why}")
        st.log(f"{e['slug']} {e['device']}: ingest stopped: {why}")
        st.save()
    return why is None


def cmd_ingest(a):
    st = State(a.state)
    if st.d["mode"] == "logits":
        die("a logits-mode state has no performance to ingest: fleet/final_pass.py card builds the final cards from it")
    todo = [(n, e) for n, e in st.jobs.items() if e["stage"] == "collected" and e.get("decision") == "promote"]
    # a model with a round-2 job: the second run ingests, the first is superseded
    if st.d["mode"] == "sdk-loom":   # the sweep job's rows go on the card (its ABBA confirmation settled the first job's decision)
        todo = [(n, e) for n, e in todo if not e.get("prev")]
    else:
        todo = [(n, e) for n, e in todo if not e.get("rerun_of_me")]
    if a.slug:
        todo = [(n, e) for n, e in todo if e["slug"] in a.slug]
    if st.d["mode"] in SDK_MODES:
        todo = [(n, e) for n, e in todo if sdk_ingest_ready(st, n, e)]
    for name, e in todo:
        W = os.path.join(st.work, "cards", f"{e['slug']}.{e['device']}")
        shutil.rmtree(W, ignore_errors=True)
        os.makedirs(W)
        try:
            full = st.d["mode"] in ("card", *SDK_MODES) or bool(glob.glob(os.path.join(e["dir"], "perf_llamacpp-rocm.np1_n*.jsonl")))
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


# ---------------------------------------------------------------- package (sdk-ab: the v0.2 programs to staging, before the card)

def resolve_ref(ref):
    r = sh(["git", "rev-parse", "--verify", "-q", ref], check=False, cwd=ROOT)
    return r.stdout.strip() if r.returncode == 0 else None


def canonical_build(st, slugs, ref):
    """programs-only build of REF for the slugs (CPU, once for all of them; reused while the ref's commit is the same). -> (dir, commit)"""
    sha = resolve_ref(ref)
    if not sha:
        die(f"--ref {ref}: not a git ref here (git fetch origin first)")
    canon = os.path.join(st.work, "canon")
    todo = [s for s in slugs if not (os.path.isfile(os.path.join(canon, s, ".ref")) and open(os.path.join(canon, s, ".ref")).read() == sha
                                      and os.path.isfile(os.path.join(canon, s, "model.json")))]
    if todo:
        r = sh([PY, os.path.join(HERE, "build_fleet.py"), "--ref", sha, "--out", canon, "--with-gen", "--slugs", ",".join(todo)], check=False)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
        for s in todo:
            if os.path.isfile(os.path.join(canon, s, "model.json")) and any(f.endswith(".loom") for f in os.listdir(os.path.join(canon, s, "program"))):
                open(os.path.join(canon, s, ".ref"), "w").write(sha)
    return canon, sha


def package_one(st, slug, canon, sha, dry, env):
    """One model: snapshot the staging repo, assemble the v0.2 package, upload it. -> {revision, programs_sha256, base} (revision None in a dry run)"""
    org = st.d["opts"]["new_org"]
    repo = f"{org}/{slug}"
    want = card_lib.programs_sha256(os.path.join(canon, slug))
    if not want:
        raise RuntimeError("the canonical build has no programs")
    W = os.path.join(st.work, "package", slug)
    # the snapshot is the repo's head (the plan pinned the A/B's old side to the head at plan time; card commits may have landed since): the
    # package replaces the programs, and --parent-commit = the head makes a concurrent publish fail the upload instead of being overwritten
    rev = hf_head(repo)
    snap, out = os.path.join(W, "snap"), os.path.join(W, "pkg")
    hf_snapshot_small(repo, rev, snap)
    alts = {card_lib.gpu_preset_name(sdk_package.load(os.path.join(snap, "model.json")), g) for g in card_lib.GPUS if card_lib.GPUS[g].get("preset_suffix")}
    changes = sdk_package.assemble(snap, os.path.join(canon, slug), out, alts)
    got = card_lib.programs_sha256(out)
    if got != want:
        raise RuntimeError(f"assembled package programs {str(got)[:12]} differ from the canonical build's {want[:12]}")
    if sdk_package.same_files(snap, out):   # the head already holds exactly this package (programs AND llm_config pin, model.json, ...)
        return {"revision": rev, "programs_sha256": want, "base": sha, "already": True}
    msg = f"Loom package rebuilt for hrx-loom {SDK_NEW} from loom-skills {sha[:10]} (card files unchanged until the card step)"
    r = sh([PY, f"{CARD}/publish_package.py", "upload", out, "--repo", slug, "--org", org, "--parent-commit", rev, "--message", msg]
           + (["--dry-run"] if dry else []), check=False, env=env)
    if r.returncode:
        raise RuntimeError(f"upload: {(r.stdout + r.stderr)[-400:]}")
    print(f"{slug}: {r.stdout.strip().splitlines()[-1][:300]}\n   programs_sha256 {got}; model.json added by v0.2: {', '.join(p for _, p in changes) or 'nothing'}")
    return {"revision": None if dry else hf_head(repo), "programs_sha256": got, "base": sha}


def cmd_package(a):
    st = State(a.state)
    if st.d["mode"] not in SDK_MODES:
        die("package is the sdk-ab / sdk-loom step (other modes publish no package)")
    org = st.d["opts"]["new_org"]
    if org != STAGING_ORG or "/" in org.strip("/"):
        die(f"package writes {STAGING_ORG} only")
    ok = ("collected", "ingested", "published")
    if a.slugs:   # a named model is packaged whatever its stage (a model waiting for a confirming run, a dry run)
        picked = [(n, e) for n, e in st.jobs.items() if e["slug"] in a.slugs and e["stage"] not in ("skipped", "unsupported")]
    else:
        picked = [(n, e) for n, e in st.jobs.items() if e["stage"] in ok and e.get("decision") == "promote"]
    slugs = sorted({e["slug"] for _, e in picked})
    missing = sorted(set(a.slugs or ()) - set(slugs))
    if missing:
        die(f"no job in the state for: {', '.join(missing)}")
    if not slugs:
        return print("nothing to package")
    canon, sha = canonical_build(st, slugs, a.ref or ("origin/" + st.d["opts"]["base"] if resolve_ref("origin/" + st.d["opts"]["base"]) else st.d["opts"]["base"]))
    env = dict(os.environ, PATH=LOOMBIN + os.pathsep + os.environ["PATH"])
    for slug in slugs:
        jobs = [(n, e) for n, e in st.jobs.items() if e["slug"] == slug]
        if not a.dry_run and all(e.get("package") for _, e in jobs):
            print(f"{slug}: package already published ({jobs[0][1]['package']['revision'][:10]})")
            continue
        try:
            if not os.path.isfile(os.path.join(canon, slug, ".ref")):
                raise RuntimeError("canonical build failed: " + open(os.path.join(canon, slug, "build.log")).read()[-300:])
            res = package_one(st, slug, canon, sha, a.dry_run, env)
            for _, e in picked:   # a job that measured other programs than the canonical build's must not be ingested against them
                built = job_programs(e) if e["slug"] == slug else None
                if built and built != res["programs_sha256"]:
                    raise RuntimeError(f"the job {e['device']} built programs {built[:12]}, the canonical build of {sha[:10]} has {res['programs_sha256'][:12]} (the base moved since the job was submitted)")
        except (RuntimeError, SystemExit, ValueError) as ex:
            print(f"{slug}: package stopped: {str(ex)[-400:]}")
            if not a.dry_run:
                for _, e in jobs:
                    e.update(stage="needs_decision", reason=f"package: {str(ex)[-300:]}")
                st.log(f"{slug}: package stopped")
                st.save()
            continue
        if not a.dry_run:
            for _, e in jobs:
                e["package"] = {**res, "at": now()}
                if str(e.get("reason", "")).startswith("package:"):
                    e.pop("reason")
            st.log(f"{slug}: v0.2 package at {res['revision'][:10]} (programs {res['programs_sha256'][:12]})")
            st.save()


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


# ---------------------------------------------------------------- rerender

def cmd_rerender(a):
    """Re-render the published staging cards: per slug fetch the repo's head without weights (model.json, lock, llm_config.json, program/, info.json),
    render README.md / REPORT.md with the current render_card.py, and (unless --dry-run) upload only the files that changed (card-only, parent = the
    fetched head). Nothing is measured: info.json is the card's own."""
    slugs = fleet_slugs() if a.all else a.slugs
    if not slugs:
        die("rerender needs --slugs S... or --all")
    bad = [s for s in slugs if s not in set(fleet_slugs())]
    if bad:
        die(f"rerender: not fleet models: {' '.join(bad)}")
    work = os.path.abspath(a.work)
    env = dict(os.environ, PATH=LOOMBIN + os.pathsep + os.environ["PATH"])
    changed = failed = 0
    for slug in slugs:
        repo = f"{STAGING_ORG}/{slug}"
        d = os.path.join(work, slug)
        if os.path.exists(d):
            die(f"{d} exists: rerender works in a fresh --work directory (it deletes each card's folder when done)")
        try:
            head = hf_head(repo)
            hf_snapshot_small(repo, head, d)
            rd = lambda f: open(os.path.join(d, f), encoding="utf-8", newline="").read() if os.path.isfile(os.path.join(d, f)) else None
            before = {f: rd(f) for f in ("README.md", "REPORT.md")}
            r = sh([PY, f"{CARD}/render_card.py", d], check=False, env=env)
            if r.returncode:
                raise RuntimeError("render failed: " + (r.stderr or r.stdout)[-300:])
            now_ = {f: rd(f) for f in before}
            diff = [f for f in before if before[f] != now_[f]]
            if not diff:
                print(f"{slug}: unchanged")
                continue
            changed += 1
            if a.dry_run:
                print(f"{slug}: would upload {', '.join(diff)}")
                for f in diff:   # the changed lines, to review before the real run (template changes since the card was published come along)
                    for ln in difflib.unified_diff((before[f] or "").splitlines(), (now_[f] or "").splitlines(), f"{f} (published)", f"{f} (rerendered)", n=0, lineterm=""):
                        print("    " + ln[:220])
                continue
            r = sh([PY, f"{CARD}/publish_package.py", "upload", d, "--repo", slug, "--org", STAGING_ORG, "--card-only", "--parent-commit", head,
                    "--message", "Card: serve line re-rendered (no re-measurement)"], check=False, env=env)
            if r.returncode:
                raise RuntimeError("upload failed: " + (r.stdout + r.stderr)[-300:])
            print(f"{slug}: uploaded {', '.join(diff)}")
        except Exception as e:   # one model's failure does not stop the others
            failed += 1
            print(f"{slug}: FAILED {e}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print(f"{len(slugs)} cards: {changed} {'would change' if a.dry_run else 'changed'}, {failed} failed")
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------- status

def cmd_status(a):
    st = State(a.state)
    print_table(st)
    for n, e in sorted(st.jobs.items(), key=lambda kv: (kv[1]["device"], kv[1]["slug"])):
        print(f"  {e['slug'][:46]:46s} {e['device']:13s} r{e['round']} {e['stage']:14s} {e.get('decision') or '':16s} {(e.get('reason') or '')[:70]}")
        if e.get("perf"):
            print(f"  {'':46s} {'':13s} {'':2s} perf v0.2 vs v0.1: {sdk_line(e)}"
                  + (f"  (ABBA-confirmed: {', '.join(sorted(e['confirmed']))})" if e.get("confirmed") else ""))
    if a.status_json:
        write_status(st, a.status_json)


def write_status(st, path):
    """models.<slug>.<gpu> = {stage, verdict, run}: what the dashboard reads (a round-2 job's entry wins over its first run's); sdk-ab also
    models.<slug>.perf.<gpu> = {decode, prefill, longctx, 4stream} in percent, positive = v0.2 faster."""
    with locked(path):
        d = json.load(open(path))
        for n, e in sorted(st.jobs.items(), key=lambda kv: kv[1]["round"]):
            m = d.setdefault("models", {}).setdefault(e["slug"], {})
            verdict = {"promote": "pass", "improved-pending": "rerun", "hold": "regressed"}.get(e.get("decision") or "", e.get("decision"))
            if e["stage"] == "skipped":
                verdict = "unsupported"
            m[GPU_KEY.get(e["device"], e["device"])] = {"stage": STATUS_STAGE.get(e["stage"], "queued"), "verdict": verdict, "run": e.get("run")}
            if e.get("perf"):   # sdk-ab: % delta v0.2 vs v0.1 per metric (positive = v0.2 faster), mean of the two runs of a confirmed pair
                m.setdefault("perf", {})[GPU_KEY.get(e["device"], e["device"])] = dict(e["perf"])
                if e.get("confirmed"):   # sdk-loom: the rows whose delta is the ABBA job's (same job), the others are cross-job against the old card
                    m.setdefault("perf_confirmed", {})[GPU_KEY.get(e["device"], e["device"])] = sorted(e["confirmed"])
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
            paths = ([st.jobs[e["prev"]]["cmp"]] if prev_cmp(st, e) else []) + [e["cmp"]]
            decision, reasons, res = decide(st, name, e, paths)
            e["decision"], e["reason"] = decision, "; ".join(reasons)
            if decision == "promote" or (decision in ("rerun", "improved-pending") and e["round"] == 1):
                e["stage"] = "collected"     # round 1: `rerun` queues the confirming job, as collect does
            else:
                e["stage"] = "needs_decision"
            e["accepted"] = f"near-tie text accepted: {a.note}"
            st.log(f"{e['slug']} {e['device']}: near-tie accepted -> {e['stage']} ({decision})")
    st.save()


def cmd_accept_regression(a):
    """sdk-ab / sdk-loom: publish a model whose slowdown was CONFIRMED (ABBA) with its own numbers, the regression explained (a note naming the
    upstream write-up or the cause): the card shows the new SDK's measurement as it is (publish partial progress); the delta line keeps the
    negative numbers. Only a confirmed perf regression is accepted: gate / func / ktest failures, degraded runners and package stops are not."""
    st = State(a.state)
    if st.d["mode"] not in SDK_MODES:
        die("accept-regression is for the SDK modes (sdk-ab, sdk-loom)")
    done = []
    for name, e in st.jobs.items():
        if e["slug"] not in a.slug or e["stage"] != "needs_decision":
            continue
        # the job whose rows go on the card: sdk-loom = the sweep job (not its ABBA confirmation); sdk-ab = the confirming round-2 job
        if (e.get("prev") if st.d["mode"] == "sdk-loom" else e.get("rerun_of_me")):
            continue
        why = e.get("reason") or ""
        # the hold reasons that mean "a slowdown was CONFIRMED": sdk-loom = its ABBA confirmation was slower; sdk-ab = both runs were slower
        confirmed = ("ABBA confirmation" in why and "slower than" in why) if st.d["mode"] == "sdk-loom" else ("confirmed regression > " in why)
        if not confirmed or any(x in why for x in ("gate failed", "degraded runner", "package:", "missing", "undecided")):
            print(f"{e['slug']} {e['device']}: not a confirmed perf regression, not accepted ({why[:120]})")
            continue
        if st.d["mode"] == "sdk-loom":   # the confirmed rows report the ABBA numbers (same job), as a promoted confirmation does
            abba = next((j for j in st.jobs.values() if j.get("prev") == name and j.get("perf_abba")), None)
            if abba:
                rows = e.get("confirm_rows") or []
                e["confirmed"] = {k: abba["perf_abba"][k] for k in rows if abba["perf_abba"].get(k) is not None}
                e["perf"] = {**(e.get("perf") or {}), **e["confirmed"]}
        e.update(stage="collected", decision="promote", accepted_regression=f"{a.note} (was: {why[:300]})")
        st.log(f"{e['slug']} {e['device']}: confirmed regression accepted for publishing: {a.note}")
        done.append(f"{e['slug']} {e['device']}")
    st.save()
    print("accepted: " + (", ".join(done) or "nothing"))


def _ktest_fail_only(e):
    """the job's gate failures when they are kernel tests that RAN and failed some cases (summary FAIL + a "N/M passed" log with N < M), else (None, why)"""
    fails = json.load(open(e["cmp"])).get("gate_failures") or [] if e.get("cmp") else []
    if not fails or not all(f.startswith("ktest:") for f in fails):
        return None, "not held by kernel tests alone (" + (", ".join(fails) or (e.get("reason") or "")[:120]) + ")"
    s = json.load(open(os.path.join(e["dir"], "summary.json"))).get("results", {}) if e.get("dir") else {}
    for f in fails:
        log = os.path.join(e.get("dir") or "", f"ktest_{f.split(':', 1)[1]}.log")
        m = re.findall(r"(\d+)/(\d+) passed", read(log, "")) if os.path.exists(log) else []
        if not str(s.get(f, "")).startswith("FAIL") or not m or int(m[-1][0]) >= int(m[-1][1]):
            return None, f"{f} did not run to a result (summary {str(s.get(f, ''))[:60]!r}, no 'N/M passed' failure in its log): the card could not show it"
    return fails, ""


def _decide_without(st, name, e, sdk_loom_first):
    """the perf verdict of job e with its kernel-test hold removed (a temp copy of its cmp; the job's evidence is untouched)"""
    tmp = e["cmp"] + ".accepted.json"
    try:
        json.dump(dict(json.load(open(e["cmp"])), gate_failures=[]), open(tmp, "w"))
        if sdk_loom_first:
            return sdk_loom_decide(st, name, e, tmp)
        return sdk_decide(st, name, e, tmp, use_prev=st.d["mode"] == "sdk-ab")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def cmd_accept_gate_failure(a):
    """sdk-ab / sdk-loom: publish a model held ONLY by kernel tests that ran and failed some cases, which a person read and explained (e.g. the same case
    failed on the previous SDK's card): the card is built from the job as it is, so its gate table shows the kernel tests as failed (no gate is loosened).
    func and serving gates (func:*, gates:*) and kernel tests that did not run to a result are refused. The perf rows still get the usual verdict: a row
    outside the band is confirmed first. sdk-loom: when the targeted ABBA job is the one held by the kernel tests, its verdict is settled the same way."""
    st = State(a.state)
    if st.d["mode"] not in SDK_MODES:
        die("accept-gate-failure is for the SDK modes (sdk-ab, sdk-loom)")
    done = []
    for name, e in list(st.jobs.items()):
        if e["slug"] not in a.slug or e["stage"] != "needs_decision" or (st.d["mode"] == "sdk-loom" and e.get("prev")):
            continue
        abba = next((x for x in st.jobs.values() if x.get("prev") == name), None) if st.d["mode"] == "sdk-loom" else None
        held_by_abba = bool(abba) and "ABBA confirmation" in (e.get("reason") or "") and "gate failed" in (e.get("reason") or "")
        job = abba if held_by_abba else e
        fails, why = _ktest_fail_only(job)
        if not fails:
            print(f"{e['slug']} {e['device']}: {why}, not accepted")
            continue
        note = f"{', '.join(fails)} failed, published as failed: {a.note}"
        e["accepted_gate_failure"] = note   # the first job is the one ingested: its card message names it
        if held_by_abba:
            decision, reasons, job["perf_abba"], job["rows_pct"] = _decide_without(st, [k for k, v in st.jobs.items() if v is abba][0], abba, False)
            decision = "hold" if decision == "rerun" else decision
            e.update(stage="collected")
            sdk_loom_settle(st, abba, decision, reasons)
        else:
            decision, reasons, perf, rows = _decide_without(st, name, e, st.d["mode"] == "sdk-loom")
            e["perf"], e["rows_pct"] = perf, rows
            if st.d["mode"] == "sdk-loom":
                e["perf_cross"] = dict(perf)
            e["decision"], e["reason"] = decision, "; ".join(reasons)
            if decision == "promote" or (decision in ("rerun", "confirm") and e["round"] == 1):
                e["stage"] = "collected"
            else:
                e["stage"] = "needs_decision"
                e["reason"] = ("still undecided after the confirming run: " if decision != "hold" else
                               "confirmed regression / failed gate / text difference: ") + e["reason"]
        st.log(f"{e['slug']} {e['device']}: kernel-test failure accepted ({', '.join(fails)}): {a.note} -> {e['stage']} ({e.get('decision')})")
        done.append(f"{e['slug']} {e['device']} ({e.get('decision')})")
    st.save()
    print("accepted: " + (", ".join(done) or "nothing"))


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
    p.add_argument("--mode", choices=("ab", "card", "logits", "sdk-ab", "sdk-loom"), default="ab",
                   help="sdk-loom: the SDK migration's Loom-only sweep (one job per model and device: the new package on the new SDK, no llama.cpp, no old-card variant; deltas against the old card's Loom rows; a row outside +-3 percent gets ONE targeted sdk-ab ABBA job); sdk-ab: the SDK migration's A/B per model and device (old = the newest old-SDK revision of the staging card on its own SDK pin, 0.1.0 when pin-less; new = built from --base on the new SDK; deltas per metric); logits: the final pass's functional job of a staging package (func, ktest, logits:staging), no performance")
    p.add_argument("--bundle", nargs=2, metavar=("URL", "SHA256"), help="logits mode: the Gufo bundle tarball the job fetches (devlab_perf.sh gufo_bundle)")
    p.add_argument("--bundle-run", help="logits mode: reuse the `gufo-bundle` artifact of this devlab-perf run (the run that built it: no build in this one)")
    p.add_argument("--bundle-build", nargs=2, metavar=("EXT_REV", "GUFO_REV"), help="logits mode: the devlab-perf run BUILDS the Gufo bundle "
                   "(loom-fun/loom-gufo-extension@EXT_REV, loom-fun/gufo@GUFO_REV; Nix in Docker on a hosted runner, cached by the two revisions)")
    p.add_argument("--all", action="store_true", help="every model of the fleet manifest (a device skips the ones that do not fit)")
    p.add_argument("--old", help="ORG of the old packages (default loom-fun) or a refs.json {slug: {old, new}} (card revisions pinned: org/repo@rev)")
    p.add_argument("--new-org", default=STAGING_ORG)
    p.add_argument("--base", required=True, help="the integration branch the experiment branches are cut from and the PRs target (its devlab "
                   "scripts and knowledge base run the jobs), e.g. r9700 or the campaign's branch")
    p.add_argument("--gates", default="dev")
    p.add_argument("--batch", type=int, default=8, help="jobs per experiment PR (smaller batches finish, and can be collected, sooner)")
    p.add_argument("--no-pin", action="store_true", help="do not pin the new package to the staging head revision at plan time")
    p.add_argument("--work", help="work directory (default: STATE.d next to the state file)")
    p.add_argument("--state", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--replicas", type=int, default=1, help="ab mode: runs of the A/B per model in ONE submission (2 = the ratchet's confirming run as a parallel "
                   "replica on another runner; collect decides on the pair). A targeted change: 2")
    p.add_argument("--sequential", action="store_true", help="ab mode: 2+2 sessions, each job extends itself to 6+6 where a row is within 2 points of its reject threshold")
    p.add_argument("--no-baselines", action="store_true", help="ab mode on a device with a memory pool: no llama.cpp re-measure (a targeted change compares Loom with Loom; "
                   "about 40 percent of a small model's job)")
    p.add_argument("--priority", action="store_true", help="the batches of this plan go to the head of the next submit wave and ignore --max-inflight (a targeted job, not a sweep)")
    p.set_defaults(fn=cmd_plan)
    for name, fn in (("submit", cmd_submit), ("poll", cmd_poll), ("collect", cmd_collect), ("rerun", cmd_rerun), ("ingest", cmd_ingest),
                     ("publish", cmd_publish), ("package", cmd_package), ("status", cmd_status), ("adopt", cmd_adopt), ("accept", cmd_accept),
                     ("accept-regression", cmd_accept_regression), ("accept-gate-failure", cmd_accept_gate_failure)):
        q = sub.add_parser(name)
        q.add_argument("state")
        q.set_defaults(fn=fn)
        if name in ("submit", "rerun", "publish", "package"):
            q.add_argument("--dry-run", action="store_true")
        if name == "submit":
            q.add_argument("--max-batches", type=int)
            q.add_argument("--max-inflight", default=None, help="N jobs per device in the queue at once, or auto = the device's runner count: a sweep goes out in "
                           "waves (call submit again as jobs finish), so a priority job queued meanwhile waits at most one wave. Default: no cap")
        if name == "status":
            q.add_argument("--status-json")
        if name in ("accept", "accept-regression", "accept-gate-failure"):
            q.add_argument("slug", nargs="+")
            q.add_argument("--note", required=True, help="what was read: e.g. 'hyphen vs space, log-prob gap < band'")
        if name == "adopt":
            q.add_argument("--run", required=True)
        if name == "ingest":
            q.add_argument("--slug", action="append")
        if name == "package":
            q.add_argument("--slugs", nargs="+", help="only these models, whatever their stage (default: the collected jobs that were promoted)")
            q.add_argument("--ref", help="the git ref the canonical build is made at (default: origin/<the state's base>)")
    q = sub.add_parser("rerender")
    q.add_argument("--slugs", nargs="+")
    q.add_argument("--all", action="store_true", help="every model of the fleet manifest")
    q.add_argument("--work", required=True, help="scratch directory on disk (one card at a time, weights are never fetched)")
    q.add_argument("--dry-run", action="store_true", help="render and list the cards whose README/REPORT change; upload nothing")
    q.set_defaults(fn=cmd_rerender)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
