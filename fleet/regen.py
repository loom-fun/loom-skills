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
  regen.py evidence S.json -o EVIDENCE.json               --programs production: the sdk-evidence/1 document (per model / device / metric, fail closed)
  regen.py eligible S.json --out-dir DIR                  --programs production: the release's eligible.json / held.json / pending.json from the evidence

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
  --programs production (sdk-loom / sdk-ab): the FROZEN migration. The target SDK (--new-sdk, default the newest sdk_pin.VERIFIED_RELEASES entry) has an
  accepted production inventory skills/loom-migrate/versions/vOLD-to-vNEW-production.json ({org/repo: revision}, fleet/sdk_inventory.py; held packages in
  the sibling -held.json). Nothing is built: every job serves the production package at its inventory revision byte for byte on the target SDK (the
  ABBA confirmation: the same package on its own old pin), each variant carries the production programs hash and devlab_perf.sh refuses to measure
  anything else. plan reads each package's small files (programs hash, kind, the devices its card claims): a claimed device with a runner gets a job, a
  claimed runner-less device is recorded (no_runner: carried under the owner's policy), an image package gets an image-service job (image_sdk_compare.py:
  identical pixels on both SDKs). collect holds a job that concluded other than success, served other programs or ran another SDK wheel. package makes
  the staging repo hold exactly the production package with only the llm_config.json pin stamped (card files the production card's until ingest);
  `evidence` / `eligible` write the release's inputs (eligible only when every runner device is qualified by a passing job of those programs).
  --old-sdk target (sdk-ab / sdk-loom, --programs build): the old package is served on the target SDK too: the kit build against the frozen
  programs on the same SDK (phase 2 of a migration: --mode sdk-ab --old-sdk target --no-baselines; the baseline rows are the previous card's).
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
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CARD = os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card")
SCRIPTS = os.path.join(ROOT, ".github", "scripts")
for p in (HERE, CARD, os.path.join(ROOT, "skills", "loom-kb"), os.path.join(ROOT, "skills", "loom-setup", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import card_lib  # noqa: E402
import promote_check  # noqa: E402
import sample_lib  # noqa: E402
import sdk_package  # noqa: E402
import sdk_pin  # noqa: E402
from sdk_pin import VERIFIED_RELEASES  # noqa: E402
from publish_package import parse_upload_receipt  # noqa: E402

try:
    import fcntl
except ImportError:   # Windows: the tests import this module; a campaign runs on Linux
    fcntl = None
REPO = "loom-fun/loom-skills"
BASELINES = os.path.join(HERE, "baselines.json")   # the ratchet store: what each card says today
PY = sys.executable
STAGING_ORG = "loom-fun-staging"
LOOMBIN = os.environ.get("REGEN_LOOMBIN", "")   # optional bin dir put first on PATH for loom-fetch (default: the hrx-loom already on PATH)
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
SDK_OLD = "card"


def version_key(version):
    """(major, minor, patch) of "X.Y.Z" (a release candidate or dev suffix sorts with its release): the ordering of SDK versions."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", str(version or ""))
    if not m:
        raise ValueError(f"not an SDK version: {version!r}")
    return tuple(int(x) for x in m.groups())


SDK_NEW = sdk_pin.newest_verified()   # what a migration moves TO by default (loom-setup/scripts/sdk_pin.py VERIFIED_RELEASES)
LEGACY_NEW_SDK = "0.2.0"   # a state planned before plan recorded opts.new_sdk (the v0.1 -> v0.2 migration) targets this
VERSIONS = os.path.join(ROOT, "skills", "loom-migrate", "versions")


def release_pin(version):
    """the llm_config.json "sdk" pin of a VERIFIED release (sdk_pin.release_pin), or None for an unverified version (a legacy target)"""
    return sdk_pin.release_pin(version) if version in VERIFIED_RELEASES else None


def repo_rel(path):
    """PATH relative to the repository when it is inside it, else absolute (another drive on Windows)"""
    try:
        rel = os.path.relpath(path, ROOT)
    except ValueError:
        return os.path.abspath(path)
    return os.path.abspath(path) if rel.startswith("..") else rel


def inventory_path(version):
    """skills/loom-migrate/versions/vOLD-to-vVERSION-production.json: the accepted production inventory {org/slug: revision} of the migration TO
    version (its frozen programs are the ones a --programs production sweep measures), or None"""
    hits = sorted(glob.glob(os.path.join(VERSIONS, f"v*-to-v{version}-production.json")))
    if len(hits) > 1:
        die(f"several production inventories for {version}: {hits}")
    return hits[0] if hits else None


def load_inventory(path):
    """{org/repo: revision} of an inventory file (keys without a "/", e.g. "_drift", are its records, not packages)"""
    return {k: v for k, v in sdk_package.load(path).items() if "/" in k}


def inventory_old_version(path):
    """the outgoing SDK of an inventory file name vOLD-to-vNEW-production.json"""
    m = re.match(r"v(\d+\.\d+\.\d+)-to-v", os.path.basename(path))
    return m.group(1) if m else None


def target_sdk(state):
    return state["opts"].get("new_sdk", LEGACY_NEW_SDK)


def pinned_target(state):
    """the verified pin of the state's target SDK (a release in VERIFIED_RELEASES), else None (an unverified / legacy target)"""
    return release_pin(target_sdk(state))


def measured_sdk(job, version):
    """Require the verified Linux wheel of VERSION for both performance variants (sdk_staging.json / sdk_stagingnd.json of the job)."""
    want = release_pin(version)
    directory = job.get("dir")
    if not directory or not want:
        return False
    try:
        for variant in ("staging", "stagingnd"):
            sdk = sdk_package.load(os.path.join(directory, f"sdk_{variant}.json"))
            if any(sdk.get(key) != want[key] for key in ("hrx_loom", "wheel_sha256")):
                return False
    except (OSError, ValueError, AttributeError):
        return False
    return True


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


def new_sdk_pin(pin, target=None):
    """True for a package pinned to the SDK this migration moves TO (target, default SDK_NEW, or one of its release candidates): every other pin,
    or none, is an old card (0.1.0; Flash-Next's 0.2.0.dev0 CI build is its "previous" SDK)"""
    base = (target or SDK_NEW).split("rc")[0]   # works for a target "0.2.0rc1" and, after the final release, "0.2.0" alike
    return bool(pin) and (pin == base or str(pin).startswith(base + "rc"))


def old_sdk_revision(repo, target=None):
    """the newest commit of REPO whose llm_config.json is not pinned to the new SDK (None if there is none)"""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    for c in HfApi().list_repo_commits(repo):
        try:
            pin = (json.load(open(hf_hub_download(repo, "llm_config.json", revision=c.commit_id))).get("sdk") or {}).get("hrx_loom")
        except EntryNotFoundError:   # a commit without llm_config.json (the repo's first commits) is not a package; any other error reaches plan's die
            continue
        if not new_sdk_pin(pin, target):
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
        old_sdk = target_sdk(st) if d.get("old_sdk_target") else SDK_OLD   # --old-sdk target: the old package on the target SDK too (programs-only A/B)
        if st["mode"] == "sdk-ab":
            # the new card's own table needs the llama.cpp rows of the same job, unless --no-baselines (rule 17: the previous card's rows are reused)
            cmd += ["--sdk-ab", "--old-sdk", old_sdk, "--new-sdk", target_sdk(st)] + ([] if d.get("no_baselines") else ["--baselines"])
        elif st["mode"] == "sdk-loom":   # the new package alone; a confirming job (suffix -r2) is the sdk-ab ABBA of that model without baselines
            cmd += ["--sdk-ab", "--old-sdk", old_sdk, "--new-sdk", target_sdk(st)] + ([] if suffix else ["--loom-only"])
        if st["mode"] in SDK_MODES and d.get("programs") == "production":
            cmd += ["--programs", "production"]   # every variant serves the pinned production package (frozen programs), nothing built
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


PRODUCTION_ORG = "loom-fun"
SMALL_FILES = ["model.json", "info.json", "llm_config.json", "program/**"]   # what plan reads of a production package (never weights)


def production_facts(repo, rev, cache):
    """What a --programs production plan needs of a published package at REV (small files only, cached under CACHE/<slug>@<rev[:12]>):
    {programs_sha256, kind: text|image, claimed: [devices of its card], memory_bytes, features, sdk}"""
    d = os.path.join(cache, f"{repo.split('/')[-1]}@{rev[:12]}")
    facts = os.path.join(d, "facts.json")
    if os.path.isfile(facts):
        return sdk_package.load(facts)
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=rev, local_dir=d, allow_patterns=SMALL_FILES)
    shutil.rmtree(os.path.join(d, ".cache"), ignore_errors=True)
    model = sdk_package.load(os.path.join(d, "model.json"))
    info = sdk_package.load(os.path.join(d, "info.json")) if os.path.isfile(os.path.join(d, "info.json")) else {}
    cfg = sdk_package.load(os.path.join(d, "llm_config.json")) if os.path.isfile(os.path.join(d, "llm_config.json")) else None
    kinds = {(sv or {}).get("kind") for sv in (model.get("services") or {}).values()}
    out = {"programs_sha256": card_lib.programs_sha256(d), "kind": "image" if cfg is None and kinds == {"image"} else "text",
           "claimed": sorted((info.get("devices") or {}).keys()), "memory_bytes": (info.get("card") or {}).get("memory_bytes"),
           "gates": {d: sorted((v or {}).get("gates") or {}) for d, v in (info.get("devices") or {}).items()},   # the gates the card claims per device
           "features": (info.get("card") or {}).get("features") or [], "sdk": (cfg or {}).get("sdk"),
           # the commit that generated the package (its llm_config stamp), else the commit that generated its card
           "generator": next((c for c in (((cfg or {}).get("generator") or {}).get("skills_commit"), (info.get("versions") or {}).get("card_generator"))
                              if c and re.fullmatch(r"[0-9a-f]{40}", str(c))), None)}
    if not out["programs_sha256"]:
        raise RuntimeError(f"{repo}@{rev[:10]}: no program/ files")
    json.dump(out, open(facts, "w"), indent=1)
    return out


def load_held(path):
    """held.json {slug: {reason, ...}} (or a list of slugs): packages out of this release; plan refuses them and --all skips them"""
    if not path:
        return {}
    d = sdk_package.load(path)
    d = {s: {} for s in d} if isinstance(d, list) else d
    return {k.split("/")[-1]: v for k, v in d.items()}


def image_jobs(st, device, slugs, refs, old_sdk):
    """(perf.json document, [job]) of image packages: one job per package on the device's runner (devlab_perf.sh job "kind": "image",
    image_sdk_compare.py): the frozen package on old_sdk and on the target SDK, ABBA, identical pixels required"""
    key = card_lib.registry.device(device)["runner"]["key"]
    jobs = [{"name": f"{key[0]}img-{s.lower().replace('.', '-')}"[:40], "runner": key, "kind": "image", "card": refs[s]["new"],
             "memory_bytes": 0, "variants": [], "steps": [],
             "image": {"old_sdk": old_sdk, "new_sdk": target_sdk(st), "programs_sha256": refs[s]["programs_sha256"], "preset": None}} for s in slugs]
    doc = {"_comment": f"Image packages (fleet/regen.py --programs production): the frozen package on hrx-loom {old_sdk} and {target_sdk(st)}, same runner, "
                       "ABBA sessions, identical pixels (image_sdk_compare.py)", "runner": key, "memory_bytes": 0, "card": jobs[0]["card"] if jobs else None,
           "jobs": jobs}
    return doc, jobs


def cmd_plan(a):
    for k, v in (("new_sdk", None), ("programs", "build"), ("old_sdk", "card"), ("held", None), ("image_device", "strix-halo"), ("llamacpp", None)):
        if not hasattr(a, k):   # a caller that builds its own namespace (fleet/final_pass.py) without the SDK-mode options
            setattr(a, k, v)
    new_sdk = a.new_sdk or SDK_NEW
    try:
        version_key(new_sdk)
    except ValueError as x:
        die(str(x))
    if new_sdk != SDK_NEW and a.mode not in SDK_MODES:
        die("--new-sdk selects the target of sdk-ab or sdk-loom")
    production = a.programs == "production"
    if production and a.mode not in SDK_MODES:
        die("--programs production is a form of --mode sdk-loom / sdk-ab")
    if a.old_sdk == "target" and (a.mode not in SDK_MODES or production):
        die("--old-sdk target compares a BUILT package with the old one on the same SDK: --mode sdk-ab / sdk-loom with --programs build")
    # a verified target with a production inventory (skills/loom-migrate/versions/vOLD-to-vNEW-production.json): the old side is the production
    # package at its accepted revision, never a staging history walk
    inventory = inventory_path(new_sdk) if a.mode in SDK_MODES and release_pin(new_sdk) else None
    accepted = load_inventory(inventory) if inventory else None
    if production and not accepted:
        die(f"--programs production needs a verified SDK {new_sdk} (sdk_pin.VERIFIED_RELEASES) and its inventory {VERSIONS}/v*-to-v{new_sdk}-production.json")
    sibling = inventory and inventory.replace("-production.json", "-held.json")
    held = load_held(a.held or (sibling if production and os.path.isfile(sibling) else None))   # default: the inventory's sibling held.json
    if production:
        org_slugs = sorted(r.split("/", 1)[1] for r in accepted)
        slugs = [s for s in org_slugs if s not in held] if a.all else a.slugs
        bad = [s for s in slugs if s in held]
        if bad:
            die(f"held out of this release ({a.held}): {', '.join(bad)}")
    else:
        slugs = fleet_slugs() if a.all else a.slugs
        if accepted and a.all:
            slugs = [slug for slug in slugs if f"{PRODUCTION_ORG}/{slug}" in accepted]
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
        nightly = {m["repo"].split("/")[-1] for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]}
        llamacpp = dict(x.split("=", 1) for x in a.llamacpp or [])
        if set(llamacpp) - (set(slugs) - nightly):
            die(f"--llamacpp names a package that is not planned or that .github/nightly-models.json lists: {sorted(set(llamacpp) - (set(slugs) - nightly))}")
        for s in slugs:
            repo = f"{a.new_org}/{s}"
            if accepted:
                published = f"{PRODUCTION_ORG}/{s}"
                if published not in accepted:
                    die(f"{published}: absent from the accepted SDK {new_sdk} production inventory ({repo_rel(inventory)})")
                revision = accepted[published]
                reference = f"{published}@{revision}"
                refs[s] = {"old": reference, "new": reference}
                if production:   # the frozen package: its programs hash (checked by the job before measuring), kind, the devices its card claims
                    try:
                        f = production_facts(published, revision, os.path.join(work, "production"))
                    except Exception as x:
                        die(f"{published}@{revision[:10]}: cannot read the production package ({x}); rerun plan")
                    refs[s].update(programs_sha256=f["programs_sha256"], kind=f["kind"], claimed=f["claimed"], generator=f.get("generator"),
                                   gates=f.get("gates") or {})
                    if s not in nightly and f["kind"] == "text":   # a hand-written package the nightly list does not hold: its card's facts
                        refs[s]["model"] = {"repo": published, "slug": s.lower(), "memory_bytes": f["memory_bytes"], "kit": None, "features": f["features"],
                                            **({"llamacpp": llamacpp[s]} if s in llamacpp else {})}
                continue
            try:
                head, old = hf_head(repo), old_sdk_revision(repo, new_sdk)
            except Exception as x:   # one transient Hub failure: a clear message, not a traceback
                die(f"{repo}: cannot resolve its revisions on the Hub ({x}); rerun plan")
            if not old:
                die(f"{repo}: no revision with an old-SDK package (pin other than {new_sdk}) in its history")
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
                   "new_sdk": new_sdk, **({"old_sdk_target": True} if a.old_sdk == "target" else {}),
                   **({"programs": "production", "inventory": repo_rel(inventory),
                                           "old_sdk": inventory_old_version(inventory), "held": held} if production else {}),
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
        dev_slugs, images = slugs, []
        if production:   # the devices the production card claims: those with a runner get a job here; an image package runs on --image-device
            dev_slugs = [s for s in slugs if refs[s]["kind"] == "text" and dev in refs[s]["claimed"]]
            images = [s for s in slugs if refs[s]["kind"] == "image" and dev == a.image_device]
            for s in slugs:
                if refs[s]["kind"] == "text" and dev not in refs[s]["claimed"]:
                    st["jobs"][f"skipped:{s}:{dev}"] = {"slug": s, "device": dev, "stage": "skipped", "round": 1,
                                                        "reason": "the production card does not claim this device"}
        chunks = [(dev_slugs[i:i + a.batch], False) for i in range(0, len(dev_slugs), a.batch)]
        chunks += [(images[i:i + a.batch], True) for i in range(0, len(images), a.batch)]
        for chunk, is_image in chunks:
            doc, jobs, skipped = image_jobs(st, dev, chunk, refs, st["opts"]["old_sdk"]) + ([],) if is_image else gen_jobs(st, dev, chunk)
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
                                         **({"source_card": e["old"]} if accepted else {}),
                                         **({"programs_sha256": e["programs_sha256"], "kind": e["kind"]} if production else {}),
                                         "aa": a.mode not in SDK_MODES and bool(e.get("old") and e.get("old") == e.get("new")),   # old == new: the A/B shows noise only
                                         **({"old": e["old"]} if a.mode == "sdk-loom" and e.get("old") else {})}   # the old card the cross-job deltas are taken against
                if rep_no == 1:
                    first[slug_of(j)] = j["name"]
            for j in jobs:   # a replica is a confirming run submitted with the first: the same links `rerun` makes (prev, rerun_of_me)
                if j.get("replica", 1) > 1:
                    f = first[slug_of(j)]
                    st["jobs"][j["name"]].update(prev=f, replica=j["replica"])
                    st["jobs"][f]["rerun_of_me"] = j["name"]
    if production:   # a device the card claims that has no DevLab runner: carried under the owner's policy, never measured here
        for s in slugs:
            for d in refs[s]["claimed"]:
                if d in card_lib.registry.devices() and not card_lib.has_runner(d):
                    st["jobs"][f"no_runner:{s}:{d}"] = {"slug": s, "device": d, "stage": "skipped", "round": 1, "no_runner": True,
                                                        "reason": "no DevLab runner: carried from the production card (rows marked with their SDK)"}
        st["opts"]["refs_all"] = {s: {k: v for k, v in e.items() if k != "model"} for s, e in refs.items()}
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
            "Jobs: " + ", ".join(f"`{j}`" for j in jobs) + "\n\n```loom-scope\nkind: tooling\n```\n\n"
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
    stem, n = branch, 1
    while sh(["gh", "api", f"repos/{REPO}/git/ref/heads/{branch}"], check=False).returncode == 0:   # a re-submitted batch (a job the runner lost)
        n += 1
        branch = f"{stem}-s{n}"
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
    seen = []   # (batch id, run, jobs): the network part first, then ONE locked read-modify-write, so a step that saved the state meanwhile
    for b in st.d["batches"]:   # (collect, rerun, accept: possibly in another shell) keeps its changes
        if b["stage"] == "planned":
            continue
        run = b.get("run") or find_run(b)
        if run:
            seen.append((b["id"], run, gh_json(["run", "view", str(run), "-R", REPO, "--json", "status,conclusion,jobs"])["jobs"]))
    with locked(st.path):
        st.d = json.load(open(st.path))
        for bid, run, jobs in seen:
            b = next(x for x in st.d["batches"] if x["id"] == bid)
            b["run"] = run
            apply_run_jobs(st, b, run, jobs)
        tmp = st.path + ".tmp"
        json.dump(st.d, open(tmp, "w"), indent=1, sort_keys=True)
        os.replace(tmp, st.path)
    pending += sum(len(b["jobs"]) for b in st.d["batches"] if b["stage"] == "planned")
    for e in st.jobs.values():
        if e["stage"] in ("planned", "submitted", "queued", "running"):
            pending += 1
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
    if st.d["opts"].get("programs") == "production":   # every card row needs its delta explained: a row outside the band is confirmed too (by row name)
        for row, x in m.items():
            r = x.get("new_vs_card")
            if row not in SDK_METRICS.values() and r and abs(r - 1) > SDK_REGRESSION:
                moved[row] = round((r - 1) * 100, 1)
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


def image_verdict(d, e):
    """An image package's job (image_sdk_compare.py): promote only when both SDKs produced deterministic outputs, the pixels are identical and the
    programs are the expected ones (summary "pixels" PASS); the result's facts go to e["image"]."""
    s = json.load(open(os.path.join(d, "summary.json"))).get("results", {})
    try:
        r = json.load(open(os.path.join(d, "image_result.json")))
        e["image"] = {k: r.get(k) for k in ("preset", "request_sha256", "pixels_match", "programs_match", "max_abs_channel_diff", "time_ratio", "error")}
        e["image"].update({v: {"median_s": (r["runs"][v] or {}).get("median_s"), "deterministic": (r["runs"][v] or {}).get("deterministic"),
                               "rgb_sha256": sorted({h for x in (r["runs"][v] or {}).get("sessions", []) for h in x["rgb_sha256"]})} for v in ("old", "new")})
    except (OSError, ValueError, KeyError):
        pass
    if str(s.get("pixels", "")).startswith("PASS") and all(str(s.get(f"image:{v}", "")).startswith("PASS") for v in ("old", "new")):
        return "promote", []
    return "hold", [f"{k}: {str(v)[:160]}" for k, v in s.items() if not str(v).startswith("PASS")] or ["the image comparison did not pass"]


def kernel_tests_not_applicable(st, e, d):
    """--programs production: kernel tests the kit cannot GENERATE for this package (make_kernel_tests.py ended before a result: ktest ERROR, a
    hand-written port's tensors) are not applicable when the production card claims no kernel-test gate on this device: the 0.2.0 card had none,
    so the frozen card has none either. Recorded on the job (not_applicable) and removed from the compare's gate failures; a kernel test that
    RAN and failed, or one the card claims, still holds."""
    claimed = (((st.d["opts"].get("refs_all") or {}).get(e["slug"]) or {}).get("gates") or {}).get(e["device"])
    if claimed is None or "kernel_tests" in claimed:
        return
    summ = json.load(open(os.path.join(d, "summary.json"))).get("results", {})
    cmp = json.load(open(e["cmp"]))
    na = [k for k in cmp.get("gate_failures") or [] if k.startswith("ktest:") and str(summ.get(k, "")).startswith("ERROR")]
    if na:
        cmp["gate_failures"] = [k for k in cmp["gate_failures"] if k not in na]
        cmp["not_applicable"] = {k: "the kit's kernel-test generator does not cover this package and its production card claims no kernel tests" for k in na}
        json.dump(cmp, open(e["cmp"], "w"), indent=1)
        e["not_applicable"] = sorted(na)


def frozen_guard(st, e):
    """--programs production, fail closed: why this job cannot qualify the frozen package (None when it can): it must have concluded success, served
    exactly the production programs and run the verified target SDK."""
    if e.get("conclusion") and e["conclusion"] != "success":
        return f"the job concluded {e['conclusion']} ({e.get('job_url')})"
    if e.get("kind") == "image":
        return None   # image_sdk_compare.py checked the programs hash of both fetches and the SDKs were installed by version (sdk_old/new.json)
    built = job_programs(e)
    if built != e.get("programs_sha256"):
        return f"the job served programs {str(built)[:12]}, not the production programs {str(e.get('programs_sha256'))[:12]}"
    if not measured_sdk(e, target_sdk(st.d)):
        return f"the job's SDK is not the verified hrx-loom {target_sdk(st.d)} wheel (sdk_staging.json / sdk_stagingnd.json)"
    adm = admission(e.get("dir")) if not e.get("prev") else {"result": "n/a"}   # an ABBA confirmation measures rows only
    if adm is None:
        return "the job recorded no native-context admission probe (admit:staging)"
    if adm["result"] == "REFUSED":
        return f"the native-context admission probe was refused: {adm.get('detail') or adm.get('tokens')}"
    return None


def collect_one(st, name, e):
    d = download(st, name, e)
    e["dir"] = d
    mode = st.d["mode"]
    if e["stage"] == "failed" and not os.path.exists(os.path.join(d, "summary.json")):
        e.update(stage="needs_decision" if e["round"] > 1 else "collected", decision="rerun" if e["round"] == 1 else "hold",
                 reason=f"the job {e.get('conclusion')} before writing summary.json (see {e.get('job_url')})")
        return
    production = st.d["opts"].get("programs") == "production"
    if e.get("kind") == "image":
        decision, reasons = image_verdict(d, e)
    elif mode == "sdk-ab":
        e["cmp"] = compare(st, name, e, d)
        decision, reasons, e["perf"], e["rows_pct"] = sdk_decide(st, name, e, e["cmp"])
    elif mode == "sdk-loom" and not e.get("prev"):   # the sweep job: cross-job deltas against the old card's Loom rows
        e["cmp"] = compare(st, name, e, d)
        if production:
            kernel_tests_not_applicable(st, e, d)
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
    if production and decision != "hold":
        why = frozen_guard(st, e)
        if why:
            decision, reasons = "hold", [why] + reasons
            if e.get("prev") and mode == "sdk-loom":   # an ABBA confirmation that cannot count stops its first job too
                sdk_loom_settle(st, e, "hold", [why])
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
    abba = {**(e.get("rows_pct") or {}), **(e.get("perf_abba") or {})}   # metric keys (decode, prefill, ...) and card row names (prefill_2k, ...)
    rows = f.get("confirm_rows") or []
    slower = {k: abba[k] for k in rows if k not in SDK_METRICS and abba.get(k) is not None and abba[k] < -SDK_REGRESSION * 100}
    if decision == "promote" and slower:   # a card row the ratchet's four metrics do not cover, confirmed slower in the same job
        decision, reasons = "hold", [f"slower than -{SDK_REGRESSION:.0%} in the same-job ABBA: " + ", ".join(f"{k} {v:+.1f} %" for k, v in slower.items())] + list(reasons)
    if decision == "promote":
        f["confirmed"] = {k: abba[k] for k in rows if abba.get(k) is not None}
        f["confirmed_by"] = e.get("job_url")
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
                                  **({key: st.jobs[first[s]][key] for key in ("old", "source_card", "programs_sha256", "kind")
                                      if key in st.jobs[first[s]]})}
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


def card_message(e, dev, run, loom_only, sdk, version=SDK_NEW):
    """the publish message of a card made from a job: where the numbers come from, the SDK delta line, an accepted slowdown's explanation"""
    msg = f"{card_lib.GPUS[dev]['runner']} results measured on this package (run {run})"
    if loom_only:
        msg += (f"; hrx-loom {version} Loom rows vs the previous card's Loom rows (another job, same runner type): " + sdk_line(e)
                + ("; confirmed by an ABBA job (same job): " + ", ".join(f"{k} {v:+.1f} %" for k, v in e["confirmed"].items()) if e.get("confirmed") else "")
                + "; baseline rows are the previous card's")
    elif sdk:
        msg += f"; hrx-loom {version} vs the previous card on its own SDK pin (same job, ABBA): " + sdk_line(e)
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
    pin = pinned_target(st.d) if sdk else None   # a verified target release: the staging head must carry exactly its pin
    verified = bool(pin)
    # sdk-loom, or an sdk-ab job planned with --no-baselines (no llama.cpp files): the Loom rows are the job's, the baseline rows the previous card's
    loom_only = st.d["mode"] == "sdk-loom" or (st.d["mode"] == "sdk-ab" and not glob.glob(os.path.join(d, "perf_llamacpp-rocm.np1_n*.jsonl"))
                                               and bool(st.d["opts"].get("no_baselines")))
    # sdk-ab: the job measured a package BUILT from the branch; the card is made on the staging head, which must hold exactly those programs (the
    # v0.2 package published before this card-only step): make_info.py refuses a different programs hash
    out = os.path.join(W, "out")
    dry = INGEST_DRY   # ingest --dry-run (--programs production): the card is made on the locally assembled frozen package
    if dry:
        rev = None
        frozen_assemble(e["source_card"], target_sdk(st.d), out)
        e = dict(e, package={"programs_sha256": card_lib.programs_sha256(out)})
    else:
        rev = hf_head(f"{st.d['opts']['new_org']}/{slug}") if sdk else e["card_revision"].split("@")[-1]
        hf_snapshot(f"{st.d['opts']['new_org']}/{slug}", rev, out)
    if sdk:   # the package step's programs, whatever the head's revision is now (another device's card may have been published since)
        got = card_lib.programs_sha256(out)
        if got != e["package"]["programs_sha256"]:
            raise RuntimeError(f"the staging head {rev[:10]} holds programs {str(got)[:12]}, not the published v0.2 package's {e['package']['programs_sha256'][:12]}: "
                               f"another publish replaced them; run `regen.py package` again")
    if verified:
        try:
            published_pin = sdk_package.load(os.path.join(out, "llm_config.json")).get("sdk") or {}
        except (OSError, ValueError, AttributeError) as ex:
            raise RuntimeError(f"cannot read the SDK pin from staging head {str(rev)[:10]}") from ex
        if any(published_pin.get(key) != value for key, value in pin.items()):
            raise RuntimeError(f"the staging head {str(rev)[:10]} no longer has the verified SDK {target_sdk(st.d)} pin")
    shutil.copy(os.path.join(out, "info.json"), os.path.join(W, "info_prev.json"))
    prev = json.load(open(os.path.join(W, "info_prev.json")))
    baseline_info = "info_prev.json"
    if verified and e.get("source_card"):   # the inventory's production card is the old side of the delta
        source, revision = e["source_card"].split("@", 1)
        baseline_info = hf_file(source, "info.json", revision)
    mem = prev["card"]["memory_bytes"]
    run, url, end = job_facts(st, e)
    src = ["--source-kind", "devlab", "--run", run, "--url", url]
    rd = lambda f: read(os.path.join(d, f))
    g = lambda pat: sorted(glob.glob(os.path.join(d, pat)))
    hrx = os.path.abspath(os.path.join(d, f"sdk_{V}.json")) if verified else "hrx.json"
    identity = [f"{S}/pull_devlab.py", "--identity", d, "--gpu", dev, "--device-out", "dev.json"]
    run_tool(identity + ([] if verified else ["--hrx-out", hrx]), W)
    llm, vk = g("perf_llamacpp-rocm.np1_n*.jsonl"), g("perf_llamacpp-vulkan.np1_n*.jsonl")
    c4l, c4v = g("c4_llamacpp-rocm.nodraft_n*.txt"), g("c4_llamacpp-vulkan.perslot.nodraft_n*.txt")
    vl, c4vl = g("perf_vllm_n*.jsonl"), g("c4_vllm_n*.txt")   # a safetensors card: vLLM ROCm is the baseline engine (devlab_perf.sh vllm: steps)
    fleet_cfg = os.path.join(ROOT, "fleet", slug, "llm_config.json")   # a package outside the manifest (a hand-written one): its own config
    st_card = (json.load(open(fleet_cfg if os.path.isfile(fleet_cfg) else os.path.join(out, "llm_config.json"))).get("source") or {}).get("kind") == "safetensors"
    production = st.d["opts"].get("programs") == "production"
    drafts = g(f"metrics_{V}_*.txt") if production and make_info_supports("perf", "--draft-metrics") else []
    if loom_only:   # the job measured Loom alone: the baseline rows (llama.cpp ROCm and Vulkan, or vLLM) are the previous card's, with their own run and date
        run_tool([f"{S}/make_info.py", "perf", "--gpu", dev, "--loom-jsonl", f"{d}/perf_{V}.jsonl", "--c4-loom", f"{d}/c4_{ND}.txt",
                  "--baseline-from-info", baseline_info, "--package", out, *src, "--device", "dev.json", "--hrx-loom", hrx,
                  "--fetched-programs", rd("programs_sha256_staging.txt"), *(["--draft-metrics", *drafts] if drafts else []), "-o", "ev_perf.json"], W)
    else:
        if not st_card and not (llm and c4l):
            raise RuntimeError("the job has no llama.cpp ROCm files (a job without --baselines cannot make this device's table)")
        baseline = card_lib.registry.methodology()["baselines"][dev]
        perf = [f"{S}/make_info.py", "perf", "--gpu", dev, "--loom-jsonl", f"{d}/perf_{V}.jsonl", "--c4-loom", f"{d}/c4_{ND}.txt", "--package", out, *src,
                "--device", "dev.json", "--hrx-loom", hrx, "--fetched-programs", rd("programs_sha256_staging.txt" if sdk else "programs_sha256.txt"),
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
    if e.get("no_reference") and gfiles and gfiles[0].endswith("gates.json"):   # accept-no-reference: the job's gates did not run; the card keeps
        gj = json.load(open(gfiles[0]))   # the previous ones (marked as not re-run on this SDK), only the admission probe is this job's
        gfiles[0] = os.path.join(W, "gates_not_run.json")
        json.dump({"gates": {}, **{k: gj[k] for k in ("ts", "level", "context_admit") if k in gj}}, open(gfiles[0], "w"), indent=1)
    adm = os.path.join(d, f"admit_{V}.json")
    if gfiles and gfiles[0].endswith("gates.json") and os.path.isfile(adm):
        # the frozen job's admission probe (admit:V, the served preset's native context; the gates ran at the dev level, which has none): the card
        # reads it from the gates file's context_admit, so the job's gates.json is passed with it (nothing else changes)
        gj = json.load(open(gfiles[0]))
        if not gj.get("context_admit"):
            gj["context_admit"] = {k: v for k, v in json.load(open(adm)).items() if k in ("admitted", "tokens", "context", "detail")}
            gfiles[0] = os.path.join(W, "gates_with_admission.json")
            json.dump(gj, open(gfiles[0], "w"), indent=1)
    if gfiles:
        run_tool([f"{S}/make_info.py", "gates", "--gpu", dev, "--time", end, *gfiles, "--package", out, *src, "--device", "dev.json", "--hrx-loom", hrx,
                  "-o", "ev_gates.json"], W)
        ev.append("ev_gates.json")
    run_tool([f"{S}/make_info.py", "roofline", "--gpu", dev, "--device", card_lib.registry.device(dev)["roofline"]["default"], "--package", out,
              "--evidence", "ev_perf.json", "-o", "ev_roof.json"], W)
    ev.append("ev_roof.json")
    if production:   # the frozen package carries no generator stamp of this branch: the production card's, named as such
        gen = ((st.d["opts"].get("refs_all") or {}).get(slug) or {}).get("generator")
        if gen:
            json.dump({"evidence": "loom-card-evidence/2", "gpu": dev, "versions": {"package": {"skills_commit": {"commit": gen, "recovered": True,
                                                                                                        "basis": "production card"}}}},
                      open(os.path.join(W, "ev_versions.json"), "w"), indent=1)
            ev.append("ev_versions.json")
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
    if production and e.get("source_card") and make_info_supports("build", "--previous-card"):   # the delta line against the production card
        build += ["--previous-card", f"card:{e['source_card']}={baseline_info}"]
    if production and make_info_supports("build", "--sdk-evidence"):   # regression explanations and every cited job's conclusion
        json.dump(build_evidence(st), open(os.path.join(W, "evidence.json"), "w"), indent=1)
        build += ["--sdk-evidence", os.path.join(W, "evidence.json")]
    run_tool(build + ["--memory-bytes", str(mem), "--from-info", "info_prev.json", *(["--reset-performance"] if loom_only else []), "--evidence", *ev,
                      "-o", "info_new.json"], W)
    shutil.copy(os.path.join(W, "info_new.json"), os.path.join(out, "info.json"))
    msg = card_message(e, dev, run, loom_only, sdk, target_sdk(st.d))
    parent = rev if verified or dry else hf_head(f"{st.d['opts']['new_org']}/{slug}")
    return out, parent, msg


_MAKE_INFO_HELP = {}
INGEST_DRY = False   # ingest --dry-run: ingest_full assembles the frozen package locally instead of reading the staging head


def make_info_supports(sub, flag):
    """whether this checkout's make_info.py SUB takes FLAG (the card generator gains options independently of regen)"""
    if sub not in _MAKE_INFO_HELP:
        r = subprocess.run([PY, os.path.join(CARD, "make_info.py"), sub, "--help"], capture_output=True, text=True)
        _MAKE_INFO_HELP[sub] = r.stdout + r.stderr
    return flag in _MAKE_INFO_HELP[sub]


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
    version = target_sdk(st.d)
    if not pkg:
        why = f"the {version} package is not published yet: run `regen.py package` first"
    elif release_pin(version) and (not built or not measured_sdk(e, version)):
        why = f"the job has no matching programs hash and verified {version} SDK identities"
    elif built and built != pkg["programs_sha256"]:
        why = f"the published {version} package's programs ({pkg['programs_sha256'][:12]}) are not the ones the job built and measured ({built[:12]})"
    if why:
        e.update(stage="needs_decision", reason=f"ingest: {why}")
        st.log(f"{e['slug']} {e['device']}: ingest stopped: {why}")
        st.save()
    return why is None


def cmd_ingest(a):
    st = State(a.state)
    if st.d["mode"] == "logits":
        die("a logits-mode state has no performance to ingest: fleet/final_pass.py card builds the final cards from it")
    todo = [(n, e) for n, e in st.jobs.items() if e["stage"] == "collected" and e.get("decision") == "promote" and e.get("kind") != "image"]   # an image
    # package's card is not an info.json table: its evidence is the job (regen.py evidence), its README the image tooling's
    # a model with a round-2 job: the second run ingests, the first is superseded
    if st.d["mode"] == "sdk-loom":   # the sweep job's rows go on the card (its ABBA confirmation settled the first job's decision)
        todo = [(n, e) for n, e in todo if not e.get("prev")]
    else:
        todo = [(n, e) for n, e in todo if not e.get("rerun_of_me")]
    if a.slug:
        todo = [(n, e) for n, e in todo if e["slug"] in a.slug]
    dry = getattr(a, "dry_run", False)
    if dry and st.d["opts"].get("programs") != "production":
        die("ingest --dry-run builds a --programs production card on the local frozen package (other modes need the staging package)")
    if st.d["mode"] in SDK_MODES and not dry:
        todo = [(n, e) for n, e in todo if sdk_ingest_ready(st, n, e)]
    global INGEST_DRY
    INGEST_DRY = dry
    if not dry and st.d["mode"] in SDK_MODES:   # one card per model at a time: each device's card is built on the staging head the previous one published
        busy = {e["slug"] for e in st.jobs.values() if e["stage"] == "ingested"}
        one = []
        for n, e in todo:
            if e["slug"] not in busy:
                busy.add(e["slug"])
                one.append((n, e))
        todo = one
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
            if dry:
                print(f"{e['slug']} {e['device']}: dry-run ingest failed: {str(ex)[-600:]}")
                continue
            e.update(stage="needs_decision", reason=f"ingest failed: {str(ex)[-400:]}")
            st.log(f"{e['slug']} {e['device']}: ingest failed")
            st.save()
            continue
        if dry:   # nothing recorded: the card is only built (and checked) for review
            print(f"{e['slug']} {e['device']}: dry-run card at {out}")
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


def package_one(st, slug, canon, sha, dry, env, prefill_changes=None, source_card=None):
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
    if source_card:
        source_repo, _, source_revision = source_card.partition("@")
        if source_repo != f"loom-fun/{slug}" or not re.fullmatch(r"[a-f0-9]{40}", source_revision):
            raise ValueError(f"invalid pinned production package: {source_card}")
        hf_snapshot_small(source_repo, source_revision, snap)
    else:
        hf_snapshot_small(repo, rev, snap)
    alts = {card_lib.gpu_preset_name(sdk_package.load(os.path.join(snap, "model.json")), g) for g in card_lib.GPUS if card_lib.GPUS[g].get("preset_suffix")}
    changes = sdk_package.assemble(snap, os.path.join(canon, slug), out, alts, prefill_changes)
    got = card_lib.programs_sha256(out)
    if got != want:
        raise RuntimeError(f"assembled package programs {str(got)[:12]} differ from the canonical build's {want[:12]}")
    version = target_sdk(st.d)
    reference = snap
    if source_card:
        reference = os.path.join(W, "staging")
        hf_snapshot_small(repo, rev, reference)
        qualification = os.path.join(reference, card_lib.QUICK_START_QUALIFICATION)
        if os.path.isfile(qualification):
            # Keep the receipt. Card generation applies it only to the exact measured settings.
            shutil.copy2(qualification, os.path.join(out, card_lib.QUICK_START_QUALIFICATION))
    if release_pin(version):   # the pin is the emitter's (the kit at --ref writes it into llm_config.json): a different one stops the model
        got_pin = sdk_package.load(os.path.join(out, "llm_config.json")).get("sdk") or {}
        if any(got_pin.get(k) != v for k, v in release_pin(version).items()):
            raise RuntimeError(f"the canonical build of {sha[:10]} pins hrx-loom {got_pin.get('hrx_loom')}, not the verified {version} "
                               f"(the kit's emitter must write the target pin; nothing is stamped over it)")
    if sdk_package.same_files(reference, out):   # the head already holds exactly this package (programs AND llm_config pin, model.json, ...)
        return {"revision": rev, "programs_sha256": want, "base": sha, "already": True}
    msg = f"Loom package rebuilt for hrx-loom {version} from loom-skills {sha[:10]} (card files unchanged until the card step)"
    r = sh([PY, f"{CARD}/publish_package.py", "upload", out, "--repo", slug, "--org", org, "--parent-commit", rev, "--message", msg]
           + (["--dry-run"] if dry else []), check=False, env=env)
    if r.returncode:
        raise RuntimeError(f"upload: {(r.stdout + r.stderr)[-400:]}")
    receipt = None if dry else parse_upload_receipt(r.stdout, repo, rev, (env or {}).get("HF_ENDPOINT"))
    print(f"{slug}: {r.stdout.strip().splitlines()[-1][:300]}\n   programs_sha256 {got}; model.json changes: {', '.join(k + ' ' + p for k, p in changes) or 'nothing'}")
    return {"revision": None if dry else receipt["commit"], "programs_sha256": got, "base": sha,
            **({"receipt": receipt} if receipt else {})}


WEIGHT_PATTERNS = re.compile(r"(^|/)parameter/|\.(gguf|safetensors|bin|mmproj)$")


def staging_to_main(value):
    """llm_config.json's knowledge-base evidence strings name the branch a fact was verified on; a frozen package was verified on what is now
    main, so "staging main" reads "main" (DECISIONS D6 b: the only text change of a frozen llm_config.json besides the pin)"""
    if isinstance(value, dict):
        return {k: staging_to_main(v) for k, v in value.items()}
    if isinstance(value, list):
        return [staging_to_main(v) for v in value]
    return value.replace("staging main", "main") if isinstance(value, str) else value


def frozen_assemble(source, version, out):
    """The frozen package of SOURCE (org/repo@rev) in OUT: its files at that revision (no weights), llm_config.json with only the "sdk" pin of
    VERSION and the staging-main evidence wording changed (staging_to_main). -> its programs hash"""
    source_repo, _, source_rev = source.partition("@")
    hf_snapshot_small(source_repo, source_rev, out)
    cfg = os.path.join(out, "llm_config.json")
    config = staging_to_main(sdk_package.load(cfg))
    config["sdk"] = release_pin(version)
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
        f.write("\n")
    return card_lib.programs_sha256(out)


def package_frozen(st, slug, jobs, dry):
    """--programs production: the staging package IS the production package at its inventory revision, byte for byte (programs, model files,
    tokenizer, card files until `ingest` replaces the card), with only llm_config.json's "sdk" pin stamped to the target release. Every selected job of
    the model must have promoted on exactly the production programs with the verified SDK. The upload makes the staging repo hold exactly these files
    (one commit, parent = the staging head at the time: a concurrent publish fails it; files absent from the production package are deleted).
    -> {revision, programs_sha256, base, deleted}"""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    org, version = st.d["opts"]["new_org"], target_sdk(st.d)
    if org != STAGING_ORG:
        raise RuntimeError(f"package writes {STAGING_ORG} only")
    sources = {e.get("source_card") for e in jobs}
    want = {e.get("programs_sha256") for e in jobs}
    if len(sources) != 1 or None in sources or len(want) != 1 or None in want:
        raise RuntimeError("no single pinned production package / programs hash in the selected jobs")
    (source,), (want,) = sources, want
    source_repo, _, source_rev = source.partition("@")
    for e in jobs:
        if e.get("kind") == "image":
            raise RuntimeError("an image package has no llm_config pin: its card is migrated by the image tooling (loom-port-image), not here")
        if e.get("decision") != "promote" or job_programs(e) != want or not measured_sdk(e, version):
            raise RuntimeError(f"the {e['device']} job did not promote on the production programs {want[:12]} with the verified {version} SDK")
    api = HfApi()
    weights = [f for f in api.list_repo_files(source_repo, revision=source_rev) if WEIGHT_PATTERNS.search(f)]
    if weights:
        raise RuntimeError(f"the production package holds weight files in its repo ({', '.join(weights[:3])}): not a frozen-path package")
    W = os.path.join(st.work, "package", slug)
    out = os.path.join(W, "pkg")
    got = frozen_assemble(source, version, out)
    if got != want:
        raise RuntimeError(f"the production snapshot's programs {str(got)[:12]} are not the measured {want[:12]}")
    repo = f"{org}/{slug}"
    head = hf_head(repo)
    files = sorted(os.path.relpath(os.path.join(r, n), out).replace(os.sep, "/") for r, ds, fs in os.walk(out) for n in fs)
    extras = sorted(f for f in api.list_repo_files(repo, revision=head) if f not in files and f != ".gitattributes")
    staged = os.path.join(W, "staging")
    hf_snapshot_small(repo, head, staged)
    if sdk_package.same_files(staged, out) and not [f for f in extras if f not in ("README.md", "info.json", "REPORT.md")]:
        # the head already holds this package (another device's job packaged it; its card step may have replaced the card files since: keep them)
        shutil.rmtree(W, ignore_errors=True)
        return {"revision": head, "programs_sha256": got, "base": source, "deleted": [], "already": True}
    print(f"{slug}: staging {head[:10]} <- {source_repo}@{source_rev[:10]} ({len(files)} files, programs {got[:12]}, pin hrx-loom {version}); "
          f"delete {len(extras)}: {', '.join(extras[:8])}{' ...' if len(extras) > 8 else ''}")
    if dry:
        return {"revision": None, "programs_sha256": got, "base": source, "deleted": extras}
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(out, f)) for f in files] + [CommitOperationDelete(path_in_repo=f) for f in extras]
    info = api.create_commit(repo, operations=ops, parent_commit=head,
                             commit_message=f"Loom package: {source_repo}@{source_rev[:10]} (frozen programs), pinned to hrx-loom {version}",
                             commit_description="Programs and model files byte-identical to the production package; only the SDK pin changes. "
                                                "The card files are the production card's until this device's card step.")
    shutil.rmtree(W, ignore_errors=True)
    return {"revision": info.oid, "programs_sha256": got, "base": source, "deleted": extras}


def cmd_package_frozen(st, a, picked):
    slugs = sorted({e["slug"] for _, e in picked if e.get("kind") != "image"})   # an image card is migrated by the image tooling (its job is the evidence)
    for slug in slugs:
        jobs = [e for _, e in picked if e["slug"] == slug and not e.get("prev")]   # the sweep jobs (an ABBA confirmation is not a card job)
        if not a.dry_run and all(e.get("package") for e in jobs):
            print(f"{slug}: package already published ({jobs[0]['package']['revision'][:10]})")
            continue
        try:
            res = package_frozen(st, slug, jobs, a.dry_run)
        except (RuntimeError, ValueError) as ex:
            print(f"{slug}: package stopped: {str(ex)[-400:]}")
            if not a.dry_run:
                for e in jobs:
                    e.update(stage="needs_decision", reason=f"package: {str(ex)[-300:]}")
                st.log(f"{slug}: package stopped")
                st.save()
            continue
        if not a.dry_run:
            for e in jobs:
                e["package"] = {**res, "at": now()}
            st.log(f"{slug}: frozen package at {res['revision'][:10]} (programs {res['programs_sha256'][:12]})")
            st.save()


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
    if st.d["opts"].get("programs") == "production":
        if a.prefill_transition or a.ref:
            die("--programs production packages the production revision: no --ref / --prefill-transition")
        return cmd_package_frozen(st, a, picked)
    transitions = {}
    for spec in a.prefill_transition:
        try:
            slug, preset, old, new_value = spec.split(":")
            old, new_value = int(old), int(new_value)
            if slug not in slugs or min(old, new_value) < 1 or old == new_value or preset in transitions.get(slug, {}):
                raise ValueError()
        except ValueError:
            die("--prefill-transition requires a selected SLUG:PRESET:OLD:NEW with distinct positive values")
        transitions.setdefault(slug, {})[preset] = (old, new_value)
    canon, sha = canonical_build(st, slugs, a.ref or ("origin/" + st.d["opts"]["base"] if resolve_ref("origin/" + st.d["opts"]["base"]) else st.d["opts"]["base"]))
    env = dict(os.environ, PATH=(LOOMBIN + os.pathsep if LOOMBIN else "") + os.environ["PATH"])
    for slug in slugs:
        jobs = [(n, e) for n, e in st.jobs.items() if e["slug"] == slug]
        if not a.dry_run and all(e.get("package") for _, e in jobs):
            print(f"{slug}: package already published ({jobs[0][1]['package']['revision'][:10]})")
            continue
        try:
            if not os.path.isfile(os.path.join(canon, slug, ".ref")):
                raise RuntimeError("canonical build failed: " + open(os.path.join(canon, slug, "build.log")).read()[-300:])
            want = card_lib.programs_sha256(os.path.join(canon, slug))
            if not want:
                raise RuntimeError("the canonical build has no programs")
            source = None
            if any(e.get("source_card") for _, e in picked if e["slug"] == slug):
                sources = {e.get("source_card") for _, e in picked if e["slug"] == slug}
                if len(sources) != 1 or None in sources:
                    raise RuntimeError("no single pinned production package in the selected evidence")
                source, = sources
            for _, e in picked:   # validate measured programs before package_one can upload anything
                built = job_programs(e) if e["slug"] == slug else None
                if source and e["slug"] == slug and not a.dry_run:
                    if e.get("decision") != "promote" or not built or not measured_sdk(e, target_sdk(st.d)):
                        raise RuntimeError(f"the {e['device']} job has no passing decision, programs hash, or verified {target_sdk(st.d)} SDK identities")
                if built and built != want:
                    raise RuntimeError(f"the job {e['device']} built programs {built[:12]}, the canonical build of {sha[:10]} has {want[:12]} (the base moved since the job was submitted)")
                if e["slug"] == slug and slug in transitions:
                    model_path = os.path.join(e.get("dir", ""), "model_staging.json")
                    if not built or not os.path.isfile(model_path):
                        raise RuntimeError("a prefill transition requires matching measured programs and model_staging.json for every selected job")
                    measured = sdk_package.load(model_path)
                    canonical = sdk_package.load(os.path.join(canon, slug, "model.json"))
                    for preset in transitions[slug]:
                        def execution(model):
                            result = dict(model.get("presets", {}).get(preset, {}).get("execution", {}))
                            result.pop("pool_capacity", None)
                            return result
                        if not execution(canonical) or execution(measured) != execution(canonical):
                            raise RuntimeError(f"the job {e['device']} did not measure the canonical {preset} execution")
            res = package_one(st, slug, canon, sha, a.dry_run, env, transitions.get(slug), source_card=source)
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
            st.log(f"{slug}: {target_sdk(st.d)} package at {res['revision'][:10]} (programs {res['programs_sha256'][:12]})")
            st.save()


# ---------------------------------------------------------------- publish (card only, staging only)

def cmd_publish(a):
    st = State(a.state)
    org = st.d["opts"]["new_org"]
    if org != STAGING_ORG or "/" in org.strip("/") or org == "loom-fun":
        die(f"publish writes {STAGING_ORG} only")
    env = dict(os.environ, PATH=(LOOMBIN + os.pathsep if LOOMBIN else "") + os.environ["PATH"])
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
            try:   # the staging commit this card landed as (eligible.json's staging_revision)
                receipt = parse_upload_receipt(r.stdout, f"{org}/{e['slug']}", e["parent"], (env or {}).get("HF_ENDPOINT"))
                e["published_revision"] = receipt["commit"]
            except Exception:   # an older publish_package.py without a receipt line: the revision is read from the Hub by `eligible`
                pass
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
    env = dict(os.environ, PATH=(LOOMBIN + os.pathsep if LOOMBIN else "") + os.environ["PATH"])
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
            print(f"  {'':46s} {'':13s} {'':2s} SDK performance delta: {sdk_line(e)}"
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


# ---------------------------------------------------------------- evidence / eligible (--programs production: the release's inputs, fail closed)

EVIDENCE_ROWS = [k for k in card_lib.METRIC_KEYS]   # decode ... streams4, spec_acceptance


def job_id(e):
    """the numeric job id of a collected job's URL (.../job/ID), or None"""
    m = re.search(r"/job/(\d+)", e.get("job_url") or "")
    return m.group(1) if m else None


def admission(d):
    """the native-context admission probe of a job (gates.sh context_admit in gates_staging/gates.json, its log): {result, tokens, served_context,
    warnings, log} or None when the job ran no gates"""
    g = os.path.join(d or "", "gates_staging", "gates.json")
    ca, logf = None, os.path.join("gates_staging", "gates_context_admit.log")
    try:
        ca = json.load(open(g)).get("context_admit")
    except (OSError, ValueError):
        pass
    if not ca:   # the job's own admit:staging step (devlab_perf.sh, ctx_admit.py --out): the frozen jobs' probe at the gates' dev level
        try:
            ca, logf = json.load(open(os.path.join(d or "", "admit_staging.json"))), "admit_staging.txt"
        except (OSError, ValueError):
            return None
    if not ca:
        return None
    log = read(os.path.join(d, logf), "") or ""
    warn = sorted({ln.strip()[:200] for ln in log.splitlines() if "WARN" in ln or "did not cancel" in ln})
    return {"result": "ADMITTED" if ca.get("admitted") else "REFUSED", "tokens": ca.get("tokens"), "served_context": ca.get("context"),
            "detail": ca.get("detail"), "warnings": warn, "log": logf}


def device_evidence(st, e, info_cache):
    """sdk-evidence/1 entry of one collected sweep job (notes/evidence_schema.md)"""
    d = e.get("dir")
    ev = {"run": e.get("run"), "job": job_id(e), "job_url": e.get("job_url"), "conclusion": e.get("conclusion"), "artifact": f"devlab-perf-{e.get('name')}"}
    if e.get("kind") == "image":
        ev["image"] = e.get("image")
        return ev
    built = job_programs(e)
    ev.update(programs_sha256=built, programs_match=bool(built) and built == e.get("programs_sha256"))
    try:
        sdk = sdk_package.load(os.path.join(d, "sdk_staging.json"))
        ev["sdk"] = {k: sdk.get(k) for k in ("hrx_loom", "wheel_sha256", "hrx_system_sha")}
        ev["sdk"]["verified"] = measured_sdk(e, target_sdk(st.d))
    except (OSError, ValueError, TypeError):
        ev["sdk"] = None
    cmp = sdk_package.load(e["cmp"]) if e.get("cmp") and os.path.isfile(e["cmp"]) else {}
    m = cmp.get("metrics") or {}
    old_ref = e.get("old") or e.get("source_card")
    old_rows, session = {}, None
    if old_ref:
        if old_ref not in info_cache:
            repo, _, rev = old_ref.partition("@")
            info_cache[old_ref] = card_lib.load_info(hf_file(repo, "info.json", rev or None))
        view = card_lib.gpu_view(info_cache[old_ref], e["device"])
        sessions = (view.get("performance") or {}).get("sessions", [])
        for k, (row, i) in card_lib.card_metrics(view).items():
            old_rows[k] = row.get("loom")
            if k == "decode":
                session = {key: sessions[i].get(key) for key in ("date", "source", "programs_sha256")}
    acc = cmp.get("acceptance") or {}
    new_acc = [x["new"] for x in acc.values() if x.get("new") is not None]
    new_rows = {k: (m.get(k) or {}).get("new") for k in EVIDENCE_ROWS if k != "spec_acceptance"}
    new_rows["spec_acceptance"] = round(statistics.median(new_acc), 4) if new_acc else None
    ev["old"] = {"session": session, "rows": {k: old_rows.get(k) for k in EVIDENCE_ROWS}}
    ev["new"] = {"rows": new_rows}
    ev["delta_pct"] = {k: round((new_rows[k] / old_rows[k] - 1) * 100, 1) for k in EVIDENCE_ROWS
                       if new_rows.get(k) and old_rows.get(k) and k != "spec_acceptance"}
    conf = e.get("confirmed") or {}
    ev["confirmed"] = {SDK_METRICS.get(k, k): v for k, v in conf.items()}
    if e.get("confirmed_by"):
        ev["confirmed_by"] = e["confirmed_by"]
    exact = {k: (m.get(k) or {}).get("new_vs_card") for k in ev["delta_pct"]}
    ev["regressions"] = []
    for k in sorted(ev["delta_pct"]):
        if not exact.get(k) or exact[k] - 1 >= -SDK_REGRESSION:
            continue
        c = ev["confirmed"].get(k)
        expl = None
        if c is not None and c >= -SDK_REGRESSION * 100 and e.get("confirmed_by"):   # the ratchet's same-job ABBA settled it: a generated explanation
            expl = {"text": f"same-job ABBA (the same programs on hrx-loom {o_sdk(st)} and {target_sdk(st.d)}): {c:+.1f} %, within the band; "
                            f"the {ev['delta_pct'][k]:+.1f} % against the previous card is the difference between two jobs", "url": e["confirmed_by"]}
        ev["regressions"].append({"row": k, "pct": ev["delta_pct"][k], "confirmed_abba_pct": c, "explanation": expl})
    summ = {}
    try:
        summ = json.load(open(os.path.join(d, "summary.json"))).get("results", {})
    except (OSError, ValueError, TypeError):
        pass
    try:
        gj = json.load(open(os.path.join(d, "gates_staging", "gates.json")))
        ev["gates"] = {k: ("PASS" if g.get("pass") else "FAIL") for k, g in (gj.get("gates") or {}).items()}
    except (OSError, ValueError, TypeError):
        ev["gates"] = None
    ev["ktest"] = next((re.findall(r"\d+/\d+", v)[-1] for k, v in summ.items() if k.startswith("ktest:") and re.findall(r"\d+/\d+", v)), None)
    ev["func"] = next((v[:120] for k, v in summ.items() if k.startswith("func:")), None)
    ev["admission"] = admission(d)
    raw = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d or "", "metrics_staging_*.txt")))
    ev["spec_acceptance"] = {"old": old_rows.get("spec_acceptance"), "new": new_rows["spec_acceptance"], "per_row": {k: x.get("new") for k, x in acc.items()},
                             "raw": raw}
    return ev


def o_sdk(st):
    return st.d["opts"].get("old_sdk") or "the old SDK"


def device_status(e):
    if e["stage"] == "skipped":
        return "no_runner" if e.get("no_runner") else "does_not_fit" if "fit" in (e.get("reason") or "") else "not_claimed"
    if e["stage"] in ("planned", "submitted", "queued", "running", "done", "failed"):
        return "pending"
    if e["stage"] == "needs_decision" or e.get("decision") == "hold":
        return "failed" if e.get("decision") == "hold" else "needs_decision"
    if e.get("decision") == "promote":
        return "qualified"
    if e["stage"] == "collected" and e.get("decision") in ("confirm", "rerun", "improved-pending"):
        return "confirming"   # its same-job ABBA confirmation is queued or running (`rerun`)
    return "needs_decision"


def build_evidence(st):
    """the sdk-evidence/1 document of a --programs production state (notes/evidence_schema.md in the campaign; one entry per model and device)"""
    o = st.d["opts"]
    if o.get("programs") != "production":
        die("evidence / eligible need a --programs production state")
    refs = o.get("refs_all") or {}
    inv = os.path.join(ROOT, o["inventory"])
    doc = {"schema": "sdk-evidence/1", "generated": now(), "state": st.path,
           "harness": {"branch": o.get("base"), "commit": resolve_ref("origin/" + o["base"]) if o.get("base") else None},
           "sdk": {"old": o.get("old_sdk"), "new": target_sdk(st.d), "new_pin": release_pin(target_sdk(st.d))},
           "inventory": {"path": o["inventory"], "sha256": hashlib.sha256(open(inv, "rb").read()).hexdigest() if os.path.isfile(inv) else None},
           "models": {}, "held": {}}
    held = o.get("held") or {}
    accepted = load_inventory(inv) if os.path.isfile(inv) else {}
    for s in held:
        doc["held"][s] = {**(held[s] if isinstance(held, dict) else {}), "production": f"{PRODUCTION_ORG}/{s}",
                          "production_revision": accepted.get(f"{PRODUCTION_ORG}/{s}"),
                          "tag_revision": accepted.get(f"{PRODUCTION_ORG}/{s}")}   # the production head that gets the outgoing tag
    info_cache = {}
    sweep = {n: e for n, e in st.jobs.items() if not e.get("prev")}   # an ABBA confirmation's numbers live on its first job (confirmed)
    for name, e in sorted(sweep.items(), key=lambda kv: (kv[1]["slug"], kv[1]["device"])):
        r = refs.get(e["slug"]) or {}
        src = (r.get("old") or e.get("source_card") or "@").split("@")
        mod = doc["models"].setdefault(e["slug"], {"production": src[0], "production_revision": src[1], "programs_sha256": r.get("programs_sha256"),
                                                   "kind": r.get("kind", e.get("kind", "text")), "claimed_devices": r.get("claimed", []), "devices": {},
                                                   **({"versions": {"package": {"skills_commit": {"commit": r["generator"], "recovered": True,
                                                                                                  "basis": "production card"}}}}
                                                      if r.get("generator") else {})})
        status = device_status(e)
        dv = {"status": status, "reason": e.get("reason")}
        if status not in ("no_runner", "not_claimed", "does_not_fit", "pending"):
            dv.update(device_evidence(st, dict(e, name=name), info_cache))
        if e.get("package"):
            dv["package"] = {k: e["package"].get(k) for k in ("revision", "programs_sha256", "base")}
        if e.get("published_revision"):
            dv["published_revision"] = e["published_revision"]
        if e.get("accepted_regression"):
            dv["accepted_regression"] = e["accepted_regression"]
        if e.get("no_reference"):
            dv["gates_not_run"] = e["no_reference"]
        if e.get("not_applicable"):
            dv["not_applicable"] = e["not_applicable"]
        mod["devices"][e["device"]] = dv
    return doc


def cmd_evidence(a):
    st = State(a.state)
    doc = build_evidence(st)
    json.dump(doc, open(a.out, "w"), indent=1, sort_keys=True)
    by = {}
    for m in doc["models"].values():
        for dv in m["devices"].values():
            by[dv["status"]] = by.get(dv["status"], 0) + 1
    print(f"evidence: {len(doc['models'])} models, devices {dict(sorted(by.items()))}: {a.out}")


def eligibility(doc):
    """(eligible, held, pending) of an sdk-evidence/1 document. Fail closed: a model is eligible only when every device its card claims that has a
    DevLab runner is `qualified` (a passing job of the shipped programs on the verified SDK) and the staging card is published; device_keys = the
    qualified devices; qualification run + job per device, never null. Runner-less claimed devices are listed as carried (owner policy)."""
    eligible, pending = {}, {}
    held = {s: dict(v, reason=v.get("reason") or "held out of this release") for s, v in doc.get("held", {}).items()}
    for slug, m in sorted(doc["models"].items()):
        devs = m["devices"]
        runner = {d: v for d, v in devs.items() if v["status"] not in ("no_runner", "not_claimed")}
        bad = {d: f"{v['status']}: {(v.get('reason') or '')[:160]}" for d, v in runner.items() if v["status"] != "qualified"}
        if not runner:
            bad["-"] = "no device with a DevLab runner was qualified"
        qual = {d: v for d, v in runner.items() if v["status"] == "qualified"}
        for d, v in qual.items():
            if not (v.get("run") and v.get("job")):
                bad[d] = "qualified without a run and job id"
            elif v.get("conclusion") != "success":
                bad[d] = f"the qualification job concluded {v.get('conclusion')}"
            elif m["kind"] == "text" and not v.get("programs_match"):
                bad[d] = "the qualification job served other programs"
        staged = sorted({v.get("published_revision") for v in qual.values() if v.get("published_revision")})
        unpublished = sorted(d for d, v in qual.items() if not v.get("published_revision"))
        if not bad and m["kind"] == "text" and unpublished:
            bad["staging"] = f"the staging card is not published for {', '.join(unpublished)} yet"
        if bad:
            pending[slug] = {"production": m["production"], "reasons": bad}
            continue
        eligible[slug] = {
            "production": m["production"], "target_revision": m["production_revision"], "frozen_production_revision": m["production_revision"],
            "programs_sha256": m["programs_sha256"], "device_keys": sorted(qual),
            "qualification": {d: {"run": int(v["run"]), "job": int(v["job"]), "url": v["job_url"]} for d, v in sorted(qual.items())},
            "carried_devices": sorted(d for d, v in devs.items() if v["status"] == "no_runner"),
            "staging": f"{STAGING_ORG}/{slug}", "staging_revision": staged[-1] if staged else None, "kind": m["kind"]}
    return eligible, held, pending


def cmd_eligible(a):
    st = State(a.state)
    doc = build_evidence(st)
    eligible, held, pending = eligibility(doc)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, d in (("eligible.json", eligible), ("held.json", held), ("pending.json", pending)):
        json.dump(d, open(os.path.join(a.out_dir, name), "w"), indent=1, sort_keys=True)
    print(f"eligible {len(eligible)}, held {len(held)}, pending {len(pending)}: {a.out_dir}")
    for s, p in pending.items():
        print(f"  pending {s}: " + "; ".join(f"{k}: {v}" for k, v in p["reasons"].items())[:300])


def cmd_verify(a):
    """--programs production: check each published card on the staging head (small files): the frozen programs, the target pin, this device's
    session from the job, and lint_card.py --form staging. Records e["verified"] = {revision, ok, problems}; prints the problems."""
    st = State(a.state)
    pin = pinned_target(st.d)
    for name, e in st.jobs.items():
        if e["stage"] != "published" or (e.get("verified") or {}).get("ok"):
            continue
        repo = f"{st.d['opts']['new_org']}/{e['slug']}"
        d = os.path.join(st.work, "verify", e["slug"])
        problems = []
        try:
            head = hf_head(repo)
            hf_snapshot_small(repo, head, d)
            if card_lib.programs_sha256(d) != e.get("programs_sha256"):
                problems.append("staging programs are not the production programs")
            got = (sdk_package.load(os.path.join(d, "llm_config.json")).get("sdk") or {})
            if pin and any(got.get(k) != v for k, v in pin.items()):
                problems.append(f"llm_config pin {got.get('hrx_loom')} is not the verified {target_sdk(st.d)}")
            info = card_lib.load_info(os.path.join(d, "info.json"))
            srcs = {s_.get("url") for s_ in (info.get("sources") or {}).values()}
            if e.get("job_url") not in srcs:
                problems.append(f"the card does not cite this device's job {e.get('job_url')}")
            r = sh([PY, os.path.join(CARD, "lint_card.py"), d, "--form", "staging"], check=False)
            problems += [ln.split(": ", 1)[-1] for ln in r.stdout.splitlines() if "[carried]" not in ln and ": [" in ln]
        except Exception as x:
            problems.append(f"verify failed: {x}")
        e["verified"] = {"revision": locals().get("head"), "ok": not problems, "problems": problems, "at": now()}
        print(f"{e['slug']} {e['device']}: " + ("ok" if not problems else "; ".join(problems)[:600]))
        shutil.rmtree(d, ignore_errors=True)
        st.save()


def cmd_accept_no_reference(a):
    """--programs production: a model whose gates could not run because the reference engine cannot serve its checkpoint (gates.sh: llama-server
    would not start, "references still pending after 5 rounds"; e.g. an architecture only a hand-made reference serves) is decided on its other
    evidence; its card carries the previous gates, shown not re-run on this SDK (the job's gate results are not ingested). The note names why.
    A gate that ran against a reference and failed is never accepted here."""
    st = State(a.state)
    if st.d["opts"].get("programs") != "production":
        die("accept-no-reference is for --programs production states")
    done = []
    for name, e in st.jobs.items():
        if e["slug"] not in a.slug or e["stage"] != "needs_decision" or e.get("prev") or not e.get("cmp"):
            continue
        fails = json.load(open(e["cmp"])).get("gate_failures") or []
        out = read(os.path.join(e.get("dir") or "", "gates_staging.out"), "") or ""
        if not fails or any(not f.startswith("gates:") for f in fails) or "references still pending after 5 rounds" not in out \
                or "could not start llama-server" not in out:
            print(f"{e['slug']} {e['device']}: not a reference that could not start ({', '.join(fails) or 'no gate failure'}), not accepted")
            continue
        decision, reasons, perf, rows = _decide_without(st, name, e, True)
        e.update(perf=perf, rows_pct=rows, perf_cross=dict(perf), decision=decision, reason="; ".join(reasons), no_reference=a.note)
        e["stage"] = "collected" if decision in ("promote", "confirm") else "needs_decision"
        st.log(f"{e['slug']} {e['device']}: gates not run (no reference): {a.note} -> {e['stage']} ({decision})")
        done.append(f"{e['slug']} {e['device']} ({decision})")
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
    p.add_argument("--new-sdk", default=None,
                   help=f"SDK target of sdk-ab or sdk-loom (default: the newest verified release, {SDK_NEW}; sdk_pin.VERIFIED_RELEASES)")
    p.add_argument("--programs", choices=("build", "production"), default="build",
                   help="SDK modes: build = the new package is built in the job from --base (the kit's programs); production = the FROZEN form: every "
                        "job serves the production package at its inventory revision (skills/loom-migrate/versions/vOLD-to-vNEW-production.json) "
                        "byte for byte, programs hash checked by the job, only the SDK differs; image packages get an image-service job")
    p.add_argument("--old-sdk", choices=("card", "target"), default="card",
                   help="SDK modes: the SDK the OLD package is served with: card = its own pin (the SDK A/B), target = the --new-sdk too (a programs-only A/B "
                        "on the new SDK, e.g. the kit build against the frozen production programs: --mode sdk-ab --old-sdk target --no-baselines)")
    p.add_argument("--held", help="--programs production: held.json {slug: {reason}} of packages out of this release (refused by name, skipped by --all; "
                                  "default: the inventory's sibling vOLD-to-vNEW-held.json)")
    p.add_argument("--llamacpp", action="append", metavar="SLUG=TAG", help="--programs production: the llama.cpp ROCm reference build of a package "
                   ".github/nightly-models.json does not list (a hand-written port whose architecture needs a newer build), e.g. "
                   "unsloth-Inkling-2-Mini-loom=b1384 (skills/loom-kb/arch/llm/moe/inkling2/porting.md)")
    p.add_argument("--image-device", default="strix-halo", help="--programs production: the device whose runner compares an image package (default %(default)s)")
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
            q.add_argument("--dry-run", action="store_true", help="--programs production: build each card on the locally assembled frozen package "
                                                                  "(no staging package needed), record nothing; for review and lint_card.py")
        if name == "package":
            q.add_argument("--slugs", nargs="+", help="only these models, whatever their stage (default: the collected jobs that were promoted)")
            q.add_argument("--ref", help="the git ref the canonical build is made at (default: origin/<the state's base>)")
            q.add_argument("--prefill-transition", action="append", default=[], metavar="SLUG:PRESET:OLD:NEW",
                           help="explicit measured prefill change; every selected job must provide matching programs and preset execution")
    q = sub.add_parser("verify", help="--programs production: check the published staging cards (programs, pin, cited job, lint_card.py)")
    q.add_argument("state")
    q.set_defaults(fn=cmd_verify)
    q = sub.add_parser("accept-no-reference", help="--programs production: gates that could not run (no reference engine for the checkpoint)")
    q.add_argument("state")
    q.add_argument("slug", nargs="+")
    q.add_argument("--note", required=True, help="why no reference serves it, e.g. 'stock llama.cpp b1384 cannot load the inkling2 GGUF (KB inkling2/porting.md)'")
    q.set_defaults(fn=cmd_accept_no_reference)
    for name, fn, help_ in (("evidence", cmd_evidence, "the sdk-evidence/1 document of a --programs production state (per model / device / metric)"),
                            ("eligible", cmd_eligible, "eligible.json / held.json / pending.json of the release from a --programs production state (fail closed)")):
        q = sub.add_parser(name, help=help_)
        q.add_argument("state")
        if name == "evidence":
            q.add_argument("-o", "--out", required=True)
        else:
            q.add_argument("--out-dir", required=True)
        q.set_defaults(fn=fn)
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
