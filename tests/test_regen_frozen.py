"""fleet/regen.py --programs production (the frozen SDK migration), the version-generic SDK constants, the evidence / eligible generator,
ab_jobs.py --programs production and the image comparison's request (no network, no GPU)."""
import contextlib
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(ROOT, "fleet"), os.path.join(ROOT, ".github", "scripts")):
    sys.path.insert(0, p)
import regen  # noqa: E402
import ab_jobs  # noqa: E402
import image_sdk_compare  # noqa: E402

TEXT = "unsloth-Qwen3.5-4B-loom"
IMAGE = "Tongyi-MAI-Z-Image-Turbo-loom"
STRIX_ONLY = "example-HandWritten-Strix-loom"   # a package .github/nightly-models.json does not list
HELD = "SenseNova-U1.6-MoT-R9700-loom"
H = "6a47" + "0" * 60
REV = {TEXT: "64bf40259117d99f3e6a9f43b7e80fb030300697"}


def facts(repo, rev, cache):
    slug = repo.split("/")[-1]
    if slug == IMAGE:
        return {"programs_sha256": "i" * 64, "kind": "image", "claimed": [], "memory_bytes": None, "features": [], "sdk": None}
    if slug == STRIX_ONLY:
        return {"programs_sha256": "k" * 64, "kind": "text", "claimed": ["strix-halo"], "memory_bytes": 33000000000, "features": ["draft"],
                "sdk": {"hrx_loom": "0.2.0"}}
    return {"programs_sha256": H, "kind": "text", "claimed": ["gfx1150", "gfx1201-16gb", "radeon-r9700", "strix-halo"], "memory_bytes": 28000000000,
            "features": ["draft"], "sdk": {"hrx_loom": "0.2.0"}}


class VersionGeneric(unittest.TestCase):
    def test_newest_verified_release_is_the_default_target(self):
        self.assertEqual(regen.SDK_NEW, max(regen.VERIFIED_RELEASES, key=regen.version_key))
        with mock.patch.dict(regen.VERIFIED_RELEASES, {"0.10.0": {"hrx_system_sha": "x", "wheels": {"manylinux_2_39_x86_64": "y"}}}):
            self.assertEqual(regen.sdk_pin.newest_verified(), "0.10.0")   # numeric ordering, not string ordering
            self.assertEqual(regen.release_pin("0.10.0")["wheel_sha256"], "y")
        self.assertGreater(regen.version_key("0.10.0"), regen.version_key("0.9.1"))
        with self.assertRaises(ValueError):
            regen.version_key("card")

    def test_release_pin_comes_from_the_verified_table(self):
        pin = regen.release_pin("0.3.0")
        rel = regen.VERIFIED_RELEASES["0.3.0"]
        self.assertEqual(pin, {"hrx_loom": "0.3.0", "source": "pypi", "hrx_system_sha": rel["hrx_system_sha"],
                               "wheel_sha256": rel["wheels"]["manylinux_2_39_x86_64"]})
        self.assertIsNone(regen.release_pin("0.2.0"))   # not verified here: the legacy staging-history path

    def test_no_hard_coded_target_version_in_regen(self):
        src = open(os.path.join(ROOT, "fleet", "regen.py")).read()
        self.assertNotIn("SDK_030", src)
        self.assertEqual(re.findall(r'==\s*"0\.\d+\.\d+"', src), [])
        self.assertNotIn('"0.3.0"', src)

    def test_legacy_state_without_a_recorded_target(self):
        self.assertEqual(regen.target_sdk({"opts": {}}), regen.LEGACY_NEW_SDK)
        self.assertEqual(regen.target_sdk({"opts": {"new_sdk": "0.3.0"}}), "0.3.0")

    def test_new_sdk_pin_follows_the_target(self):
        self.assertTrue(regen.new_sdk_pin("0.3.0", "0.3.0"))
        self.assertFalse(regen.new_sdk_pin("0.2.0", "0.3.0"))
        self.assertTrue(regen.new_sdk_pin("0.2.0rc1", "0.2.0"))


def fixture_versions(d):
    """a versions/ directory with an accepted 0.3.0 inventory (a drift record included) and its held list"""
    inv = {f"loom-fun/{s}": "a" * 40 for s in (TEXT, IMAGE, STRIX_ONLY, HELD, "unsloth-Qwen3-0.6B-loom")}
    inv[f"loom-fun/{TEXT}"] = REV[TEXT]
    inv["_drift"] = {"moved": {}}
    json.dump(inv, open(os.path.join(d, "v0.2.0-to-v0.3.0-production.json"), "w"))
    json.dump({HELD: {"reason": "owner decision"}}, open(os.path.join(d, "v0.2.0-to-v0.3.0-held.json"), "w"))
    return d


class Inventory(unittest.TestCase):
    def test_inventory_of_the_target_and_its_old_version(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(regen, "VERSIONS", fixture_versions(d)):
            path = regen.inventory_path("0.3.0")
            self.assertTrue(path.endswith("v0.2.0-to-v0.3.0-production.json"))
            self.assertEqual(regen.inventory_old_version(path), "0.2.0")
            inv = regen.load_inventory(path)
            self.assertEqual(len(inv), 5)   # "_drift" is a record, not a package
            self.assertEqual(set(regen.load_held(path.replace("-production.json", "-held.json"))), {HELD})
            self.assertIsNone(regen.inventory_path("9.9.9"))

    def test_the_repository_inventory_pins_full_revisions(self):
        path = regen.inventory_path(regen.SDK_NEW)
        inv = regen.load_inventory(path)
        self.assertGreaterEqual(len(inv), 61)
        self.assertTrue(all(k.startswith("loom-fun/") and re.fullmatch(r"[0-9a-f]{40}", v) for k, v in inv.items()))

    def test_drift(self):
        import sdk_inventory
        d = sdk_inventory.drift({"a/x": "1", "a/y": "2", "_drift": {}}, {"a/x": "1", "a/y": "3", "a/z": "4"})
        self.assertEqual(d, {"added": {"a/z": "4"}, "removed": {}, "moved": {"a/y": {"from": "2", "to": "3"}}})


class AbJobsFrozen(unittest.TestCase):
    def doc(self):
        ref = f"loom-fun/{TEXT}@{REV[TEXT]}"
        return {"_comment": "x", "jobs": [{"name": "rab-x", "card": ref, "variants": [
            {"name": "fun", "card": ref, "execution": {}, "cap_pool": 1},
            {"name": "staging", "context": 4096},
            {"name": "funnd", "card": ref, "execution": {"draft_depth": 0}},
            {"name": "stagingnd", "execution": {"draft_depth": 0}}], "steps": []}]}

    def test_every_variant_serves_the_pinned_package_with_its_hash(self):
        ref = f"loom-fun/{TEXT}@{REV[TEXT]}"
        doc = ab_jobs.frozen(self.doc(), "card", "0.3.0", {TEXT: {"old": ref, "new": ref, "programs_sha256": H}})
        j = doc["jobs"][0]
        self.assertEqual(j["sdk"], "0.3.0")
        self.assertNotIn("admit:staging", j["steps"])   # no gates in this job: no probe added
        by = {v["name"]: v for v in j["variants"]}
        for name in ("staging", "stagingnd"):
            self.assertNotIn("build", by[name])
            self.assertNotIn("card", by[name])   # the job's card: the pinned production revision
            self.assertEqual((by[name]["sdk"], by[name]["programs_sha256"]), ("0.3.0", H))
        self.assertEqual(by["staging"]["context"], 4096)   # the generator's serving settings stay
        for name, new in (("fun", "staging"), ("funnd", "stagingnd")):
            self.assertEqual(by[name]["card"], ref)
            self.assertEqual(by[name]["sdk"], "card")
            self.assertEqual({k: v for k, v in by[name].items() if k not in ("name", "card", "sdk")},
                             {k: v for k, v in by[new].items() if k not in ("name", "sdk")})   # same settings: the SDK is the only difference
        lo = ab_jobs.loom_only(doc)
        self.assertEqual([v["name"] for v in lo["jobs"][0]["variants"]], ["staging", "stagingnd"])

    def test_refuses_without_the_production_hash(self):
        with self.assertRaises(SystemExit):
            ab_jobs.frozen(self.doc(), "card", "0.3.0", {})

    def test_programs_production_needs_sdk_ab(self):
        with self.assertRaises(SystemExit):
            ab_jobs.main(["--device", "strix-halo", "--programs", "production", TEXT])

    def test_model_entry_fallback(self):
        self.assertEqual(ab_jobs.model_entry({}, {STRIX_ONLY: {"model": {"memory_bytes": 1}}}, STRIX_ONLY), {"memory_bytes": 1})
        with self.assertRaises(SystemExit):
            ab_jobs.model_entry({}, {}, STRIX_ONLY)


class PlanFrozen(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = os.path.join(self.tmp, "s.json")
        self.calls = []

    def gen(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        dev = cmd[cmd.index("--device") + 1]
        key = regen.card_lib.registry.device(dev)["runner"]["key"]
        refs = json.load(open(cmd[cmd.index("--refs") + 1]))
        slugs = [c for c in cmd if c in refs]
        jobs = [{"name": f"{key[0]}ab-{s.lower()}"[:40], "card": refs[s]["new"], "variants": [], "steps": ["perf:staging:1"]} for s in slugs]
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"jobs": jobs}), "")

    def plan(self, *slugs, extra=()):
        versions = fixture_versions(tempfile.mkdtemp(dir=self.tmp))
        with mock.patch.object(regen, "sh", self.gen), mock.patch.object(regen, "production_facts", facts), mock.patch.object(regen, "VERSIONS", versions), \
                contextlib.redirect_stdout(io.StringIO()):
            regen.main(["plan", "--base", "sdk030", "--device", "strix-halo", "--device", "radeon-r9700", "--mode", "sdk-loom", "--programs", "production",
                        "--state", self.state, *extra, *slugs])
        return regen.State(self.state)

    def test_plan_pins_production_and_claims(self):
        st = self.plan(TEXT, IMAGE, STRIX_ONLY)
        o = st.d["opts"]
        self.assertEqual((o["programs"], o["new_sdk"], o["old_sdk"]), ("production", regen.SDK_NEW, "0.2.0"))
        self.assertEqual(set(o["held"]), {HELD})
        for cmd in self.calls:
            self.assertIn("--programs", cmd)
            self.assertEqual(cmd[cmd.index("--programs") + 1], "production")
            self.assertEqual(cmd[cmd.index("--new-sdk") + 1], regen.SDK_NEW)
            self.assertIn("--loom-only", cmd)
        text = {n: e for n, e in st.jobs.items() if e["slug"] == TEXT and e["stage"] == "planned"}
        self.assertEqual(sorted(e["device"] for e in text.values()), ["radeon-r9700", "strix-halo"])
        for e in text.values():
            self.assertEqual(e["programs_sha256"], H)
            self.assertEqual(e["source_card"], f"loom-fun/{TEXT}@{REV[TEXT]}")
            self.assertEqual(e["old"], e["source_card"])
        carried = {e["device"] for e in st.jobs.values() if e.get("no_runner")}
        self.assertEqual(carried, {"gfx1150", "gfx1201-16gb"})
        # the Strix-only card gets no R9700 job; its refs carry a nightly-shaped model entry (it is not in nightly-models.json)
        self.assertEqual(st.jobs[f"skipped:{STRIX_ONLY}:radeon-r9700"]["reason"], "the production card does not claim this device")
        refs = json.load(open(os.path.join(st.work, "refs_strix-halo.json")))
        self.assertEqual(refs[STRIX_ONLY]["model"]["memory_bytes"], 33000000000)
        self.assertNotIn("llamacpp", refs[STRIX_ONLY]["model"])

        # the image package: one image job on Strix Halo, none on the R9700
        img = [e for e in st.jobs.values() if e["slug"] == IMAGE]
        self.assertEqual([(e["device"], e["kind"]) for e in img], [("strix-halo", "image")])
        batch = next(b for b in st.d["batches"] if b["jobs"] == [n for n, e in st.jobs.items() if e["slug"] == IMAGE])
        job = json.load(open(batch["perf"]))["jobs"][0]
        self.assertEqual(job["kind"], "image")
        self.assertEqual(job["image"], {"old_sdk": "0.2.0", "new_sdk": regen.SDK_NEW, "programs_sha256": "i" * 64, "preset": None})

    def test_llamacpp_override_of_an_unlisted_package(self):
        st = self.plan(STRIX_ONLY, extra=("--llamacpp", f"{STRIX_ONLY}=b1384"))
        self.assertEqual(json.load(open(os.path.join(st.work, "refs_strix-halo.json")))[STRIX_ONLY]["model"]["llamacpp"], "b1384")
        with self.assertRaises(SystemExit):   # a package nightly-models.json lists keeps its own build
            self.plan(TEXT, extra=("--llamacpp", f"{TEXT}=b1384"))

    def test_held_and_unknown_packages_are_refused(self):
        with self.assertRaises(SystemExit):
            self.plan(HELD)
        with self.assertRaises(SystemExit):
            self.plan("not-a-production-package-loom")

    def test_all_is_the_inventory_minus_held(self):
        st = self.plan(extra=("--all",))
        slugs = {e["slug"] for e in st.jobs.values()}
        self.assertEqual(slugs, {TEXT, IMAGE, STRIX_ONLY, "unsloth-Qwen3-0.6B-loom"})   # the inventory minus the held package

    def test_production_needs_an_sdk_mode_and_old_sdk_target_needs_a_build(self):
        with self.assertRaises(SystemExit), mock.patch.object(regen, "sh", self.gen), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["plan", "--base", "b", "--device", "strix-halo", "--programs", "production", "--state", self.state, TEXT])
        with self.assertRaises(SystemExit), mock.patch.object(regen, "sh", self.gen), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["plan", "--base", "b", "--device", "strix-halo", "--mode", "sdk-loom", "--programs", "production", "--old-sdk", "target",
                        "--state", self.state, TEXT])

    def test_old_sdk_target_serves_the_old_package_on_the_target(self):
        st = {"mode": "sdk-ab", "opts": {"gates": "dev", "new_org": regen.STAGING_ORG, "old_org": "loom-fun", "new_sdk": "0.3.0", "old_sdk_target": True,
                                         "no_baselines": True}}
        cmd = [str(c) for c in regen.generator_cmd(st, "radeon-r9700", [TEXT])]
        self.assertEqual(cmd[cmd.index("--old-sdk") + 1], "0.3.0")
        self.assertNotIn("--baselines", cmd)


class CollectFrozen(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        open(os.path.join(self.d, "programs_sha256_staging.txt"), "w").write(H + "\n")
        for v in ("staging", "stagingnd"):
            json.dump(regen.release_pin("0.3.0"), open(os.path.join(self.d, f"sdk_{v}.json"), "w"))
        self.st = SimpleNamespace(d={"mode": "sdk-loom", "opts": {"new_sdk": "0.3.0", "programs": "production"}})

    def test_guard(self):
        e = {"dir": self.d, "programs_sha256": H, "conclusion": "success"}
        self.assertIn("no native-context admission", regen.frozen_guard(self.st, e))
        json.dump({"admitted": False, "tokens": 100, "context": 262144, "detail": "HTTP 400"}, open(os.path.join(self.d, "admit_staging.json"), "w"))
        self.assertIn("refused", regen.frozen_guard(self.st, e))
        json.dump({"admitted": True, "tokens": 262066, "context": 262144}, open(os.path.join(self.d, "admit_staging.json"), "w"))
        open(os.path.join(self.d, "admit_staging.txt"), "w").write("WARN: the server did not cancel the request within 60s\nADMITTED\n")
        self.assertEqual(regen.admission(self.d)["warnings"], ["WARN: the server did not cancel the request within 60s"])
        self.assertIsNone(regen.frozen_guard(self.st, e))
        self.assertIn("concluded failure", regen.frozen_guard(self.st, dict(e, conclusion="failure")))
        no_conclusion = {k: v for k, v in e.items() if k != "conclusion"}   # retro review: a missing conclusion used to pass the guard
        self.assertIn("without a recorded conclusion", regen.frozen_guard(self.st, no_conclusion))
        self.assertIn("not the production programs", regen.frozen_guard(self.st, dict(e, programs_sha256="b" * 64)))
        json.dump({"hrx_loom": "0.3.0", "wheel_sha256": "x"}, open(os.path.join(self.d, "sdk_stagingnd.json"), "w"))
        self.assertIn("not the verified", regen.frozen_guard(self.st, e))

    def test_image_verdict(self):
        res = {"results": {"image:old": "PASS hrx-loom 0.2.0", "image:new": "PASS hrx-loom 0.3.0", "pixels": "PASS pixels identical"}}
        json.dump(res, open(os.path.join(self.d, "summary.json"), "w"))
        json.dump({"preset": "image", "pixels_match": True, "programs_match": True, "runs": {
            "old": {"median_s": 7.0, "deterministic": True, "sessions": [{"rgb_sha256": ["a"]}]},
            "new": {"median_s": 6.9, "deterministic": True, "sessions": [{"rgb_sha256": ["a"]}]}}}, open(os.path.join(self.d, "image_result.json"), "w"))
        e = {}
        self.assertEqual(regen.image_verdict(self.d, e), ("promote", []))
        self.assertEqual(e["image"]["new"]["rgb_sha256"], ["a"])
        res["results"]["pixels"] = "FAIL pixels differ"
        json.dump(res, open(os.path.join(self.d, "summary.json"), "w"))
        self.assertEqual(regen.image_verdict(self.d, {})[0], "hold")


class Eligibility(unittest.TestCase):
    def dev(self, status="qualified", **kw):
        return {"status": status, "run": "45", "job": "138", "job_url": "https://github.com/x/actions/runs/45/job/138", "conclusion": "success",
                "programs_match": True, "published_revision": "s" * 40, **kw}

    def doc(self, devices):
        return {"models": {TEXT: {"production": f"loom-fun/{TEXT}", "production_revision": REV[TEXT], "programs_sha256": H, "kind": "text",
                                  "claimed_devices": sorted(devices), "devices": devices}},
                "held": {HELD: {"reason": "owner", "tag_revision": "t" * 40}}}

    HEAD = staticmethod(lambda repo: "s" * 40)

    def test_every_runner_device_qualified(self):
        el, held, pend = regen.eligibility(self.doc({"strix-halo": self.dev(), "radeon-r9700": self.dev(run="46", job="139"),
                                                     "gfx1150": {"status": "no_runner"}}), self.HEAD)
        row = el[TEXT]
        self.assertEqual(row["device_keys"], ["radeon-r9700", "strix-halo"])
        self.assertEqual(row["qualification"]["radeon-r9700"]["run"], 46)
        self.assertEqual(row["carried_devices"], ["gfx1150"])
        self.assertEqual((row["programs_sha256"], row["target_revision"], row["staging_revision"]), (H, REV[TEXT], "s" * 40))
        self.assertEqual(held[HELD]["tag_revision"], "t" * 40)
        self.assertEqual(pend, {})
        self.assertEqual(row["gates_not_rerun"], {})

    def test_staging_revision_is_the_latest_publish_and_the_live_head(self):
        # retro review: sorted(revisions)[-1] picked the alphabetically largest sha ("f..."), not the latest publish ("a...")
        devs = {"strix-halo": self.dev(published_revision="a" * 40, published="2026-11-25T03:00:00Z"),
                "radeon-r9700": self.dev(run="46", job="139", published_revision="f" * 40, published="2026-11-25T01:00:00Z")}
        el, _, pend = regen.eligibility(self.doc(devs), lambda repo: "a" * 40)
        self.assertEqual(el[TEXT]["staging_revision"], "a" * 40)
        el, _, pend = regen.eligibility(self.doc(devs), lambda repo: "b" * 40)   # published since by something else
        self.assertEqual(el, {})
        self.assertIn("is not the latest published revision", pend[TEXT]["reasons"]["staging"])
        el, _, pend = regen.eligibility(self.doc(devs))   # never eligible without the live head
        self.assertEqual(el, {})
        self.assertIn("not checked", pend[TEXT]["reasons"]["staging"])

        def down(repo):
            raise OSError("503")
        self.assertIn("cannot read the live staging head", regen.eligibility(self.doc(devs), down)[2][TEXT]["reasons"]["staging"])

    def test_gates_not_rerun_listed(self):
        el, _, _ = regen.eligibility(self.doc({"strix-halo": self.dev(gates_not_run="no stock reference")}), self.HEAD)
        self.assertEqual(el[TEXT]["gates_not_rerun"], {"strix-halo": "no stock reference"})

    def image_doc(self, image):
        dev = {**self.dev(), "image": image}
        dev.pop("programs_match")
        return {"models": {IMAGE: {"production": f"loom-fun/{IMAGE}", "production_revision": "a" * 40, "programs_sha256": "i" * 64, "kind": "image",
                                   "claimed_devices": [], "devices": {"strix-halo": dev}}}, "held": {}}

    def test_image_rows_carry_their_pixel_verdict(self):
        good = {"preset": "image", "request_sha256": "r" * 64, "pixels_match": True, "programs_match": True, "max_abs_channel_diff": 0,
                "old": {"deterministic": True}, "new": {"deterministic": True}, "time_ratio": 1.0}
        el, _, _ = regen.eligibility(self.image_doc(good), self.HEAD)
        q = el[IMAGE]["qualification"]["strix-halo"]
        self.assertEqual((q["run"], q["job"]), (45, 138))
        self.assertEqual(q["image"], {"preset": "image", "request_sha256": "r" * 64, "pixels_match": True, "programs_match": True,
                                      "max_abs_channel_diff": 0, "deterministic": {"old": True, "new": True}})
        for bad in ({**good, "pixels_match": False}, {**good, "max_abs_channel_diff": 3}, {**good, "programs_match": False},
                    {**good, "new": {"deterministic": False}}, None):
            el, _, pend = regen.eligibility(self.image_doc(bad), self.HEAD)
            self.assertEqual(el, {}, bad)
            self.assertIn(IMAGE, pend)

    def test_fail_closed(self):
        for bad in (self.dev("failed"), self.dev(conclusion="failure"), self.dev(run=None), self.dev(programs_match=False),
                    self.dev("pending"), self.dev(published_revision=None)):
            el, _, pend = regen.eligibility(self.doc({"strix-halo": self.dev(), "radeon-r9700": bad}), self.HEAD)
            self.assertEqual(el, {}, bad)
            self.assertIn(TEXT, pend)
        el, _, pend = regen.eligibility(self.doc({"gfx1150": {"status": "no_runner"}}), self.HEAD)
        self.assertEqual(el, {})


class PackageFrozen(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.prod = os.path.join(self.tmp, "prod")
        os.makedirs(os.path.join(self.prod, "program"))
        open(os.path.join(self.prod, "program", "llm.loom"), "w").write("frozen")
        for f in ("model.json", "tokenizer.json", "README.md", "info.json"):
            open(os.path.join(self.prod, f), "w").write(f)
        json.dump({"sdk": {"hrx_loom": "0.2.0"}}, open(os.path.join(self.prod, "llm_config.json"), "w"))
        self.want = regen.card_lib.programs_sha256(self.prod)
        self.job_dir = os.path.join(self.tmp, "job")
        os.makedirs(self.job_dir)
        open(os.path.join(self.job_dir, "programs_sha256_staging.txt"), "w").write(self.want)
        for v in ("staging", "stagingnd"):
            json.dump(regen.release_pin("0.3.0"), open(os.path.join(self.job_dir, f"sdk_{v}.json"), "w"))
        self.st = SimpleNamespace(work=self.tmp, d={"opts": {"new_org": regen.STAGING_ORG, "new_sdk": "0.3.0"}})
        self.job = {"slug": TEXT, "device": "strix-halo", "dir": self.job_dir, "decision": "promote", "programs_sha256": self.want,
                    "source_card": f"loom-fun/{TEXT}@{REV[TEXT]}"}

    def run_package(self, staging_files, repo_files=(), dry=False, job=None):
        commits = []

        class Api:
            def list_repo_files(self, repo, revision=None):
                return list(repo_files) if repo.startswith("loom-fun/") else list(staging_files)

            def create_commit(self, repo, operations, parent_commit, commit_message, commit_description):
                for o in operations:   # the files are read at commit time (the work dir is removed afterwards)
                    if o.kind == "add":
                        o.content = open(o.path_or_fileobj).read()
                commits.append((repo, operations, parent_commit))
                return SimpleNamespace(oid="n" * 40)

        def snap(repo, rev, out):
            shutil.rmtree(out, ignore_errors=True)
            if repo.startswith("loom-fun/"):
                shutil.copytree(self.prod, out)
            else:
                os.makedirs(out)
                for f in staging_files:
                    os.makedirs(os.path.dirname(os.path.join(out, f)) or out, exist_ok=True)
                    open(os.path.join(out, f), "w").write("old " + f)
        hub = SimpleNamespace(HfApi=Api, CommitOperationAdd=lambda path_in_repo, path_or_fileobj: SimpleNamespace(
            kind="add", path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj),
            CommitOperationDelete=lambda path_in_repo: SimpleNamespace(kind="delete", path_in_repo=path_in_repo))
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), mock.patch.object(regen, "hf_snapshot_small", snap), \
                mock.patch.object(regen, "hf_head", lambda r: "h" * 40), contextlib.redirect_stdout(io.StringIO()):
            res = regen.package_frozen(self.st, TEXT, [job or self.job], dry)
        return res, commits

    def test_staging_becomes_the_production_package_with_the_new_pin(self):
        res, commits = self.run_package(["program/llm.loom", "program/fused_extra.loom", "README.md", ".gitattributes"])
        self.assertEqual(res["programs_sha256"], self.want)
        self.assertEqual(res["deleted"], ["program/fused_extra.loom"])   # the kit-built leftovers go
        repo, ops, parent = commits[0]
        self.assertEqual((repo, parent), (f"{regen.STAGING_ORG}/{TEXT}", "h" * 40))
        add = {o.path_in_repo: o for o in ops if o.kind == "add"}
        self.assertEqual(json.loads(add["llm_config.json"].content)["sdk"], regen.release_pin("0.3.0"))
        self.assertEqual(add["program/llm.loom"].content, "frozen")

    def test_refusals(self):
        with self.assertRaisesRegex(RuntimeError, "did not promote"):
            self.run_package([], job=dict(self.job, decision="hold"))
        with self.assertRaisesRegex(RuntimeError, "did not promote"):
            self.run_package([], job=dict(self.job, programs_sha256="z" * 64))
        with self.assertRaisesRegex(RuntimeError, "weight files"):
            self.run_package([], repo_files=["parameter/x.gguf"])
        with self.assertRaisesRegex(RuntimeError, "image"):
            self.run_package([], job=dict(self.job, kind="image"))

    def test_dry_run_uploads_nothing(self):
        res, commits = self.run_package(["README.md"], dry=True)
        self.assertEqual((res["revision"], commits), (None, []))

    def test_a_model_with_only_its_abba_confirmation_promoted_is_skipped(self):
        """the sweep job waits for a decision while its ABBA job is promoted: nothing to package for that model (no crash), the next model is"""
        abba = dict(self.job, stage="collected", prev="first")
        other = dict(self.job, slug="other", stage="collected", package={"revision": "p" * 40})
        st = SimpleNamespace(d={"opts": {}}, save=lambda: None, log=lambda m: None)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            regen.cmd_package_frozen(st, SimpleNamespace(dry_run=False), [("abba", abba), ("o", other)])
        self.assertIn("other: package already published", out.getvalue())
        self.assertNotIn(TEXT, out.getvalue())


class ImageRequest(unittest.TestCase):
    def test_text_to_image_and_upscale_requests(self):
        m = {"model": {"id": "z"}, "services": {"image": {"kind": "image", "interface": {"input": "text prompt"}}},
             "presets": {"image": {"service": "image", "execution": {"width": 1024, "height": 768}, "request": {"prompt": "p", "seed": 42}}}}
        body, size = image_sdk_compare.request_body(m, "image")
        self.assertEqual((body["prompt"], body["seed"], body["model"], size), ("p", 42, "z", (1024, 768)))
        m["services"]["image"]["interface"]["input"] = "base64 of the raw RGB input image in the prompt"
        body, size = image_sdk_compare.request_body(m, "image")
        self.assertEqual((body["size"], size), ("1024x768", (1024, 768)))   # the preset's own geometry from a quarter-size input
        self.assertEqual(image_sdk_compare.abba(2), ["old", "new", "new", "old"])


class DevlabPerfFrozenCheck(unittest.TestCase):
    def test_the_job_refuses_other_programs_before_measuring(self):
        src = open(os.path.join(ROOT, ".github", "scripts", "devlab_perf.sh")).read()
        check = src.index("vwant=$(qv")
        self.assertLess(check, src.index("pkg_preset() {"))   # inside the variant setup loop, before any serve
        self.assertIn('kind\') or \'\')")" = image ]', src)
        self.assertIn('metrics_${p1}_', src)
        self.assertIn('${SERVED_PRESET:+--preset "$SERVED_PRESET"}', src)   # the probe uses the served (device) preset's context


if __name__ == "__main__":
    unittest.main()


class ProductionFacts(unittest.TestCase):
    def test_small_files_give_hash_kind_claims_and_generator(self):
        with tempfile.TemporaryDirectory() as cache:
            def snapshot_download(repo, revision, local_dir, allow_patterns):
                self.assertNotIn("*.gguf", allow_patterns)
                os.makedirs(os.path.join(local_dir, "program"))
                open(os.path.join(local_dir, "program", "llm.loom"), "w").write("x")
                json.dump({"services": {"chat": {"kind": "text"}}, "presets": {}}, open(os.path.join(local_dir, "model.json"), "w"))
                json.dump({"card": {"memory_bytes": 5, "features": ["draft"]}, "devices": {"strix-halo": {}, "gfx1150": {}, "radeon-r9700": {"state": "not_a_target", "reason": "x"}},
                           "versions": {"card_generator": "38b84928d947d45615439ffe54dc3d66899f7b0b"}}, open(os.path.join(local_dir, "info.json"), "w"))
                json.dump({"sdk": {"hrx_loom": "0.2.0"}, "generator": {"skills_commit": "unknown"}}, open(os.path.join(local_dir, "llm_config.json"), "w"))
            with mock.patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download)}):
                f = regen.production_facts(f"loom-fun/{STRIX_ONLY}", "b" * 40, cache)
            self.assertEqual((f["kind"], f["claimed"], f["memory_bytes"]), ("text", ["gfx1150", "strix-halo"], 5))
            # retro review: an "unknown" stamp fell back to the card generator's commit, which made the card, not the programs
            self.assertIsNone(f["generator"])
            self.assertTrue(f["programs_sha256"])
            facts_path = glob.glob(os.path.join(cache, "*", "facts.json"))[0]
            legacy = dict(json.load(open(facts_path)), generator="38b84928d947d45615439ffe54dc3d66899f7b0b")
            legacy.pop("generator_source")
            json.dump(legacy, open(facts_path, "w"))   # a cache written before the rule: re-read from the cached llm_config.json
            self.assertIsNone(regen.production_facts(f"loom-fun/{STRIX_ONLY}", "b" * 40, cache)["generator"])
            json.dump({"generator": {"skills_commit": "c" * 40}}, open(os.path.join(os.path.dirname(facts_path), "llm_config.json"), "w"))
            legacy["generator_source"] = None
            json.dump(legacy, open(facts_path, "w"))
            self.assertEqual(regen.production_facts(f"loom-fun/{STRIX_ONLY}", "b" * 40, cache)["generator"], "c" * 40)


class PackageGenerator(unittest.TestCase):
    def test_recovered_or_hand_written(self):
        self.assertEqual(regen.package_generator("a" * 40, True), {"commit": "a" * 40, "recovered": True, "basis": "production package llm_config.json"})
        self.assertEqual(regen.package_generator(None, False), {"commit": None, "basis": "hand-written package, no generator"})
        self.assertEqual(regen.package_generator(None, True), {"commit": None, "basis": regen.NO_STAMP})

    def test_legacy_ref_with_a_card_generator_commit(self):
        with tempfile.TemporaryDirectory() as work:
            ref = {"old": f"loom-fun/{TEXT}@{REV[TEXT]}", "generator": "38b84928d947d45615439ffe54dc3d66899f7b0b"}   # planned before the rule
            st = SimpleNamespace(d={"work": work, "opts": {"refs_all": {TEXT: ref}}})
            self.assertIsNone(regen.ref_generator(st, TEXT))   # no cached llm_config.json: no commit
            d = os.path.join(work, "production", f"{TEXT}@{REV[TEXT][:12]}")
            os.makedirs(d)
            json.dump({"generator": {"skills_commit": "unknown"}}, open(os.path.join(d, "llm_config.json"), "w"))
            self.assertIsNone(regen.ref_generator(st, TEXT))
            json.dump({"generator": {"skills_commit": "d" * 40}}, open(os.path.join(d, "llm_config.json"), "w"))
            self.assertEqual(regen.ref_generator(st, TEXT), "d" * 40)
            st.d["opts"]["refs_all"][TEXT] = {"generator": None, "generator_source": "llm_config"}
            self.assertIsNone(regen.ref_generator(st, TEXT))


class FrozenAssembleAndLint(unittest.TestCase):
    def test_only_the_pin_and_the_staging_main_wording_change(self):
        cfg = {"sdk": {"hrx_loom": "0.2.0"}, "resolved": {"x": {"evidence": "verified on staging main (run 1)"}}, "gguf": "m.gguf"}
        self.assertEqual(regen.staging_to_main(cfg)["resolved"]["x"]["evidence"], "verified on main (run 1)")
        with tempfile.TemporaryDirectory() as d:
            def snap(repo, rev, out):
                self.assertEqual((repo, rev), (f"loom-fun/{TEXT}", REV[TEXT]))
                os.makedirs(os.path.join(out, "program"))
                open(os.path.join(out, "program", "llm.loom"), "w").write("frozen")
                json.dump(cfg, open(os.path.join(out, "llm_config.json"), "w"))
            out = os.path.join(d, "pkg")
            with mock.patch.object(regen, "hf_snapshot_small", snap):
                h = regen.frozen_assemble(f"loom-fun/{TEXT}@{REV[TEXT]}", "0.3.0", out)
            got = json.load(open(os.path.join(out, "llm_config.json")))
            self.assertEqual(got, {**regen.staging_to_main(cfg), "sdk": regen.release_pin("0.3.0")})
            self.assertEqual(h, regen.card_lib.programs_sha256(out))


class KernelTestsNotApplicable(unittest.TestCase):
    def test_a_generator_error_without_a_claimed_kernel_gate(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"results": {"ktest:staging": "ERROR", "gates:staging:dev": "GATES PASS"}}, open(os.path.join(d, "summary.json"), "w"))
            cmp = os.path.join(d, "cmp.json")
            st = SimpleNamespace(d={"opts": {"refs_all": {STRIX_ONLY: {"gates": {"strix-halo": ["agent", "golden"], "radeon-r9700": ["kernel_tests"]}}}}})
            for dev, want in (("strix-halo", []), ("radeon-r9700", ["ktest:staging"])):
                json.dump({"gate_failures": ["ktest:staging"]}, open(cmp, "w"))
                e = {"slug": STRIX_ONLY, "device": dev, "cmp": cmp}
                regen.kernel_tests_not_applicable(st, e, d)
                self.assertEqual(json.load(open(cmp))["gate_failures"], want, dev)
            st.d["opts"]["refs_all"][TEXT] = {"gates": {"strix-halo": ["agent", "golden"]}}   # a fleet (kit-generated) package: the kit makes its tests
            json.dump({"gate_failures": ["ktest:staging"]}, open(cmp, "w"))
            regen.kernel_tests_not_applicable(st, {"slug": TEXT, "device": "strix-halo", "cmp": cmp}, d)
            self.assertEqual(json.load(open(cmp))["gate_failures"], ["ktest:staging"])
            json.dump({"results": {"ktest:staging": "FAIL (3/5 passed)"}}, open(os.path.join(d, "summary.json"), "w"))   # ran and failed: still a failure
            json.dump({"gate_failures": ["ktest:staging"]}, open(cmp, "w"))
            regen.kernel_tests_not_applicable(st, {"slug": STRIX_ONLY, "device": "strix-halo", "cmp": cmp}, d)
            self.assertEqual(json.load(open(cmp))["gate_failures"], ["ktest:staging"])


class AcceptNoReference(unittest.TestCase):
    def test_only_gates_whose_reference_could_not_start(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        d = os.path.join(tmp, "job")
        os.makedirs(d)
        cmp = os.path.join(tmp, "cmp.json")
        json.dump({"gate_failures": ["gates:staging:dev"], "metrics": {k: {"new": 100.0, "card": 100.0, "new_vs_card": 1.0} for k in regen.SDK_METRICS.values()}},
                  open(cmp, "w"))
        state = os.path.join(tmp, "s.json")
        job = {"slug": STRIX_ONLY, "device": "strix-halo", "stage": "needs_decision", "round": 1, "cmp": cmp, "dir": d}
        json.dump({"work": tmp, "mode": "sdk-loom", "opts": {"programs": "production", "new_sdk": "0.3.0"}, "jobs": {"j": job}, "batches": []}, open(state, "w"))
        os.makedirs(os.path.join(d, "gates_staging"))

        def gates(**passed):
            json.dump({"verdict": "FAIL", "gates": {n: {"pass": p} for n, p in passed.items()}}, open(os.path.join(d, "gates_staging", "gates.json"), "w"))
        gates(golden=False, agent=True)
        open(os.path.join(d, "gates_staging.out"), "w").write("== golden: 3 llama.cpp references pending\n")
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["accept-no-reference", state, STRIX_ONLY, "--note", "n"])
        self.assertEqual(regen.State(state).jobs["j"]["stage"], "needs_decision")   # a reference that ran: never accepted
        pending = "could not start llama-server: x\n== golden: references still pending after 5 rounds (FAIL)\n"
        open(os.path.join(d, "gates_staging.out"), "w").write(pending)
        gates(golden=False, agent=False)   # retro review: agent failed against a recorded reference; the log's pending line used to accept it all
        with contextlib.redirect_stdout(io.StringIO()) as out:
            regen.main(["accept-no-reference", state, STRIX_ONLY, "--note", "n"])
        self.assertEqual(regen.State(state).jobs["j"]["stage"], "needs_decision")
        self.assertIn("agent failed with a reference recorded", out.getvalue())
        os.remove(os.path.join(d, "gates_staging", "gates.json"))   # which gates failed is not recorded: not accepted
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["accept-no-reference", state, STRIX_ONLY, "--note", "n"])
        self.assertEqual(regen.State(state).jobs["j"]["stage"], "needs_decision")
        gates(golden=False, agent=True)
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["accept-no-reference", state, STRIX_ONLY, "--note", "no stock reference for inkling2"])
        e = regen.State(state).jobs["j"]
        self.assertEqual((e["stage"], e["decision"], e["no_reference"]), ("collected", "promote", "no stock reference for inkling2"))


class AcceptShownNotEligible(unittest.TestCase):
    """accept-gate-failure --shown-not-eligible: serving gates that ran and failed on the new SDK are published as failed, never eligible"""
    def test_failed_gates_shown_and_the_device_never_qualifies(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        d = os.path.join(tmp, "job")
        os.makedirs(os.path.join(d, "gates_staging"))
        cmp = os.path.join(tmp, "cmp.json")
        metrics = {k: {"new": 100.0, "card": 100.0, "new_vs_card": 1.0} for k in regen.SDK_METRICS.values()}
        state = os.path.join(tmp, "s.json")
        held = {"slug": STRIX_ONLY, "device": "strix-halo", "stage": "needs_decision", "decision": "hold", "round": 1, "cmp": cmp, "dir": d,
                "reason": "confirmed regression / failed gate / text difference: gate failed: gates:staging:dev"}

        def job(fails=("gates:staging:dev",), admitted=True, programs="production", **passed):
            json.dump({"gate_failures": list(fails), "metrics": metrics}, open(cmp, "w"))
            json.dump({"verdict": "FAIL", "context_admit": {"admitted": admitted}, "gates": {n: {"pass": p} for n, p in passed.items()}},
                      open(os.path.join(d, "gates_staging", "gates.json"), "w"))
            json.dump({"work": tmp, "mode": "sdk-loom", "opts": {"programs": programs, "new_sdk": "0.3.0"}, "jobs": {"j": dict(held)}, "batches": []},
                      open(state, "w"))

        def accept(*flags):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                regen.main(["accept-gate-failure", state, STRIX_ONLY, "--note", "new on 0.3.0; root cause open", *flags])
            return regen.State(state).jobs["j"], out.getvalue()

        open(os.path.join(d, "gates_staging.out"), "w").write("could not start llama-server: x\n== golden: references still pending after 5 rounds (FAIL)\n")
        job(consistency=False, spec=False, golden=False, agent=True)
        self.assertEqual(accept()[0]["stage"], "needs_decision")   # without the flag: kernel tests only
        e, _ = accept("--shown-not-eligible")
        self.assertEqual((e["stage"], e["decision"]), ("collected", "promote"))
        self.assertEqual((e["shown_failed"]["failed"], e["shown_failed"]["not_run"]), (["consistency", "spec"], ["golden"]))
        self.assertIn("not eligible for the release", e["accepted_gate_failure"])
        self.assertIn("golden: not re-run", regen.card_message(e, "strix-halo", "9", True, True))
        self.assertEqual(regen.device_status(e), "failed")   # published, still never qualified: eligible lists it as pending
        self.assertEqual(json.load(open(cmp))["gate_failures"], ["gates:staging:dev"])   # the job's evidence is untouched
        # refused: a func failure, a refused admission check, only reference-pending gates (accept-no-reference), a state not --programs production
        for kw, word in ((dict(fails=("gates:staging:dev", "func:staging"), consistency=False), "func:staging"),
                         (dict(admitted=False, consistency=False), "admission"), (dict(golden=False, agent=True), "accept-no-reference")):
            job(**kw)
            e, out = accept("--shown-not-eligible")
            self.assertEqual(e["stage"], "needs_decision", kw)
            self.assertIn(word, out)
        job(programs="build", consistency=False)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            accept("--shown-not-eligible")


class LaneOwned(unittest.TestCase):
    """slugs a lane owns end to end: the sweep never packages, ingests, publishes, verifies, confirms or submits them"""
    def test_sidecar_and_opts_skip_every_publishing_step(self):
        with tempfile.TemporaryDirectory() as t:
            state = os.path.join(t, "s.json")
            jobs = {"l": {"slug": "lane-loom", "device": "strix-halo", "stage": "ingested", "round": 1, "batch": 0},
                    "c": {"slug": "lane-loom", "device": "radeon-r9700", "stage": "collected", "decision": "confirm", "round": 1, "batch": 0},
                    "o": {"slug": "other-loom", "device": "strix-halo", "stage": "planned", "round": 1, "batch": 1}}
            json.dump({"work": t, "mode": "sdk-loom", "opts": {"programs": "production", "new_org": regen.STAGING_ORG, "lane_owned": ["x-loom"]},
                       "jobs": jobs, "batches": [{"id": 0, "device": "strix-halo", "jobs": ["l"], "stage": "planned"},
                                                 {"id": 1, "device": "strix-halo", "jobs": ["o"], "stage": "planned"}]}, open(state, "w"))
            open(state + ".lane_owned", "w").write("# lanes\nlane-loom  # qwen35 lane\n")
            st = regen.State(state)
            self.assertEqual(regen.lane_owned(st), {"x-loom", "lane-loom"})
            submitted, ran = [], []
            with mock.patch.object(regen, "submit_batch", lambda st, b, dry: submitted.append(b["id"])), \
                    mock.patch.object(regen, "sh", lambda *a, **k: ran.append(a) or SimpleNamespace(returncode=0, stdout="", stderr="")), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                regen.main(["submit", state])
                regen.main(["rerun", state])
                regen.main(["publish", state])
                regen.main(["package", state])
            self.assertEqual(submitted, [1])
            self.assertEqual(ran, [])
            self.assertIn("no confirming runs needed", out.getvalue())
            self.assertEqual(regen.State(state).jobs["l"]["stage"], "ingested")
            with self.assertRaises(SystemExit) as ex:
                regen.main(["package", state, "--slugs", "lane-loom"])
            self.assertIn("lane-owned", str(ex.exception.code))


class KitVerdict(unittest.TestCase):
    """Phase 2: ship the kit programs of a model only when every device's A/B passes; else keep the frozen ones and re-queue its cards."""
    def write(self, t, name, doc):
        p = os.path.join(t, name)
        json.dump({"work": t, **doc}, open(p, "w"))
        return p

    def p2(self, t, name, jobs):
        for n, e in jobs.items():
            cmp = self.write(t, f"{n}.cmp.json", e.pop("cmpdoc", {"text": {"status": "MATCH", "prompts": 3, "identical": 3}, "gate_failures": []}))
            e.update(cmp=cmp, job_url=f"https://github.com/loom-fun/loom-skills/actions/runs/1/job/{n}")
        return self.write(t, name, {"mode": "sdk-ab", "opts": {"old_sdk_target": True, "base": "sdk030"}, "batches": [], "jobs": jobs})

    def test_ship_keep_and_wait(self):
        with tempfile.TemporaryDirectory() as t:
            pub = {"stage": "published", "decision": "promote", "verified": {"ok": True}}
            sweep = self.write(t, "sweep.json", {"mode": "sdk-loom", "opts": {"programs": "production"}, "batches": [], "jobs": {
                "a-r": {"slug": "a", "device": "radeon-r9700", **pub}, "a-s": {"slug": "a", "device": "strix-halo", **pub},
                "b-s": {"slug": "b", "device": "strix-halo", **pub}, "c-s": {"slug": "c", "device": "strix-halo", "stage": "running"}}})
            ok = {"stage": "collected", "decision": "promote", "round": 1, "rows_pct": {"decode": -1.0, "prefill_8k": 2.0}}
            r = self.p2(t, "p2r.json", {"ra": {"slug": "a", "device": "radeon-r9700", **ok}})
            s = self.p2(t, "p2s.json", {
                "sa": {"slug": "a", "device": "strix-halo", **ok},
                "sb": {"slug": "b", "device": "strix-halo", **ok, "rows_pct": {"decode": 1.0, "decode_32k": -4.4},
                       "cmpdoc": {"text": {"status": "DIFF", "prompts": 3, "identical": 2}, "gate_failures": []}},
                "sc": {"slug": "c", "device": "strix-halo", **ok}})
            out = os.path.join(t, "kv.json")
            with contextlib.redirect_stdout(io.StringIO()):
                regen.main(["kit-verdict", sweep, r, s, "--out", out])
            kv = json.load(open(out))
            self.assertEqual(kv["a"], {"ship": True})
            self.assertFalse(kv["b"]["ship"])
            self.assertEqual(kv["b"]["reason"], "Strix greedy text differs on 1/3 prompts; Strix decode_32k -4.4 %".replace("Strix", regen.card_lib.GPUS["strix-halo"]["short"]))
            self.assertNotIn("c", kv)   # its migration-only card is not published yet
            sw = regen.State(sweep)
            self.assertEqual(sw.d["opts"]["kit_kept"]["b"]["url"], "https://github.com/loom-fun/loom-skills/actions/runs/1/job/sb")
            self.assertEqual((sw.jobs["b-s"]["stage"], "verified" in sw.jobs["b-s"]), ("collected", False))
            self.assertEqual(sw.jobs["a-s"]["stage"], "published")
            self.assertEqual(regen.State(r).d["opts"]["ship"], ["a"])
            self.assertEqual(regen.State(s).d["opts"]["ship"], ["a"])
            with contextlib.redirect_stdout(io.StringIO()):   # ingest: a model kit-verdict has not shipped waits; a shipped one needs its package
                regen.main(["ingest", s])
            p2 = regen.State(s).jobs
            self.assertEqual((p2["sc"]["stage"], p2["sc"].get("reason")), ("collected", None))
            self.assertEqual(p2["sa"]["stage"], "needs_decision")
            self.assertIn("package is not published yet", p2["sa"]["reason"])


class Share(unittest.TestCase):
    """One job per model and device: the --old-frozen phase-2 job is also the frozen package's migration job."""
    def test_old_frozen_job(self):
        ref = f"loom-fun/{TEXT}@{REV[TEXT]}"
        doc = AbJobsFrozen().doc()
        doc["jobs"][0]["steps"] = ["perf:fun:1", "perf:staging:1", "func:staging", "ktest:staging", "gates:staging:dev"]
        ab_jobs.old_frozen(doc, {TEXT: {"old": ref, "new": ref, "programs_sha256": H}}, "dev")
        j = doc["jobs"][0]
        by = {v["name"]: v for v in j["variants"]}
        self.assertEqual((by["fun"]["programs_sha256"], by["funnd"]["programs_sha256"]), (H, H))
        self.assertNotIn("programs_sha256", by["staging"])
        for step in ("func:fun", "ktest:fun", "gates:fun:dev", "admit:fun", "admit:staging"):
            self.assertIn(step, j["steps"])
        with self.assertRaises(SystemExit):
            ab_jobs.old_frozen(AbJobsFrozen().doc(), {}, "dev")
        with self.assertRaises(SystemExit):
            ab_jobs.main(["--device", "strix-halo", "--sdk-ab", "--old-sdk", "0.2.0", "--new-sdk", "0.3.0", "--old-frozen", TEXT])

    def test_view_is_the_frozen_side_as_a_migration_job(self):
        with tempfile.TemporaryDirectory() as t:
            src, dst = os.path.join(t, "job"), os.path.join(t, "view")
            os.makedirs(os.path.join(src, "gates_fun"))
            os.makedirs(os.path.join(src, "gates_staging"))
            for f in ("perf_fun.jsonl", "perf_staging.jsonl", "c4_funnd.txt", "c4_stagingnd.txt", "programs_sha256_fun.txt", "programs_sha256_staging.txt",
                      "sdk_fun.json", "sdk_funnd.json", "admit_fun.json", "admit_staging.json", "metrics_fun_000.txt", "perf-fun.log", "runner.txt",
                      "gates_fun/gates.json", "gates_staging/gates.json", "build_staging.log", "text_fun.json"):
                open(os.path.join(src, f), "w").write(f)
            json.dump({"job": "sab-x", "results": {"perf:fun:1": "fun decode_tps [1] median 1", "perf:staging:1": "staging decode_tps [2] median 2",
                                                   "c4:funnd": "funnd c4 median 3", "textmatch:fun": "x", "func:fun": "pass", "func:staging": "fail",
                                                   "gates:fun:dev": "GATES PASS", "admit:fun": "ADMITTED"}}, open(os.path.join(src, "summary.json"), "w"))
            open(os.path.join(src, "results.tsv"), "w").write("perf:fun:1\tfun decode\nperf:staging:1\tstaging decode\n")
            regen.phase1_view(src, dst)
            got = sorted(os.path.relpath(os.path.join(r, f), dst).replace(os.sep, "/") for r, _, fs in os.walk(dst) for f in fs)
            self.assertEqual(got, sorted(["admit_staging.json", "c4_stagingnd.txt", "gates_staging/gates.json", "metrics_staging_000.txt", "perf-staging.log",
                                          "perf_staging.jsonl", "programs_sha256_staging.txt", "results.tsv", "runner.txt", "sdk_staging.json",
                                          "sdk_stagingnd.json", "summary.json", "text_staging.json"]))
            self.assertEqual(open(os.path.join(dst, "perf_staging.jsonl")).read(), "perf_fun.jsonl")   # the frozen side's file
            self.assertEqual(open(os.path.join(dst, "gates_staging/gates.json")).read(), "gates_fun/gates.json")
            self.assertEqual(json.load(open(os.path.join(dst, "summary.json")))["results"],
                             {"perf:staging:1": "staging decode_tps [1] median 1", "c4:stagingnd": "stagingnd c4 median 3", "func:staging": "pass",
                              "gates:staging:dev": "GATES PASS", "admit:staging": "ADMITTED"})
            self.assertEqual(open(os.path.join(dst, "results.tsv")).read(), "perf:staging:1\tfun decode\n")

    def test_share_and_feed(self):
        with tempfile.TemporaryDirectory() as t:
            def write(name, doc):
                p = os.path.join(t, name)
                json.dump({"work": t, **doc}, open(p, "w"))
                return p
            sweep = write("sweep.json", {"mode": "sdk-loom", "opts": {"programs": "production"}, "log": [], "batches": [
                {"id": 0, "device": "strix-halo", "jobs": ["sab-a", "sab-b"], "stage": "submitted", "run": "11"},
                {"id": 1, "device": "strix-halo", "jobs": ["sab-c"], "stage": "planned"}], "jobs": {
                "sab-a": {"slug": "a", "device": "strix-halo", "stage": "queued", "batch": 0, "round": 1},
                "sab-b": {"slug": "b", "device": "strix-halo", "stage": "queued", "batch": 0, "round": 1},
                "sab-c": {"slug": "c", "device": "strix-halo", "stage": "planned", "batch": 1, "round": 1}}})
            p2 = write("p2.json", {"mode": "sdk-ab", "opts": {"old_sdk_target": True, "old_frozen": True}, "log": [], "batches": [], "jobs": {
                "sab-a": {"slug": "a", "device": "strix-halo", "stage": "planned", "round": 1},
                "sab-b": {"slug": "b", "device": "strix-halo", "stage": "planned", "round": 1},
                "sab-c": {"slug": "c", "device": "strix-halo", "stage": "planned", "round": 1}}})
            calls = []
            with mock.patch.object(regen, "sh", lambda cmd, check=True: calls.append(cmd)), contextlib.redirect_stdout(io.StringIO()):
                regen.main(["share", sweep, p2, "--cancel"])
            sw, q = regen.State(sweep), regen.State(p2)
            self.assertEqual({n: e["stage"] for n, e in sw.jobs.items()}, {"sab-a": "shared", "sab-b": "shared", "sab-c": "shared"})
            self.assertEqual([b["jobs"] for b in sw.d["batches"]], [[], []])
            self.assertEqual(sw.d["batches"][1]["stage"], "dropped")
            with mock.patch.object(regen, "gh_json", lambda args: {"jobs": []}), contextlib.redirect_stdout(io.StringIO()):
                try:
                    regen.main(["poll", sweep])   # a dropped batch (no branch, no run) is not looked up
                except SystemExit:
                    pass
            self.assertEqual(calls, [["gh", "api", "-X", "POST", f"repos/{regen.REPO}/actions/runs/11/force-cancel"]])
            self.assertEqual(q.jobs["sab-a"]["phase1"], {"state": sw.path, "job": "sab-a"})
            # the phase-2 job of a is collected: its frozen side is the migration job's result; b's job failed: it stops for a decision
            d = os.path.join(t, "jobdir")
            os.makedirs(d)
            json.dump({"job": "sab-a", "results": {}}, open(os.path.join(d, "summary.json"), "w"))
            with contextlib.redirect_stdout(io.StringIO()):
                regen.feed_phase1({**q.jobs["sab-a"], "dir": d, "conclusion": "success", "run": "22", "job_url": "u"})
                regen.feed_phase1({**q.jobs["sab-b"], "dir": d, "conclusion": "failure", "job_url": "v"})
            sw = regen.State(sweep)
            self.assertEqual((sw.jobs["sab-a"]["stage"], sw.jobs["sab-a"]["run"], sw.jobs["sab-a"]["job_url"]), ("done", "22", "u"))
            self.assertEqual(regen.download(sw, "sab-a", sw.jobs["sab-a"]), sw.jobs["sab-a"]["shared_dir"])
            self.assertEqual(sw.jobs["sab-b"]["stage"], "needs_decision")


class RooflineStub(unittest.TestCase):
    def test_stub_only_when_the_checkpoint_is_not_cached(self):
        dump = sorted(glob.glob(os.path.join(ROOT, "fleet", "unsloth-Qwen3-0.6B-loom", "gguf", "*.json.gz")))[0]
        name = os.path.basename(dump)[:-len(".json.gz")]
        ck = {"repo": "unsloth/Qwen3-0.6B-GGUF", "revision": "r" * 40, "file": name}
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {"HF_HOME": os.path.join(t, "hf")}), \
                mock.patch.object(regen.card_lib, "load_package", lambda p: {}), \
                mock.patch.object(regen.card_lib, "package_facts", lambda f: {"checkpoint": ck}):
            p = regen.roofline_stub("unsloth-Qwen3-0.6B-loom", t, t)
            self.assertEqual(os.path.basename(p), name)   # the card names the checkpoint file, not a stub path
            with open(p, "rb") as f:
                self.assertEqual(f.read(4), b"GGUF")
            snap = os.path.join(t, "hf", "hub", "models--unsloth--Qwen3-0.6B-GGUF", "snapshots", "r" * 40)
            os.makedirs(snap)
            open(os.path.join(snap, name), "w").close()
            self.assertIsNone(regen.roofline_stub("unsloth-Qwen3-0.6B-loom", t, t))   # cached: make_info reads the real file
            self.assertIsNone(regen.roofline_stub("no-such-model", t, t))

    def test_a_package_outside_the_fleet_reads_the_header_from_the_hub(self):
        import gguf_stub
        dump = sorted(glob.glob(os.path.join(ROOT, "fleet", "unsloth-Qwen3-0.6B-loom", "gguf", "*.json.gz")))[0]
        name = os.path.basename(dump)[:-len(".json.gz")]
        ck = {"repo": "someone/Hand-GGUF", "revision": "r" * 40, "file": name}
        seen = []

        def remote(url, blobs):
            seen.append((url, blobs))
            return gguf_stub.load(dump)
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {"HF_HOME": os.path.join(t, "hf")}), \
                mock.patch.object(regen.card_lib, "load_package", lambda p: {}), \
                mock.patch.object(regen.card_lib, "package_facts", lambda f: {"checkpoint": ck}), \
                mock.patch.object(gguf_stub, "dump", remote), mock.patch.object(gguf_stub, "write_stub", lambda d, path, blobs=None: path):
            self.assertEqual(os.path.basename(regen.roofline_stub("someone-Hand-loom", t, t)), name)
            self.assertEqual(seen[0][0], f"https://huggingface.co/someone/Hand-GGUF/resolve/{'r' * 40}/{name}")
            self.assertTrue(seen[0][1].startswith(t))   # the card's own blob store, never the fleet's
            seen.clear()
            ck["file"] = "missing.gguf"   # a fleet package without its dump: no Hub fallback
            self.assertIsNone(regen.roofline_stub("unsloth-Qwen3-0.6B-loom", t, t))
            self.assertEqual(seen, [])


class ConfirmationRows(unittest.TestCase):
    def test_a_row_the_sweep_job_never_had_is_not_missing(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "cmp.json")
            json.dump({"metrics": {"decode": {"old": 63.3, "new": 62.5, "ratio": 0.987}, "prefill_2k": {"old": 1416.5, "new": 1371.7, "ratio": 0.968},
                                   "streams4": {"old": 109.5, "new": 108.4, "ratio": 0.99}}, "gate_failures": [], "text": {"status": "MATCH"}}, open(p, "w"))
            first = {"rows_pct": {"decode": -2.0, "prefill_2k": -3.9, "streams4": -1.1}, "round": 1}
            st = SimpleNamespace(d={"mode": "sdk-loom", "opts": {}}, jobs={"a": first})
            d, r, *_ = regen.sdk_decide(st, "a-r2", {"prev": "a", "round": 2}, p, use_prev=False)
            self.assertFalse(any("missing" in x for x in r), r)   # no 8k prefill on this card: not measured, not missing
            first["rows_pct"]["prefill_8k"] = -1.0
            d, r, *_ = regen.sdk_decide(st, "a-r2", {"prev": "a", "round": 2}, p, use_prev=False)
            self.assertTrue(any("missing" in x for x in r), r)   # the sweep job measured it: the confirmation must too


class AcceptRegressionRows(unittest.TestCase):
    def test_confirmed_card_rows_and_the_abba_job(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "s.json")
            json.dump({"work": t, "mode": "sdk-loom", "opts": {"programs": "production"}, "batches": [], "log": [], "jobs": {
                "a": {"slug": "m", "device": "radeon-r9700", "stage": "needs_decision", "round": 1, "confirm_rows": ["prefill_1k", "prefill_2k"],
                      "perf": {"prefill": -1.0}, "reason": "ABBA confirmation (job 9): slower than -3% in the same-job ABBA: prefill_2k -3.2 %"},
                "a-r2": {"slug": "m", "device": "radeon-r9700", "stage": "collected", "round": 2, "prev": "a", "job_url": "https://x/job/2",
                         "perf_abba": {"decode": -1.3, "prefill": None}, "rows_pct": {"prefill_1k": -2.1, "prefill_2k": -3.2}}}}, open(p, "w"))
            with contextlib.redirect_stdout(io.StringIO()):
                regen.main(["accept-regression", p, "m", "--note", "confirmed; root cause open"])
            e = regen.State(p).jobs["a"]
            self.assertEqual((e["stage"], e["confirmed"], e["confirmed_by"]), ("collected", {"prefill_1k": -2.1, "prefill_2k": -3.2}, "https://x/job/2"))


class Explain(unittest.TestCase):
    def test_recorded_and_changed_cards_requeued(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "s.json")
            json.dump({"work": t, "mode": "sdk-loom", "opts": {"programs": "production"}, "batches": [], "log": [], "jobs": {
                "a": {"slug": "m", "device": "strix-halo", "stage": "published", "verified": {"ok": False}, "round": 1},
                "b": {"slug": "m", "device": "radeon-r9700", "stage": "published", "round": 1},
                "c": {"slug": "n", "device": "strix-halo", "stage": "published", "round": 1}}}, open(p, "w"))
            f = os.path.join(t, "x.json")
            doc = {"m": {"card": {"strix-halo": {"decode_2k": {"text": "acceptance of its prompts", "url": "https://x/job/1"}}}}}
            json.dump(doc, open(f, "w"))
            with contextlib.redirect_stdout(io.StringIO()):
                regen.main(["explain", p, f])
            st = regen.State(p)
            self.assertEqual({n: e["stage"] for n, e in st.jobs.items()}, {"a": "collected", "b": "published", "c": "published"})
            self.assertNotIn("verified", st.jobs["a"])
            self.assertEqual(regen.explanations(st.d, "m"), doc["m"])
            self.assertEqual(regen.explanations(st.d, "n"), {})
            st.jobs["a"]["stage"] = "published"
            st.save()
            with contextlib.redirect_stdout(io.StringIO()):
                regen.main(["explain", p, f])   # unchanged: nothing re-queued
            self.assertEqual(regen.State(p).jobs["a"]["stage"], "published")


class OldFrozenGates(unittest.TestCase):
    def test_the_frozen_sides_gates_do_not_decide_the_kit_build(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "cmp.json")
            m = {"decode": {"old": 1.0, "new": 1.0, "ratio": 1.0}, "prefill_8k": {"old": 1.0, "new": 1.0, "ratio": 1.0},
                 "decode_32k": {"old": 1.0, "new": 1.0, "ratio": 1.0}, "streams4": {"old": 1.0, "new": 1.0, "ratio": 1.0}}
            json.dump({"metrics": m, "gate_failures": ["ktest:fun"], "text": {"status": "MATCH"}}, open(p, "w"))
            st = SimpleNamespace(d={"mode": "sdk-ab", "opts": {"old_frozen": True, "old_sdk_target": True}}, jobs={})
            self.assertEqual(regen.sdk_decide(st, "j", {"round": 1}, p)[0], "promote")
            json.dump({"metrics": m, "gate_failures": ["ktest:fun", "gates:staging:dev"], "text": {"status": "MATCH"}}, open(p, "w"))
            d, r, *_ = regen.sdk_decide(st, "j", {"round": 1}, p)
            self.assertEqual((d, r[0]), ("hold", "gate failed: gates:staging:dev"))

    def test_devlab_ktest_reads_its_own_variants_package(self):
        text = open(os.path.join(ROOT, ".github", "scripts", "devlab_perf.sh")).read()
        self.assertIn('[ -f "$D/$p1/cfg.json" ] || cp "$D/cfg.json" "$D/$p1/cfg.json"', text)


class ConstrainedEvidence(unittest.TestCase):
    def test_replay_is_scoped_and_uses_candidate(self):
        opts = {"constrained_gates": {TEXT: {"artifact": "artifact", "job": "job.json", "variant": "sample"}}}
        with mock.patch.object(regen, "run_tool") as run:
            for slug, gpu in [(IMAGE, "radeon-r9700"), (TEXT, "strix-halo")]:
                self.assertEqual(regen.constrained_evidence(opts, slug, gpu, "candidate", "work", None), [])
            with self.assertRaisesRegex(RuntimeError, "pinned production"):
                regen.constrained_evidence(opts, TEXT, "radeon-r9700", "candidate", "work", None)
            run.assert_not_called()
            args = (opts, TEXT, "radeon-r9700", "candidate", "work", "production/info.json")
            self.assertEqual(regen.constrained_evidence(*args), ["ev_constrained.json"])
            run.assert_called_once_with([f"{regen.CARD}/make_info.py", "emulated-gates", os.path.abspath("artifact"),
                                         "--job", os.path.abspath("job.json"), "--variant", "sample", "--package",
                                         "candidate", "-o", "ev_constrained.json"], "work")
            run.side_effect = RuntimeError("package programs differ")
            with self.assertRaisesRegex(RuntimeError, "programs differ"):
                regen.constrained_evidence(*args)
