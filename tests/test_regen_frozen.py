"""fleet/regen.py --programs production (the frozen SDK migration), the version-generic SDK constants, the evidence / eligible generator,
ab_jobs.py --programs production and the image comparison's request (no network, no GPU)."""
import contextlib
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
STRIX_ONLY = "unsloth-Inkling-2-Mini-loom"
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
        # the image package: one image job on Strix Halo, none on the R9700
        img = [e for e in st.jobs.values() if e["slug"] == IMAGE]
        self.assertEqual([(e["device"], e["kind"]) for e in img], [("strix-halo", "image")])
        batch = next(b for b in st.d["batches"] if b["jobs"] == [n for n, e in st.jobs.items() if e["slug"] == IMAGE])
        job = json.load(open(batch["perf"]))["jobs"][0]
        self.assertEqual(job["kind"], "image")
        self.assertEqual(job["image"], {"old_sdk": "0.2.0", "new_sdk": regen.SDK_NEW, "programs_sha256": "i" * 64, "preset": None})

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

    def test_every_runner_device_qualified(self):
        el, held, pend = regen.eligibility(self.doc({"strix-halo": self.dev(), "radeon-r9700": self.dev(run="46", job="139"),
                                                     "gfx1150": {"status": "no_runner"}}))
        row = el[TEXT]
        self.assertEqual(row["device_keys"], ["radeon-r9700", "strix-halo"])
        self.assertEqual(row["qualification"]["radeon-r9700"]["run"], 46)
        self.assertEqual(row["carried_devices"], ["gfx1150"])
        self.assertEqual((row["programs_sha256"], row["target_revision"], row["staging_revision"]), (H, REV[TEXT], "s" * 40))
        self.assertEqual(held[HELD]["tag_revision"], "t" * 40)
        self.assertEqual(pend, {})

    def test_fail_closed(self):
        for bad in (self.dev("failed"), self.dev(conclusion="failure"), self.dev(run=None), self.dev(programs_match=False),
                    self.dev("pending"), self.dev(published_revision=None)):
            el, _, pend = regen.eligibility(self.doc({"strix-halo": self.dev(), "radeon-r9700": bad}))
            self.assertEqual(el, {}, bad)
            self.assertIn(TEXT, pend)
        el, _, pend = regen.eligibility(self.doc({"gfx1150": {"status": "no_runner"}}))
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


class ImageRequest(unittest.TestCase):
    def test_text_to_image_and_upscale_requests(self):
        m = {"model": {"id": "z"}, "services": {"image": {"kind": "image", "interface": {"input": "text prompt"}}},
             "presets": {"image": {"service": "image", "execution": {"width": 1024, "height": 768}, "request": {"prompt": "p", "seed": 42}}}}
        body, size = image_sdk_compare.request_body(m, "image")
        self.assertEqual((body["prompt"], body["seed"], body["model"], size), ("p", 42, "z", (1024, 768)))
        m["services"]["image"]["interface"]["input"] = "base64 of the raw RGB input image in the prompt"
        body, size = image_sdk_compare.request_body(m, "image")
        self.assertEqual((body["size"], size), ("256x64", (256, 64)))
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
                json.dump({"card": {"memory_bytes": 5, "features": ["draft"]}, "devices": {"strix-halo": {}, "gfx1150": {}},
                           "versions": {"card_generator": "38b84928d947d45615439ffe54dc3d66899f7b0b"}}, open(os.path.join(local_dir, "info.json"), "w"))
                json.dump({"sdk": {"hrx_loom": "0.2.0"}, "generator": {"skills_commit": "unknown"}}, open(os.path.join(local_dir, "llm_config.json"), "w"))
            with mock.patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download)}):
                f = regen.production_facts(f"loom-fun/{STRIX_ONLY}", "b" * 40, cache)
            self.assertEqual((f["kind"], f["claimed"], f["memory_bytes"]), ("text", ["gfx1150", "strix-halo"], 5))
            self.assertEqual(f["generator"], "38b84928d947d45615439ffe54dc3d66899f7b0b")   # an "unknown" stamp falls back to the card's generator
            self.assertTrue(f["programs_sha256"])
