"""fleet/regen.py: plan / submit / poll / collect / rerun / ingest / publish / status on fixtures (no network, no GPU).

The fixture job (tests/data/regen/job_e2b) is a downloaded R9700 job of this campaign (an A/A of one staging revision); the job list
(perf_r9700_aa.json) is what ab_jobs.py wrote for it."""
import contextlib
import io
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "fleet"))
import regen  # noqa: E402
import sdk_package  # noqa: E402

DATA = os.path.join(ROOT, "tests", "data", "regen")
JOB = "rab-unsloth-gemma-4-e2b-it-loom"
SLUG = "unsloth-gemma-4-E2B-it-loom"
STRIX_SLUG = "unsloth-Qwen3-4B-Instruct-2507-loom"


def fake_generator(device="radeon-r9700", names=None):
    """What the job generators print: the fixture job list (re-named for other devices)."""
    doc = json.load(open(os.path.join(DATA, "perf_r9700_aa.json")))
    doc["jobs"] = [dict(j) for j in doc["jobs"][:1]]
    return doc


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = os.path.join(self.tmp, "s.json")
        self.calls = []

    def gen(self, cmd, **kw):
        """stands in for the generator subprocess; other commands are recorded"""
        self.calls.append([str(c) for c in cmd])
        if any(str(c).endswith(("ab_jobs.py", "r9700_jobs.py", "device_jobs.py")) for c in cmd):
            slugs = [c for c in cmd if str(c).endswith("-loom")]
            doc = fake_generator()
            if slugs and slugs[0] != SLUG:
                doc["jobs"][0].update(name=("sab-" + slugs[0].lower())[:40], card=f"loom-fun-staging/{slugs[0]}@abc")
            if "--suffix=-r2" in cmd:
                doc["jobs"][0]["name"] = doc["jobs"][0]["name"][:37] + "-r2"
            return subprocess.CompletedProcess(cmd, 0, json.dumps(doc), "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def plan(self, *args, slugs=(SLUG,), device="radeon-r9700"):
        with mock.patch.object(regen, "sh", self.gen), mock.patch.object(regen, "hf_head", lambda repo: "f" * 40), \
                mock.patch.object(regen, "old_sdk_revision", lambda repo, target=None: "e" * 40), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["plan", "--base", "r9700", "--device", device, "--state", self.state, *args, *slugs])
        return regen.State(self.state)

    def collected(self, **over):
        """a state whose one job is finished and its fixture directory downloaded"""
        st = self.plan("--old", os.path.join(self.tmp, "refs.json")) if False else self.plan()
        e = st.jobs[JOB]
        e.update(stage="done", run="1", job_url="https://github.com/loom-fun/loom-skills/actions/runs/1/job/2", completed="2026-11-14T21:32:24Z", **over)
        st.save()
        return st

    def collect(self):
        def download(st, name, e):
            d = regen.job_dir(st, name)
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(os.path.join(DATA, "job_e2b"), d)
            return d
        info = os.path.join(self.tmp, "ref_info.json")      # the reference card of the runner guard (R9700 jobs read it): no results = not judged
        json.dump({"schema": "loom-card/3", "devices": {}}, open(info, "w"))
        with mock.patch.object(regen, "download", download), mock.patch.object(regen, "hf_file", lambda *a, **k: info), \
                contextlib.redirect_stdout(io.StringIO()):
            regen.main(["collect", self.state])
        return regen.State(self.state)


class PlanTest(Base):
    def test_plan_writes_batches_and_state(self):
        st = self.plan()
        self.assertEqual(st.d["mode"], "ab")
        self.assertEqual(st.jobs[JOB]["stage"], "planned")
        self.assertEqual(len(st.d["batches"]), 1)
        doc = json.load(open(st.d["batches"][0]["perf"]))
        self.assertEqual([j["name"] for j in doc["jobs"]], [JOB])

    def test_ab_on_a_device_without_a_published_table_is_an_aa_of_the_pinned_revision(self):
        st = self.plan()
        refs = json.load(open(os.path.join(st.work, "refs_radeon-r9700.json")))
        self.assertEqual(refs[SLUG]["old"], refs[SLUG]["new"])
        self.assertTrue(refs[SLUG]["new"].endswith("@" + "f" * 40))
        self.assertTrue(st.jobs[JOB]["aa"])
        self.assertIn("--refs", self.calls[0])

    def test_strix_compares_with_the_published_card(self):
        st = self.plan(device="strix-halo", slugs=(STRIX_SLUG,))
        e = next(iter(st.jobs.values()))
        self.assertFalse(e["aa"])
        self.assertNotIn("--baselines", self.calls[0])   # Strix llama.cpp is not re-measured
        refs = json.load(open(os.path.join(st.work, "refs_strix-halo.json")))
        self.assertNotIn("old", refs[STRIX_SLUG])

    def test_scarce_runners_first_and_batches(self):
        a = regen.device_order("radeon-r9700")
        b = regen.device_order("strix-halo")
        self.assertLess(a, b)

    def test_batch_size_splits(self):
        st = self.plan("--batch", "1", slugs=(SLUG, STRIX_SLUG))
        self.assertEqual(len(st.d["batches"]), 2)

    def test_unfit_models_are_skipped_not_failed(self):
        def gen(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, "", "no job: x do not fit the R9700")
        with mock.patch.object(regen, "sh", gen), mock.patch.object(regen, "hf_head", lambda r: "f" * 40), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["plan", "--base", "r9700", "--device", "radeon-r9700", "--state", self.state, SLUG])
        st = regen.State(self.state)
        self.assertEqual(next(iter(st.jobs.values()))["stage"], "skipped")

    def test_card_mode_names_a_job_per_device(self):
        with mock.patch.object(regen, "generator_cmd", lambda st, d, s, suffix="": ["x", "ab_jobs.py"] + s):
            st = self.plan("--mode", "card")
        self.assertTrue(any(n.startswith("rc-") for n in st.jobs), list(st.jobs))

    def test_only_staging_is_a_target(self):
        with self.assertRaises(SystemExit), mock.patch.object(regen, "sh", self.gen):
            regen.main(["plan", "--base", "r9700", "--device", "radeon-r9700", "--state", self.state, "--new-org", "loom-fun", SLUG])


class SubmitPollTest(Base):
    def test_submit_dry_run_touches_nothing(self):
        self.plan()
        buf = io.StringIO()
        with mock.patch.object(regen, "gh_api", side_effect=AssertionError("network")), contextlib.redirect_stdout(buf):
            regen.main(["submit", self.state, "--dry-run"])
        self.assertIn("devlab-perf", buf.getvalue())
        self.assertEqual(regen.State(self.state).jobs[JOB]["stage"], "planned")

    def test_submit_creates_branch_file_pr_label(self):
        self.plan()
        seen = []

        def api(method, path, fields=None, jq=None):
            seen.append((method, path.split("repos/loom-fun/loom-skills/")[-1].split("?")[0], dict(fields or {})))
            return {".object.sha": "a" * 40, ".sha": "b" * 40, ".number": "77"}.get(jq)
        with mock.patch.object(regen, "gh_api", api), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["submit", self.state])
        self.assertEqual([(m, p) for m, p, _ in seen], [("GET", "git/ref/heads/r9700"), ("POST", "git/refs"), ("GET", "contents/.github/devlab/perf.json"),
                                                         ("PUT", "contents/.github/devlab/perf.json"), ("POST", "pulls"), ("POST", "issues/77/labels")])
        self.assertEqual(seen[4][2]["base"], "r9700")   # plan --base: the experiment branch is cut from it and targets it
        self.assertIn("kind: tooling", seen[4][2]["body"])
        import scope_lint
        scope, errors = scope_lint.parse_block(seen[4][2]["body"])
        self.assertEqual(errors, [])
        self.assertEqual(scope["scopes"], [])   # tooling has no selector; "none" would name a nonexistent KB node
        self.assertEqual(seen[5][2], {"labels[]": "devlab-perf"})
        st = regen.State(self.state)
        self.assertEqual((st.jobs[JOB]["stage"], st.jobs[JOB]["pr"]), ("submitted", 77))

    def test_poll_maps_job_states_and_exit_code(self):
        self.plan()
        st = regen.State(self.state)
        st.d["batches"][0].update(stage="submitted", branch="b", pr=1)
        st.save()
        jobs = {"status": "in_progress", "jobs": [{"name": JOB, "status": "in_progress", "conclusion": None, "url": "u", "startedAt": "t0", "completedAt": None}]}

        def gj(args):
            return [{"databaseId": 5, "createdAt": "t", "status": "x", "event": "pull_request"}] if args[1] == "list" else jobs
        with mock.patch.object(regen, "gh_json", gj), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as c:
                regen.main(["poll", self.state])
            self.assertEqual(c.exception.code, 10)
            self.assertEqual(regen.State(self.state).jobs[JOB]["stage"], "running")
            jobs["jobs"][0].update(status="completed", conclusion="success", completedAt="t1")
            with self.assertRaises(SystemExit) as c:
                regen.main(["poll", self.state])
            self.assertEqual(c.exception.code, 0)
        e = regen.State(self.state).jobs[JOB]
        self.assertEqual((e["stage"], e["run"], e["completed"]), ("done", "5", "t1"))


    def test_poll_follows_an_adopted_batch_without_a_branch(self):
        """regen.py adopt --run (a batch submitted by hand, e.g. while the contents API failed) records the run but no branch: poll still follows it"""
        self.plan()
        st = regen.State(self.state)
        st.d["batches"][0].update(stage="submitted", run="9")
        st.save()
        jobs = {"jobs": [{"name": JOB, "status": "completed", "conclusion": "success", "url": "u", "startedAt": "t0", "completedAt": "t1"}]}
        with mock.patch.object(regen, "gh_json", lambda args: jobs), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as c:
                regen.main(["poll", self.state])
            self.assertEqual(c.exception.code, 0)
        self.assertEqual(regen.State(self.state).jobs[JOB]["stage"], "done")

class CollectTest(Base):
    def test_aa_job_in_band_is_promoted(self):
        self.collected()
        e = self.collect().jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "promote"))
        self.assertTrue(os.path.exists(e["cmp"]))
        self.assertEqual(e["card_revision"].split("@")[-1], "512e9bb19d98b83a7c995a135e4c036c0ad0190e")

    def test_failed_gate_stops_that_model(self):
        self.collected()
        orig = regen.download

        def download(st, name, e):
            d = orig(st, name, e) if False else regen.job_dir(st, name)
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(os.path.join(DATA, "job_e2b"), d)
            s = json.load(open(os.path.join(d, "summary.json")))
            s["results"]["gates:staging:dev"] = "GATES FAIL level=dev {'golden': 'FAIL'}"
            json.dump(s, open(os.path.join(d, "summary.json"), "w"))
            return d
        info = os.path.join(self.tmp, "ref_info.json")
        json.dump({"schema": "loom-card/3", "devices": {}}, open(info, "w"))
        with mock.patch.object(regen, "download", download), mock.patch.object(regen, "hf_file", lambda *a, **k: info), \
                contextlib.redirect_stdout(io.StringIO()):
            regen.main(["collect", self.state])
        e = regen.State(self.state).jobs[JOB]
        self.assertEqual(e["stage"], "needs_decision")
        self.assertIn("gate", e["reason"])

    def test_row_below_the_published_value_asks_for_one_confirming_run_then_stops(self):
        self.collected()
        bp = os.path.join(self.tmp, "baselines.json")   # the card says 400, the job measured ~271
        json.dump({"models": {SLUG: {"radeon-r9700": {"decode": {"value": 400.0, "run": "x"}}}}}, open(bp, "w"))
        with mock.patch.object(regen, "BASELINES", bp):
            e = self.collect().jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "rerun"))
        self.assertIn("below the published card value", e["reason"])

    def test_rerun_makes_a_round_2_batch_with_prev(self):
        self.collected()
        st = regen.State(self.state)
        e = st.jobs[JOB]
        e.update(stage="collected", decision="rerun", cmp="/x/cmp.json")
        st.save()
        with mock.patch.object(regen, "sh", self.gen), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["rerun", self.state])
        st = regen.State(self.state)
        r2 = [n for n in st.jobs if n.endswith("-r2")]
        self.assertEqual(len(r2), 1)
        self.assertEqual((st.jobs[r2[0]]["prev"], st.jobs[r2[0]]["round"], st.jobs[r2[0]]["aa"]), (JOB, 2, True))
        self.assertEqual((st.d["batches"][-1]["round"], st.d["batches"][-1]["priority"]), (2, True))   # targeted: not behind the waves
        # a second call does not queue another
        with mock.patch.object(regen, "sh", self.gen), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["rerun", self.state])
        self.assertEqual(len([n for n in regen.State(self.state).jobs if n.endswith("-r2")]), 1)

    def test_card_mode_verdict_needs_every_step_and_passing_gates(self):
        d = os.path.join(self.tmp, "card")
        os.makedirs(d)
        ok = {"perf:card:1": "x", "c4:nodraft": "x", "func:card": "pass", "ktest:card": "PASS (1/1)", "gates:card:dev": "GATES PASS"}
        json.dump({"results": ok}, open(os.path.join(d, "summary.json"), "w"))
        self.assertEqual(regen.card_mode_verdict(d, {})[0], "promote")
        json.dump({"results": {**ok, "ktest:card": "FAIL 3/72"}}, open(os.path.join(d, "summary.json"), "w"))
        self.assertEqual(regen.card_mode_verdict(d, {})[0], "hold")
        del ok["c4:nodraft"]
        json.dump({"results": ok}, open(os.path.join(d, "summary.json"), "w"))
        self.assertEqual(regen.card_mode_verdict(d, {})[0], "rerun")


class IngestPublishTest(Base):
    def test_sdk_030_ingest_records_the_serving_wheel_and_rejects_a_stale_pin(self):
        st = SimpleNamespace(d={"mode": "sdk-loom", "opts": {"new_org": regen.STAGING_ORG, "new_sdk": "0.3.0"}})
        job = {"dir": self.tmp, "device": "strix-halo", "slug": SLUG, "source_card": f"loom-fun/{SLUG}@{'a' * 40}",
               "package": {"programs_sha256": "b" * 64}}
        with open(os.path.join(self.tmp, "programs_sha256_staging.txt"), "w") as f:
            f.write("b" * 64)
        with open(os.path.join(self.tmp, "ktest_staging.log"), "w"):
            pass
        pin = regen.release_pin("0.3.0")

        def snapshot(repo, revision, out):
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "llm_config.json"), "w") as f:
                json.dump({"sdk": pin}, f)
            with open(os.path.join(out, "info.json"), "w") as f:
                json.dump({"card": {"memory_bytes": 123}, "devices": {}}, f)

        def tool(args, cwd):
            if "build" in args:
                shutil.copy(os.path.join(cwd, "info_prev.json"), os.path.join(cwd, "info_new.json"))

        with mock.patch.object(regen, "hf_snapshot", snapshot), mock.patch.object(regen, "hf_head", return_value="c" * 40), \
                mock.patch.object(regen.card_lib, "programs_sha256", return_value="b" * 64), \
                mock.patch.object(regen, "hf_file", return_value="frozen-info.json"), \
                mock.patch.object(regen, "job_facts", return_value=("1", "https://example.com/run/job", "2026-11-23")), \
                mock.patch.object(regen, "run_tool", side_effect=tool) as tools:
            _, parent, _ = regen.ingest_full(st, JOB, job, self.tmp)
            self.assertEqual(parent, "c" * 40)
            commands = [call.args[0] for call in tools.call_args_list]
            self.assertNotIn("--hrx-out", commands[0])
            for action in ("perf", "gates"):
                command = next(cmd for cmd in commands if cmd[1] == action)
                self.assertEqual(command[command.index("--hrx-loom") + 1], os.path.join(self.tmp, "sdk_staging.json"))
            pin = {"hrx_loom": "0.2.0"}
            tools.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "no longer has the verified SDK 0.3.0 pin"):
                regen.ingest_full(st, JOB, job, self.tmp)
            tools.assert_not_called()

    def ingested(self, prev_devices=None):
        self.collected()
        self.collect()
        calls = []

        def hf_snapshot(repo, rev, out):
            os.makedirs(out)
            json.dump({"card": {"memory_bytes": 123}, "devices": prev_devices or {"strix-halo": {}}}, open(os.path.join(out, "info.json"), "w"))
            open(os.path.join(out, "README.md"), "w").write("card")
            calls.append(("snapshot", repo, rev))

        def run_tool(args, cwd):
            calls.append(("tool", [str(a) for a in args]))
            if "build" in args:
                shutil.copy(os.path.join(cwd, "info_prev.json"), os.path.join(cwd, "info_new.json"))
            return ""
        with mock.patch.object(regen, "hf_snapshot", hf_snapshot), mock.patch.object(regen, "hf_head", lambda r: "p" * 40), \
                mock.patch.object(regen, "run_tool", run_tool), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["ingest", self.state])
        return regen.State(self.state), calls

    def test_ingest_runs_the_recipe_in_order_on_the_measured_revision(self):
        st, calls = self.ingested()
        self.assertEqual(calls[0], ("snapshot", "loom-fun-staging/" + SLUG, "512e9bb19d98b83a7c995a135e4c036c0ad0190e"))
        tools = [os.path.basename(c[1][0]) + ":" + (c[1][1] if len(c[1]) > 1 and not c[1][1].startswith("-") else "") for c in calls[1:] if c[0] == "tool"]
        self.assertEqual(tools[:2], ["pull_devlab.py:", "make_info.py:perf"])
        self.assertEqual([t for t in tools if t.startswith("make_info.py")], ["make_info.py:perf", "make_info.py:gates", "make_info.py:roofline", "make_info.py:build"])
        perf = next(c[1] for c in calls if c[0] == "tool" and "perf" in c[1])
        self.assertIn("perf_staging.jsonl", " ".join(perf))
        self.assertIn("c4_stagingnd.txt", " ".join(perf))
        build = next(c[1] for c in calls if c[0] == "tool" and "build" in c[1])
        self.assertEqual(build[build.index("--memory-bytes") + 1], "123")
        self.assertEqual(build[build.index("--gpus") + 1], "strix-halo,radeon-r9700")   # the card's GPUs, registry order, plus this device
        e = st.jobs[JOB]
        self.assertEqual((e["stage"], e["parent"]), ("ingested", "p" * 40))

    def test_ingest_of_one_device_keeps_the_other_devices_published_table(self):
        _, calls = self.ingested({"strix-halo": {"performance": {"sessions": []}}})
        build = next(c[1] for c in calls if c[0] == "tool" and "build" in c[1])
        self.assertEqual([build[i + 1] for i, a in enumerate(build) if a == "--perf-from"], ["strix-halo=info_prev.json"])

    def test_ingest_without_another_device_table_passes_no_perf_from(self):
        _, calls = self.ingested({"strix-halo": {}})
        self.assertNotIn("--perf-from", next(c[1] for c in calls if c[0] == "tool" and "build" in c[1]))

    def test_publish_dry_run_is_card_only_to_staging_with_the_parent(self):
        self.ingested()
        seen = []

        def sh(cmd, **kw):
            seen.append([str(c) for c in cmd])
            return subprocess.CompletedProcess(cmd, 0, "dry run: ok", "")
        with mock.patch.object(regen, "sh", sh), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["publish", self.state, "--dry-run"])
        cmd = seen[0]
        for flag in ("--card-only", "--dry-run"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--org") + 1], "loom-fun-staging")
        self.assertEqual(cmd[cmd.index("--parent-commit") + 1], "p" * 40)
        self.assertEqual(regen.State(self.state).jobs[JOB]["stage"], "ingested")   # a dry run publishes nothing

    def test_publish_refuses_other_orgs(self):
        self.ingested()
        st = regen.State(self.state)
        st.d["opts"]["new_org"] = "loom-fun"
        st.save()
        with self.assertRaises(SystemExit):
            regen.main(["publish", self.state, "--dry-run"])


class StatusTest(Base):
    def test_status_json_gets_stage_verdict_run_per_gpu(self):
        self.collected()
        self.collect()
        p = os.path.join(self.tmp, "status.json")
        json.dump({"models": {SLUG: {"strix": {"stage": "queued"}}}}, open(p, "w"))
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["status", self.state, "--status-json", p])
        m = json.load(open(p))["models"][SLUG]
        self.assertEqual(m["r9700"], {"stage": "measured", "verdict": "pass", "run": "1"})
        self.assertEqual(m["strix"], {"stage": "queued"})   # the other GPU's entry is untouched


class SdkAbTest(Base):
    """--mode sdk-ab: the SDK migration's A/B (old = the manifest's staging card on 0.1.0, new = built on 0.2.0): plan, ratchet, deltas, status."""

    def plan_sdk(self, **kw):
        return self.plan("--mode", "sdk-ab", "--new-sdk", "0.2.0", **kw)

    def cmp_file(self, name, ratios, gates=None, text="MATCH", degraded=None):
        """a strix_ab_compare.py result with the given new/old ratio per row"""
        p = os.path.join(self.tmp, f"{name}.json")
        json.dump({"job": name, "metrics": {k: {"old": 100.0, "new": round(100.0 * r, 2) if r else None, "ratio": r, "status": "PASS" if r else "MISSING"}
                               for k, r in ratios.items()}, "gate_failures": gates or [],
                   "text": {"status": text, "prompts": 3, "identical": 3 if text == "MATCH" else 1}, "degraded_runner": degraded or {}}, open(p, "w"))
        return p

    def e(self, st, name=JOB, **kw):
        e = st.jobs[name]
        e.update(stage="done", run="7", job_url="https://x/job/1", completed="2026-11-16T12:00:00Z", **kw)
        st.save()
        return e

    def collect_with(self, ratios, name=JOB, **kw):
        """collect the (finished) job `name` as a job whose compare says `ratios`"""
        def download(st, n, e):
            d = regen.job_dir(st, n)
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(os.path.join(DATA, "job_e2b"), d)
            return d
        cmp = self.cmp_file(name, ratios, **kw)
        with mock.patch.object(regen, "download", download), mock.patch.object(regen, "compare", lambda st, n, e, d: cmp), \
                contextlib.redirect_stdout(io.StringIO()):
            regen.main(["collect", self.state])
        return regen.State(self.state)

    def package_recorded(self, e, h, rev="h" * 40):
        st = regen.State(self.state)
        st.jobs[e["name"] if "name" in e else JOB]["package"] = {"revision": rev, "programs_sha256": h, "base": "c" * 40}
        st.save()

    GOOD = {"decode": 1.02, "decode_2k": 1.01, "prefill_8k": 1.05, "decode_32k": 0.99, "streams4": 1.0}

    def test_plan_pins_old_and_new_to_the_staging_head(self):
        st = self.plan_sdk()
        refs = json.load(open(os.path.join(st.work, "refs_radeon-r9700.json")))
        # old = the newest old-SDK revision (mocked "e"), new = the head at plan time (mocked "f"), the same on every device
        self.assertEqual(refs[SLUG], {"old": f"loom-fun-staging/{SLUG}@{'e' * 40}", "new": f"loom-fun-staging/{SLUG}@{'f' * 40}"})
        self.assertFalse(st.jobs[JOB]["aa"])   # same revision, but two SDKs: not a noise-only A/A
        cmd = self.calls[0]
        for flag in ("--sdk-ab", "--baselines"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--old-sdk") + 1], "card")
        self.assertEqual(cmd[cmd.index("--new-sdk") + 1], "0.2.0")

    def test_sdk_030_pins_production_and_persists_the_target(self):
        st = self.plan("--mode", "sdk-loom", "--new-sdk", "0.3.0")
        self.assertEqual(st.d["opts"]["new_sdk"], "0.3.0")
        reference = f"loom-fun/{SLUG}@3503abcf7cfff7bc58bd819ea32ccab8c18c075d"
        self.assertEqual(st.jobs[JOB]["source_card"], reference)
        with open(os.path.join(st.work, "refs_radeon-r9700.json")) as f:
            self.assertEqual(json.load(f)[SLUG], {"old": reference, "new": reference})
        for suffix in ("", "-r2"):
            cmd = regen.generator_cmd(st.d, "radeon-r9700", [SLUG], suffix)
            self.assertEqual(cmd[cmd.index("--new-sdk") + 1], "0.3.0")
            self.assertEqual("--loom-only" in cmd, not suffix)

    def test_sdk_030_requires_both_verified_serving_wheels(self):
        job = {"dir": self.tmp}
        self.assertFalse(regen.measured_sdk(job, "0.3.0"))
        first = os.path.join(self.tmp, "sdk_staging.json")
        second = os.path.join(self.tmp, "sdk_stagingnd.json")
        for path, pin in ((first, regen.release_pin("0.3.0")), (second, {"hrx_loom": "0.3.0"})):
            with open(path, "w") as f:
                json.dump(pin, f)
        self.assertFalse(regen.measured_sdk(job, "0.3.0"))
        with open(second, "w") as f:
            json.dump(regen.release_pin("0.3.0"), f)
        self.assertTrue(regen.measured_sdk(job, "0.3.0"))

    def test_sdk_ab_takes_no_old_org(self):
        with self.assertRaises(SystemExit), mock.patch.object(regen, "sh", self.gen):
            regen.main(["plan", "--base", "sdk020", "--device", "radeon-r9700", "--mode", "sdk-ab", "--old", "loom-fun", "--state", self.state, SLUG])

    def test_a_row_neither_side_served_is_not_measured_not_missing(self):
        st = self.plan_sdk()
        e = st.jobs[JOB]
        e.update(round=1)
        p = os.path.join(self.tmp, "na.json")
        json.dump({"metrics": {"decode": {"old": 100.0, "new": 101.0, "ratio": 1.01}, "prefill_8k": {"old": 100.0, "new": 100.0, "ratio": 1.0},
                               "streams4": {"old": 100.0, "new": 100.0, "ratio": 1.0}}, "gate_failures": [], "text": {"status": "MATCH"}}, open(p, "w"))
        self.assertEqual(regen.sdk_decide(st, JOB, e, p)[0], "promote")   # no decode_32k row on either side
        m = json.load(open(p)); m["metrics"]["decode_32k"] = {"old": 50.0, "new": None, "ratio": None}; json.dump(m, open(p, "w"))
        self.assertEqual(regen.sdk_decide(st, JOB, e, p)[0], "rerun")   # the old side served it, the new one did not
        json.dump({"metrics": {}, "gate_failures": [], "text": {"status": "MATCH"}}, open(p, "w"))
        self.assertEqual(regen.sdk_decide(st, JOB, e, p)[0], "rerun")   # nothing measured on either side (a failed perf step): never a promote
        json.dump({"metrics": {"decode": {"old": 100.0, "new": 101.0, "ratio": 1.01}, "prefill_8k": {"old": 100.0, "new": 100.0, "ratio": 1.0},
                               "streams4": {"old": 100.0, "new": 100.0, "ratio": 1.0}, "decode_32k": {"old": None, "new": 50.0, "ratio": None}},
                   "gate_failures": [], "text": {"status": "MATCH"}}, open(p, "w"))
        d, r, *_ = regen.sdk_decide(st, JOB, e, p)
        self.assertEqual(d, "promote")
        self.assertTrue(any("only on the new SDK" in x for x in r))

    @unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "needs huggingface_hub")
    def test_old_sdk_revision_walks_the_history(self):
        from huggingface_hub.utils import EntryNotFoundError
        commits = [mock.Mock(commit_id=c) for c in ("c3", "c2", "c1", "c0")]
        pins = {"c3": {"sdk": {"hrx_loom": regen.SDK_NEW}}, "c2": {"sdk": {"hrx_loom": "0.1.0"}}, "c1": {}}
        def dl(repo, name, revision):
            if revision not in pins:
                raise EntryNotFoundError("no llm_config.json")
            p = os.path.join(self.tmp, f"{revision}.json"); json.dump(pins[revision], open(p, "w")); return p
        api = mock.Mock(); api.list_repo_commits.return_value = commits
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch("huggingface_hub.hf_hub_download", dl):
            self.assertEqual(regen.old_sdk_revision("o/r"), "c2")
            pins["c2"] = {"sdk": {"hrx_loom": regen.SDK_NEW}}; pins["c1"] = {"sdk": {"hrx_loom": regen.SDK_NEW}}
            self.assertIsNone(regen.old_sdk_revision("o/r"))   # c0 has no llm_config.json: not a package
        def boom(repo, name, revision):
            raise OSError("429")
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch("huggingface_hub.hf_hub_download", boom):
            with self.assertRaises(OSError):   # a Hub failure is not "no package": plan dies with a clear message instead
                regen.old_sdk_revision("o/r")

    def test_round_two_with_nothing_measured_holds(self):
        st = self.plan_sdk()
        e = st.jobs[JOB]
        e.update(round=2, prev=JOB)
        e["perf"] = {"decode": 1.0, "prefill": 1.0, "longctx": 1.0, "4stream": 1.0}
        p = os.path.join(self.tmp, "empty.json")
        json.dump({"metrics": {}, "gate_failures": [], "text": {"status": "MATCH"}}, open(p, "w"))
        self.assertEqual(regen.sdk_decide(st, JOB, e, p)[0], "hold")

    def test_new_sdk_pins_are_the_target_and_its_candidates(self):
        base = regen.SDK_NEW.split("rc")[0]
        self.assertTrue(regen.new_sdk_pin(base) and regen.new_sdk_pin(base + "rc1"))
        self.assertFalse(regen.new_sdk_pin(None) or regen.new_sdk_pin("0.1.0") or regen.new_sdk_pin(base + ".dev0"))

    def test_deltas_in_percent_positive_is_v02_faster(self):
        self.e(self.plan_sdk())
        e = self.collect_with(self.GOOD).jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "promote"))
        self.assertEqual(e["perf"], {"decode": 2.0, "prefill": 5.0, "longctx": -1.0, "4stream": 0.0})   # longctx -1 %: inside the 3 % ratchet
        self.assertEqual(e["rows_pct"]["decode_2k"], 1.0)

    def test_regression_over_3_percent_asks_for_one_confirming_run(self):
        self.e(self.plan_sdk())
        e = self.collect_with({**self.GOOD, "prefill_8k": 0.96}).jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "rerun"))
        self.assertIn("prefill -4.0 %", e["reason"])

    def test_missing_metric_is_a_rerun_not_a_pass(self):
        self.e(self.plan_sdk())
        e = self.collect_with({**self.GOOD, "streams4": None}).jobs[JOB]   # the old side served the row, the new one did not
        self.assertEqual(e["decision"], "rerun")
        self.assertIn("4stream", e["reason"])

    def test_failed_gate_stops_the_model(self):
        self.e(self.plan_sdk())
        e = self.collect_with(self.GOOD, gates=["gates:staging:dev"]).jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("needs_decision", "hold"))
        self.assertIn("gate failed", e["reason"])

    def test_text_difference_is_reported_not_held(self):
        self.e(self.plan_sdk())
        e = self.collect_with(self.GOOD, text="DIFF").jobs[JOB]
        self.assertEqual(e["decision"], "promote")
        self.assertIn("greedy text differs", e["reason"])

    def confirm(self, first, second):
        st = self.plan_sdk()
        self.e(st)
        self.collect_with(first)
        with mock.patch.object(regen, "sh", self.gen), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["rerun", self.state])
        st = regen.State(self.state)
        r2 = next(n for n in st.jobs if n.endswith("-r2"))
        self.assertEqual("--suffix=-r2" in self.calls[-1], True)
        self.assertEqual(st.jobs[r2]["prev"], JOB)
        st.jobs[r2].update(stage="done", run="8", job_url="https://x/job/2", completed="2026-11-16T14:00:00Z")
        st.save()
        return self.collect_with(second, name=r2).jobs[r2]

    def test_confirmed_regression_is_needs_decision_with_both_numbers(self):
        e = self.confirm({**self.GOOD, "prefill_8k": 0.96}, {**self.GOOD, "prefill_8k": 0.94})
        self.assertEqual((e["stage"], e["decision"]), ("needs_decision", "hold"))
        self.assertIn("prefill -4.0 % / -6.0 %", e["reason"])
        self.assertEqual(e["perf"]["prefill"], -5.0)   # the mean of the two runs

    def test_regression_not_repeated_is_promoted_with_the_mean(self):
        e = self.confirm({**self.GOOD, "prefill_8k": 0.96}, {**self.GOOD, "prefill_8k": 1.02})
        self.assertEqual((e["stage"], e["decision"]), ("collected", "promote"))
        self.assertEqual(e["perf"]["prefill"], -1.0)

    def test_degraded_runner_reruns(self):
        self.e(self.plan_sdk())
        e = self.collect_with(self.GOOD, degraded={"decode": 0.8}).jobs[JOB]
        self.assertEqual(e["decision"], "rerun")

    def test_status_json_gets_perf_per_gpu(self):
        self.e(self.plan_sdk())
        self.collect_with({**self.GOOD, "prefill_8k": 0.9})   # round 1 regression: its perf is already there
        p = os.path.join(self.tmp, "status.json")
        json.dump({"models": {SLUG: {"strix": {"stage": "queued"}, "perf": {"strix": {"decode": 1.0}}}}}, open(p, "w"))
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["status", self.state, "--status-json", p])
        m = json.load(open(p))["models"][SLUG]
        self.assertEqual(m["perf"]["r9700"], {"decode": 2.0, "prefill": -10.0, "longctx": -1.0, "4stream": 0.0})
        self.assertEqual(m["perf"]["strix"], {"decode": 1.0})   # the other GPU's numbers stay
        self.assertEqual(m["r9700"]["stage"], "measured")

    def test_ingest_makes_the_v02_card_on_the_staging_head_with_the_delta_line(self):
        self.e(self.plan_sdk())
        e = self.collect_with(self.GOOD).jobs[JOB]
        open(os.path.join(e["dir"], "programs_sha256_staging.txt"), "w").write("b" * 64 + "\n")   # the programs the job BUILT (devlab_perf.sh)
        self.package_recorded(e, "b" * 64)
        calls = []

        def hf_snapshot(repo, rev, out):
            calls.append(("snapshot", repo, rev))
            os.makedirs(os.path.join(out, "program"), exist_ok=True)
            open(os.path.join(out, "program", "llm.loom"), "w").write("x")
            json.dump({"card": {"memory_bytes": 1}, "devices": {}}, open(os.path.join(out, "info.json"), "w"))
            open(os.path.join(out, "README.md"), "w").write("x")

        def run_tool(args, cwd):
            calls.append(("tool", [str(a) for a in args]))
            if "build" in args and "-o" in args:
                shutil.copy(os.path.join(cwd, "info_prev.json"), os.path.join(cwd, "info_new.json"))
            return ""
        with mock.patch.object(regen, "hf_snapshot", hf_snapshot), mock.patch.object(regen, "hf_head", lambda r: "h" * 40), \
                mock.patch.object(regen.card_lib, "programs_sha256", lambda d: "b" * 64), \
                mock.patch.object(regen, "run_tool", run_tool), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["ingest", self.state])
        self.assertEqual(calls[0], ("snapshot", "loom-fun-staging/" + SLUG, "h" * 40))   # the head (v0.2 package), not the v0.1 revision the A/B started from
        perf = next(c[1] for c in calls if c[0] == "tool" and "perf" in c[1] and "--loom-jsonl" in c[1])
        self.assertEqual(perf[perf.index("--fetched-programs") + 1], "b" * 64)   # make_info refuses a head whose programs are not the ones measured
        e = regen.State(self.state).jobs[JOB]
        self.assertEqual(e["stage"], "ingested")
        self.assertIn("decode +2.0 %", e["message"])
        self.assertIn("prefill +5.0 %", e["message"])

    def ingest_stopped(self, h_package, h_job, head=None):
        self.e(self.plan_sdk())
        e = self.collect_with(self.GOOD).jobs[JOB]
        open(os.path.join(e["dir"], "programs_sha256_staging.txt"), "w").write(h_job + "\n")
        if h_package:
            self.package_recorded(e, h_package)
        with mock.patch.object(regen, "hf_snapshot", side_effect=AssertionError("no snapshot of a model that stops")), \
                mock.patch.object(regen, "hf_head", lambda r: "h" * 40), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["ingest", self.state])
        return regen.State(self.state).jobs[JOB]

    def test_ingest_stops_on_a_programs_hash_other_than_the_jobs(self):
        e = self.ingest_stopped("a" * 64, "b" * 64)
        self.assertEqual(e["stage"], "needs_decision")
        self.assertIn("not the ones the job built", e["reason"])

    def test_ingest_stops_when_no_package_was_published(self):
        e = self.ingest_stopped(None, "b" * 64)
        self.assertEqual(e["stage"], "needs_decision")
        self.assertIn("regen.py package", e["reason"])

    def test_ingest_stops_when_the_head_lost_the_package_programs(self):
        self.e(self.plan_sdk())
        e = self.collect_with(self.GOOD).jobs[JOB]
        open(os.path.join(e["dir"], "programs_sha256_staging.txt"), "w").write("b" * 64 + "\n")
        self.package_recorded(e, "b" * 64)

        def snap(repo, rev, out):
            os.makedirs(os.path.join(out, "program"))
            open(os.path.join(out, "program", "llm.loom"), "w").write("someone else's")
        with mock.patch.object(regen, "hf_snapshot", snap), mock.patch.object(regen, "hf_head", lambda r: "h" * 40), \
                contextlib.redirect_stdout(io.StringIO()):
            regen.main(["ingest", self.state])
        e = regen.State(self.state).jobs[JOB]
        self.assertEqual(e["stage"], "needs_decision")
        self.assertIn("package", e["reason"])


class SdkLoomTest(Base):
    """--mode sdk-loom: the Loom-only sweep. The job's Loom rows are compared with the OLD card's Loom rows (another job); a row outside +-3 % either way
    gets ONE targeted ABBA job (the sdk-ab job without baselines) whose verdict decides; ingest takes the llama.cpp rows from the previous card."""
    e = SdkAbTest.e
    package_recorded = SdkAbTest.package_recorded

    def plan_loom(self):
        return self.plan("--mode", "sdk-loom", "--new-sdk", "0.2.0")

    def cmp_file(self, name, vs_card, gates=None):
        """a strix_ab_compare.py result of a Loom-only job: new values and new / old-card ratios (no old side)"""
        p = os.path.join(self.tmp, f"{name}.json")
        json.dump({"job": name, "metrics": {k: {"old": None, "new": 100.0 * r if r and r != "nocard" else (100.0 if r else None), "ratio": None, "status": "MISSING",
                                                "card": 100.0 if r != "nocard" else None, "new_vs_card": r if r != "nocard" else None}
                                            for k, r in vs_card.items()}, "gate_failures": gates or [], "degraded_runner": {}}, open(p, "w"))
        return p

    def test_accept_regression_publishes_only_a_confirmed_slowdown(self):
        st = self.plan_loom() if hasattr(self, "plan_loom") else self.plan("--mode", "sdk-loom", "--new-sdk", "0.2.0")
        e = st.jobs[JOB]
        e.update(stage="needs_decision", reason="confirmed regression / failed gate / text difference: ABBA confirmation (job 1): slower than -3%: prefill -7.4 %",
                 confirm_rows=["prefill"], perf={"decode": 1.0, "prefill": -6.0, "longctx": 0.0, "4stream": 0.0})
        st.jobs["abba"] = {**{k: v for k, v in e.items() if k not in ("perf", "confirm_rows")}, "slug": SLUG, "prev": JOB, "perf_abba": {"decode": 0.5, "prefill": -7.4, "longctx": None, "4stream": 0.2}}
        st.save()
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["accept-regression", self.state, SLUG, "--note", "U1: SDK runtime on Strix (notes/upstream.md)"])
        e = regen.State(self.state).jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "promote"))
        self.assertIn("U1", e["accepted_regression"])
        self.assertEqual(e["perf"]["prefill"], -7.4)   # the ABBA number on the confirmed row
        self.assertEqual(e["confirmed"], {"prefill": -7.4})
        st = regen.State(self.state); st.jobs[JOB].update(stage="needs_decision", reason="gate failed: ktest:staging"); st.save()
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["accept-regression", self.state, SLUG, "--note", "x"])
        self.assertEqual(regen.State(self.state).jobs[JOB]["stage"], "needs_decision")   # a gate failure is never accepted

    def test_accept_regression_sdk_ab_needs_both_runs_confirmed(self):
        st = self.plan("--mode", "sdk-ab", "--new-sdk", "0.2.0")
        e = st.jobs[JOB]
        e.update(stage="needs_decision", round=2, reason="confirmed regression / failed gate / text difference: still undecided after the confirming run: slower than -3%: decode -4.0 %")
        st.save()
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["accept-regression", self.state, SLUG, "--note", "x"])
        self.assertEqual(regen.State(self.state).jobs[JOB]["stage"], "needs_decision")   # never confirmed
        st = regen.State(self.state)
        st.jobs[JOB].update(reason="confirmed regression / failed gate / text difference: confirmed regression > 3% in both runs: decode -4.0 % / -4.2 %")
        st.save()
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["accept-regression", self.state, SLUG, "--note", "SDK"])
        self.assertEqual(regen.State(self.state).jobs[JOB]["stage"], "collected")

    def _gate_job(self, e, fails, metrics, ktest="FAIL (67/68 passed; 2 skipped)", log="67/68 passed\n", tag="j"):
        d = os.path.join(self.tmp, "gate_" + tag)
        os.makedirs(d, exist_ok=True)
        json.dump({"results": {"ktest:staging": ktest}}, open(os.path.join(d, "summary.json"), "w"))
        if log is not None:
            open(os.path.join(d, "ktest_staging.log"), "w").write(log)
        cmp = os.path.join(d, "cmp.json")
        json.dump({"gate_failures": fails, "metrics": metrics}, open(cmp, "w"))
        e.update(dir=d, cmp=cmp)
        return cmp

    def _accept_gate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["accept-gate-failure", self.state, SLUG, "--note", "same case failed on the 0.1.0 card"])
        return regen.State(self.state)

    def test_accept_gate_failure_only_kernel_tests_that_ran(self):
        st = self.plan("--mode", "sdk-loom", "--new-sdk", "0.2.0")
        metrics = {r: {"new": 100.0, "card": 100.0, "new_vs_card": 1.0} for r in regen.SDK_METRICS.values()}
        held = dict(stage="needs_decision", decision="hold", reason="confirmed regression / failed gate / text difference: gate failed: ktest:staging")
        cmp = self._gate_job(st.jobs[JOB], ["ktest:staging"], metrics)
        st.jobs[JOB].update(held); st.save()
        e = self._accept_gate().jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "promote"))
        self.assertIn("ktest:staging failed, published as failed", e["accepted_gate_failure"])
        self.assertEqual(json.load(open(cmp))["gate_failures"], ["ktest:staging"])   # the job's evidence is untouched: the card shows the failure
        self.assertIn("published as failed", regen.card_message(e, "strix-halo", "9", True, True))
        # refused: a func / serving gate; kernel tests that did not run to a result (ERROR, or no "N/M passed" failure in the log)
        for fails, kt, log in ((["ktest:staging", "func:staging"], "FAIL (67/68 passed)", "67/68 passed\n"), (["ktest:staging"], "ERROR", None),
                               (["ktest:staging"], "FAIL", "Traceback\n")):
            st = regen.State(self.state)
            self._gate_job(st.jobs[JOB], fails, metrics, kt, log, tag=kt[:4] + str(len(fails)))
            st.jobs[JOB].update(held); st.save()
            self.assertEqual(self._accept_gate().jobs[JOB]["stage"], "needs_decision", (fails, kt))
        # perf rows outside the band still ask for the confirming job
        st = regen.State(self.state)
        self._gate_job(st.jobs[JOB], ["ktest:staging"], {**metrics, "prefill_8k": {"new": 90.0, "card": 100.0, "new_vs_card": 0.9}}, tag="slow")
        st.jobs[JOB].update(held); st.save()
        e = self._accept_gate().jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "confirm"))

    def test_accept_gate_failure_settles_a_held_abba_confirmation(self):
        st = self.plan("--mode", "sdk-loom", "--new-sdk", "0.2.0")
        e = st.jobs[JOB]
        e.update(stage="needs_decision", decision="confirm", confirm_rows=["prefill"], perf_cross={"decode": 0.0, "prefill": -5.0, "longctx": 0.0, "4stream": 0.0},
                 reason="ABBA confirmation (job 7): gate failed: ktest:staging (ABBA prefill -1.0 %)")
        abba = {k: v for k, v in e.items() if k not in ("confirm_rows", "perf_cross")}
        abba.update(prev=JOB, round=2, run="7", stage="needs_decision", decision="hold")
        ab = {r: {"old": 100.0, "new": 99.0, "ratio": 0.99} for r in regen.SDK_METRICS.values()}
        self._gate_job(abba, ["ktest:staging"], ab, tag="abba")
        st.jobs["abba"] = abba
        st.save()
        e = self._accept_gate().jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "promote"))
        self.assertEqual(e["confirmed"], {"prefill": -1.0})
        self.assertIn("published as failed", e["accepted_gate_failure"])

    def test_accept_gate_failure_sdk_ab_round2(self):
        st = self.plan("--mode", "sdk-ab", "--new-sdk", "0.2.0")
        e = st.jobs[JOB]
        ab = {r: {"old": 100.0, "new": 100.0, "ratio": 1.0} for r in regen.SDK_METRICS.values()}
        self._gate_job(e, ["ktest:staging"], ab, tag="ab2")
        e.update(stage="needs_decision", decision="hold", round=2, reason="confirmed regression / failed gate / text difference: gate failed: ktest:staging")
        st.save()
        e = self._accept_gate().jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "promote"))
        st = regen.State(self.state)
        self._gate_job(st.jobs[JOB], ["ktest:staging"], {**ab, "decode": {"old": 100.0, "new": 90.0, "ratio": 0.9}}, tag="ab2s")
        st.jobs[JOB].update(stage="needs_decision", decision="hold"); st.save()
        e = self._accept_gate().jobs[JOB]
        self.assertEqual(e["stage"], "needs_decision")   # round 2 slower than the band: a person decides, as collect does
        self.assertTrue(e["reason"].startswith("still undecided after the confirming run: "))

    def test_card_message_per_mode_and_accepted_note(self):
        e = {"perf": {"decode": 1.0, "prefill": -7.4, "longctx": 0.0, "4stream": 0.2}, "confirmed": {"prefill": -7.4},
             "accepted_regression": "U1: SDK runtime (notes) (was: ...)"}
        m = regen.card_message(e, "strix-halo", "9", True, True)
        self.assertIn("another job, same runner type", m)
        self.assertIn("baseline rows are the previous card's", m)   # sdk-loom also migrates safetensors cards whose baseline is vLLM
        self.assertNotIn("llama.cpp", m)
        self.assertNotIn("own SDK pin", m)
        self.assertIn("confirmed slowdown published as measured: U1: SDK runtime (notes)", m)
        m = regen.card_message({k: v for k, v in e.items() if k != "confirmed"}, "strix-halo", "9", False, True)
        self.assertIn("own SDK pin (same job, ABBA): decode", m)
        self.assertIn("confirmed slowdown published as measured", m)
        self.assertNotIn("confirmed slowdown", regen.card_message({"perf": e["perf"]}, "strix-halo", "9", False, True))

    def test_real_compare_marks_a_card_row_the_job_did_not_measure(self):
        """through .github/scripts/strix_ab_compare.py: a Loom-only job without the 32k rows vs a card that has them -> longctx is confirmed, not n/a"""
        d = os.path.join(self.tmp, "job")
        shutil.copytree(os.path.join(DATA, "job_e2b"), d)
        for f in os.listdir(d):
            if "fun" in f:
                os.remove(os.path.join(d, f))
        p = os.path.join(d, "perf_staging.jsonl")   # drop the 32k rows: the job did not measure long context
        strip = lambda o: {k: strip(v) for k, v in o.items() if "32k" not in k} if isinstance(o, dict) else o
        open(p, "w").write("".join(json.dumps(strip(json.loads(l))) + "\n" for l in open(p) if l.strip()))
        info = os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card", "test", "flint", "expected", "info.json")
        out = os.path.join(self.tmp, "cmp.json")
        subprocess.run([sys.executable, os.path.join(ROOT, ".github", "scripts", "strix_ab_compare.py"), d, "--json", out, "--fun-info", info,
                        "--gpu", "strix-halo"], check=True, capture_output=True, text=True)
        m = json.load(open(out))["metrics"]
        self.assertIn("decode_32k", m)
        self.assertIsNone(m["decode_32k"]["new"])
        self.assertTrue(m["decode_32k"]["card"])
        st = self.plan_loom() if hasattr(self, "plan_loom") else None
        e = {"round": 1}
        decision, reasons, *_ = regen.sdk_loom_decide(st, JOB, e, out)
        self.assertEqual(decision, "confirm")
        self.assertIn("longctx", e["confirm_rows"])

    def collect_cmp(self, cmp, name=JOB):
        def download(st, n, e):
            d = regen.job_dir(st, n)
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(os.path.join(DATA, "job_e2b"), d)
            return d
        with mock.patch.object(regen, "download", download), mock.patch.object(regen, "compare", lambda st, n, e, d: cmp), \
                contextlib.redirect_stdout(io.StringIO()):
            regen.main(["collect", self.state])
        return regen.State(self.state)

    def collect_loom(self, vs_card, name=JOB, **kw):
        return self.collect_cmp(self.cmp_file(name, vs_card, **kw), name)

    IN = {"decode": 1.02, "decode_2k": 1.05, "prefill_8k": 0.98, "decode_32k": 1.0, "streams4": 1.01}

    def test_plan_makes_loom_only_jobs_and_remembers_the_old_card(self):
        st = self.plan_loom()
        cmd = next(c for c in self.calls if any(x.endswith("ab_jobs.py") for x in c))
        for flag in ("--sdk-ab", "--loom-only"):
            self.assertIn(flag, cmd)
        self.assertNotIn("--baselines", cmd)
        self.assertEqual(st.jobs[JOB]["old"], f"loom-fun-staging/{SLUG}@" + "e" * 40)
        self.assertEqual(st.d["mode"], "sdk-loom")

    def test_rows_inside_the_band_promote_with_cross_job_deltas(self):
        self.e(self.plan_loom())
        e = self.collect_loom(self.IN).jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("collected", "promote"))
        self.assertEqual(e["perf"], {"decode": 2.0, "prefill": -2.0, "longctx": 0.0, "4stream": 1.0})

    def test_a_row_outside_the_band_either_way_asks_for_one_abba_job(self):
        for row, r, text in (("prefill_8k", 0.96, "prefill -4.0 %"), ("decode", 1.05, "decode +5.0 %")):
            self.setUp()
            self.e(self.plan_loom())
            e = self.collect_loom({**self.IN, row: r}).jobs[JOB]
            self.assertEqual((e["stage"], e["decision"]), ("collected", "confirm"), row)
            self.assertIn(text, e["reason"])
            self.assertEqual(e["confirm_rows"], [{"decode": "decode", "prefill_8k": "prefill"}[row]])

    def test_a_row_the_old_card_has_no_value_for_is_reported_not_confirmed(self):
        self.e(self.plan_loom())
        e = self.collect_loom({**self.IN, "streams4": "nocard"}).jobs[JOB]
        self.assertEqual(e["decision"], "promote")
        self.assertIn("4stream", e["reason"])
        self.assertIsNone(e["perf"]["4stream"])

    def test_a_row_the_old_card_has_that_the_job_did_not_measure_is_confirmed(self):
        self.e(self.plan_loom())
        e = self.collect_loom({**self.IN, "decode_32k": None}).jobs[JOB]
        self.assertEqual((e["decision"], e["confirm_rows"]), ("confirm", ["longctx"]))
        self.assertIn("longctx missing", e["reason"])

    def test_a_failed_gate_holds_and_nothing_measured_holds(self):
        self.e(self.plan_loom())
        e = self.collect_loom(self.IN, gates=["ktest:staging"]).jobs[JOB]
        self.assertEqual((e["stage"], e["decision"]), ("needs_decision", "hold"))
        self.setUp()
        self.e(self.plan_loom())
        e = self.collect_loom({k: None for k in self.IN}).jobs[JOB]
        self.assertEqual(e["decision"], "hold")

    def abba(self, first, second):
        """round 1 (cross-job) then its ABBA job; returns the state after both are collected"""
        self.e(self.plan_loom())
        self.collect_loom(first)
        with mock.patch.object(regen, "sh", self.gen), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["rerun", self.state])
        cmd = self.calls[-1]
        self.assertIn("--suffix=-r2", cmd)
        self.assertIn("--sdk-ab", cmd)
        self.assertNotIn("--loom-only", cmd)   # the confirming job is the ABBA: old card on its pin against the new package
        self.assertNotIn("--baselines", cmd)
        st = regen.State(self.state)
        r2 = next(n for n in st.jobs if n.endswith("-r2"))
        self.assertEqual((st.jobs[r2]["prev"], st.jobs[r2]["old"]), (JOB, st.jobs[JOB]["old"]))
        st.jobs[r2].update(stage="done", run="8", job_url="https://x/job/2", completed="2026-11-16T14:00:00Z")
        st.save()
        cmp = SdkAbTest.cmp_file(self, "abba", second)
        return self.collect_cmp(cmp, r2), r2

    def test_abba_job_decides_the_confirmed_rows_and_their_deltas_replace_the_cross_job_ones(self):
        st, r2 = self.abba({**self.IN, "prefill_8k": 0.96, "decode": 1.06}, {"decode": 1.04, "prefill_8k": 0.985, "decode_32k": 1.0, "streams4": 1.0, "decode_2k": 1.0})
        f = st.jobs[JOB]
        self.assertEqual((f["stage"], f["decision"]), ("collected", "promote"))
        self.assertEqual(f["confirmed"], {"decode": 4.0, "prefill": -1.5})
        self.assertEqual(f["perf"], {"decode": 4.0, "prefill": -1.5, "longctx": 0.0, "4stream": 1.0})   # longctx and 4stream stay cross-job
        self.assertEqual(f["perf_cross"]["decode"], 6.0)
        self.assertNotIn("perf", st.jobs[r2])   # the confirming job reports through the first one
        p = os.path.join(self.tmp, "status.json")
        json.dump({"models": {}}, open(p, "w"))
        with contextlib.redirect_stdout(io.StringIO()):
            regen.main(["status", self.state, "--status-json", p])
        m = json.load(open(p))["models"][SLUG]
        self.assertEqual(m["perf"]["r9700"], f["perf"])
        self.assertEqual(m["perf_confirmed"]["r9700"], ["decode", "prefill"])

    def test_abba_job_slower_than_3_percent_stops_the_model_with_the_numbers(self):
        st, r2 = self.abba({**self.IN, "prefill_8k": 0.96}, {"decode": 1.0, "prefill_8k": 0.95, "decode_32k": 1.0, "streams4": 1.0, "decode_2k": 1.0})
        f = st.jobs[JOB]
        self.assertEqual(f["stage"], "needs_decision")
        self.assertIn("prefill -5.0 %", f["reason"])
        self.assertEqual((st.jobs[r2]["stage"], st.jobs[r2]["decision"]), ("needs_decision", "hold"))

    def test_ingest_takes_the_llama_rows_from_the_previous_card_and_names_the_confirmed_rows(self):
        st, _ = self.abba({**self.IN, "prefill_8k": 0.96}, {"decode": 1.0, "prefill_8k": 0.99, "decode_32k": 1.0, "streams4": 1.0, "decode_2k": 1.0})
        e = st.jobs[JOB]
        open(os.path.join(e["dir"], "programs_sha256_staging.txt"), "w").write("b" * 64 + "\n")
        self.package_recorded(e, "b" * 64)
        calls = []

        def hf_snapshot(repo, rev, out):
            os.makedirs(os.path.join(out, "program"), exist_ok=True)
            open(os.path.join(out, "program", "llm.loom"), "w").write("x")
            json.dump({"card": {"memory_bytes": 1}, "devices": {}}, open(os.path.join(out, "info.json"), "w"))
            open(os.path.join(out, "README.md"), "w").write("x")

        def run_tool(args, cwd):
            calls.append([str(a) for a in args])
            if "build" in args and "-o" in args:
                shutil.copy(os.path.join(cwd, "info_prev.json"), os.path.join(cwd, "info_new.json"))
            return ""
        with mock.patch.object(regen, "hf_snapshot", hf_snapshot), mock.patch.object(regen, "hf_head", lambda r: "h" * 40), \
                mock.patch.object(regen.card_lib, "programs_sha256", lambda d: "b" * 64), \
                mock.patch.object(regen, "run_tool", run_tool), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["ingest", self.state])
        perf = next(c for c in calls if "perf" in c and "--loom-jsonl" in c)
        self.assertIn("--baseline-from-info", perf)
        self.assertNotIn("--llama-jsonl", perf)
        self.assertFalse([c for c in calls if "--backend" in c])   # no Vulkan re-measure: its session stays as published
        build = next(c for c in calls if "build" in c and "--evidence" in c)
        self.assertIn("--reset-performance", build)
        e = regen.State(self.state).jobs[JOB]
        self.assertEqual(e["stage"], "ingested")
        self.assertIn("another job", e["message"])
        self.assertIn("confirmed by an ABBA job (same job): prefill -1.0 %", e["message"])

    def test_package_and_ingest_use_the_first_job_only(self):
        st, r2 = self.abba({**self.IN, "prefill_8k": 0.96}, {"decode": 1.0, "prefill_8k": 0.99, "decode_32k": 1.0, "streams4": 1.0, "decode_2k": 1.0})
        self.assertEqual((st.jobs[r2]["stage"], st.jobs[r2]["decision"]), ("collected", "promote"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            regen.main(["ingest", self.state, "--slug", SLUG])   # no package recorded: the first job stops, the confirming job is not ingested
        st = regen.State(self.state)
        self.assertEqual(st.jobs[JOB]["stage"], "needs_decision")
        self.assertEqual(st.jobs[r2]["stage"], "collected")


def v01_model():
    """a published (v0.1.0 era) model.json: pool_capacity, origins, a parameter file origin, a per-GPU preset"""
    return {"schema_version": 1, "model": {"id": "a/b"}, "programs": {"source": {"inputs": ["program/control.loom"]}},
            "variants": {"gguf": {"services": {"chat": {"config": {"llm.card": True}, "parameters": {"model": {"files": [
                {"path": "parameter/m.gguf", "format": "gguf", "origin": "@checkpoint/m.gguf"}]}}}}}},
            "origins": {"checkpoint": {"kind": "hf", "repository": "a/b-GGUF", "revision": "r" * 40}},
            "presets": {"chat": {"execution": {"context_capacity": 4096, "pool_capacity": 8192, "draft_depth": 3}},
                        "chat-r9700": {"execution": {"context_capacity": 4096, "pool_capacity": 8192, "draft_depth": 2}}}}


def v02_model():
    """what the canonical build writes for it: no pool_capacity, no origins, a new config key"""
    m = v01_model()
    m.pop("origins")
    m["presets"].pop("chat-r9700")
    m["presets"]["chat"]["execution"].pop("pool_capacity")
    m["variants"]["gguf"]["services"]["chat"]["config"]["llm.logits"] = False
    m["variants"]["gguf"]["services"]["chat"]["parameters"]["model"]["files"][0].pop("origin")
    m["programs"]["source"]["inputs"].append("llm_config.json")
    return m


class PackageTest(Base):
    """regen.py package: the v0.2 programs onto the staging repo (offline: a fake snapshot, a fake canonical build, the upload as a dry run)"""
    PUBLISHED_CARD = ("README.md", "info.json", "REPORT.md", "tokenizer.json", "model.json.lock")

    def write_tree(self, d, programs, model, gen=None, extra=()):
        os.makedirs(os.path.join(d, "program"), exist_ok=True)
        for n, t in programs.items():
            open(os.path.join(d, "program", n), "w").write(t)
        for n, t in (gen or {}).items():
            os.makedirs(os.path.join(d, "gen"), exist_ok=True)
            open(os.path.join(d, "gen", n), "w").write(t)
        open(os.path.join(d, "weights.loom"), "w").write("w")
        json.dump(model, open(os.path.join(d, "model.json"), "w"))
        json.dump({"gguf": "m.gguf"}, open(os.path.join(d, "llm_config.json"), "w"))
        for f in extra:
            open(os.path.join(d, f), "w").write(f)

    def setUp(self):
        super().setUp()
        self.snap = os.path.join(self.tmp, "snap")
        self.write_tree(self.snap, {"control.loom": "v01", "stale.loom": "gone"}, v01_model(), {"flash2.loom": "old gen"}, self.PUBLISHED_CARD)
        self.canon = os.path.join(self.tmp, "canon")
        self.write_tree(os.path.join(self.canon, SLUG), {"control.loom": "v02"}, v02_model(), {"flash2.loom": "new gen"})
        open(os.path.join(self.canon, SLUG, ".ref"), "w").write("c" * 40)
        self.want = regen.card_lib.programs_sha256(os.path.join(self.canon, SLUG))
        self.st = self.plan("--mode", "sdk-ab", "--new-sdk", "0.2.0")

    def run_package(self, *args, head="a" * 40, new_head="d" * 40, snap_programs=None, upload_stdout=None, source_of=None):
        uploads = []
        heads = iter([head, new_head])

        def sh(cmd, **kw):
            if cmd[0] == "git":   # the default --ref lookup
                return subprocess.CompletedProcess(cmd, 1, "", "")
            uploads.append([str(c) for c in cmd])
            repo = f"{regen.STAGING_ORG}/{SLUG}"
            receipt = {"repo": repo, "parent_commit": head, "commit": "b" * 40,
                       "url": f"https://huggingface.co/{repo}/commit/" + "b" * 40}
            stdout = ("dry run: would upload 12 files to https://huggingface.co/x" if "--dry-run" in cmd
                      else json.dumps(receipt) if upload_stdout is None else upload_stdout)
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        def snapshot(repo, rev, out):
            uploads.append(["snapshot", repo, rev])
            shutil.rmtree(out, ignore_errors=True)   # as hf_snapshot_small does
            shutil.copytree(source_of(repo, rev) if source_of else self.snap, out)
        out = io.StringIO()
        with mock.patch.object(regen, "sh", sh), mock.patch.object(regen, "hf_head", lambda r: next(heads)), \
                mock.patch.object(regen, "hf_snapshot_small", snapshot), \
                mock.patch.object(regen, "canonical_build", lambda st, slugs, ref: (self.canon, "c" * 40)), contextlib.redirect_stdout(out):
            regen.main(["package", self.state, *args])
        return regen.State(self.state), uploads, out.getvalue()

    def test_dry_run_lists_the_upload_and_the_programs_hash_and_changes_nothing(self):
        st, calls, out = self.run_package("--dry-run", "--slugs", SLUG)
        up = next(c for c in calls if "upload" in c)
        self.assertIn("--dry-run", up)
        self.assertEqual(up[up.index("--org") + 1], "loom-fun-staging")
        self.assertEqual(up[up.index("--parent-commit") + 1], "a" * 40)   # the snapshot's revision
        self.assertNotIn("--card-only", up)
        self.assertIn("hrx-loom 0.2.0", up[up.index("--message") + 1])
        self.assertIn(self.want, out)
        self.assertNotIn("package", st.jobs[JOB])

    def test_package_dir_has_the_canonical_programs_and_keeps_the_card(self):
        self.run_package("--dry-run", "--slugs", SLUG)
        pkg = os.path.join(regen.State(self.state).work, "package", SLUG, "pkg")
        self.assertEqual(open(os.path.join(pkg, "program", "control.loom")).read(), "v02")
        self.assertFalse(os.path.exists(os.path.join(pkg, "program", "stale.loom")))
        self.assertEqual(open(os.path.join(pkg, "gen", "flash2.loom")).read(), "new gen")
        for f in self.PUBLISHED_CARD:
            self.assertTrue(os.path.isfile(os.path.join(pkg, f)), f)
        self.assertEqual(regen.card_lib.programs_sha256(pkg), self.want)
        m = json.load(open(os.path.join(pkg, "model.json")))
        self.assertEqual(m["origins"], v01_model()["origins"])   # what publish_package.py prepare wrote stays
        self.assertEqual(m["variants"]["gguf"]["services"]["chat"]["parameters"]["model"]["files"][0]["origin"], "@checkpoint/m.gguf")
        self.assertNotIn("pool_capacity", json.dumps(m))
        self.assertFalse(m["variants"]["gguf"]["services"]["chat"]["config"]["llm.logits"])   # the v0.2 field the canonical build adds

    def test_a_production_source_card_keeps_the_staging_card_files(self):
        """the package assembled from the pinned production revision (frozen-source phase) ships the staging repo's card files until the
        card step: the production README / info.json name loom-fun, and a card file the staging head lacks is not carried over"""
        st = regen.State(self.state)
        st.jobs[JOB]["source_card"] = f"loom-fun/{SLUG}@{'e' * 40}"
        st.save()
        staging = os.path.join(self.tmp, "staging_head")
        self.write_tree(staging, {"control.loom": "v01"}, v01_model(), None, ("tokenizer.json",))
        for f in ("README.md", "info.json"):
            open(os.path.join(staging, f), "w").write("staging " + f)
        production = self.snap

        def pick(repo, rev):
            return production if repo.startswith("loom-fun/") else staging
        self.run_package("--dry-run", "--slugs", SLUG, source_of=pick)
        pkg = os.path.join(regen.State(self.state).work, "package", SLUG, "pkg")
        for f in ("README.md", "info.json"):
            self.assertEqual(open(os.path.join(pkg, f)).read(), "staging " + f)
        self.assertFalse(os.path.exists(os.path.join(pkg, "REPORT.md")))   # production-only: never shipped on staging
        self.assertEqual(open(os.path.join(pkg, "program", "control.loom")).read(), "v02")
        self.assertEqual(regen.card_lib.programs_sha256(pkg), self.want)

    def test_real_run_records_the_new_revision_and_hash_and_is_idempotent(self):
        st, calls, _ = self.run_package("--slugs", SLUG)
        up = next(c for c in calls if "upload" in c)
        self.assertNotIn("--dry-run", up)
        pk = st.jobs[JOB]["package"]
        self.assertEqual((pk["revision"], pk["programs_sha256"], pk["base"]), ("b" * 40, self.want, "c" * 40))
        self.assertEqual(pk["receipt"]["commit"], pk["revision"])   # later HEAD would be d*40
        st, calls, out = self.run_package("--slugs", SLUG)   # nothing left to do: no snapshot, no upload
        self.assertFalse(calls)
        self.assertIn("already published", out)

    def test_a_head_that_already_holds_the_package_is_recorded_not_uploaded(self):
        st, calls, _ = self.run_package("--dry-run", "--slugs", SLUG)   # assembles the package into the state's work dir
        pkg = os.path.join(st.work, "package", SLUG, "pkg")
        shutil.rmtree(self.snap)
        shutil.copytree(pkg, self.snap)   # the head now holds exactly that package
        st, calls, _ = self.run_package("--slugs", SLUG)
        self.assertFalse([c for c in calls if "upload" in c])
        self.assertEqual(st.jobs[JOB]["package"]["revision"], "a" * 40)

    def test_missing_receipt_stops_after_one_upload_without_retry(self):
        st, calls, out = self.run_package("--slugs", SLUG, upload_stdout="")
        self.assertEqual(len([c for c in calls if "upload" in c]), 1)
        self.assertEqual(st.jobs[JOB]["stage"], "needs_decision")
        self.assertNotIn("package", st.jobs[JOB])
        self.assertIn("write may have occurred; reconcile before retry", out)

    def test_wrong_receipt_binding_stops_after_one_upload_without_retry(self):
        repo = f"{regen.STAGING_ORG}/{SLUG}"
        for field, wrong in (("repo", "other/model"), ("parent_commit", "e" * 40)):
            with self.subTest(field=field):
                receipt = {"repo": repo, "parent_commit": "a" * 40, "commit": "b" * 40,
                           "url": f"https://huggingface.co/{repo}/commit/" + "b" * 40}
                receipt[field] = wrong
                st, calls, out = self.run_package("--slugs", SLUG, upload_stdout=json.dumps(receipt))
                self.assertEqual(len([c for c in calls if "upload" in c]), 1)
                self.assertEqual(st.jobs[JOB]["stage"], "needs_decision")
                self.assertNotIn("package", st.jobs[JOB])
                self.assertIn("returned commit receipt", out)

    def test_a_card_steps_gpu_preset_on_the_head_is_kept_and_compares_equal(self):
        st, calls, _ = self.run_package("--dry-run", "--slugs", SLUG)
        pkg = os.path.join(st.work, "package", SLUG, "pkg")
        m = json.load(open(os.path.join(pkg, "model.json")))
        first = next(iter(m["presets"]))
        m["presets"][first + "-r9700"] = dict(m["presets"][first], execution={**m["presets"][first].get("execution", {}), "context_capacity": 4096})
        json.dump(m, open(os.path.join(pkg, "model.json"), "w"), indent=1)   # the R9700 card step added its preset to the head
        shutil.rmtree(self.snap)
        shutil.copytree(pkg, self.snap)
        st, calls, _ = self.run_package("--dry-run", "--slugs", SLUG)
        out = json.load(open(os.path.join(st.work, "package", SLUG, "pkg", "model.json")))
        self.assertIn(first + "-r9700", out["presets"])   # carried over, not stripped by a re-upload

    def test_same_programs_but_another_sdk_pin_is_uploaded(self):
        # the head holds the canonical programs but an older llm_config.json pin (a package re-pinned from a pre-release): upload, not "already"
        shutil.rmtree(os.path.join(self.snap, "program"))
        shutil.copytree(os.path.join(self.canon, SLUG, "program"), os.path.join(self.snap, "program"))
        st, calls, _ = self.run_package("--slugs", SLUG)
        self.assertTrue([c for c in calls if "upload" in c])

    def test_default_selection_is_the_promoted_collected_jobs(self):
        st, calls, out = self.run_package("--dry-run")
        self.assertIn("nothing to package", out)   # the planned job is neither collected nor promoted
        self.assertFalse(calls)

    def test_model_json_with_an_unexpected_difference_stops_the_model(self):
        m = v02_model()
        m["presets"]["chat"]["execution"]["draft_depth"] = 5   # a serving setting that changed: not the migration
        json.dump(m, open(os.path.join(self.canon, SLUG, "model.json"), "w"))
        st, calls, out = self.run_package("--slugs", SLUG)
        self.assertIn("changed presets.chat.execution.draft_depth", out)
        self.assertFalse([c for c in calls if "upload" in c])
        self.assertEqual(st.jobs[JOB]["stage"], "needs_decision")
        self.assertIn("package", st.jobs[JOB]["reason"])
        self.assertFalse(st.jobs[JOB].get("package"))

    def prefill_transition(self, measured=True):
        for directory, value in ((self.snap, 1024), (os.path.join(self.canon, SLUG), 2048)):
            path = os.path.join(directory, "model.json")
            model = sdk_package.load(path)
            model["presets"]["chat"]["execution"]["prefill_capacity"] = value
            with open(path, "w") as f:
                json.dump(model, f)
        if measured:
            st = regen.State(self.state)
            d = os.path.join(self.tmp, "measured")
            os.makedirs(d)
            with open(os.path.join(d, "programs_sha256_staging.txt"), "w") as f:
                f.write(self.want + "\n")
            shutil.copyfile(os.path.join(self.canon, SLUG, "model.json"), os.path.join(d, "model_staging.json"))
            st.jobs[JOB]["dir"] = d
            st.save()
            return d

    def test_changed_prefill_is_still_refused_by_default(self):
        self.prefill_transition()
        st, calls, out = self.run_package("--slugs", SLUG)
        self.assertIn("changed presets.chat.execution.prefill_capacity", out)
        self.assertFalse([c for c in calls if "upload" in c])

    def test_exact_prefill_transition_requires_measured_settings_and_preserves_origins(self):
        self.prefill_transition()
        st, calls, _ = self.run_package("--slugs", SLUG, "--prefill-transition", f"{SLUG}:chat:1024:2048")
        self.assertEqual(len([c for c in calls if "upload" in c]), 1)
        pkg = os.path.join(st.work, "package", SLUG, "pkg")
        model = sdk_package.load(os.path.join(pkg, "model.json"))
        self.assertEqual(model["presets"]["chat"]["execution"]["prefill_capacity"], 2048)
        self.assertEqual(model["origins"], v01_model()["origins"])
        self.assertEqual(open(os.path.join(pkg, "model.json.lock")).read(), "model.json.lock")

    def test_prefill_transition_without_measurement_stops_before_snapshot(self):
        self.prefill_transition(measured=False)
        st, calls, out = self.run_package("--slugs", SLUG, "--prefill-transition", f"{SLUG}:chat:1024:2048")
        self.assertIn("requires matching measured programs", out)
        self.assertEqual(calls, [])

    def test_prefill_transition_with_other_measured_execution_stops_before_snapshot(self):
        d = self.prefill_transition()
        path = os.path.join(d, "model_staging.json")
        model = sdk_package.load(path)
        model["presets"]["chat"]["execution"]["context_capacity"] = 8192
        with open(path, "w") as f:
            json.dump(model, f)
        st, calls, out = self.run_package("--slugs", SLUG, "--prefill-transition", f"{SLUG}:chat:1024:2048")
        self.assertIn("did not measure the canonical chat execution", out)
        self.assertEqual(calls, [])

    def test_prefill_transition_wrong_old_value_does_not_authorize_upload(self):
        self.prefill_transition()
        st, calls, out = self.run_package("--slugs", SLUG, "--prefill-transition", f"{SLUG}:chat:512:2048")
        self.assertIn("expected 512->2048, found 1024->2048", out)
        self.assertFalse([c for c in calls if "upload" in c])

    def test_prefill_transition_does_not_allow_other_unmeasured_model_changes(self):
        d = self.prefill_transition()
        path = os.path.join(self.canon, SLUG, "model.json")
        model = sdk_package.load(path)
        model["presets"]["chat"]["execution"]["context_capacity"] = 8192
        with open(path, "w") as f:
            json.dump(model, f)
        shutil.copyfile(path, os.path.join(d, "model_staging.json"))
        st, calls, out = self.run_package("--slugs", SLUG, "--prefill-transition", f"{SLUG}:chat:1024:2048")
        self.assertIn("changed presets.chat.execution.context_capacity", out)
        self.assertFalse([c for c in calls if "upload" in c])

    def test_a_job_that_built_other_programs_stops_the_model(self):
        st = regen.State(self.state)
        d = os.path.join(self.tmp, "jobdir")
        os.makedirs(d)
        open(os.path.join(d, "programs_sha256_staging.txt"), "w").write("e" * 64 + "\n")
        st.jobs[JOB]["dir"] = d
        st.save()
        with mock.patch.object(regen, "package_one", wraps=regen.package_one) as package:
            st, calls, out = self.run_package("--slugs", SLUG)
        package.assert_not_called()   # the mismatch must stop before even snapshot/assembly, let alone upload
        self.assertEqual(calls, [])
        self.assertIn("built programs eeeeeeeeeeee", out)
        self.assertEqual(st.jobs[JOB]["stage"], "needs_decision")
        self.assertFalse(st.jobs[JOB].get("package"))

    def test_a_job_matching_the_canonical_programs_is_uploaded(self):
        st = regen.State(self.state)
        d = os.path.join(self.tmp, "jobdir")
        os.makedirs(d)
        open(os.path.join(d, "programs_sha256_staging.txt"), "w").write(self.want + "\n")
        st.jobs[JOB]["dir"] = d
        st.save()
        with mock.patch.object(regen, "package_one", wraps=regen.package_one) as package:
            st, calls, _ = self.run_package("--slugs", SLUG)
        package.assert_called_once()
        self.assertEqual(len([c for c in calls if "upload" in c]), 1)
        self.assertEqual(st.jobs[JOB]["package"]["programs_sha256"], self.want)

    def test_a_weight_file_without_an_origin_is_refused(self):
        m = v01_model()
        m["variants"]["gguf"]["services"]["chat"]["parameters"]["model"]["files"].append({"path": "parameter/new.gguf", "format": "gguf"})
        with self.assertRaises(ValueError):
            from_new = v02_model()
            from_new["variants"]["gguf"]["services"]["chat"]["parameters"]["model"]["files"].append({"path": "parameter/new.gguf", "format": "gguf"})
            sdk_package.carry_over(m, from_new)

    def test_only_sdk_ab_states_package(self):
        st = regen.State(self.state)
        st.d["mode"] = "ab"
        st.save()
        with self.assertRaises(SystemExit):
            regen.main(["package", self.state])


if __name__ == "__main__":
    unittest.main()
