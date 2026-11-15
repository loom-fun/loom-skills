"""fleet/regen.py: plan / submit / poll / collect / rerun / ingest / publish / status on fixtures (no network, no GPU).

The fixture job (tests/data/regen/job_e2b) is a downloaded R9700 job of this campaign (an A/A of one staging revision); the job list
(perf_r9700_aa.json) is what ab_jobs.py wrote for it."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "fleet"))
import regen  # noqa: E402

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
                contextlib.redirect_stdout(io.StringIO()):
            regen.main(["plan", "--device", device, "--state", self.state, *args, *slugs])
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
        with mock.patch.object(regen, "download", download), contextlib.redirect_stdout(io.StringIO()):
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
            regen.main(["plan", "--device", "radeon-r9700", "--state", self.state, SLUG])
        st = regen.State(self.state)
        self.assertEqual(next(iter(st.jobs.values()))["stage"], "skipped")

    def test_card_mode_names_a_job_per_device(self):
        with mock.patch.object(regen, "generator_cmd", lambda st, d, s, suffix="": ["x", "ab_jobs.py"] + s):
            st = self.plan("--mode", "card")
        self.assertTrue(any(n.startswith("rc-") for n in st.jobs), list(st.jobs))

    def test_only_staging_is_a_target(self):
        with self.assertRaises(SystemExit), mock.patch.object(regen, "sh", self.gen):
            regen.main(["plan", "--device", "radeon-r9700", "--state", self.state, "--new-org", "loom-fun", SLUG])


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
        self.assertEqual([(m, p) for m, p, _ in seen], [("GET", "git/ref/heads/refactor"), ("POST", "git/refs"), ("GET", "contents/.github/devlab/perf.json"),
                                                         ("PUT", "contents/.github/devlab/perf.json"), ("POST", "pulls"), ("POST", "issues/77/labels")])
        self.assertEqual(seen[4][2]["base"], "refactor")
        self.assertIn("kind: tooling", seen[4][2]["body"])
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
        with mock.patch.object(regen, "download", download), contextlib.redirect_stdout(io.StringIO()):
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
        self.assertEqual(st.d["batches"][-1]["round"], 2)
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


if __name__ == "__main__":
    unittest.main()
