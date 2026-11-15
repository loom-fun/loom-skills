"""The fast lane of a narrow change: replicated confirming runs in one submission (ab_jobs.py --replicas), the sequential 2+2 -> 6+6
extension inside a job (--sequential, ab_extend.py, devlab_perf.sh), waves and priority in regen.py submit, runner names in promote_check."""
import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, ".github", "scripts")
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "fleet"))
sys.path.insert(0, os.path.dirname(__file__))
import card_lib  # noqa: E402
import ab_jobs  # noqa: E402
import ab_extend  # noqa: E402
import promote_check  # noqa: E402
import regen  # noqa: E402
import test_regen  # noqa: E402

VL = "unsloth-Qwen3-VL-8B-Instruct-loom"   # the R2b model
FAST = "unsloth-Qwen3-0.6B-loom"


def gen(args, tps=60.0):
    old = card_lib.published_decode
    card_lib.published_decode = lambda slug, org="loom-fun": tps
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ab_jobs.main(args)
    finally:
        card_lib.published_decode = old
    return json.loads(buf.getvalue())


class Replicas(unittest.TestCase):
    def test_one_replica_is_the_unchanged_document(self):
        a = gen(["--device", "radeon-r9700", VL])
        self.assertEqual(a, gen(["--device", "radeon-r9700", "--replicas", "1", VL]))
        self.assertEqual(len(a["jobs"]), 1)
        self.assertNotIn("replica", a["jobs"][0])
        self.assertFalse(any(s.startswith("extend") for s in a["jobs"][0]["steps"]))

    def test_two_replicas_are_two_parallel_jobs_with_rotated_order(self):
        d = gen(["--device", "radeon-r9700", "--replicas", "2", VL])
        j1, j2 = d["jobs"]
        self.assertEqual(j2["name"], j1["name"] + "-r2")
        self.assertEqual((j2["replica"], j1["runner"], j2["runner"]), (2, "r9700", "r9700"))
        self.assertEqual(len({j["name"] for j in d["jobs"]}), 2)
        self.assertNotEqual(j1["steps"][:4], j2["steps"][:4])   # ABBA rotated
        self.assertEqual(sorted(j1["steps"]), sorted(j2["steps"]))   # the same measurements
        self.assertEqual({k: v for k, v in j1.items() if k not in ("name", "steps")}, {k: v for k, v in j2.items() if k not in ("name", "steps", "replica")})

    def test_replica_of_a_confirming_suffix_and_long_names(self):
        d = gen(["--device", "radeon-r9700", "--replicas", "2", VL])
        self.assertTrue(all(len(j["name"]) <= 40 for j in d["jobs"]))


class Sequential(unittest.TestCase):
    def test_start_2_plus_2_and_one_extend_step(self):
        j = gen(["--device", "radeon-r9700", "--sequential", VL])["jobs"][0]
        s = j["steps"]
        self.assertEqual(sum(x.startswith("perf:") and not x.endswith("long") for x in s), 4)   # 2 sessions per side
        self.assertEqual(sum(x.endswith("long") for x in s), 4)
        self.assertEqual([x for x in s if x.startswith("extend")], ["extend:6:2:0"])
        self.assertLess(s.index("extend:6:2:0"), s.index("textmatch:fun"))
        self.assertEqual(ab_jobs.card_lib.registry.methodology()["sequential"]["extend_to"], 6)

    def test_a_fast_model_keeps_its_six_sessions(self):
        j = gen(["--device", "radeon-r9700", "--sequential", FAST], tps=300.0)["jobs"][0]
        self.assertFalse(any(x.startswith("extend") for x in j["steps"]))
        self.assertEqual(sum(x.startswith("perf:") and not x.endswith("long") for x in j["steps"]), 12)

    def test_extension_adds_only_rows_near_a_threshold(self):
        # decode_32k reject 0.92; prefill_8k 0.95; decode 0.92
        near = ab_extend.groups_to_extend({"decode": 1.0, "decode_32k": 0.93, "prefill_8k": 1.0, "streams4": 0.5})
        self.assertEqual(near, {"long": ["decode_32k"]})
        self.assertEqual(ab_extend.groups_to_extend({"decode": 1.01, "prefill_8k": 0.99}), {})
        self.assertEqual(ab_extend.groups_to_extend({"decode": 0.90}), {"short": ["decode"]})   # just below the reject line is within 2 points too

    def test_extension_steps_continue_the_abba_order_to_6_plus_6(self):
        st = ab_extend.steps({"long": ["decode_32k"]}, to=6, have=2, rot=0)
        self.assertEqual(len(st), 8)   # 4 more sessions per side
        self.assertTrue(all(x.endswith(":1:long") for x in st))
        first4 = [f"perf:{v}:1:long" for v in ab_jobs.abba(2, start=1)]
        full = [f"perf:{v}:1:long" for v in ab_jobs.abba(6, start=1)]
        self.assertEqual(first4 + st, full)
        self.assertEqual(ab_extend.steps({}, 6, 2, 0), [])

    def test_devlab_perf_parses_and_has_the_extend_step(self):
        sh = os.path.join(SCRIPTS, "devlab_perf.sh")
        if sys.platform != "win32":   # the runner's bash is a Linux one
            subprocess.run(["bash", "-n", sh], check=True)
        with open(sh, encoding="utf-8") as f:
            self.assertIn("extend)", f.read())


class Ratchet(unittest.TestCase):
    def test_runners_of_the_pair_are_recorded(self):
        c = {"metrics": {"decode": {"ratio": 1.0, "old": 1, "new": 1, "old_sessions": [1, 1], "new_sessions": [1, 1]}}, "gate_status": {}, "text": {"status": "MATCH"}}
        r = promote_check.check("m", "radeon-r9700", [dict(c, runner="a"), dict(c, runner="b")], meth=None)
        self.assertEqual((r["runners"], r["same_runner"], r["decision"]), (["a", "b"], False, "promote"))
        r = promote_check.check("m", "radeon-r9700", [dict(c, runner="a"), dict(c, runner="a")], meth=None)
        self.assertTrue(r["same_runner"])


class RegenReplicasAndWaves(test_regen.Base):
    def gen(self, cmd, **kw):
        self.calls.append([str(c) for c in cmd])
        if any(str(c).endswith("ab_jobs.py") for c in cmd):
            doc = test_regen.fake_generator()
            j = dict(doc["jobs"][0])
            doc["jobs"] = [j]
            if "--replicas" in cmd:
                doc["jobs"].append(dict(j, name=j["name"] + "-r2", replica=2))
            return subprocess.CompletedProcess(cmd, 0, json.dumps(doc), "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def test_default_plan_is_unchanged(self):
        st = self.plan()
        self.assertEqual(len(st.jobs), 1)
        self.assertNotIn("--replicas", self.calls[0])
        self.assertNotIn("--sequential", self.calls[0])
        self.assertNotIn("replicas", st.d["opts"])
        self.assertNotIn("priority", st.d["batches"][0])

    def test_replicas_plan_links_the_pair_in_one_batch(self):
        st = self.plan("--replicas", "2", "--sequential", "--no-baselines", "--priority")
        self.assertIn("--replicas", self.calls[0])
        self.assertIn("--sequential", self.calls[0])
        self.assertNotIn("--baselines", self.calls[0])
        j1, j2 = test_regen.JOB, test_regen.JOB + "-r2"
        self.assertEqual(len(st.d["batches"]), 1)
        self.assertEqual(st.d["batches"][0]["jobs"], [j1, j2])
        self.assertTrue(st.d["batches"][0]["priority"])
        self.assertEqual((st.jobs[j2]["prev"], st.jobs[j2]["round"], st.jobs[j1]["round"]), (j1, 2, 1))
        self.assertEqual(st.jobs[j1]["rerun_of_me"], j2)   # the first run is superseded by the pair's decision: no further rerun is queued

    def test_replica_waits_for_its_first_run_to_be_collected(self):
        st = self.plan("--replicas", "2")
        j1, j2 = test_regen.JOB, test_regen.JOB + "-r2"
        st.jobs[j2].update(stage="done", run="1")
        st.save()
        with mock.patch.object(regen, "download", side_effect=AssertionError("collected before its first run")), contextlib.redirect_stdout(io.StringIO()):
            regen.main(["collect", self.state])
        self.assertEqual(regen.State(self.state).jobs[j2]["stage"], "done")

    def submit(self, *args):
        out = io.StringIO()
        with mock.patch.object(regen, "submit_batch", lambda st, b, dry: (b.update(stage="submitted"), [st.jobs[n].update(stage="submitted") for n in b["jobs"]], st.save())), \
                contextlib.redirect_stdout(out):
            regen.main(["submit", self.state, *args])
        return regen.State(self.state), out.getvalue()

    def two_batches(self, *extra):
        self.plan("--batch", "1", *extra, slugs=(test_regen.SLUG, "unsloth-other-loom"))

    def test_without_a_cap_every_batch_goes_out(self):
        self.two_batches()
        st, _ = self.submit()
        self.assertEqual([b["stage"] for b in st.d["batches"]], ["submitted"] * 2)

    def test_cap_submits_waves(self):
        self.two_batches()
        st, out = self.submit("--max-inflight", "1")
        self.assertEqual([b["stage"] for b in st.d["batches"]], ["submitted", "planned"])
        self.assertIn("waits for the next wave", out)
        for e in st.jobs.values():
            e["stage"] = "done"   # the wave finished
        st.save()
        st, _ = self.submit("--max-inflight", "1")
        self.assertEqual([b["stage"] for b in st.d["batches"]], ["submitted", "submitted"])

    def test_priority_batches_go_first_and_ignore_the_cap(self):
        self.two_batches()
        st = regen.State(self.state)
        st.d["batches"][1]["priority"] = True
        st.save()
        st, _ = self.submit("--max-inflight", "1", "--max-batches", "1")
        self.assertEqual([b["stage"] for b in st.d["batches"]], ["planned", "submitted"])

    def test_max_inflight_auto_is_the_device_runner_count(self):
        self.assertEqual(regen.max_inflight("auto", "radeon-r9700"), 3)
        self.assertEqual(regen.max_inflight("auto", "strix-halo"), 0)
        self.assertEqual(regen.max_inflight(None, "radeon-r9700"), 0)
        self.assertEqual(regen.max_inflight("4", "radeon-r9700"), 4)


if __name__ == "__main__":
    unittest.main()
