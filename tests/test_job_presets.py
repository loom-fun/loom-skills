"""job_presets.py + ab_jobs.py / device_jobs.py --preset: iterate (no gates, smoke for func), final (func + ktest + gates once, new variant only),
full (the default, unchanged); the R9700 runner count comes from device.json."""
import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, ".github", "scripts")
sys.path.insert(0, SCRIPTS)
import job_presets  # noqa: E402
from test_ab_jobs import run_script  # noqa: E402

SLUG = "amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom"
R9 = "radeon-r9700"


def gen(script, *args):
    # Preset tests use the existing synthetic-rate helper, never live metadata.
    return json.loads(run_script(script, ["--device", R9, *args]))["jobs"][0]


class Presets(unittest.TestCase):
    FULL = ["func:staging", "ktest:staging", "gates:staging:dev", "perf:fun:1", "perf:staging:1", "c4:stagingnd", "textmatch:fun",
            "llama@rocm.np1:3", "llamac4@rocm.nodraft", "vllm:3", "vllmc4:3"]

    def test_full_is_identity(self):
        self.assertEqual(job_presets.apply("full", self.FULL, "staging"), self.FULL)

    def test_iterate_has_no_gates_no_func_no_baselines(self):
        s = job_presets.apply("iterate", self.FULL, "staging")
        self.assertEqual(s, ["smoke:staging", "ktest:staging", "perf:fun:1", "perf:staging:1", "c4:stagingnd", "textmatch:fun"])

    def test_final_is_func_ktest_gates_once(self):
        self.assertEqual(job_presets.apply("final", self.FULL, "staging", "dev"), ["func:staging", "ktest:staging", "gates:staging:dev"])
        self.assertEqual(job_presets.apply("final", self.FULL, "card", ""), ["func:card", "ktest:card", "gates:card:dev"])

    def test_unknown_preset(self):
        with self.assertRaises(ValueError):
            job_presets.apply("fast", [], "x")

    def test_generators(self):
        it = gen("ab_jobs.py", "--preset", "iterate", "--old-org", "loom-fun-staging", SLUG)
        self.assertFalse([s for s in it["steps"] if s.startswith(("gates", "func"))])
        self.assertIn("smoke:staging", it["steps"])
        fin = gen("ab_jobs.py", "--preset", "final", "--old-org", "loom-fun-staging", SLUG)
        self.assertEqual(fin["steps"], ["func:staging", "ktest:staging", "gates:staging:dev"])
        self.assertEqual([v["name"] for v in fin["variants"]], ["staging"])   # the unchanged base is not built, let alone gated
        full = gen("ab_jobs.py", "--old-org", "loom-fun-staging", "--gates", "dev", SLUG)
        self.assertIn("func:staging", full["steps"])
        self.assertIn("gates:staging:dev", full["steps"])
        c_it = gen("device_jobs.py", "--preset", "iterate", SLUG)
        self.assertEqual([s.split(":")[0] for s in c_it["steps"]][:2], ["smoke", "ktest"])
        self.assertFalse([s for s in c_it["steps"] if s.startswith(("gates", "func", "vllm", "llama"))])
        c_fin = gen("device_jobs.py", "--preset", "final", SLUG)
        self.assertEqual(c_fin["steps"], ["func:card", "ktest:card", "gates:card:dev"])
        self.assertEqual([v["name"] for v in c_fin["variants"]], ["card"])

    def test_runner_count_is_the_measured_one(self):
        d = json.load(open(os.path.join(ROOT, "skills", "loom-kb", "device", "rdna4", "radeon-r9700", "device.json")))
        self.assertEqual(d["runner"]["concurrent_runners"], 8)
        self.assertNotIn("about 3 concurrent runners", open(os.path.join(ROOT, "skills", "loom-fleet", "reference", "campaign.md")).read())


if __name__ == "__main__":
    unittest.main()
