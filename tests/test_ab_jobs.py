"""ab_jobs.py (device-generic A/B jobs): the Strix wrapper's perf.json is byte-identical to the former strix_ab_jobs.py, the R9700 jobs
serve the new card exactly as r9700_jobs.py does, card references pin revisions, devlab_perf.sh parses."""
import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, ".github", "scripts")
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
sys.path.insert(0, SCRIPTS)
import card_lib  # noqa: E402
import ab_jobs  # noqa: E402

FAST, SLOW = "unsloth-Qwen3-0.6B-loom", "ggml-org-Qwen3-4B-loom"


def run_script(name, args):
    """stdout of .github/scripts/NAME ARGS with the published decode rates stubbed (no network): the 0.6B is fast, the rest are not."""
    old = (card_lib.published_decode, sys.argv)
    card_lib.published_decode = lambda slug, org="loom-fun": 300.0 if "0.6B" in slug else 60.0
    sys.argv = [name] + args
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(SCRIPTS, name), run_name="__main__")
    finally:
        card_lib.published_decode, sys.argv = old
    return buf.getvalue()


GOLD = [(["unsloth-Qwen3-0.6B-loom", "ggml-org-Qwen3-4B-loom"]), (["--baselines", "--suffix=-r2", "--reps", "4", "--llama-reps", "3", "--gates", "fast", SLOW]),
        (["--old-org", "loom-fun-x", "--new-org", "y", FAST])]


class StrixWrapper(unittest.TestCase):
    def test_byte_identical_to_the_former_script(self):
        for i, args in enumerate(GOLD, 1):
            gold = open(os.path.join(ROOT, "tests", "data", "ab_jobs", f"strix_{i}.json")).read()
            self.assertEqual(run_script("strix_ab_jobs.py", args), gold, args)
            self.assertEqual(run_script("ab_jobs.py", ["--device", "strix-halo"] + args), gold, args)


class R9700(unittest.TestCase):
    def jobs(self, *args):
        return json.loads(run_script("ab_jobs.py", ["--device", "radeon-r9700", "--fit", os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"), *args]))

    def test_new_variants_are_r9700_jobs_card_and_nodraft(self):
        fit = os.path.join(ROOT, ".github", "devlab", "r9700_fit.json")
        for slug in (FAST, SLOW, "InternScience-Sage-A1.5-loom", "bartowski-Ornith-1.5-35B-A3B-loom"):
            ref = json.loads(run_script("r9700_jobs.py", ["--fit", fit, slug]))["jobs"][0]
            ab = self.jobs(slug)["jobs"][0]
            v = {x["name"]: x for x in ab["variants"]}
            self.assertEqual(v["staging"], {**ref["variants"][0], "name": "staging"}, slug)
            self.assertEqual(v["stagingnd"], {**ref["variants"][1], "name": "stagingnd"}, slug)
            for k in ("require_preset_r9700", "memory_bytes_r9700", "served_context", "llama_kv_rocm", "card", "kit", "llamacpp", "llamacpp_vulkan"):
                self.assertEqual(ab.get(k), ref.get(k), (slug, k))

    def test_old_variants_get_the_same_settings(self):
        j = self.jobs("--old-rev", "abc", "--new-rev", "def", "--baselines", FAST)["jobs"][0]
        v = {x["name"]: x for x in j["variants"]}
        self.assertEqual(j["card"], f"loom-fun-staging/{FAST}@def")
        self.assertEqual(v["fun"]["card"], f"loom-fun/{FAST}@abc")
        self.assertEqual(v["fun"]["execution"], {"prefill_capacity": 4096})
        self.assertEqual(v["funnd"]["execution"], {"prefill_capacity": 4096, "draft_depth": 0})
        self.assertGreater(v["fun"]["cap_pool"], 0)
        self.assertNotIn("require_preset_r9700", v["fun"])
        self.assertIn("llama@vulkan.np1:6", j["steps"])   # fast model: 6 sessions, llama.cpp as many runs
        self.assertEqual(j["steps"].count("perf:fun:1"), 6)

    def test_no_fit_no_job(self):
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "ab_jobs.py"), "--device", "radeon-r9700", "bartowski-Ornith-1.5-122B-loom"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)

    def test_refs_map_wins(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({SLOW: {"old": f"loom-fun-staging/{SLOW}@aaa", "new": f"loom-fun-staging/{SLOW}"}}, f)
            f.flush()
            j = self.jobs("--refs", f.name, SLOW)["jobs"][0]
        self.assertEqual(j["card"], f"loom-fun-staging/{SLOW}")
        self.assertEqual(j["variants"][0]["card"], f"loom-fun-staging/{SLOW}@aaa")


class DevlabPerf(unittest.TestCase):
    def test_syntax_and_card_pin(self):
        p = os.path.join(SCRIPTS, "devlab_perf.sh")
        self.assertEqual(subprocess.run(["bash", "-n", p]).returncode, 0)
        s = open(p).read()
        for needle in ("card_revision.txt", "card_revision_$v.txt", "want=${ref#*@}", "cap_pool"):
            self.assertIn(needle, s)


if __name__ == "__main__":
    unittest.main()
