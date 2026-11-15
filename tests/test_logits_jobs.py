"""logits_jobs.py: ab_jobs.py's A/B job with the new side built from a ref and the logits/gufo steps added; devlab_perf.sh knows the steps."""
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, ".github", "scripts")
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
sys.path.insert(0, SCRIPTS)
import card_lib  # noqa: E402
import logits_jobs  # noqa: E402

SLUG = "unsloth-Qwen3-0.6B-loom"


def jobs(*args):
    old, card_lib.published_decode = card_lib.published_decode, lambda slug, org="loom-fun": 300.0
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            logits_jobs.main(list(args))
    finally:
        card_lib.published_decode = old
    return json.loads(buf.getvalue())


class LogitsJobs(unittest.TestCase):
    def test_strix(self):
        j = jobs("--device", "strix-halo", "--ref", "origin/x", "--bundle", "https://e/x.tgz", "ab" * 32, SLUG)["jobs"][0]
        v = {x["name"]: x for x in j["variants"]}
        self.assertEqual((v["staging"]["ref"], v["stagingnd"]["ref"], "ref" in v["fun"], "build" in v["fun"]), ("origin/x", "origin/x", False, False))
        self.assertEqual(j["steps"][-2:], ["logits:staging", "gufo:staging"])
        self.assertIn("perf:fun:1", j["steps"])
        self.assertEqual(j["bundle"], {"url": "https://e/x.tgz", "sha256": "ab" * 32})

    def test_bundle_is_built_in_the_run(self):
        j = jobs("--device", "strix-halo", "--ref", "origin/x", "--bundle-build", "f822f2b", "b1e06141", SLUG)["jobs"][0]
        self.assertEqual(j["bundle"], {"build": {"extension": "loom-fun/loom-gufo-extension@f822f2b", "gufo": "loom-fun/gufo@b1e06141"}})

    def test_workflow_builds_and_hands_over_the_bundle(self):
        import re
        wf = open(os.path.join(ROOT, ".github", "workflows", "devlab-perf.yml")).read()
        self.assertIn("build_bundle.sh", wf)
        self.assertIn("name: gufo-bundle", wf)
        self.assertIn("GUFO_BUNDLE=", wf)
        self.assertNotRegex(wf, r"huggingface\.co.*upload|gh release")

    def test_r9700_rebuilds_the_device_preset(self):
        j = jobs("--device", "radeon-r9700", "--ref", "origin/x", "--no-gufo", "--no-ab", SLUG)["jobs"][0]
        v = {x["name"]: x for x in j["variants"]}
        self.assertEqual(v["staging"]["build_args"], ["--prefill-device", "radeon-r9700=4096"])
        self.assertEqual([s.split(":")[0] for s in j["steps"]], ["func", "ktest", "logits"])

    def test_script_has_the_steps(self):
        sh = open(os.path.join(SCRIPTS, "devlab_perf.sh")).read()
        self.assertIn("    logits)", sh)
        self.assertIn("    gufo)", sh)
        self.assertEqual(subprocess.run(["bash", "-n", os.path.join(SCRIPTS, "devlab_perf.sh")]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
