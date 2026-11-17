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
    old, card_lib.published_decode = card_lib.published_decode, lambda slug, org="loom-fun", gpu="strix-halo": 300.0
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

    def test_bundle_of_an_earlier_run_is_reused(self):
        j = jobs("--device", "strix-halo", "--ref", "origin/x", "--bundle-run", "43700000000", SLUG)["jobs"][0]
        self.assertEqual(j["bundle"], {"run": "43700000000"})

    def test_workflow_downloads_the_artifact_of_that_run(self):
        wf = open(os.path.join(ROOT, ".github", "workflows", "devlab-perf.yml")).read()
        for needle in ("bundle_run: ${{ steps.pick.outputs.bundle_run }}", "run-id: ${{ needs.jobs.outputs.bundle_run }}",
                       "if: needs.bundle.result == 'success' || needs.jobs.outputs.bundle_run != ''", "retention-days: 30"):
            self.assertIn(needle, wf)

    def test_bundle_pins_default(self):
        j = jobs("--device", "strix-halo", "--ref", "origin/x", SLUG)["jobs"][0]
        pins = json.load(open(os.path.join(ROOT, ".github", "devlab", "gufo_bundle.json")))
        self.assertEqual(j["bundle"], {"build": {"extension": pins["extension"], "gufo": pins["gufo"]}})
        import re
        self.assertTrue(all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", pins[k]) for k in ("extension", "gufo")), "full commit ids: actions/checkout needs them")
        self.assertNotIn("bundle", jobs("--device", "strix-halo", "--ref", "origin/x", "--no-logits", SLUG)["jobs"][0])

    def test_replicas(self):
        d = jobs("--device", "strix-halo", "--ref", "origin/x", "--replicas", "2", SLUG)["jobs"]
        self.assertEqual([j["name"] for j in d], ["sab-unsloth-qwen3-0-6b-loom", "sab-unsloth-qwen3-0-6b-loom-r2"])
        self.assertTrue(all(any(v.get("ref") == "origin/x" for v in j["variants"]) and j["steps"][-1] == "gufo:staging" for j in d))

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

    @unittest.skipIf(os.name == "nt", "bash script")
    def test_pkg_preset_names_the_preset_a_gpu_serves(self):
        """A card with several presets must be opened with one named (embed_probe, `gufo serve --loom-preset`): Strix its first, the R9700 `<first>-r9700`."""
        import re, tempfile
        sh = open(os.path.join(SCRIPTS, "devlab_perf.sh")).read()
        fn = re.search(r"^pkg_preset\(\) \{.*?^\}$", sh, re.M | re.S).group(0)
        with tempfile.TemporaryDirectory() as d:
            for name, presets in (("two", ["chat", "chat-r9700"]), ("one", ["chat"]), ("first_only", ["chat", "other"])):
                json.dump({"presets": {p: {} for p in presets}}, open(os.path.join(d, name + ".json"), "w"))

            def run(model, rdev):
                r = subprocess.run(["bash", "-c", f'T={sys.executable}; RDEV="{rdev}"; RSUF=-r9700; {fn}\npkg_preset {os.path.join(d, model)}.json'],
                                   capture_output=True, text=True)
                return r.stdout.strip()
            self.assertEqual((run("two", ""), run("two", "radeon-r9700")), ("chat", "chat-r9700"))
            self.assertEqual((run("one", ""), run("one", "radeon-r9700")), ("", ""))
            self.assertEqual(run("first_only", "radeon-r9700"), "chat")
        self.assertIn('--preset "$(pkg_preset', sh)
        self.assertIn("EXTRA_GUFO_ARGS=", sh)

    def test_script_has_the_steps(self):
        sh = open(os.path.join(SCRIPTS, "devlab_perf.sh")).read()
        self.assertIn("    logits)", sh)
        self.assertIn("    gufo)", sh)
        if os.name != "nt":   # CRLF checkouts / a different bash on Windows runners
            self.assertEqual(subprocess.run(["bash", "-n", os.path.join(SCRIPTS, "devlab_perf.sh")]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
