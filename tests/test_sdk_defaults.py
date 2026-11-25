"""Every emitter and default targets the newest verified SDK release (sdk_pin.VERIFIED_RELEASES), not a literal version."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "loom-setup" / "scripts"))
import sdk_pin  # noqa: E402


class NewestRelease(unittest.TestCase):
    def test_newest_is_numeric_not_lexical(self):
        saved = dict(sdk_pin.VERIFIED_RELEASES)
        try:
            for v in ("0.3.10", "0.3.9"):
                sdk_pin.VERIFIED_RELEASES[v] = saved["0.3.0"]
            self.assertEqual(sdk_pin.newest_verified(), "0.3.10")
        finally:
            sdk_pin.VERIFIED_RELEASES.clear()
            sdk_pin.VERIFIED_RELEASES.update(saved)

    def test_release_pin_matches_the_verified_wheel(self):
        v = sdk_pin.newest_verified()
        pin = sdk_pin.release_pin()
        rel = sdk_pin.VERIFIED_RELEASES[v]
        self.assertEqual(pin, {"hrx_loom": v, "source": "pypi", "hrx_system_sha": rel["hrx_system_sha"],
                               "wheel_sha256": rel["wheels"]["manylinux_2_39_x86_64"]})


class FreshStarter(unittest.TestCase):
    def test_a_fresh_starter_kit_emits_the_newest_pin(self):
        """apply_example.sh copies the starter out of the repository: the copy must still emit the newest verified pin and sha256."""
        with tempfile.TemporaryDirectory() as d:
            kit = os.path.join(d, "kit")
            subprocess.run(["bash", str(ROOT / "skills" / "loom-port-llm" / "reference" / "apply_example.sh"), "starter", kit],
                           check=True, capture_output=True)
            out = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import kit_source, json; "
                                  "print(json.dumps(kit_source.SDK_PIN))", kit], check=True, capture_output=True, text=True).stdout
            self.assertEqual(json.loads(out), sdk_pin.release_pin())
            self.assertEqual(json.loads(out)["wheel_sha256"], sdk_pin.VERIFIED_RELEASES[sdk_pin.newest_verified()]["wheels"]["manylinux_2_39_x86_64"])
            with open(os.path.join(kit, "KIT_SOURCE"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["sdk_pin"], sdk_pin.release_pin())

    def test_build_stamps_the_kit_pin(self):
        t = (ROOT / "skills" / "loom-port-llm" / "starter" / "build.py").read_text()
        self.assertIn('bench["sdk"] = dict(SDK_PIN)', t)


class Defaults(unittest.TestCase):
    def test_kbench_job_pins_the_newest(self):
        doc = json.loads((ROOT / ".github" / "devlab" / "moe_q4_0-kbench.json").read_text())
        self.assertEqual(doc["sdk"], sdk_pin.newest_verified())

    def test_installer_default_is_the_newest(self):
        t = (ROOT / "skills" / "loom-setup" / "scripts" / "install_loom.sh").read_text()
        self.assertIn(f"which=${{2:-{sdk_pin.newest_verified()}}}", t)

    def test_ab_jobs_new_sdk_default(self):
        sys.path.insert(0, str(ROOT / ".github" / "scripts"))
        import ab_jobs
        self.assertEqual(ab_jobs.parser().get_default("new_sdk"), sdk_pin.newest_verified())


if __name__ == "__main__":
    unittest.main()
