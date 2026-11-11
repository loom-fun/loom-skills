import json
import sys
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/loom-setup/scripts"
PIN = {"hrx_loom": "0.2.0.dev0", "source": "ci", "artifact_id": 13628412621, "wheel_sha256": "ab" * 32}


def run(*args, **env):
    return subprocess.run(["python3", str(SCRIPTS / "sdk_pin.py"), *args], capture_output=True, text=True,
                          env={**os.environ, **env})


class SdkPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.cfg = Path(self.tmp) / "llm_config.json"
        self.root = Path(self.tmp) / "loom"
        self.root.mkdir()

    def install(self, **have):
        (self.root / "loom-install.json").write_text(json.dumps(have))

    def test_show(self):
        self.cfg.write_text(json.dumps({"sdk": PIN}))
        self.assertEqual(run("show", str(self.cfg)).stdout.strip(), "ci:13628412621")
        self.cfg.write_text(json.dumps({"sdk": {"hrx_loom": "0.1.1", "source": "pypi"}}))
        self.assertEqual(run("show", str(self.cfg)).stdout.strip(), "0.1.1")
        self.cfg.write_text("{}")
        self.assertEqual(run("show", str(self.cfg)).stdout.strip(), "")

    def test_matches_by_wheel_hash(self):
        self.cfg.write_text(json.dumps({"sdk": PIN}))
        self.assertNotEqual(run("matches", str(self.root), str(self.cfg)).returncode, 0)   # nothing installed
        self.install(hrx_loom="0.1.0", source="pypi", wheel_sha256="cd" * 32)
        self.assertNotEqual(run("matches", str(self.root), str(self.cfg)).returncode, 0)
        self.install(hrx_loom="0.2.0.dev0", source="ci", artifact_id=13628412621, wheel_sha256="ab" * 32)
        self.assertEqual(run("matches", str(self.root), str(self.cfg)).returncode, 0)

    def test_no_pin_never_matches(self):
        self.cfg.write_text("{}")
        self.install(hrx_loom="0.1.0", source="pypi")
        self.assertNotEqual(run("matches", str(self.root), str(self.cfg)).returncode, 0)

    def test_stamp_refreshes_the_pin(self):
        self.cfg.write_text(json.dumps({"model_id": "m", "sdk": {"hrx_loom": "old", "source": "pypi"}}))
        self.install(hrx_loom="0.2.0.dev1", source="ci", artifact_id=None, hrx_system_sha="f" * 40, wheel_sha256="ef" * 32)
        self.assertEqual(run("stamp", str(self.root), str(self.cfg)).returncode, 0)
        c = json.loads(self.cfg.read_text())
        self.assertEqual(c["model_id"], "m")
        self.assertEqual(c["sdk"]["hrx_loom"], "0.2.0.dev1")
        self.assertNotIn("artifact_id", c["sdk"])


@unittest.skipUnless(shutil.which("bash") and sys.platform != "win32", "bash on a POSIX host (fake venv layout, POSIX paths)")
class EnsureSdkTests(unittest.TestCase):
    def test_pinned_root_is_separate_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "llm_config.json"
            cfg.write_text(json.dumps({"sdk": PIN}))
            # a pinned install already in place (fake venvs): ensure_sdk.sh finds it by identity and installs nothing
            pinned = Path(tmp) / "loom-13628412621"
            for p in (".venv/bin/loom-serve", ".tools/bin/python"):
                (pinned / p).parent.mkdir(parents=True, exist_ok=True)
                (pinned / p).write_text("")
                (pinned / p).chmod(0o755)
            (pinned / "loom-install.json").write_text(json.dumps({"hrx_loom": "0.2.0.dev0", "wheel_sha256": "ab" * 32,
                                                                   "source": "ci", "artifact_id": 13628412621}))
            env = {k: v for k, v in os.environ.items() if k != "LOOM_VERSION"}
            r = subprocess.run(["bash", str(SCRIPTS / "ensure_sdk.sh"), str(Path(tmp) / "loom"), str(cfg)],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), str(pinned))


if __name__ == "__main__":
    unittest.main()
