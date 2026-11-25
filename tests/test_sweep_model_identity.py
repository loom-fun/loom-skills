"""Fit/sweep rebuilds must resolve the selected package's model-scoped choices."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PORT = Path(__file__).resolve().parents[1] / "skills/loom-port-llm"
sys.path.insert(0, str(PORT))
import sweep_compile as sc


class SweepIdentity(unittest.TestCase):
    def test_selected_identity_reaches_build_even_with_other_extra_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "input"
            pkg.mkdir()
            (pkg / "llm_config.json").write_text(json.dumps({
                "source": {"kind": "safetensors", "repo": "a/b", "revision": "r"},
                "model_id": "stale-cfg-id",
            }))
            (pkg / "model.json").write_text('{"model": {"id": "selected-id"}}')
            calls = []

            def run(cmd, cwd, env=None, **kw):
                calls.append(cmd)
                if cmd[1].endswith("inspect_safetensors.py"):
                    Path(cmd[3]).write_text('{"arch": "gemma4"}')
                return 0, ""

            with patch.object(sc, "origin_repos", return_value={}), patch.object(sc, "sh", run), \
                    patch.object(sc.shutil, "copytree", side_effect=lambda a, b: Path(b).mkdir()):
                ok, error, _ = sc.build("selected-model-loom", str(pkg), str(root / "work"),
                                        ("--prefill", "2048", "--slug", "unrelated-model", "--model-id", "unrelated-id"))
            self.assertTrue(ok, error)
            build = calls[-1]
            self.assertEqual(build[-4:], ["--slug", "selected-model-loom", "--model-id", "selected-id"])
            self.assertIn("2048", build)


class SweepCopiedKit(unittest.TestCase):
    def test_a_plain_starter_copy_records_its_sdk_pin(self):
        """A package without a kit example is rebuilt from a copy of the starter: the copy has no loom-setup beside it, so
        kit_source.py write must record the SDK pin in KIT_SOURCE (as apply_example.sh and devlab_perf.sh do)."""
        real_sh = sc.sh
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "input"
            pkg.mkdir()
            (pkg / "llm_config.json").write_text(json.dumps({"source": {"kind": "safetensors", "repo": "a/b", "revision": "r"}}))

            def run(cmd, cwd, env=None, **kw):
                if cmd[1].endswith("kit_source.py"):
                    return real_sh(cmd, cwd, env)
                if cmd[1].endswith("inspect_safetensors.py"):
                    Path(cmd[3]).write_text('{"arch": "gemma4"}')
                return 0, ""

            with patch.object(sc, "origin_repos", return_value={}), patch.object(sc, "sh", run):
                ok, error, _ = sc.build("m-loom", str(pkg), str(root / "work"))
            self.assertTrue(ok, error)
            src = json.loads((root / "work" / "kit" / "KIT_SOURCE").read_text())
            self.assertTrue(src.get("sdk_pin"))
