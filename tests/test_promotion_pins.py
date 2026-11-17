"""Immutable source selection without Hub requests or publication."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "skills/loom-benchmark/scripts/card"
sys.path.insert(0, str(CARD))
SPEC = importlib.util.spec_from_file_location("promotion_pins", CARD / "promote_staging.py")
PROMOTE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROMOTE)
SLUG = "example-loom"
SOURCE = "loom-fun-staging/" + SLUG
PIN, HEAD, TARGET = "a" * 40, "b" * 40, "c" * 40


class PromotionPinsTests(unittest.TestCase):
    def flow(self, directory, pin=None, execute=False):
        api = Mock()
        api.model_info.side_effect = [SimpleNamespace(sha=pin or HEAD, siblings=[]),
                                      SimpleNamespace(sha=TARGET, siblings=[])]

        def download(repo, revision, local_dir, ignore_patterns):
            self.assertEqual(repo, SOURCE)
            p = Path(local_dir)
            p.mkdir(parents=True)
            (p / "info.json").write_text(json.dumps({"repo": SOURCE}))
            (p / "README.md").write_text("A generated card.\n")
            (p / "REPORT.md").write_text("A generated report.\n")

        fetch = Mock(side_effect=download)
        run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
        # The fake module ensures these tests never import a live Hub client or perform network I/O.
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=fetch)}), \
                patch.object(PROMOTE.subprocess, "run", run):
            result = PROMOTE.promote(api, SLUG, directory, execute, pin)
        return result, api, fetch, run

    def test_pin_reaches_metadata_snapshot_and_preserves_destination_guard(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "pins.json"
            fleet = {f"model-{i}": PIN for i in range(52)}
            manifest.write_text(json.dumps(fleet))
            self.assertEqual(PROMOTE.staging_revisions(manifest, list(fleet)), fleet)
            manifest.write_text(json.dumps({SLUG: PIN}))
            pins = PROMOTE.staging_revisions(manifest, [SLUG])
            result, api, fetch, run = self.flow(str(Path(td) / "work"), pins[SLUG], execute=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["requested_staging_revision"], PIN)
        self.assertEqual(result["staging_commit"], PIN)
        self.assertEqual(result["target_commit"], TARGET)
        self.assertEqual(api.model_info.call_args_list[0].kwargs, {"files_metadata": True, "revision": PIN})
        self.assertEqual(api.model_info.call_args_list[1].kwargs, {"files_metadata": True})
        self.assertEqual(fetch.call_args.kwargs["revision"], PIN)
        commands = [call.args[0] for call in run.call_args_list]
        upload = commands[-1]
        self.assertEqual(upload[-2:], ["--parent-commit", TARGET])
        self.assertIn("--dry-run", commands[-2])

    def test_bad_manifests_and_resolved_revision_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pins.json"
            for value in ([], {}, {"other": PIN}, {SLUG: PIN, "extra": HEAD},
                          {SLUG: None}, {SLUG: "main"}, {SLUG: "a" * 39}, {SLUG: "z" * 40}):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        PROMOTE.staging_revisions(path, [SLUG])
            path.write_text("{broken")
            with self.assertRaises(ValueError):
                PROMOTE.staging_revisions(path, [SLUG])
            client = Mock()
            with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(HfApi=client)}), \
                    patch.object(sys, "argv", ["promote_staging.py", "--work", td,
                                              "--staging-revisions", str(path), SLUG]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    PROMOTE.main()
            self.assertEqual(error.exception.code, 2)
            client.assert_not_called()
        api = Mock()
        api.model_info.return_value = SimpleNamespace(sha=HEAD, siblings=[])
        fetch = Mock()
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=fetch)}):
            with self.assertRaisesRegex(ValueError, "resolved.*refusing promotion"):
                PROMOTE.promote(api, SLUG, "unused-work-directory", False, PIN)
        fetch.assert_not_called()
        self.assertEqual(api.model_info.call_count, 1)

    def test_default_still_uses_live_source_head_without_revision_override(self):
        self.assertEqual(PROMOTE.staging_revisions(None, [SLUG]), {})
        with tempfile.TemporaryDirectory() as td:
            result, api, fetch, run = self.flow(td)
        self.assertTrue(result["ok"])
        self.assertNotIn("requested_staging_revision", result)
        self.assertEqual(result["staging_commit"], HEAD)
        self.assertEqual(api.model_info.call_args_list[0].kwargs, {"files_metadata": True})
        self.assertEqual(fetch.call_args.kwargs["revision"], HEAD)
        self.assertIn("--dry-run", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
