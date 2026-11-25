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
PSPEC = importlib.util.spec_from_file_location("safe_publish", CARD / "publish_package.py")
PUBLISH = importlib.util.module_from_spec(PSPEC)
PSPEC.loader.exec_module(PUBLISH)
SLUG = "example-loom"
SOURCE = "loom-fun-staging/" + SLUG
PIN, HEAD, TARGET, RETURNED = "a" * 40, "b" * 40, "c" * 40, "d" * 40


class PromotionPinsTests(unittest.TestCase):
    card_problems = []

    def flow(self, directory, pin=None, execute=False, target_extras=False):
        api = Mock()
        api.endpoint = "https://huggingface.co"
        api.model_info.side_effect = [SimpleNamespace(sha=pin or HEAD, siblings=[]),
                                      SimpleNamespace(sha=TARGET, siblings=[SimpleNamespace(rfilename="independent.txt", lfs=None, blob_id=HEAD)] if target_extras else []),
                                      SimpleNamespace(sha=TARGET)]

        def download(repo, revision, local_dir, ignore_patterns):
            self.assertEqual(repo, SOURCE)
            p = Path(local_dir)
            p.mkdir(parents=True)
            (p / "info.json").write_text(json.dumps({"repo": SOURCE}))
            (p / "README.md").write_text("A generated card.\n")
            (p / "REPORT.md").write_text("A generated report.\n")

        fetch = Mock(side_effect=download)
        def command(cmd, **kwargs):
            receipt = {"repo": "loom-fun/" + SLUG, "parent_commit": TARGET, "commit": RETURNED,
                       "url": "https://huggingface.co/loom-fun/" + SLUG + "/commit/" + RETURNED}
            stdout = json.dumps(receipt) if "upload" in cmd and "--dry-run" not in cmd else ""
            return subprocess.CompletedProcess(cmd, 0, stdout, "")
        run = Mock(side_effect=command)
        # The fake module ensures these tests never import a live Hub client or perform network I/O.
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=fetch)}), \
                patch.object(PROMOTE.subprocess, "run", run), \
                patch.object(PROMOTE, "card_file_problems", lambda directory: self.card_problems):
            result = PROMOTE.promote(api, SLUG, directory, execute, pin, TARGET if execute else None)
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
        self.assertEqual(api.model_info.call_args_list[1].kwargs, {"files_metadata": True, "revision": TARGET})
        self.assertEqual(result["receipt"]["commit"], RETURNED)
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

    def test_target_map_and_live_target_movement_refuse_before_composition(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "targets.json"
            for value in ({}, {"other": TARGET}, {SLUG: "main"}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    PROMOTE.revision_map(path, [SLUG], "--target-revisions")
        api = Mock()
        api.model_info.side_effect = [SimpleNamespace(sha=PIN, siblings=[]),
                                      SimpleNamespace(sha=TARGET, siblings=[]), SimpleNamespace(sha=HEAD)]
        fetch = Mock()
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=fetch)}):
            with self.assertRaisesRegex(ValueError, "target moved"):
                PROMOTE.promote(api, SLUG, "unused", True, PIN, TARGET)
        fetch.assert_not_called()
        with self.assertRaisesRegex(ValueError, "explicitly reviewed target"):
            PROMOTE.promote(api, SLUG, "unused", True, PIN)

    def test_missing_upload_receipt_is_failure_not_success(self):
        # flow normally supplies a receipt; replace JSON decoding only for that returned upload response.
        decoder = PROMOTE.json.loads
        with patch.object(PROMOTE.json, "loads", side_effect=lambda s, *a, **k: {} if '"parent_commit"' in s else decoder(s, *a, **k)):
            with tempfile.TemporaryDirectory() as td:
                result, _, _, _ = self.flow(td, PIN, execute=True)
        self.assertFalse(result["ok"])
        self.assertIn("invalid/missing", result["executed"])

    def test_execute_cli_requires_target_map_before_hub_client(self):
        client = Mock()
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(HfApi=client)}), \
                patch.dict(PROMOTE.os.environ, {"PROMOTE_CONFIRM": "loom-fun"}), \
                patch.object(sys, "argv", ["promote_staging.py", "--execute", "--work", "unused", SLUG]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                PROMOTE.main()
        self.assertEqual(error.exception.code, 2)
        client.assert_not_called()

    def test_promotion_remote_extras_refuse_execute_without_deletion(self):
        with tempfile.TemporaryDirectory() as td:
            result, _, _, run = self.flow(td, PIN, execute=True, target_extras=True)
        self.assertFalse(result["ok"])
        self.assertEqual(result["files"]["removed"], ["independent.txt"])
        self.assertEqual(result["executed"], "skipped: checks failed")
        self.assertFalse(any("upload" in c.args[0] and "--dry-run" not in c.args[0] for c in run.call_args_list))

    def test_card_file_problems_block_upload(self):
        self.card_problems = ["sdk-qualification.json is not a card file"]
        with tempfile.TemporaryDirectory() as td:
            result, _, _, run = self.flow(str(Path(td) / "work"), PIN, execute=True)
        self.assertFalse(result["ok"])
        self.assertEqual(result["executed"], "skipped: checks failed")
        self.assertFalse(any("upload" in c.args[0] and "--dry-run" not in c.args[0] for c in run.call_args_list))

    def test_missing_check_is_a_problem(self):
        with patch.object(PROMOTE, "HERE", "/nonexistent"):
            self.assertIn("missing", PROMOTE.card_file_problems("/tmp")[0])


class UploadSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("README.md", "REPORT.md", "info.json", "model.json", "model.json.lock", "llm_config.json"):
            (self.root / name).write_text("{}")
        self.args = SimpleNamespace(out=str(self.root), repo=SLUG, org="loom-fun", dry_run=False, card_only=False,
                                    allow_dirty=False, parent_commit=TARGET, init_new_repo=False, message=None)
        self.api = Mock()
        self.api.endpoint = "https://huggingface.co"
        self.api.model_info.return_value = SimpleNamespace(sha=TARGET, siblings=[])
        self.api.upload_folder.return_value = SimpleNamespace(oid=RETURNED, commit_url="https://huggingface.co/loom-fun/" + SLUG + "/commit/" + RETURNED)
        self.client = Mock(return_value=self.api)
        class NotFound(Exception):
            pass
        self.NotFound = NotFound
        self.hub = {"huggingface_hub": SimpleNamespace(HfApi=self.client),
                    "huggingface_hub.errors": SimpleNamespace(RepositoryNotFoundError=NotFound)}

    def upload(self):
        output = io.StringIO()
        with patch.dict(sys.modules, self.hub), patch.object(PUBLISH, "check_card_text", return_value=None), \
                patch.object(PUBLISH, "refuse_dirty"), patch.object(PUBLISH, "card_model", return_value=[]), \
                contextlib.redirect_stdout(output):
            PUBLISH.cmd_upload(self.args)
        return output.getvalue()

    def test_existing_requires_parent_and_target_movement_never_mutates(self):
        self.args.parent_commit = None
        with self.assertRaises(SystemExit):
            self.upload()
        self.client.assert_not_called()
        self.args.parent_commit = TARGET
        self.api.model_info.return_value.sha = HEAD
        with self.assertRaises(SystemExit):
            self.upload()
        self.api.create_repo.assert_not_called()
        self.api.upload_folder.assert_not_called()

    def test_cas_conflict_has_no_second_try_or_unguarded_fallback(self):
        self.api.upload_folder.side_effect = RuntimeError("CAS conflict")
        with self.assertRaisesRegex(RuntimeError, "CAS conflict"):
            self.upload()
        self.assertEqual(self.api.upload_folder.call_count, 1)
        self.assertEqual(self.api.upload_folder.call_args.kwargs["parent_commit"], TARGET)
        self.api.create_repo.assert_not_called()

    def test_bad_returned_receipt_fails_without_retry(self):
        for receipt in (SimpleNamespace(oid=None, commit_url=None),
                        SimpleNamespace(oid=RETURNED, commit_url="https://huggingface.co/wrong/repo/commit/" + RETURNED)):
            self.api.upload_folder.return_value = receipt
            self.api.upload_folder.reset_mock()
            with self.assertRaisesRegex(SystemExit, "write may have occurred"):
                self.upload()
            self.assertEqual(self.api.upload_folder.call_count, 1)

    def test_exact_returned_receipt_survives_later_head_movement(self):
        self.api.model_info.side_effect = [SimpleNamespace(sha=TARGET, siblings=[]), SimpleNamespace(sha=HEAD)]
        receipt = json.loads(self.upload())
        self.assertEqual(receipt, {"repo": "loom-fun/" + SLUG, "parent_commit": TARGET, "commit": RETURNED,
                                   "url": "https://huggingface.co/loom-fun/" + SLUG + "/commit/" + RETURNED})
        self.assertEqual(self.api.model_info("loom-fun/" + SLUG).sha, HEAD)
        self.assertEqual(receipt["commit"], RETURNED)

    def test_no_remote_extra_or_stray_is_deleted(self):
        self.api.model_info.return_value.siblings = [SimpleNamespace(rfilename=x) for x in ("independent.txt", ".loom-fetch.lock", "ev_old.json")]
        with self.assertRaisesRegex(SystemExit, "deletion is not supported"):
            self.upload()
        self.api.upload_folder.assert_not_called()
        self.args.card_only = True
        self.upload()
        kwargs = self.api.upload_folder.call_args.kwargs
        self.assertNotIn("delete_patterns", kwargs)
        self.assertEqual(kwargs["allow_patterns"], list(PUBLISH.CARD))

    def test_dry_run_stays_offline_without_parent(self):
        self.args.parent_commit = None
        self.args.dry_run = True
        self.assertIn("dry run", self.upload())
        self.client.assert_not_called()

    def test_explicit_new_repo_uses_verified_seed_parent_cas(self):
        self.args.parent_commit = None
        self.args.init_new_repo = True
        self.api.model_info.side_effect = [self.NotFound(), SimpleNamespace(sha=TARGET, siblings=[SimpleNamespace(rfilename=".gitattributes")])]
        self.api.list_repo_commits.return_value = [SimpleNamespace(commit_id=TARGET)]
        receipt = json.loads(self.upload())
        self.api.create_repo.assert_called_once_with("loom-fun/" + SLUG, repo_type="model", exist_ok=False)
        self.assertEqual(receipt["parent_commit"], TARGET)
        self.assertEqual(self.api.upload_folder.call_args.kwargs["parent_commit"], TARGET)

    def test_existing_or_unseeded_or_independently_modified_new_repo_refuses(self):
        self.args.parent_commit = None
        self.args.init_new_repo = True
        with self.assertRaisesRegex(SystemExit, "already exists"):
            self.upload()
        self.api.create_repo.assert_not_called()
        for seed in (SimpleNamespace(sha=None, siblings=[]),
                     SimpleNamespace(sha=TARGET, siblings=[SimpleNamespace(rfilename="README.md")])):
            self.api.model_info.side_effect = [self.NotFound(), seed]
            with self.assertRaisesRegex(SystemExit, "no verifiable seed"):
                self.upload()
        self.api.model_info.side_effect = [self.NotFound(), SimpleNamespace(sha=TARGET, siblings=[])]
        self.api.list_repo_commits.return_value = [SimpleNamespace(commit_id=TARGET), SimpleNamespace(commit_id=HEAD)]
        with self.assertRaisesRegex(SystemExit, "beyond its initial seed"):
            self.upload()
        self.api.upload_folder.assert_not_called()

    def test_new_repo_create_conflict_never_falls_back_to_existing(self):
        self.args.parent_commit = None
        self.args.init_new_repo = True
        self.api.model_info.side_effect = [self.NotFound()]
        self.api.create_repo.side_effect = RuntimeError("409 concurrent creation")
        with self.assertRaisesRegex(RuntimeError, "409"):
            self.upload()
        self.api.create_repo.assert_called_once()
        self.api.upload_folder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
