"""XSJ publication checks the generated artifacts and qualified launch before contacting the Hub."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

CARD = Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts/card"
sys.path.insert(0, str(CARD))
import card_lib  # noqa: E402
import make_info  # noqa: E402
import publish_package  # noqa: E402
import render_card  # noqa: E402

GPU = "gfx1152"
SLUG = "unsloth-Qwen3-0.6B-loom"
PRESET = "chat-gfx1152"
RUN = "20261120-160817-s2j9"
REPO = f"loom-fun/{SLUG}"
PARENT = "a" * 40


class CardPublication(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.pkg = self.tmp / "package"
        shutil.copytree(CARD / "test/r9700/pkg", self.pkg)
        self.digest = card_lib.programs_sha256(self.pkg)
        results = self.tmp / "results"
        results.mkdir()
        self.results_file = results / f"{GPU}.json"
        source = {"kind": "xsj", "run": RUN, "url": None, "hrx_loom": {"hrx_loom": "0.2.0"},
                  "device": {"arch": GPU, "gpu": "AMD Radeon 860M", "host": "amd-krk-1",
                             "cpu": "AMD Ryzen AI 7 350", "os": "Windows 11", "driver": "32.0.31041.1004"}}
        self.gate = {"name": "consistency", "status": "pass", "level": "full", "time": "2026-11-20T16:19:00",
                     "detail": None, "source": source, "programs_sha256": self.digest}
        self.entry = {"fits": True, "date": "2026-11-20", "max_context": 32768,
                      "execution": {"rows": 4, "prefill_capacity": 512},
                      "serve_args": ["--pool_backing=fixed", "--parameter_mode=preload"],
                      "evidence": [{"evidence": card_lib.EVIDENCE_ID, "gpu": GPU, "gates": [self.gate]}]}
        self.write_results()
        self.enterContext(patch.dict(os.environ, {"LOOM_DEVICE_RESULTS": str(results)}))
        self.enterContext(patch.dict(sys.modules, {"huggingface_hub": None}))

    def write_results(self):
        card_lib.write_json(self.results_file, {"schema": card_lib.DEVICE_RESULTS_ID, "device": GPU,
                                               "gpu": "AMD Radeon 860M",
                                               "models": {SLUG: self.entry}})

    def build(self, *args):
        make_info.main(["build", str(self.pkg), "--repo", REPO, "--upstream", "none",
                        "--generator-commit", "0" * 40, "--report", "generated", *map(str, args)])
        self.render()

    def render(self):
        render_card.main([str(self.pkg)])

    def upload(self, *options):
        publish_package.main(["upload", str(self.pkg), "--repo", REPO, *options])

    def test_matching_configuration_reaches_mock_upload_and_uses_immutable_parent(self):
        remote = self.tmp / "parent"
        shutil.copytree(self.pkg, remote)
        self.build()
        self.upload("--card-only", "--dry-run")
        self.upload("--dry-run")
        api = Mock(endpoint="https://huggingface.co")
        api.model_info.return_value = SimpleNamespace(sha=PARENT, siblings=[])
        api.upload_folder.return_value = SimpleNamespace(oid="b" * 40,
                                                        commit_url=f"{api.endpoint}/{REPO}/commit/" + "b" * 40)
        hub = SimpleNamespace(HfApi=Mock(return_value=api), snapshot_download=Mock(return_value=remote),
                              hf_hub_download=Mock(return_value=remote / "model.json"))
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.upload("--card-only", "--parent-commit", PARENT)
        hub.snapshot_download.assert_called_once_with(REPO, revision=PARENT, allow_patterns=["program/*.loom"])
        hub.hf_hub_download.assert_called_once_with(REPO, "model.json", revision=PARENT)
        self.assertEqual(api.upload_folder.call_args.kwargs["parent_commit"], PARENT)
        self.assertEqual(api.upload_folder.call_args.kwargs["allow_patterns"], [*publish_package.CARD, "model.json"])

    def test_changed_preset_rejected_even_after_regeneration(self):
        self.build()
        path = self.pkg / "model.json"
        for key, value in (("rows", 5), ("context_capacity", 33792), ("max_tokens", 2048)):
            with self.subTest(key=key):
                original = path.read_bytes()
                model = card_lib.load_json(path)
                model["presets"][PRESET]["execution"][key] = value
                card_lib.write_json(path, model)
                self.assertEqual(card_lib.programs_sha256(self.pkg), self.digest)
                self.render()
                with self.assertRaisesRegex(SystemExit, f"{PRESET} differs from the qualified XSJ preset"):
                    self.upload("--card-only", "--dry-run")
                path.write_bytes(original)

    def test_changed_serve_flags_rejected_in_dry_run_and_real_upload(self):
        self.build()
        path = self.pkg / "info.json"
        info = card_lib.load_json(path)
        info["devices"][GPU]["serve_args"][0] = "--pool_backing=elastic"
        info["devices"][GPU]["quick_start"]["args"][0] = "--pool_backing=elastic"
        card_lib.write_json(path, info)
        self.render()
        for options in (("--card-only", "--dry-run"), ("--parent-commit", PARENT)):
            with self.subTest(options=options), self.assertRaisesRegex(SystemExit, "Quick start or serving settings differ"):
                self.upload(*options)

    def test_added_package_serve_flag_rejected(self):
        self.build()
        path = self.pkg / "llm_config.json"
        cfg = card_lib.load_json(path)
        cfg["serve"] = {"args": ["--pool_backing=elastic"]}
        card_lib.write_json(path, cfg)
        self.render()
        with self.assertRaisesRegex(SystemExit, "Quick start or serving settings differ"):
            self.upload("--dry-run")

    def test_both_generated_files_are_checked(self):
        self.build()
        for name in ("README.md", "REPORT.md"):
            with self.subTest(name=name):
                path = self.pkg / name
                original = path.read_bytes()
                path.write_bytes(original + b"Manual edit\n")
                with self.assertRaisesRegex(SystemExit, f"{name}: missing or stale"):
                    self.upload("--dry-run")
                path.unlink()
                with self.assertRaisesRegex(SystemExit, f"{name}: missing or stale"):
                    self.upload("--card-only", "--parent-commit", PARENT)
                path.write_bytes(original)

    def test_missing_or_different_run_fails_closed(self):
        self.build()
        with patch.dict(os.environ, {"LOOM_DEVICE_RESULTS": "none"}):
            with self.assertRaisesRegex(SystemExit, "requires qualified device results"):
                self.upload("--dry-run")
        self.gate["source"]["run"] = "20261120-170000-alt1"
        self.write_results()
        with self.assertRaisesRegex(SystemExit, "requires qualified device results"):
            self.upload("--dry-run")

    def test_historical_xsj_gate_does_not_require_current_results(self):
        self.build()
        path = self.pkg / "program/service.loom"
        path.write_bytes(path.read_bytes() + b"\n// new revision\n")
        self.build("--from-info", self.pkg / "info.json", "--device-results", "none")
        self.assertIn("a previous build", (self.pkg / "README.md").read_text(encoding="utf-8"))
        with patch.dict(os.environ, {"LOOM_DEVICE_RESULTS": "none"}):
            self.upload("--dry-run")

    def test_non_xsj_card_uses_existing_path(self):
        self.build("--device-results", "none", "--gpus", "strix-halo")
        (self.pkg / "REPORT.md").write_text("legacy bytes\n", encoding="utf-8")
        with patch.object(card_lib, "load_device_results", side_effect=AssertionError("per-device validation")):
            self.upload("--dry-run")


if __name__ == "__main__":
    unittest.main()
