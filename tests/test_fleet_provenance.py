"""Canonical package builds retain authentic provenance without changing comparison programs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fleet"))
import build_fleet as fleet  # noqa: E402


class FleetProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def git(self, repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    def repo(self, name):
        repo = self.root / name
        repo.mkdir()
        self.git(repo, "init", "-q")
        return repo

    def commit(self, repo):
        self.git(repo, "add", ".")
        self.git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture", "--no-gpg-sign")
        return self.git(repo, "rev-parse", "HEAD")

    def test_plain_kit_records_source_repo_not_output_ancestor_and_reads_sdk_environment(self):
        source = self.repo("source-A")
        skills = source / "skills/loom-port-llm"
        starter = skills / "starter"
        starter.mkdir(parents=True)
        shutil.copy(ROOT / "skills/loom-port-llm/starter/kit_source.py", starter)
        wanted = self.commit(source)
        outer = self.repo("output-B")
        (outer / "skills/loom-port-llm").mkdir(parents=True)
        (outer / "fixture").write_text("another repository")
        other = self.commit(outer)
        self.assertNotEqual(wanted, other)
        work = outer / "build"
        work.mkdir()
        venv = self.root / "sdk"
        dist = venv / "lib/python3.12/site-packages/hrx_loom-0.2.0.dist-info"
        dist.mkdir(parents=True)
        record = b"fixture SDK distribution record\n"
        (dist / "RECORD").write_bytes(record)
        with mock.patch.dict(os.environ, VENV=str(venv)):
            os.environ.pop("LOOM_ROOT", None)
            kit = fleet.kit_dir(str(skills), None, str(work))
            stamp = json.loads(subprocess.check_output([sys.executable, str(Path(kit) / "kit_source.py"), "stamp", kit], text=True))
        self.assertEqual(stamp["skills_commit"], wanted)
        self.assertFalse(stamp["dirty"])
        self.assertFalse(stamp["kit"]["modified"])
        self.assertEqual(stamp["hrx_loom"], {"hrx_loom": "0.2.0", "fingerprint": hashlib.sha256(record).hexdigest()[:16], "verified": False})
        with mock.patch.dict(os.environ):
            os.environ.pop("VENV", None)
            os.environ.pop("LOOM_ROOT", None)
            unknown = json.loads(subprocess.check_output([sys.executable, str(Path(kit) / "kit_source.py"), "stamp", kit], text=True))
        self.assertEqual(unknown["skills_commit"], wanted)
        self.assertEqual(unknown["hrx_loom"], "unknown")   # no SDK identity is invented from the source pin

    def test_plain_legacy_kit_without_provenance_tool_still_copies(self):
        skills = self.root / "legacy"
        (skills / "starter").mkdir(parents=True)
        (skills / "starter/build.py").write_text("legacy")
        work = self.root / "work"
        work.mkdir()
        kit = Path(fleet.kit_dir(str(skills), None, str(work)))
        self.assertEqual((kit / "build.py").read_text(), "legacy")
        self.assertFalse((kit / "KIT_SOURCE").exists())

    def test_package_build_retains_stamp_while_default_comparison_strips_it(self):
        inputs = self.root / "inputs"
        model = inputs / "model"
        model.mkdir(parents=True)
        (model / "llm_config.json").write_text('{"gguf":"model.gguf"}')
        (model / "tokenizer_min.json").write_text("{}")
        stamp = {"skills_commit": "a" * 40, "dirty": False, "hrx_loom": {"hrx_loom": "0.2.0"}}
        sdk = {"hrx_loom": "0.2.0", "wheel_sha256": "b" * 64}

        def kit_dir(skills, example, work):
            kit = Path(work) / "kit"
            kit.mkdir()
            (kit / "build.py").write_text('"--slug"')
            return str(kit)

        def sh(cmd, cwd, env=None):
            work = Path(cwd)
            if cmd[1].endswith("inspect_gguf.py"):
                (work / "fresh.json").write_text("{}")
            elif cmd[1].endswith("build.py"):
                pkg = work / "pkg"
                (pkg / "program").mkdir(parents=True)
                (pkg / "program/model.loom").write_text("// identical program bytes\n")
                (pkg / "gen").mkdir()
                (pkg / "gen/helper.loom").write_text("// helper\n")
                (pkg / "model.json").write_text("{}")
                (pkg / "llm_config.json").write_text(json.dumps({"generator": stamp, "sdk": sdk}))
            else:
                self.fail(str(cmd))
            return 0, ""

        outputs = []
        with mock.patch.object(fleet, "HERE", str(inputs)), mock.patch.object(fleet, "kit_dir", kit_dir), mock.patch.object(fleet, "sh", sh):
            for preserve in (False, True):
                out = self.root / str(preserve)
                _, error, _ = fleet.build_one("model", {"files": {}, "build": {}, "model_id": "Model"}, "skills", str(out), "1", False, with_gen=preserve)
                self.assertIsNone(error)
                cfg = json.loads((out / "model/llm_config.json").read_text())
                self.assertEqual(cfg.get("generator"), stamp if preserve else None)
                self.assertEqual(cfg["sdk"], sdk)
                self.assertEqual((out / "model/gen").exists(), preserve)
                outputs.append((out / "model/program/model.loom").read_bytes())
        self.assertEqual(*outputs)


if __name__ == "__main__":
    unittest.main()
