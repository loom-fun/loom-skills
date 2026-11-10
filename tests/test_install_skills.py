"""CPU acceptance checks for the shared suite and native discovery adapters."""
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_skills", REPO / "scripts/install_skills.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class ManagedInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="loom portability ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        for name in ("loom", "loom-setup"):
            skill = self.source / name
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test skill\n---\n", encoding="utf-8")
        self.dest = self.base / "installed skills"

    def install(self, uninstall=False):
        return installer.install([self.dest], uninstall, self.source)

    def test_update_and_uninstall_preserve_unrelated_skills(self):
        self.install()
        unrelated = self.dest / "other-skill.txt"
        unrelated.write_text("keep", encoding="utf-8")
        cache = self.dest / "loom/__pycache__"
        cache.mkdir()
        (cache / "helper.pyc").write_bytes(b"cache")
        (self.source / "loom/new.txt").write_text("updated", encoding="utf-8")
        self.install()
        self.assertEqual((self.dest / "loom/new.txt").read_text(), "updated")
        self.install(uninstall=True)
        self.assertEqual(list(self.dest.iterdir()), [unrelated])

    def test_unmanaged_collision_is_preflighted_across_all_roots(self):
        collision = self.dest / "loom-setup"
        collision.mkdir(parents=True)
        sentinel = collision / "my-notes.txt"
        sentinel.write_text("keep", encoding="utf-8")
        empty_root = self.base / "other-agent"
        with self.assertRaisesRegex(ValueError, "Unmanaged"):
            installer.install([empty_root, self.dest], source=self.source)
        self.assertFalse(empty_root.exists())
        self.assertEqual(sentinel.read_text(), "keep")

    def test_modified_or_added_files_block_update_and_removal(self):
        for modification in ("SKILL.md", "notes.txt"):
            with self.subTest(modification=modification):
                self.dest = self.base / modification.replace(".", "-")
                self.install()
                changed = self.dest / "loom" / modification
                changed.write_text("local changes", encoding="utf-8")
                for uninstall in (False, True):
                    with self.assertRaisesRegex(ValueError, "Locally modified"):
                        self.install(uninstall)
                self.assertEqual(changed.read_text(), "local changes")

    def test_source_overlap_is_rejected(self):
        for root in (self.source, self.source.parent, self.source / "nested"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "overlaps"):
                installer.install([root], source=self.source)

    def test_symlinks_are_not_followed_or_removed(self):
        self.dest.mkdir()
        target = self.dest / "loom"
        try:
            target.symlink_to(self.source / "loom", target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation is unavailable on this Windows host")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.install()
        self.assertTrue((self.source / "loom/SKILL.md").exists())

    def test_failed_replacement_restores_previous_skill(self):
        self.install()
        original = (self.dest / "loom/SKILL.md").read_bytes()
        rename = Path.rename

        def fail_new(path, target):
            if path.name == "new":
                raise OSError("simulated rename failure")
            return rename(path, target)

        with patch.object(Path, "rename", fail_new), self.assertRaises(OSError):
            self.install()
        self.assertEqual((self.dest / "loom/SKILL.md").read_bytes(), original)


class SharedSuiteTests(unittest.TestCase):
    def test_complete_real_suite_from_unrelated_working_directory(self):
        # Exercise each native profile without touching the user's installed skills.
        with tempfile.TemporaryDirectory(prefix="loom path with spaces ") as temp:
            home = Path(temp).resolve()
            unrelated = home / "unrelated cwd"
            unrelated.mkdir()
            with patch.dict(os.environ, {}, clear=True):
                roots = installer.destinations("all", home)
                self.assertEqual(roots, [home / ".claude/skills", home / ".agents/skills"])
                for agent, directory in (("claude", ".claude"), ("codex", ".agents"), ("pi", ".agents")):
                    self.assertEqual(installer.destinations(agent, home), [home / directory / "skills"])
                expected = sorted(p.parent.name for p in (REPO / "skills").glob("*/SKILL.md"))
                self.assertIn("loom-org", expected)
                self.assertEqual(installer.install(roots), len(expected))
                for root in roots:
                    self.assertEqual(sorted(p.name for p in root.iterdir()), expected)
                    for name in expected:
                        skill = root / name
                        text = (skill / "SKILL.md").read_text(encoding="utf-8")
                        self.assertIn(f"name: {name}\n", text)
                        # Verify all direct resource references against the installed bundle.
                        for suffix in re.findall(r'\$\{LOOM_SKILL_DIR\}(/[\w./-]+)', text):
                            self.assertTrue((skill / suffix.lstrip("/")).exists(), (root, name, suffix))
                    result = subprocess.run(
                        [sys.executable, "-B", str(root / "loom-improve/scripts/kpi.py"), "--help"],
                        cwd=unrelated, capture_output=True, text=True, check=True,
                    )
                    self.assertIn("summarize", result.stdout)
                # Imports/reads must not leave files that make an install look modified.
                installer.install(roots, uninstall=True)
                self.assertTrue(all(not list(root.iterdir()) for root in roots))


if __name__ == "__main__":
    unittest.main()
