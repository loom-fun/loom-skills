"""Fail-loud offline tools: build_fleet / diff_fleet refuse empty or partial builds, scope_lint --local sees the working tree,
preflight names every missing package at once."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fleet"))
sys.path.insert(0, str(ROOT / "tests"))
import build_fleet  # noqa: E402
import diff_fleet  # noqa: E402
import preflight  # noqa: E402
import scope_lint as SL  # noqa: E402
from test_scope_lint import Repo  # noqa: E402


def write(d, rel, text="x\n"):
    p = Path(d, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


class Complete(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="trialgaps_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def good(self, slug):
        write(self.d, f"{slug}/model.json", "{}")
        write(self.d, f"{slug}/program/a.loom", "kernel.def @k {}\n")

    def test_has_output(self):
        self.good("ok")
        write(self.d, "failed/build.log", "FAILED: x\n")
        self.assertTrue(build_fleet.has_output(os.path.join(self.d, "ok")))
        self.assertFalse(build_fleet.has_output(os.path.join(self.d, "failed")))

    def test_check_complete_flags_missing_failed_and_summary_failures(self):
        self.good("ok")
        write(self.d, "failed/build.log", "FAILED: x\n")
        bad = diff_fleet.check_complete(self.d, ["ok", "failed", "absent"])
        self.assertEqual(sorted(b.split(":")[0] for b in bad), ["absent", "failed"])
        write(self.d, "summary.json", json.dumps({"failures": {"ok": "boom"}}))
        self.assertTrue(any(b.startswith("ok: the build failed") for b in diff_fleet.check_complete(self.d, ["ok"])))

    def run_diff(self, base, head, *extra):
        return subprocess.run([sys.executable, str(ROOT / "fleet" / "diff_fleet.py"), base, head, *extra], capture_output=True, text=True)

    def test_diff_refuses_empty_dirs_but_not_a_complete_subset(self):
        a, b = os.path.join(self.d, "a"), os.path.join(self.d, "b")
        os.makedirs(a)
        os.makedirs(b)
        r = self.run_diff(a, b)   # the false pass of the trial: two empty builds
        self.assertEqual(r.returncode, 1)
        self.assertIn("refusing to compare incomplete builds", r.stderr)
        for side in (a, b):
            write(side, "m/model.json", "{}")
            write(side, "m/llm_config.json", "{}")
            write(side, "m/program/p.loom", "kernel.def @k {}\n")
        self.assertEqual(self.run_diff(a, b).returncode, 1)               # still partial against the 51-model manifest
        r = self.run_diff(a, b, "--allow-partial")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("identical on all models", r.stdout)


class LocalHead(unittest.TestCase):
    def test_clean_tree_is_head_dirty_tree_includes_edits_and_new_files(self):
        r = Repo()
        try:
            self.assertEqual(SL.local_head(r.d), "HEAD")
            r.write("README.md", "changed\n")
            r.write("fleet/new_file.txt", "new\n")
            sha = SL.local_head(r.d)
            self.assertNotEqual(sha, "HEAD")
            _, files = SL.changed_files(r.d, "head", sha)
            self.assertEqual(sorted(p for _, p in files), ["README.md", "fleet/new_file.txt"])
            self.assertEqual(subprocess.run(["git", "-C", r.d, "status", "--porcelain"], capture_output=True, text=True).stdout.count("\n"), 2)   # tree untouched
        finally:
            r.close()


class Preflight(unittest.TestCase):
    def test_every_missing_package_in_one_error(self):
        req = {"json": "json", "no_such_mod_a": "pip-a", "no_such_mod_b": "pip-b"}
        self.assertEqual(preflight.missing(req), ["pip-a", "pip-b"])
        with self.assertRaises(SystemExit) as c:
            preflight.require(req)
        msg = str(c.exception)
        self.assertIn("pip-a, pip-b", msg)
        self.assertIn("pip install json pip-a pip-b", msg)
        preflight.require({"json": "json"})

    def test_required_set(self):
        self.assertEqual(sorted(preflight.REQUIRED.values()), ["gguf", "jinja2", "numpy", "pyyaml"])


if __name__ == "__main__":
    unittest.main()
