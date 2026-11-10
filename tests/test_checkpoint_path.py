import importlib.util
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("checkpoint_path", REPO / "skills/loom-port-llm/checkpoint_path.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class CheckpointPathTests(unittest.TestCase):
    def test_native_python_cli_path_is_usable_by_bash(self):
        bash = shutil.which("bash")
        if os.name == "nt":
            git = shutil.which("git")
            candidate = Path(git).parent.parent / "bin/bash.exe" if git else None
            bash = str(candidate) if candidate and candidate.is_file() else None
        if not bash:
            self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory(prefix="checkpoint shell spaces ") as temp:
            source = Path(temp) / "model.gguf"
            source.write_bytes(b"weights")
            log = Path(temp) / "download.log"
            target = Path(temp) / "work/model.gguf"
            env = dict(os.environ, TEST_PYTHON=Path(sys.executable).as_posix(), TEST_HELPER=(REPO / "skills/loom-port-llm/checkpoint_path.py").as_posix(), TEST_LOG=log.as_posix(), TEST_TARGET=target.as_posix())
            script = 'set -eu; cached=$("$TEST_PYTHON" "$TEST_HELPER" from-log "$TEST_LOG" model.gguf); test -f "$cached"; "$TEST_PYTHON" "$TEST_HELPER" link "$cached" "$TEST_TARGET"'
            for output in (str(source), "path=" + str(source)):
                with self.subTest(output=output):
                    log.write_text("Warning: unauthenticated\n" + output + "\nhf exit 0\n", encoding="utf-8")
                    result = subprocess.run([bash, "-c", script], env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_checkpoint_fallbacks_without_symlink_privileges(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "blob"
            source.write_bytes(b"checkpoint")
            target = Path(temp) / "work/model.gguf"
            with patch.object(Path, "symlink_to", side_effect=PermissionError("no Developer Mode")):
                helper.link_checkpoint(source, target)
            self.assertTrue(target.samefile(source))
            helper.link_checkpoint(source, target)  # Repeated port is a no-op.
            copy = Path(temp) / "copy/model.gguf"
            with patch.object(Path, "symlink_to", side_effect=PermissionError()), patch.object(helper.os, "link", side_effect=OSError("cross-volume")):
                helper.link_checkpoint(source, copy)
            self.assertEqual(copy.read_bytes(), source.read_bytes())
            self.assertFalse(copy.samefile(source))

    def test_link_keeps_the_hf_snapshot_path(self):
        # publish_package.py reads repo and revision from snapshots/REV/FILE: resolving to blobs/ loses them
        if os.name == "nt":
            self.skipTest("symlinks")
        with tempfile.TemporaryDirectory() as temp:
            snap = Path(temp) / "models--o--r/snapshots" / ("a" * 40)
            blob = Path(temp) / "models--o--r/blobs/sha"
            snap.mkdir(parents=True); blob.parent.mkdir(parents=True); blob.write_bytes(b"w")
            (snap / "m.gguf").symlink_to(blob)
            target = Path(temp) / "work/m.gguf"
            helper.link_checkpoint(snap / "m.gguf", target)
            self.assertEqual(os.readlink(target), str(snap / "m.gguf"))
            self.assertEqual(target.read_bytes(), b"w")


if __name__ == "__main__":
    unittest.main()
