"""Verify explicit needle exclusion never executes a needle or fabricates a pass."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
GATES = (REPO / "skills/loom-benchmark/scripts/gates.sh").read_text(encoding="utf-8")


class GateScopeTests(unittest.TestCase):
    def test_real_gate_applicability_detects_tools_and_fails_on_bad_config(self):
        bash = shutil.which("bash")
        if os.name == "nt":
            git = shutil.which("git")
            candidate = Path(git).parent.parent / "bin/bash.exe" if git else None
            bash = str(candidate) if candidate and candidate.is_file() else None
        if not bash:
            self.skipTest("Bash unavailable")
        prefix = GATES.split("t_start=$(date +%s)", 1)[0]
        with tempfile.TemporaryDirectory(prefix="gate native paths ") as temp:
            base = Path(temp)
            scripts = base / "scripts"
            (scripts / "card").mkdir(parents=True)
            shutil.copyfile(REPO / "skills/loom-benchmark/scripts/_lib.sh", scripts / "_lib.sh")
            shutil.copyfile(REPO / "skills/loom-benchmark/scripts/card/card_lib.py", scripts / "card/card_lib.py")
            harness = scripts / "gates.sh"
            harness.write_text(prefix + '\nprintf "tools=%s" "$tools"', encoding="utf-8")
            package = base / "package"
            (package / "program").mkdir(parents=True)
            (package / "program/test.loom").write_text("test", encoding="utf-8")
            (package / "model.json").write_text(json.dumps({"presets": {"default": {"execution": {"context_capacity": 4096}}}}), encoding="utf-8")
            python = base / "sdk/.tools/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text('#!/bin/bash\nexec "$TEST_PYTHON" "$@"\n', encoding="utf-8")
            python.chmod(0o755)
            gguf = base / "model.gguf"
            gguf.write_bytes(b"unused")
            env = dict(os.environ, LOOM_ROOT=(base / "sdk").as_posix(), NO_LLAMA="1", TEST_PYTHON=Path(sys.executable).as_posix())
            config = package / "llm_config.json"
            for chat, expected in (({"call_format": "hermes"}, "tools=1"), ({"call_format": "none"}, "tools=")):
                config.write_text(json.dumps({"model_id": "test", "chat": chat}), encoding="utf-8")
                result = subprocess.run([bash, harness.as_posix(), package.as_posix(), "http://127.0.0.1:1", gguf.as_posix(), (base / "bench").as_posix()], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.endswith(expected), result.stdout)
            config.write_text("not JSON", encoding="utf-8")
            result = subprocess.run([bash, harness.as_posix(), package.as_posix(), "http://127.0.0.1:1", gguf.as_posix(), (base / "bench").as_posix()], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("tool calling unsupported", result.stdout)

    def test_direct_needle_is_disabled_before_imports_or_requests(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "skills/loom-benchmark/scripts/needle.py")],
            env=dict(os.environ, SKIP_NEEDLE="1"), capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no request sent", result.stderr)

    def test_scope_writer_records_exclusion_without_a_pass(self):
        writer = GATES.split('<<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
        for skip in ("0", "1"):
            with self.subTest(skip=skip), tempfile.TemporaryDirectory() as temp:
                subprocess.run(
                    [sys.executable, "-c", writer, "full", "test-hash", "1", "consistency", "0", "1"],
                    env=dict(os.environ, SKIP_NEEDLE=skip), cwd=temp, check=True, capture_output=True,
                )
                evidence = json.loads((Path(temp) / "gates.json").read_text())
                self.assertEqual(evidence["verdict"], "PASS")
                self.assertNotIn("needle", evidence["gates"])
                self.assertEqual("skipped" in evidence, skip == "1")
                if skip == "1":
                    self.assertEqual(set(evidence["skipped"]), {"needle", "needle_full"})

    def test_shell_gate_dispatch_never_calls_excluded_commands(self):
        bash = shutil.which("bash")
        if os.name == "nt":
            git = shutil.which("git")
            candidate = Path(git).parent.parent / "bin/bash.exe" if git else None
            bash = str(candidate) if candidate and candidate.is_file() else None
        if not bash:
            self.skipTest("Bash unavailable")
        function = "run() {" + GATES.split("run() {", 1)[1].split("\n# Speculative decoding", 1)[0]
        with tempfile.TemporaryDirectory() as temp:
            script = "set -u\ndeclare -A result secs done\norder=(); defer=0\npassed_before() { return 1; }\n"
            script += function + "\n"
            script += 'run needle bash -c "touch forbidden"\nrun needle_full bash -c "touch forbidden"\n'
            script += 'run consistency bash -c "echo real-gate > executed"\n'
            result = subprocess.run([bash, "-c", script], cwd=temp, env=dict(os.environ, SKIP_NEEDLE="1"), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((Path(temp) / "forbidden").exists())
            self.assertEqual((Path(temp) / "executed").read_text().strip(), "real-gate")
            self.assertEqual(result.stdout.count("SKIPPED"), 2)


if __name__ == "__main__":
    unittest.main()
