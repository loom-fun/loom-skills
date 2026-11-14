"""Gate scope: tool-calling applicability in gates.sh, and the retired needle gates."""
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

    def test_no_needle_gates_remain(self):
        self.assertNotIn("needle", GATES.lower())
        self.assertFalse((REPO / "skills/loom-benchmark/scripts/needle.py").exists())
        self.assertTrue((REPO / "skills/loom-benchmark/scripts/ctx_admit.py").is_file())

    def test_retired_gates_are_dropped_from_old_cards(self):
        sys.path.insert(0, str(REPO / "skills/loom-benchmark/scripts/card"))
        import card_lib
        old = {"schema": "loom-card/3", "devices": {"strix-halo": {"gates": {"golden": {"status": "pass"}, "needle": {"status": "pass"}, "needle_full": {"status": "pass"}}}}}
        self.assertEqual(sorted(card_lib.normalize(old)["devices"]["strix-halo"]["gates"]), ["golden"])
        self.assertIn("needle", old["devices"]["strix-halo"]["gates"])   # the input is not changed
        self.assertNotIn("needle", card_lib.GATE_NAMES)


if __name__ == "__main__":
    unittest.main()
