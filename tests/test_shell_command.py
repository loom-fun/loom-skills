import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shell_command", REPO / "skills/loom-benchmark/scripts/shell_command.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ShellCommandTests(unittest.TestCase):

    def test_python_executes_bash_file_with_literal_spaced_argument(self):
        with tempfile.TemporaryDirectory(prefix="native Bash spaces ") as temp:
            script = Path(temp) / "test script.sh"
            script.write_text('printf "%s" "$1"\n', encoding="utf-8")
            argument = "literal spaces; $(touch unexpected)"
            result = subprocess.run(helper.bash_script(script.as_posix(), argument), cwd=temp, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout, argument)
            self.assertFalse((Path(temp) / "unexpected").exists())


if __name__ == "__main__":
    unittest.main()
