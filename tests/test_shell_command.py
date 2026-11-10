import importlib.util
import json
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
    def test_release_metadata_larger_than_native_argument_limit(self):
        prefix = (REPO / "skills/loom-benchmark/scripts/fetch_llamacpp.sh").read_text(encoding="utf-8").split("dir=$dest/$rtag", 1)[0]
        with tempfile.TemporaryDirectory(prefix="release metadata ") as temp:
            metadata = Path(temp) / "release.json"
            metadata.write_text(json.dumps({"tag_name": "b-test", "padding": "x" * 200000, "assets": [
                {"name": f"llama-b-test-{osname}-rocm-gfx1151-x64.zip", "browser_download_url": "https://example.invalid/reference.zip", "digest": "sha256:test"}
                for osname in ("windows", "ubuntu")]}), encoding="utf-8")
            script = Path(temp) / "fetch-prefix.sh"
            script.write_text('curl() { cat "$TEST_METADATA"; }\n' + prefix + '\nprintf "%s" "$url"\n', encoding="utf-8")
            result = subprocess.run(helper.bash_script(script.as_posix(), temp, "latest", "gfx1151"), env=dict(os.environ, TEST_METADATA=metadata.as_posix(), PY=Path(sys.executable).as_posix()), capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout, "https://example.invalid/reference.zip")

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
