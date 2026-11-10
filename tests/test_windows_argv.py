import importlib.util
import os
from pathlib import Path
import subprocess
import unittest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("windows_argv", REPO / "skills/loom-benchmark/scripts/windows_argv.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@unittest.skipUnless(os.name == "nt", "Native Windows argument parser")
class WindowsArgvTests(unittest.TestCase):
    def test_native_server_command_roundtrip_preserves_quoted_paths(self):
        args = [r"C:\Users\Name With Spaces\SDK\loom-serve.exe", "--device=amdgpu", "--model=C:/model with spaces/model.json", "", 'literal "quoted" value', "trailing\\"]
        self.assertEqual(helper.split_command_line(subprocess.list2cmdline(args)), args)


if __name__ == "__main__":
    unittest.main()
