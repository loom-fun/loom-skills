"""devlab_perf.sh local-evidence pieces: gufo_path resolves a pinned file in the user's HF cache without copying it, and
power_steps.py summarizes the power log per step and per engine (busy samples only, band flags)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/devlab_perf.sh"


def function(name):
    text = SCRIPT.read_text().split(name + "() {", 1)[1]
    return name + "() {" + text.split("\n}", 1)[0] + "\n}\n"


@unittest.skipIf(os.name == "nt", "devlab_perf.sh runs on the Linux DevLab runners only")
class GufoPath(unittest.TestCase):
    def test_local_cache_pinned_revision(self):
        with tempfile.TemporaryDirectory() as t:
            snap = Path(t, "hub/models--own--M-GGUF/snapshots/abc123/")
            snap.mkdir(parents=True)
            (snap / "m.gguf").write_text("x")
            sh = function("gufo_path") + 'gufo_path "own/M-GGUF@abc123/m.gguf"'
            r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True,
                               env=dict(os.environ, LOCAL_HF_CACHE="1", HF_HOME=t, WORK=t + "/w"))
            self.assertEqual(r.stdout.strip(), str(snap / "m.gguf"))
            self.assertFalse(Path(t, "w").exists())   # never a copy of the cached file

    def test_steps_exist(self):
        s = SCRIPT.read_text()
        for k in ("    gufoperf)", "    gufoc4)", "perf.py\" \"$gurl\" gufo --kind gufo", "--speculative off --sessions 4"):
            self.assertIn(k, s)


class PowerSteps(unittest.TestCase):
    def test_summary(self):
        with tempfile.TemporaryDirectory() as t:
            with open(os.path.join(t, "power.tsv"), "w") as f:
                for i in range(100):   # 0-10 s loom busy at 123 W (one idle sample), 10-20 s gufo at 99 W
                    w = (5 if i == 3 else 123) if i < 50 else 99
                    f.write(f"{1000 + i * 0.2:.1f} {w * 1000000} {2550000000 if i < 50 else 2400000000}\n")
            with open(os.path.join(t, "step_times.tsv"), "w") as f:
                f.write("perf:card:1\t1000\t1010\ngufoperf:1\t1010\t1020\n")
            r = subprocess.run([sys.executable, "-I", str(ROOT / ".github/scripts/power_steps.py"), t, "--band", "120", "125"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            d = json.load(open(os.path.join(t, "power_steps.json")))
            self.assertEqual(d["engines"]["loom"]["median_w"], 123.0)
            self.assertEqual(d["engines"]["loom"]["busy"], 49)
            self.assertEqual(d["engines"]["gufo"]["median_ghz"], 2.4)
            self.assertNotIn("flag", d["steps"][0])
            self.assertIn("outside 120-125 W", d["steps"][1]["flag"])


if __name__ == "__main__":
    unittest.main()
