"""The opt-in logits readback (loom-skills#736) must not change default programs.

The generated @llm_advance ends with a tail guarded by `config.get @llm.logits` (card config, default off). Command programs
have no offline compile product, so the proof runs loom-opt's folding passes: with the capability off, the stage equals the same
stage without the tail (SSA names renumbered); with it on, the export launch is the only difference."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/loom-port-llm/starter"))
SLUG = "unsloth-Qwen3-0.6B-loom"
TAIL = re.compile(r"  // Opt-in logits readback: folded away.*?\n  scf\.if %logits_on \{\n.*?\n  \}\n", re.S)
CONFIG = ["--config=llm.context=4096", "--config=llm.spans=4", "--config=llm.tokens=64", "--config=llm.pool=4096"]
PASSES = ["--pass=canonicalize", "--pass=cse", "--pass=dce", "--pass=symbol-dce"]


def tool(name):
    for c in (os.environ.get(name.upper().replace("-", "_")), shutil.which(name), f"/home/jfowers/lt/loom010/.venv/bin/{name}"):
        if c and os.path.exists(c):
            return c


def norm(text):
    return re.sub(r"%\d+", "%N", text)


@unittest.skipUnless(tool("loom-opt"), "loom-opt not found (set LOOM_OPT)")
class LogitsFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gguf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("gguf not installed (fleet build)")
        cls.tmp = tempfile.mkdtemp()
        r = subprocess.run([sys.executable, str(ROOT / "fleet/build_fleet.py"), "--tree", str(ROOT), "--out", cls.tmp, "--slugs", SLUG],
                           capture_output=True, text=True)
        if r.returncode:
            raise unittest.SkipTest("fleet build failed: " + (r.stdout + r.stderr)[-300:])
        cls.program = Path(cls.tmp, SLUG, "program")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def opt(self, path, *extra):
        r = subprocess.run([tool("loom-opt"), str(path), *PASSES, *CONFIG, *extra], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        return norm(r.stdout)

    def test_default_stage_equals_stage_without_tail(self):
        text = (self.program / "llm.loom").read_text()
        self.assertEqual(len(TAIL.findall(text)), 1)
        bare = Path(self.tmp, "bare.loom")
        bare.write_text(TAIL.sub("", text))
        self.assertEqual(self.opt(self.program / "llm.loom"), self.opt(bare))

    def test_capability_on_launches_the_export(self):
        on = self.opt(self.program / "llm.loom", "--config=llm.logits=true")
        off = self.opt(self.program / "llm.loom", "--config=llm.logits=false")
        self.assertIn("kernel.launch @g_logits_export[", on)
        self.assertNotIn("kernel.launch @g_logits_export[", off)

    def test_storage_and_control_only_gain_the_guard(self):
        storage = (self.program / "storage.loom").read_text()
        self.assertIn("config.def @llm.logits = false : i1", storage)
        self.assertIn("scf.if %logits_on", storage)
        for name in ("control", "prepare", "service"):
            self.assertNotIn("llm.logits", (self.program / f"{name}.loom").read_text())


if __name__ == "__main__":
    unittest.main()
