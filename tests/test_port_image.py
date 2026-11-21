"""loom-port-image: wiring, bundled scripts, and the Philox noise generator the diffusers oracle depends on."""
import json
import py_compile
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT / "skills/loom-port-image"
SCRIPTS = SK / "scripts"

try:
    import numpy as np
except ImportError:   # the CI python has no numpy: the noise tests skip
    np = None


class Wiring(unittest.TestCase):
    """loom-port-image is wired into the suite like the other skills."""

    def test_skill_frontmatter(self):
        text = (SK / "SKILL.md").read_text(encoding="utf-8")
        head = text.split("---")[1]
        self.assertIn("name: loom-port-image", head)
        self.assertIn("description:", head)
        self.assertLess(len(text.splitlines()), 500)

    def test_listed_in_readme_router_checklist(self):
        self.assertIn("| `loom-port-image` |", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("`loom-port-image` (", (ROOT / "skills/loom/SKILL.md").read_text(encoding="utf-8"))
        self.assertIn("skills/loom-port-image/SKILL.md",
                      (ROOT / "skills/loom-fleet/reference/practices-checklist.md").read_text(encoding="utf-8"))

    def test_referenced_scripts_and_links_exist(self):
        for md in [SK / "SKILL.md", *(SK / "reference").glob("*.md")]:
            text = md.read_text(encoding="utf-8")
            for rel in re.findall(r"\]\((?!http)([^)#]+)\)", text):
                self.assertTrue((md.parent / rel).exists(), f"{md.name}: {rel}")
        for name in re.findall(r"`scripts/(\w+\.py)", (SK / "SKILL.md").read_text(encoding="utf-8")):
            self.assertTrue((SCRIPTS / name).exists(), name)

    def test_scripts_compile_and_have_usage(self):
        scripts = sorted(SCRIPTS.glob("*.py"))
        self.assertTrue(scripts)
        for s in scripts:
            py_compile.compile(str(s), cfile=str(Path(tempfile.gettempdir()) / (s.stem + ".pyc")), doraise=True)
            text = s.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("#!/usr/bin/env python3\n"), s.name)
            self.assertIn("usage: " + s.name, text, s.name)

    def test_prompt_set(self):
        cfg = json.loads((SK / "reference/prompts.json").read_text(encoding="utf-8"))
        ids = [p["id"] for p in cfg["prompts"]]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(cfg["seeds"], [0, 42, 1234])
        for size in cfg["sizes"]:
            w, h = (int(v) for v in size.split("x"))
            self.assertEqual((w % 16, h % 16), (0, 0), size)


class StageHf(unittest.TestCase):
    """stage_hf.py turns a library card into a self-contained package (standard library only)."""

    def test_flattens_motifs_and_drops_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp) / "lib"
            card_dir = lib / "model/pub/m"
            (card_dir / "program").mkdir(parents=True)
            (lib / "motif/tensor").mkdir(parents=True)
            (lib / "motif/tensor/linear.loom").write_text("// motif\n")
            (card_dir / "program/generate.loom").write_text("// program\n")
            (card_dir / "model.json.lock").write_text("{}\n")
            card = {
                "programs": {
                    "source": {"libraries": [{"path": "library/motif/tensor/linear.loom", "origin": "../.."},
                                             "program/generate.loom"]},
                    "archive": {"path": "bazel-out/generate.vmfb"},
                },
                "variants": {"bf16": {"services": {"images": {"programs": ["archive"]}}}},
            }
            (card_dir / "model.json").write_text(json.dumps(card))
            out = Path(tmp) / "out"
            subprocess.run([sys.executable, str(SCRIPTS / "stage_hf.py"), str(card_dir), str(out)],
                           check=True, capture_output=True)
            staged = json.loads((out / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(list(staged["programs"]), ["source"])
            self.assertEqual(staged["programs"]["source"]["libraries"],
                             ["library/motif/tensor/linear.loom", "program/generate.loom"])
            self.assertEqual(staged["variants"]["bf16"]["services"]["images"]["programs"], ["source"])
            self.assertTrue((out / "library/motif/tensor/linear.loom").exists())
            self.assertTrue((out / "program/generate.loom").exists())
            self.assertTrue((out / "model.json.lock").exists())


class LoomDeps(unittest.TestCase):
    """loom_deps.py resolves declared symbols to sibling program files and motif labels."""

    def run_deps(self, decl):
        with tempfile.TemporaryDirectory() as tmp:
            program = Path(tmp) / "model/pub/m/program"
            program.mkdir(parents=True)
            (Path(tmp) / "motif/tensor").mkdir(parents=True)
            (Path(tmp) / "motif/tensor/linear.loom").write_text("kernel.def public @linear(%x) {\n}\n")
            (program / "block.loom").write_text("func.def @block() {\n}\n")
            (program / "generate.loom").write_text(decl)
            return subprocess.run([sys.executable, str(SCRIPTS / "loom_deps.py"), str(program)],
                                  capture_output=True, text=True)

    def test_resolves_sibling_and_motif(self):
        r = self.run_deps("kernel.decl @linear\nfunc.decl @block\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("generate ['//motif/tensor:linear', ':block']", r.stdout)

    def test_missing_symbol_fails(self):
        r = self.run_deps("kernel.decl @nowhere\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn("MISSING @nowhere", r.stdout)


@unittest.skipIf(np is None, "numpy not installed")
class PhiloxNoise(unittest.TestCase):
    """The NumPy Philox4x32-10 draw must match Random123 and give the latent shapes the pipelines expect."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        import philox_noise
        cls.P = philox_noise

    def test_known_answers(self):
        for ctr, key, want in self.P.KAT:
            got = tuple(int(w[0]) for w in self.P.philox4x32_10(*[np.array([v], np.uint32) for v in ctr], *key))
            self.assertEqual(got, want)

    def test_layouts(self):
        self.assertEqual(self.P.LAYOUTS["flux2"](0, 256, 512).shape, (1, 128, 16, 32))
        self.assertEqual(self.P.LAYOUTS["zimage"](0, 256, 512).shape, (1, 16, 32, 64))

    def test_seeded_and_bf16(self):
        a, b = self.P.loom_noise(42, 4096), self.P.loom_noise(42, 4096)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, self.P.loom_noise(43, 4096)))
        bits = a.astype(np.float32).view(np.uint32)
        self.assertFalse((bits & 0xFFFF).any())


if __name__ == "__main__":
    unittest.main()
