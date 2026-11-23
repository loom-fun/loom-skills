"""loom-port-image: wiring, bundled scripts, and the Philox noise generator the diffusers oracle depends on."""
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

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

    def test_installed_image_route(self):
        kb = SK / "../loom-kb/kb.py"
        result = subprocess.run(
            [sys.executable, str(kb), "route", "--arch", "z-image", "--device", "strix-halo", "--topic",
             "image-porting,kernels-ir", "--relative"],
            check=True, capture_output=True, text=True)
        files = result.stdout.splitlines()
        for relative in ("device/rdna3/strix-halo/image-porting.md", "arch/image/diffusion/README.md",
                         "arch/image/diffusion/z-image/README.md", "topic/image-porting.md"):
            self.assertIn(relative, files)
        self.assertFalse(any(path.startswith("arch/llm/") for path in files))


class StageHf(unittest.TestCase):
    """stage_hf.py turns a library card into a self-contained package (standard library only)."""

    def setUp(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.lib = tmp / "lib"
        self.card_dir = self.lib / "model/pub/m"
        self.out = tmp / "out"
        (self.card_dir / "program").mkdir(parents=True)
        (self.card_dir / "assets").mkdir()
        (self.lib / "motif/tensor").mkdir(parents=True)
        (self.lib / "motif/tensor/linear.loom").write_text("// motif\n")
        (self.card_dir / "program/generate.loom").write_text("// program\n")
        (self.card_dir / "assets/shape.bin").write_bytes(b"shape")
        (self.card_dir / "assets/tokenizer.json").write_text("{}")
        (self.card_dir / "model.json.lock").write_text("{}\n")
        self.motif = {"path": "library/motif/tensor/linear.loom", "origin": "../../../motif/tensor/linear.loom"}
        self.input = {"path": "program/generate.loom", "data": ["assets/shape.bin"]}
        self.card = {
            "programs": {"source": {"libraries": [self.motif], "inputs": [self.input]}, "archive": {}},
            "variants": {"bf16": {"services": {"images": {"programs": ["source", "archive"],
                                                         "tokenizers": {"text": {"path": "assets/tokenizer.json"}}}}}},
            "presets": {"image": {"programs": "archive"}},
        }

    def run_stage(self):
        (self.card_dir / "model.json").write_text(json.dumps(self.card))
        return subprocess.run([sys.executable, str(SCRIPTS / "stage_hf.py"), str(self.card_dir), str(self.out)],
                              capture_output=True, text=True)

    def test_flattens_motifs_and_copies_local_assets(self):
        result = self.run_stage()
        self.assertEqual(result.returncode, 0, result.stderr)
        staged = json.loads((self.out / "model.json").read_text())
        self.assertEqual(list(staged["programs"]), ["source"])
        self.assertEqual(staged["programs"]["source"]["libraries"], ["library/motif/tensor/linear.loom"])
        self.assertEqual(staged["variants"]["bf16"]["services"]["images"]["programs"], ["source"])
        self.assertEqual(staged["presets"]["image"]["programs"], "source")
        for path in ("library/motif/tensor/linear.loom", "program/generate.loom", "assets/shape.bin",
                     "assets/tokenizer.json", "model.json.lock"):
            self.assertTrue((self.out / path).is_file(), path)

    def test_path_traversal_and_missing_inputs_never_publish(self):
        for filename in ("library/motif/../../outside.loom", "/tmp/file", "assets/..\\outside", "assets/missing"):
            with self.subTest(filename=filename):
                self.input["data"] = [filename]
                self.assertNotEqual(self.run_stage().returncode, 0)
                self.assertFalse(self.out.exists())

    def test_rejects_wrong_catalog_origin(self):
        self.motif["origin"] = "../../../../outside.loom"
        self.assertIn("does not match", self.run_stage().stderr)
        self.assertFalse(self.out.exists())

    def test_existing_output_stays_intact(self):
        self.out.mkdir()
        stale = self.out / "previous"
        stale.write_text("evidence")
        self.assertIn("output already exists", self.run_stage().stderr)
        self.assertEqual(list(self.out.iterdir()), [stale])


class ImageEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        import bench_images
        import oracle_diffusers
        cls.bench = bench_images
        cls.oracle = oracle_diffusers

    def test_skip_requires_identical_and_complete_oracle_evidence(self):
        case = Path(self.enterContext(tempfile.TemporaryDirectory()))
        settings = {"model": "first", "revision": "first-sha", "steps": 2, "capture_stages": True, "guidance_scale": 0}
        (case / "image.png").write_bytes(b"png")
        self.assertFalse(self.oracle.complete_case(case, settings))
        (case / "meta.json").write_text(json.dumps({"settings": settings}))
        self.assertFalse(self.oracle.complete_case(case, settings))
        for name in self.oracle.stage_names(settings):
            (case / name).write_bytes(b"tensor")
        self.assertTrue(self.oracle.complete_case(case, settings))
        for key, value in (("model", "other"), ("revision", "other"), ("steps", 4),
                           ("capture_stages", False), ("guidance_scale", 1)):
            self.assertFalse(self.oracle.complete_case(case, {**settings, key: value}), key)

    def test_local_snapshot_identity_tracks_bytes_independent_of_mtime(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        first, second = root / "a", root / "b"
        for directory in (first, second):
            directory.mkdir()
            (directory / "weights.bin").write_bytes(b"same")
        other = second / "weights.bin"
        os.utime(other, ns=(1000000000, 1000000000))
        digest = self.oracle.snapshot_digest(first)
        self.assertEqual(self.oracle.snapshot_digest(second), digest)
        other.write_bytes(b"edit")
        os.utime(other, ns=(1000000000, 1000000000))
        self.assertNotEqual(self.oracle.snapshot_digest(second), digest)

    def test_rejects_traversal_and_duplicate_case_ids(self):
        for names in (["../outside"], ["/absolute"], ["same", "Same"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.oracle.selected_cases([{"id": name} for name in names], ["384x384"], [0])
        output = Path(self.enterContext(tempfile.TemporaryDirectory())) / "out"
        for labels in (["../outside"], ["/absolute"], ["same", "Same"]):
            with self.subTest(labels=labels):
                flags = [item for label in labels for item in ("--target", f"{label}=sdserver:http://localhost")]
                result = subprocess.run([sys.executable, str(SCRIPTS / "bench_images.py"), *flags,
                                         "--steps", "8", "--out", str(output)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("target labels", result.stderr)
                self.assertFalse(output.exists())

    def test_timing_checks_card_and_sends_explicit_lemonade_options(self):
        card = Path(self.enterContext(tempfile.TemporaryDirectory())) / "model.json"
        card.write_text(json.dumps({"model": {"id": "image"}, "services": {
            "image": {"kind": "image", "interface": {"denoising_steps": 8, "guidance_scale": 0.0}}}}))
        args = SimpleNamespace(steps=8, cfg_scale=0.0, loom_guidance={}, port=8271, lemonade="http://localhost",
                               thenoise_cfg_free=[])
        loom = self.bench.Target(f"loom=loom:{card}", args)
        lemon = self.bench.Target("lemon=lemonade:model:sd-cpp:rocm", args)
        direct = self.bench.Target("direct=sdserver:http://localhost", args)
        self.assertNotIn("steps", loom.body("fox", 0, "384x384"))
        self.assertEqual((lemon.body("fox", 0, "384x384")["steps"],
                          lemon.body("fox", 0, "384x384")["cfg_scale"]), (8, 0.0))
        self.assertIn('"txt_cfg": 0.0', direct.body("fox", 0, "384x384")["prompt"])
        with patch.object(self.bench, "http", return_value={"data": [{"id": "image", "loom": {"registration": "ready"}}]}):
            self.assertEqual(loom.info()["registration"]["registration"], "ready")
        options = {"steps": 8, "cfg_scale": 0, "sd-cpp_backend": "rocm"}
        health = {"all_models_loaded": [{"model_name": "model", "recipe": "sd-cpp", "recipe_options": options}]}
        with patch.object(self.bench, "http", return_value=health) as client:
            lemon.up()
            self.assertEqual(client.call_args.args[2], {"model_name": "model", **options, "save_options": False})
            self.assertEqual(lemon.info()["loaded"]["recipe_options"], options)
            options["steps"] = 9
            with self.assertRaisesRegex(ValueError, "reported steps=9"):
                lemon.info()
        with self.assertRaisesRegex(ValueError, "omits zero guidance"):
            self.bench.Target("tn=lemonade:model:thenoise:rocm", args)
        args.thenoise_cfg_free = ["tn"]
        thenoise = self.bench.Target("tn=lemonade:model:thenoise:rocm", args)
        self.assertEqual(thenoise.body("fox", 0, "384x384")["cfg_scale"], 1.0)
        args.steps = 9
        with self.assertRaisesRegex(ValueError, "differ from requested"):
            self.bench.Target(f"loom=loom:{card}", args)

    @unittest.skipIf(np is None, "numpy not installed")
    def test_paired_scoring_checks_case_identity(self):
        import metrics

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        a, b = root / "left", root / "right"
        for side in (a, b):
            (side / "fox").mkdir(parents=True)
            meta = {"prompt": "a red fox", "seed": 0, "width": 384, "height": 384}
            (side / "fox/meta.json").write_text(json.dumps(meta))
        self.assertEqual(metrics.matching_metadata([a, b], ["fox"])["fox"]["seed"], 0)
        (b / "fox/meta.json").write_text(json.dumps({**meta, "prompt": "a brown hare"}))
        with self.assertRaisesRegex(ValueError, "differs for paired case"):
            metrics.matching_metadata([a, b], ["fox"])

    @unittest.skipIf(np is None, "numpy not installed")
    def test_paired_metrics_report_mean_and_confidence_interval(self):
        import metrics
        from PIL import Image

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        base, candidate = root / "base", root / "candidate"
        for path, value in ((base, 10), (candidate, 20)):
            for seed in (0, 42):
                case = path / f"fox_s{seed}_16x16"
                case.mkdir(parents=True)
                case.joinpath("meta.json").write_text(json.dumps(
                    {"prompt": "a red fox", "seed": seed, "width": 16, "height": 16}))
                Image.new("RGB", (16, 16), (value, 0, 0)).save(case / "image.png")

        class FixtureScorer:
            name = "test scorer"

            def __call__(self, image, prompt):
                return image.getpixel((0, 0))[0]

        with patch.object(metrics, "ClipScorer", return_value=FixtureScorer()), \
                patch.object(metrics, "PickScorer", return_value=FixtureScorer()):
            report = metrics.scores([base, candidate], "base", "cpu")
        for name in ("clip", "pick"):
            self.assertEqual(report["sets"]["candidate"][name]["vs_baseline"],
                             {"mean": 10, "ci95": [10, 10], "win_rate": 1, "n": 2})


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
