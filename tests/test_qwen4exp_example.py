import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "skills/loom-port-llm/reference/qwen4exp-example"
PKG = "model/Qwen/Qwen3.8-Flash-Next"


class Qwen4expExample(unittest.TestCase):
    def test_catalog_files_exist_and_pin_matches_fleet(self):
        cat = EX / "catalog"
        model = json.loads((cat / "model.json").read_text())
        for prog in model["programs"].values():
            for item in prog["inputs"] + prog["libraries"]:
                paths = [item] if isinstance(item, str) else [item["path"], *item.get("data", [])]
                for p in paths:
                    self.assertTrue((cat / p).is_file(), p)
        pin = json.loads((cat / "catalog.json").read_text())["sdk"]
        fleet = json.loads((ROOT / "fleet/unsloth-Qwen3.8-Flash-Next-loom/llm_config.json").read_text())["sdk"]
        self.assertEqual(pin, fleet)
        self.assertEqual(pin["hrx_loom"], "0.3.0")
        self.assertEqual(pin["source"], "pypi")

    def test_catalog_serving_flags_respect_an_explicit_null(self):
        sys.path.insert(0, str(ROOT / "skills/loom-port-llm/starter"))
        try:
            from catalog import package_serve
        finally:
            sys.path.pop(0)
        fixed = {"args": ["--pool_backing=fixed"]}
        elastic = {"args": ["--pool_backing=elastic"]}
        self.assertIsNone(package_serve({"serve": None}, {"serve": fixed}, "mtp"))
        self.assertEqual(package_serve({"serve": fixed}, {"serve": elastic}, "mtp"), elastic)
        self.assertEqual(package_serve({}, {"serve": fixed}, "mtp"), fixed)
        self.assertEqual(package_serve({"serve": elastic}, {"serve": fixed}, None), elastic)

    def test_flatten_moves_shared_files(self):
        with tempfile.TemporaryDirectory() as t:
            lib, out = Path(t, "lib"), Path(t, "out")
            for rel in (f"{PKG}/program/a.loom", f"{PKG}/program/dispatch/b.loom", "program/text/qwen3_8/c.loom"):
                (lib / rel).parent.mkdir(parents=True, exist_ok=True)
                (lib / rel).write_text(rel)
            card = {"programs": {
                "source": {"inputs": [{"path": f"{PKG}/program/a.loom", "data": [f"{PKG}/program/dispatch/b.loom"]}],
                           "libraries": ["program/text/qwen3_8/c.loom"]},
                "vision-source": {"inputs": [f"{PKG}/program/a.loom"], "libraries": []},
                "archive": {"inputs": [], "libraries": []}},
                "variants": {"v": {"services": {"s": {"programs": ["source", "archive"]}}}}}
            (lib / PKG / "model.json").write_text(json.dumps(card))
            out.mkdir()
            subprocess.run([sys.executable, str(EX / "flatten.py"), str(lib), str(out)], check=True, capture_output=True)
            m = json.loads((out / "model.json").read_text())
            self.assertEqual(m["programs"]["source"]["inputs"], [{"path": "program/a.loom", "data": ["program/dispatch/b.loom"]}])
            self.assertEqual(m["programs"]["source"]["libraries"], ["program/shared/program/text/qwen3_8/c.loom"])
            self.assertEqual(m["variants"]["v"]["services"]["s"]["programs"], ["source"])
            self.assertNotIn("archive", m["programs"])
            self.assertTrue((out / "program/shared/program/text/qwen3_8/c.loom").is_file())


if __name__ == "__main__":
    unittest.main()
