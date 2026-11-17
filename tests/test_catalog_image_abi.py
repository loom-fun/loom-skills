"""The Flash-Next catalog copy's image component implements the host's component ABI 1 exactly as the kit's generator does.

hrx-loom 0.2.0 calls `text_image_run_v1(state, stage, input, output, batch)`; the library's Flash-Next vision_control.loom (rc1, 0.2.0 and main) still declares the
3-argument form, so every image request of the rebuilt Flash-Next package failed with "model export 'text_image_run_v1' argument count mismatch: host expects 5,
source declares 3" (0.2.0 sweep: func image_input, gates vision and spec FAIL on Strix)."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "skills/loom-port-llm/reference/qwen4exp-example/catalog/program"
GEN = ROOT / "skills/loom-port-llm/starter/vision/gen_component.py"
DEF = re.compile(r"func\.def public @(text_image_\w+_v1)\(([^)]*)\)")


def arg_types(text):
    """{export: [argument types]} of the text_image_*_v1 definitions in TEXT."""
    out = {}
    for name, args in DEF.findall(text):
        out[name] = [a.split(":", 1)[1].strip() for a in args.split(", ") if a.strip()]
    return out


class ImageAbi(unittest.TestCase):
    def test_catalog_exports_match_the_generator(self):
        cat = {}
        for f in sorted(CATALOG.glob("vision_*.loom")):
            cat.update(arg_types(f.read_text()))
        gen = arg_types(GEN.read_text())
        self.assertEqual(sorted(cat), sorted(gen))
        for name in gen:
            self.assertEqual(cat[name], gen[name], name)
        self.assertEqual(len(cat["text_image_run_v1"]), 5)


if __name__ == "__main__":
    unittest.main()
