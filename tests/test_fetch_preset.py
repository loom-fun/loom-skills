"""Every rendered `loom-fetch` line names its preset (`huggingface:REPO:PRESET`, legacy `--preset=PRESET`), single-preset cards included:
adding a preset to a published card later must never break a published command (0.1.0 and 0.2.0 refuse a multi-preset card fetched without one)."""
import glob
import os
import re
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GOLDENS = os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card", "test")
FETCH = re.compile(r"^loom-fetch (\S+)(.*)$", re.M)


def missing_preset(text):
    """The fetch lines of a card text that name no preset."""
    bad = []
    for m in FETCH.finditer(text):
        src, rest = m.group(1), m.group(2)
        if src.startswith("huggingface:"):
            named = re.fullmatch(r"huggingface:[^/:\s]+/[^/:\s]+:[^:\s]+", src)
        else:   # legacy pins: the raw model.json URL plus --preset=
            named = re.search(r"--preset=\S+", rest)
        if not named:
            bad.append(m.group(0))
    return bad


class FetchPreset(unittest.TestCase):
    def test_checker_flags_missing_preset(self):
        self.assertTrue(missing_preset("loom-fetch huggingface:o/r --materialize=copy --output=x"))
        self.assertTrue(missing_preset("loom-fetch https://huggingface.co/o/r/resolve/main/model.json --output=x"))
        self.assertFalse(missing_preset("loom-fetch huggingface:o/r:chat --materialize=copy --output=x"))
        self.assertFalse(missing_preset("loom-fetch https://huggingface.co/o/r/resolve/main/model.json --preset=chat --output=x"))

    def test_every_golden_fetch_line_names_its_preset(self):
        files = glob.glob(os.path.join(GOLDENS, "*", "expected", "README.md")) + glob.glob(os.path.join(GOLDENS, "legacy", "*", "README.md"))
        seen = 0
        for f in files:
            text = open(f, encoding="utf-8").read()
            seen += len(FETCH.findall(text))
            self.assertEqual(missing_preset(text), [], f)
        self.assertGreater(seen, 8)

    def test_single_preset_fixture(self):   # sdk02: one preset (`chat`), 0.2.0 syntax
        text = open(os.path.join(GOLDENS, "sdk02", "expected", "README.md"), encoding="utf-8").read()
        self.assertIn("loom-fetch huggingface:loom-fun/unsloth-Qwen3-0.6B-loom:chat ", text)


if __name__ == "__main__":
    unittest.main()
