import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/loom-benchmark/scripts/card"))
from matched_spec import DATA, classification, drafter_file, matched_spec, row_states, violations  # noqa: E402

CARDS = json.load(open(DATA))


class ClassificationTests(unittest.TestCase):
    def test_all_cards_classified_and_no_ngram(self):
        self.assertEqual(len(CARDS), 51)
        for repo, e in CARDS.items():
            self.assertIn(e["choice"], ("mtp", "dflash", "assist", "none"), repo)
            self.assertNotIn("ngram", e["llama_spec"], repo)
            self.assertEqual(bool(e["llama_spec"]), e["choice"] != "none", repo)
            if e["choice"] != "none":
                self.assertTrue(e["provenance"], repo)   # a drafter without a public upstream is no drafter

    def test_wrong_model_drafters_are_not_used(self):
        for repo in ("loom-fun/unsloth-Qwen3-4B-Instruct-2507-loom", "loom-fun/unsloth-Qwen3-30B-A3B-Instruct-2507-loom"):
            self.assertEqual(classification(repo)["choice"], "none")

    def test_specs(self):
        self.assertIn("draft-dflash --spec-draft-model {drafter}", matched_spec("loom-fun/ggml-org-Qwen3-8B-loom") if "loom-fun/ggml-org-Qwen3-8B-loom" in CARDS else matched_spec("loom-fun/unsloth-Qwen3-8B-loom"))
        self.assertEqual(matched_spec("loom-fun/InternScience-Sage-A1.5-loom"), "--spec-type draft-mtp")
        self.assertEqual(matched_spec("loom-fun/bartowski-Qwen2.5-7B-Instruct-loom"), "")

    def test_rows_streams4_never_drafts(self):
        r = row_states("loom-fun/InternScience-Sage-A1.5-loom")
        self.assertTrue(all(v for k, v in r.items() if k != "streams4"))
        self.assertFalse(r["streams4"])
        self.assertFalse(any(row_states("loom-fun/bartowski-Qwen2.5-7B-Instruct-loom").values()))

    def test_violations_compare_the_package_with_the_classification(self):
        repo = "loom-fun/bartowski-Qwen2.5-7B-Instruct-loom"
        self.assertEqual(violations(repo, {"draft": {"kind": "ngram", "depth": 2}}), ["drafts with ngram, the classification says none"])
        self.assertEqual(violations(repo, {}), [])

    def test_drafter_file(self):
        self.assertEqual(drafter_file({"drafter": {"gguf": "a.gguf"}}), "a.gguf")
        self.assertEqual(drafter_file({"assistant": {"gguf": "b.gguf"}, "draft": {"gguf": "c.gguf"}}), "b.gguf")
        self.assertEqual(drafter_file({"draft": {"kind": "ngram"}}), None)


if __name__ == "__main__":
    unittest.main()
