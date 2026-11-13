import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts/card"))
from matched_spec import drafter_file, matched_spec  # noqa: E402


class MatchedSpecTests(unittest.TestCase):
    def test_no_draft_means_no_drafting(self):
        for cfg in ({}, {"draft": None}, {"draft": {"kind": "none", "depth": 0}}, {"draft": {"kind": "ngram", "depth": 0}}):
            self.assertEqual(matched_spec(cfg), "")

    def test_dflash_uses_the_same_drafter_and_depth(self):
        self.assertEqual(matched_spec({"draft": {"kind": "dflash", "depth": 3, "gguf": "d.gguf"}}),
                         "--spec-type draft-dflash --spec-draft-model {drafter} --spec-draft-n-max 3 -ngld 99")

    def test_assist_mtp_ngram(self):
        self.assertEqual(matched_spec({"draft": {"kind": "assist", "depth": 2}}),
                         "--spec-type draft-mtp --spec-draft-model {drafter} --spec-draft-n-max 2 -ngld 99")
        self.assertEqual(matched_spec({"draft": {"kind": "mtp", "depth": 3}}), "--spec-type draft-mtp --spec-draft-n-max 3")
        self.assertEqual(matched_spec({"draft": {"kind": "ngram", "depth": 1}}), "--spec-type ngram-simple --spec-draft-n-max 1")

    def test_unknown_kind_is_an_error_not_a_silent_fallback(self):
        with self.assertRaises(ValueError):
            matched_spec({"draft": {"kind": "eagle", "depth": 2}})

    def test_drafter_file(self):
        self.assertEqual(drafter_file({"drafter": {"gguf": "a.gguf"}}), "a.gguf")
        self.assertEqual(drafter_file({"assistant": {"gguf": "b.gguf"}, "draft": {"gguf": "c.gguf"}}), "b.gguf")
        self.assertEqual(drafter_file({"draft": {"kind": "ngram"}}), None)


if __name__ == "__main__":
    unittest.main()
