"""logits_check.py comparisons (no GPU): argmax vs the device token, llama.cpp top-5 within tolerance, default output unchanged,
the verdict, and the not-supported path."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, ".github", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts"))
import logits_check as lc  # noqa: E402
from bench_common import near_tie  # noqa: E402

V = 10


def row(best, second=None, pad=True):
    r = [-5.0] * V
    r[best] = 3.0
    if second is not None:
        r[second] = 2.5
    return r + ([-3e38] * 6 if pad else [])


class Comparisons(unittest.TestCase):
    def test_argmax(self):
        self.assertEqual(lc.check_argmax(row(4), V, 4), (True, ""))
        self.assertFalse(lc.check_argmax(row(4), V, 5)[0])
        self.assertFalse(lc.check_argmax(row(4, pad=False), V + 3, 4)[0])           # short row
        self.assertFalse(lc.check_argmax(row(4)[:V] + [9.0] * 6, V, 4)[0])           # unmasked padding
        self.assertFalse(lc.check_argmax([float("nan")] + row(4)[1:], V, 4)[0])

    def test_tie_goes_to_lowest_index(self):
        self.assertEqual(lc.argmax([1.0, 1.0, 0.0], 3), 0)

    def test_log_softmax_sums_to_one(self):
        top = lc.log_softmax_top(row(2, 3), V, k=V)
        self.assertAlmostEqual(sum(__import__("math").exp(lp) for _, lp in top), 1.0, places=6)
        self.assertEqual(top[0][0], 2)

    def test_vs_llama(self):
        rows = [row(1, 2), row(3, 4)]
        mine = {i: lp for i, lp in lc.log_softmax_top(rows[0], V)}
        good = [dict(mine), {i: lp for i, lp in lc.log_softmax_top(rows[1], V)}]
        self.assertEqual(lc.check_vs_llama(rows, V, [1, 3], good, near_tie), (2, []))
        off = [{**mine, 1: mine[1] + 0.6}, good[1]]
        n, probs = lc.check_vs_llama(rows, V, [1, 3], off, near_tie)
        self.assertEqual(n, 2)
        self.assertTrue(probs and "step 0" in probs[0])
        # a divergence llama.cpp is confident about is a problem; a near-tie is not, and ends the comparison
        confident = [{1: -0.01, 3: -6.0}]
        n, probs = lc.check_vs_llama([row(3)], V, [1], confident, near_tie)
        self.assertEqual(n, 0)
        self.assertTrue(probs)
        tie = [{1: -0.6, 3: -0.9}]
        self.assertEqual(lc.check_vs_llama([row(3)], V, [1], tie, near_tie), (0, []))

    def test_default_unchanged(self):
        self.assertEqual(lc.check_default_unchanged([1, 2, 3], [1, 2, 3], [1, 2, 3]), (True, ""))
        self.assertIn("token 1", lc.check_default_unchanged([1, 2, 3], [1, 9, 3], [1, 2, 3])[1])
        self.assertIn("flagged", lc.check_default_unchanged([1, 2, 3], [1, 2, 3], [1, 2, 4])[1])

    def test_verdict(self):
        ok = [{"name": "argmax", "ok": True}]
        self.assertTrue(lc.verdict(ok, 8, True)[0])
        self.assertFalse(lc.verdict(ok, 3, True)[0])        # too few steps compared with llama.cpp
        self.assertTrue(lc.verdict(ok, 0, False)[0])        # no llama.cpp reference asked for
        self.assertFalse(lc.verdict(ok + [{"name": "x", "ok": False, "why": "w"}], 9, True)[0])


class NotSupported(unittest.TestCase):
    def test_without_a_library(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "logits.json")
            os.environ.pop("LOOM_EMBED_LIB", None)
            class A:
                package, lib, llama, steps, prompts, preset, rows, memory_bytes = "x", None, None, 1, 1, None, 1, 0
            A.out = out
            self.assertEqual(lc.run(A), 2)
            j = json.load(open(out))
            self.assertEqual(j["verdict"], "NOT_SUPPORTED")
            self.assertNotIn("logits", j["gates"])        # never a pass or a fail without the library


if __name__ == "__main__":
    unittest.main()
