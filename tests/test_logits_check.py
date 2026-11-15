"""logits_check.py comparisons (no GPU): llama.cpp top-5 within tolerance, default output unchanged, embed_probe's line, the
verdict, and the not-supported path."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, ".github", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts"))
import logits_check as lc  # noqa: E402
try:
    from bench_common import near_tie  # noqa: E402   (needs requests: not installed in the unit-test CI job)
except ImportError:
    near_tie = None


class Comparisons(unittest.TestCase):
    @unittest.skipIf(near_tie is None, "bench_common needs requests")
    def test_vs_llama(self):
        mine = [[("Paris", -0.1), ("The", -2.5)], [(".", -0.2), ("!", -2.0)]]
        good = [{"Paris": -0.12, "The": -2.4}, {".": -0.25, "!": -2.1}]
        self.assertEqual(lc.check_vs_llama(mine, ["Paris", "."], good, near_tie), (2, []))
        off = [{"Paris": -0.12, "The": -4.5}, good[1]]       # a log-prob 2 nats away
        n, probs = lc.check_vs_llama(mine, ["Paris", "."], off, near_tie)
        self.assertEqual(n, 2)
        self.assertTrue(probs and "step 0" in probs[0])
        # a divergence llama.cpp is confident about is a problem; a near-tie is not, and ends the comparison
        n, probs = lc.check_vs_llama([[("The", -0.1)]], ["Paris"], [{"Paris": -0.01, "The": -6.0}], near_tie)
        self.assertEqual(n, 0)
        self.assertTrue(probs)
        self.assertEqual(lc.check_vs_llama([[("The", -0.9)]], ["Paris"], [{"Paris": -0.6, "The": -0.9}], near_tie), (0, []))
        self.assertTrue(lc.check_vs_llama([[]], ["Paris"], [{"Paris": -0.1}], near_tie)[1])   # no top-logprobs at all

    @unittest.skipIf(near_tie is None, "bench_common needs requests")
    def test_special_tokens_and_tail_tolerance(self):
        # llama.cpp prints the end-of-turn token as '', Gufo as the marker: the same token
        self.assertEqual(lc.check_vs_llama([[("<|im_end|>", -0.1)]], [""], [{"": -0.15}], near_tie), (1, []))
        # several tokens print as '': the best entry of each side compares (Loom: the marker and an empty-text token)
        self.assertEqual(lc.check_vs_llama([[("<|im_end|>", -1.02), ("", -8.5)]], [""], [{"": -1.03}], near_tie), (1, []))
        # tail tokens differ by up to ~0.7 nats between the engines' quantized matmuls: inside the tolerance
        self.assertEqual(lc.check_vs_llama([[("Paris", -0.1), ("A", -3.72)]], ["Paris"], [{"Paris": -0.12, "A": -3.25}], near_tie), (1, []))

    @unittest.skipIf(near_tie is None, "bench_common needs requests")
    def test_measured_engine_differences(self):
        # fleet sweep (DevLab): the same GGUF, tail log-probs 1.1-2.3 nats apart on mid-size/MoE models while top-1 agrees
        top = [("A", -0.2), ("b", -3.727), ("c", -8.409), ("d", -12.702)]
        ref = {"A": -0.25, "b": -4.763, "c": -7.309, "d": -10.357}      # 1.04 at p~0.02, 1.1 at p~7e-4, 2.3 at p~3e-5 (not compared)
        self.assertEqual(lc.check_vs_llama([top], ["A"], [ref], near_tie), (1, []))
        # the same distance on a confident token is a problem
        n, probs = lc.check_vs_llama([[("A", -0.2)]], ["A"], [{"A": -1.5}], near_tie)
        self.assertTrue(probs and "tolerance 1.0" in probs[0])
        # the worst measured tail (a MoE model, p~1e-3 on one side): 2.7 nats apart
        self.assertEqual(lc.check_vs_llama([[("A", -0.2), ("b", -9.429)]], ["A"], [{"A": -0.2, "b": -6.744}], near_tie), (1, []))
        # a wrong row (a shifted slot): the second token several nats off where the mass is
        n, probs = lc.check_vs_llama([[("A", -0.2), ("b", -6.5)]], ["A"], [{"A": -0.2, "b": -1.9}], near_tie)
        self.assertTrue(probs)

    @unittest.skipIf(near_tie is None, "bench_common needs requests")
    def test_spellings(self):
        # the end-of-sentence marker (fullwidth bars) is llama.cpp's ''; several tokens print '' and count by their best entry
        eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"
        self.assertEqual(lc.check_vs_llama([[(eos, -0.00002), ("\n", -12.0)]], [""], [{"": -0.00002, "\n": -11.9}], near_tie), (1, []))
        self.assertEqual(lc.best_by_key([("<|im_end|>", -0.0), ("<|text_end|>", -13.5)]), {"": -0.0})
        # a sentencepiece vocabulary: Gufo prints 'are' where llama.cpp prints ' are'
        self.assertEqual(lc.check_vs_llama([[("are", -0.19), ("is", -1.78)]], [" are"], [{" are": -0.18, " is": -1.76}], near_tie), (1, []))
        self.assertEqual(lc.special_key(" "), " ")      # a whitespace-only token keeps its text

    def test_tail_tolerance_tiers(self):
        self.assertEqual(lc.tail_tol(-0.1, -0.2), 1.0)
        self.assertEqual(lc.tail_tol(-2.0, -3.2), 1.5)
        self.assertEqual(lc.tail_tol(-7.0, -9.0), 3.0)
        self.assertIsNone(lc.tail_tol(-10.4, -12.7))

    @unittest.skipUnless(hasattr(os, "killpg"), "process groups are POSIX")
    def test_stop_group(self):
        import subprocess
        import time
        p = subprocess.Popen(["bash", "-c", "sleep 300 & wait"], start_new_session=True)   # a wrapper whose child holds the "GPU"
        time.sleep(0.3)
        t = time.time()
        lc.stop_group(p, grace=10)
        self.assertLess(time.time() - t, 8)
        with self.assertRaises(ProcessLookupError):
            os.killpg(p.pid, 0)

    def test_default_unchanged(self):
        self.assertEqual(lc.check_default_unchanged("Paris.", "Paris.", "Paris."), (True, ""))
        self.assertIn("character 2", lc.check_default_unchanged("Paris.", "Pa!is.", "Paris.")[1])
        self.assertIn("host path", lc.check_default_unchanged("Paris.", "Paris.", "Paris!")[1])

    def test_probe_line(self):
        self.assertEqual(lc.probe_check('{"opened": true, "failures": 0}'), (True, ""))
        self.assertFalse(lc.probe_check('{"opened": true, "failures": 2}')[0])
        self.assertIn("does not declare", lc.probe_check('{"opened": false, "error": "package does not declare llm.logits"}')[1])
        self.assertFalse(lc.probe_check("segfault")[0])

    def test_verdict(self):
        ok = [{"name": "embed_probe", "ok": True}]
        self.assertTrue(lc.verdict(ok, 8, True)[0])
        self.assertFalse(lc.verdict(ok, 3, True)[0])        # too few steps compared with llama.cpp
        self.assertTrue(lc.verdict(ok, 0, False)[0])        # no llama.cpp reference asked for
        self.assertFalse(lc.verdict(ok + [{"name": "x", "ok": False, "why": "w"}], 9, True)[0])


class NotSupported(unittest.TestCase):
    def test_without_a_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "logits.json")
            os.environ.pop("GUFO_BUNDLE", None)

            class A:
                package, bundle, llama, steps, prompts, port = "x", os.path.join(d, "nobundle"), None, 1, 1, 8181
            A.out = out
            self.assertEqual(lc.run(A), 2)
            with open(out) as f:
                j = json.load(f)
            self.assertEqual(j["verdict"], "NOT_SUPPORTED")
            self.assertNotIn("logits", j["gates"])        # never a pass or a fail without the bundle


if __name__ == "__main__":
    unittest.main()


class GufoGate(unittest.TestCase):
    def test_gate(self):
        import gufo_gate
        c = {"results": [{"check": "a", "status": "pass"}, {"check": "b", "status": "pass"}]}
        lc = {"results": [{"check": "x", "status": "pass"}]}
        self.assertEqual(gufo_gate.gate(c, lc, 0), (True, "2/2 conformance checks, 1/1 logits checks"))
        self.assertFalse(gufo_gate.gate(c, lc, 1)[0])                                                # a script failed
        self.assertFalse(gufo_gate.gate({"results": c["results"] + [{"check": "g", "status": "gap"}]}, lc, 0)[0])   # a gap is a miss with --logits
        self.assertFalse(gufo_gate.gate(c, {"results": [{"check": "x", "status": "fail"}]}, 0)[0])
        self.assertFalse(gufo_gate.gate(None, lc, 0)[0])
