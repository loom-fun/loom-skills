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

    @unittest.skipIf(near_tie is None, "bench_common needs requests")
    def test_end_markers_are_one_token(self):
        # llama.cpp ends with '' where Loom's top-1 is the template's end marker (gemma-4 '<turn|>', gemma-3 '<end_of_turn>', Laguna '</assistant>')
        for marker in ("<turn|>", "<end_of_turn>", "</assistant>"):
            top = [[("Paris", -0.0), ], [(marker, -0.22), (".", -1.64)]]
            ref = [{"Paris": -0.0}, {"": -0.217, ".": -1.636}]
            self.assertTrue(lc.check_vs_llama(top, ["Paris", ""], ref, near_tie)[1])    # without the model's markers: a mismatch
            self.assertEqual(lc.check_vs_llama(top, ["Paris", ""], ref, near_tie, eog={marker})[1], [])
            self.assertEqual(lc.check_vs_llama(top, ["Paris", ""], ref, near_tie, eog={marker})[0], 2)
        # an end marker where llama.cpp is confident of another token stays a problem; the marker's log-prob is compared like any other
        n, probs = lc.check_vs_llama([[("<turn|>", -0.1)]], ["Paris"], [{"Paris": -0.01, "": -6.0}], near_tie, eog={"<turn|>"})
        self.assertTrue(probs)
        n, probs = lc.check_vs_llama([[("<turn|>", -0.1)]], [""], [{"": -2.5}], near_tie, eog={"<turn|>"})
        self.assertTrue(probs and "tolerance" in probs[0])
        # Gufo prints a whitespace-only token as '', llama.cpp keeps its text
        self.assertEqual(lc.check_vs_llama([[("", -0.3)]], [" "], [{" ": -0.31}], near_tie), (1, []))

    @unittest.skipIf(near_tie is None, "bench_common needs requests")
    def test_leading_spaces_of_a_byte_level_vocabulary(self):
        # gpt-oss (DevLab R9700): Loom wrote '17*' where llama.cpp wrote '17 *'; with leading spaces folded the two read as one token and the
        # divergence surfaced one step later, after the texts had parted ('23' vs ' ', not a near-tie). Gufo prints this vocabulary's spaces.
        loom = [[("17", -0.0)], [("*", -0.78), (" *", -0.86), (" times", -2.48)], [("23", -0.0003), ("20", -8.45)]]
        keys = ["17", " *", " "]
        tops = [{"17": -0.0}, {" *": -0.81, "*": -0.84, " times": -2.45}, {" ": -0.001, "23": -6.9}]
        self.assertTrue(lc.keeps_spaces([loom]))
        self.assertFalse(lc.keeps_spaces([[[("are", -0.1), ("is", -2.0)]], [[(" ", -0.2)]]]))    # sentencepiece; a whitespace-only token says nothing
        n, probs = lc.check_vs_llama(loom, keys, tops, near_tie)              # folded: the divergence is found at step 2
        self.assertEqual((n, len(probs)), (2, 1))
        self.assertIn("step 2", probs[0])
        self.assertEqual(lc.check_vs_llama(loom, keys, tops, near_tie, keep_space=True), (1, []))   # step 1: a near-tie (0.03 nats)
        # stricter: a ' X' vs 'X' mismatch folding passed silently is now judged; llama.cpp sure of ' *' makes it a problem
        self.assertEqual(lc.check_vs_llama(loom, keys, [tops[0], {" *": -0.01, "*": -5.0}, tops[2]], near_tie)[0], 2)
        n, probs = lc.check_vs_llama(loom, keys, [tops[0], {" *": -0.01, "*": -5.0}, tops[2]], near_tie, keep_space=True)
        self.assertTrue(probs and "step 1" in probs[0])
        # and the two spellings' log-probs are compared each with its own, not the best of both
        n, probs = lc.check_vs_llama([[("A", -0.1), (" B", -1.2), ("B", -6.0)]], ["A"], [{"A": -0.1, " B": -1.25, "B": -1.3}], near_tie, keep_space=True)
        self.assertTrue(probs and "'B'" in probs[0])
        self.assertEqual(lc.check_vs_llama([[("A", -0.1), (" B", -1.2), ("B", -6.0)]], ["A"], [{"A": -0.1, " B": -1.25, "B": -1.3}], near_tie), (1, []))
        self.assertEqual(lc.special_key(" *", True), " *")
        self.assertEqual(lc.special_key(" *"), "*")
        self.assertEqual(lc.best_by_key([(" *", -0.6), ("*", -0.8)], True), {" *": -0.6, "*": -0.8})

    def test_eog_keys(self):
        class Tok:
            def decode(self, ids, skip_special_tokens=False):
                return {1: "<eos>", 106: "<end_of_turn>"}[ids[0]]

        class Pkg:
            chat = {"terminal": "<end_of_turn>"}
            cfg = {"eos_id": 1, "extra_eos_ids": [106]}
            tok = Tok()
        self.assertEqual(lc.eog_keys(Pkg()), {"<eos>", "<end_of_turn>"})

    def test_probe_gaps(self):
        # embed_probe compares the device's token with the row exported from the same logits: rounding makes an exact tie (gap 0), nothing more
        line = '{"opened": true, "width": 1000, "failures": 2}'
        err = ("FAIL x.c:1: step 10 row 1: argmax(row)=5 (12.250) but the device selected 7 (12.250)\n"
               "FAIL x.c:1: step 12 row 0: argmax(row)=5 (12.250) but the device selected 9 (12.220)\n")
        ok, why, gaps, problems, ends = lc.probe_judge(line, err)
        self.assertEqual((ok, why, gaps, problems), (True, "", [(10, 1, 0.0), (12, 0, 0.03)], []))
        self.assertTrue(lc.probe_check(line, err)[0])
        # a gap above the margin is a finding with step/row/gap (every gap is still recorded), not a pass
        far = ("FAIL x.c:1: step 3 row 0: argmax(row)=5 (12.250) but the device selected 7 (12.000)\n"
               "FAIL x.c:1: step 4 row 2: argmax(row)=5 (12.250) but the device selected 7 (12.250)\n")
        ok, why, gaps, problems, ends = lc.probe_judge('{"opened": true, "width": 1000, "failures": 2}', far)
        self.assertFalse(ok)
        self.assertIn("step 3 row 0: device token 0.250 nats below the row maximum", why)
        self.assertEqual(gaps, [(3, 0, 0.25), (4, 2, 0.0)])
        out = "FAIL x.c:1: step 3 row 0: argmax(row)=5 (12.250) but the device selected 1000 (-1.000)\n"
        self.assertIn("outside the row", lc.probe_judge('{"opened": true, "width": 1000, "failures": 1}', out)[1])
        pad = "FAIL x.c:1: step 3 row 0: argmax 151900 is padding (vocabulary 151665)\n"
        self.assertFalse(lc.probe_judge('{"opened": true, "width": 1000, "failures": 1}', pad)[0])
        self.assertFalse(lc.probe_judge('{"opened": true, "width": 1000, "failures": 1}', "")[0])   # no stderr: nothing to judge
        self.assertFalse(lc.probe_judge('{"opened": true, "width": 1000, "failures": 3}', err)[0])  # a failure not accounted for

    def test_probe_end_tokens(self):
        # Qwen: the host's argmax is <|endoftext|> (151643, logit 34), the device reports the chat's terminal <|im_end|> (151645, logit 1.4):
        # both end the generation, whatever their exported logits; a pair with one ordinary token is not that
        line = '{"opened": true, "width": 152000, "failures": 2}'
        err = ("FAIL x.c:127: step 9 row 0: argmax(row)=151643 (34.122) but the device selected 151645 (1.367)\n"
               "FAIL x.c:127: step 10 row 0: argmax(row)=151643 (24.807) but the device selected 151645 (6.312)\n")
        ok, why, gaps, problems, ends = lc.probe_judge(line, err, eog={151643, 151645})
        self.assertEqual((ok, gaps, problems, len(ends)), (True, [], [], 2))
        self.assertEqual(ends[0], (9, 0, 151643, 151645, 34.122, 1.367))
        self.assertFalse(lc.probe_judge(line, err, eog={151643})[0])                      # the device token is not an end token
        self.assertFalse(lc.probe_judge(line, err.replace("151645", "42"), eog={151643, 151645})[0])
        self.assertEqual(lc.eog_ids({"eos_id": 1, "extra_eos_ids": [106], "chat": {}}), {1, 106})

        class Tok:
            def token_to_id(self, t):
                return {"<end_of_turn>": 106}[t]
        self.assertEqual(lc.eog_ids({"eos_id": 1, "chat": {"terminal": "<end_of_turn>"}}, Tok()), {1, 106})

    def test_rejudge_from_saved_probe_output(self):
        out = {"verdict": "FAIL", "level": "dev", "stats": {"steps": 10, "prompts": 2},
               "checks": [{"name": "embed_probe", "ok": False, "why": "x"}, {"name": "llama", "ok": True, "compared": 12}]}
        line = '{"opened": true, "width": 152000, "failures": 1}'
        err = "FAIL x.c:127: step 9 row 0: argmax(row)=151643 (34.122) but the device selected 151645 (1.367)\n"
        new = lc.rejudge(out, line + "\n", err, 1, {151643, 151645})
        self.assertEqual(new["verdict"], "PASS")
        self.assertEqual(new["stats"]["probe_end_pairs"], 1)
        self.assertLessEqual(len(new["gates"]["logits"]["detail"]), lc.DETAIL_MAX)
        self.assertEqual(lc.rejudge(out, line + "\n", err, 1, set())["verdict"], "FAIL")    # without the end tokens it is a finding
        self.assertIn("nats below the row maximum", lc.rejudge(out, line + "\n", err, 1, set())["gates"]["logits"]["detail"] + lc.rejudge(out, line, err, 1, set())["checks"][0]["why"])

    def test_detail_is_one_short_line(self):
        self.assertEqual(lc.clip("a" * 61), "a" * 59 + "…")
        self.assertLessEqual(len(lc.clip("x: " + "why " * 40)), lc.DETAIL_MAX)
        self.assertEqual(lc.clip("argmax = device token"), "argmax = device token")

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
        self.assertEqual(gufo_gate.gate(c, lc, 0), (True, "2/2 conformance, 1/1 logits"))
        self.assertFalse(gufo_gate.gate(c, lc, 1)[0])                                                # a script failed
        self.assertFalse(gufo_gate.gate({"results": c["results"] + [{"check": "g", "status": "gap"}]}, lc, 0)[0])   # a gap is a miss with --logits
        self.assertFalse(gufo_gate.gate(c, {"results": [{"check": "x", "status": "fail"}]}, 0)[0])
        self.assertFalse(gufo_gate.gate(None, lc, 0)[0])
