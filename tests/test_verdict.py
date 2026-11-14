import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fleet"))
import promote_check as P  # noqa: E402
import sample_lib as L  # noqa: E402
import verdict as V  # noqa: E402

METH = L.load_methodology("/nonexistent/methodology.json")    # a missing file: the calibrated defaults


def cmp(ratios, sessions=None, gates=None, text="MATCH"):
    m = {}
    for r, x in ratios.items():
        e = {"old": 100.0, "new": round(100.0 * x, 1), "ratio": x}
        if sessions and r in sessions:
            e["old_sessions"], e["new_sessions"] = sessions[r]
        m[r] = e
    return {"metrics": m, "gate_status": gates or {}, "text": {"status": text}}


class Rows(unittest.TestCase):
    def v(self, row, ratio, slug="model", **kw):
        return V.model_verdict(cmp({row: ratio}, **kw), slug, None, METH)["rows"][row]["verdict"]

    def test_bands_are_per_row(self):
        self.assertEqual(self.v("prefill_8k", 0.975), "regressed")      # band 0.02
        self.assertEqual(self.v("prefill_1k", 0.975), "pass")           # band 0.03
        self.assertEqual(self.v("decode_2k", 0.965), "pass")            # band 0.04
        self.assertEqual(self.v("decode", 0.96), "regressed")
        self.assertEqual(self.v("decode", 1.05), "improved")

    def test_reject_threshold_and_extension_zone(self):
        self.assertEqual(self.v("decode_32k", 0.76), "reject")
        e = V.model_verdict(cmp({"decode_32k": 0.93}), "m", None, METH)["rows"]["decode_32k"]
        self.assertEqual(e["verdict"], "regressed")
        self.assertTrue(e["extend"])                                    # within 0.02 of the 0.92 reject threshold
        self.assertFalse(V.model_verdict(cmp({"decode_32k": 0.96}), "m", None, METH)["rows"]["decode_32k"]["extend"])

    def test_no_geomean_trading(self):
        r = V.model_verdict(cmp({"decode": 1.11, "decode_32k": 0.78}), "m", None, METH)
        self.assertEqual(r["verdict"], "reject")                        # a gain on one row never pays for a loss on another

    def test_worst_and_order(self):
        self.assertEqual(V.worst(["pass", "improved", "regressed"]), "regressed")
        self.assertEqual(V.worst(["improved", "pass"]), "pass")
        self.assertEqual(V.worst(["flag", "pass", "gate_fail"]), "gate_fail")
        self.assertEqual(V.worst([]), "pass")

    def test_gates_and_text(self):
        r = V.model_verdict(cmp({"decode": 1.0}, gates={"ktest:staging": "FAIL"}), "m", None, METH)
        self.assertEqual(r["verdict"], "gate_fail")
        self.assertEqual(V.model_verdict(cmp({"decode": 1.0}, text="DIFF"), "m", None, METH)["verdict"], "text_diff")

    def test_row_filter_and_missing(self):
        r = V.model_verdict(cmp({"decode": 0.8, "prefill_8k": 1.0}), "m", ["prefill_8k"], METH)
        self.assertEqual(list(r["rows"]), ["prefill_8k"])
        self.assertEqual(r["verdict"], "pass")
        c = cmp({"decode": 1.0})
        c["metrics"]["decode"]["ratio"] = None
        self.assertEqual(V.model_verdict(c, "m", None, METH)["rows"]["decode"]["verdict"], "missing")


class Bimodal(unittest.TestCase):
    def slow_fast(self, old, new, slug="unsloth-Qwen3-0.6B-loom"):
        ratio = round((sorted(new)[len(new) // 2]) / (sorted(old)[len(old) // 2]), 4)
        c = cmp({"decode": ratio}, sessions={"decode": (old, new)})
        return V.model_verdict(c, slug, None, METH)["rows"]["decode"]

    def test_modes_split(self):
        self.assertEqual(len(V.modes([297, 298, 330, 331])), 2)
        self.assertEqual(len(V.modes([300, 301, 302])), 1)

    def test_like_with_like(self):
        # both sides have a slow and a fast mode with 2 sessions each, no change within a mode: pass
        e = self.slow_fast([297, 298, 330, 331], [297, 299, 331, 330])
        self.assertEqual(e["verdict"], "pass")
        # both modes 8 % slower: caught per mode
        e = self.slow_fast([297, 298, 330, 331], [273, 274, 304, 305])
        self.assertIn(e["verdict"], ("regressed", "reject"))
        # the fast mode collapsed into the slow one: a change in mode frequency, flagged
        self.assertEqual(self.slow_fast([297, 298, 330, 331], [297, 298, 303, 304])["verdict"], "flag")

    def test_mode_frequency_change_is_flagged_not_passed(self):
        e = self.slow_fast([297, 298, 330, 331], [297, 298, 297, 299])      # new side lost the fast mode
        self.assertEqual(e["verdict"], "flag")
        e = self.slow_fast([297, 298, 330, 331], [297, 330, 331, 332])      # slow mode has one session only
        self.assertEqual(e["verdict"], "flag")

    def test_unimodal_sides_with_a_mode_sized_gap_are_flagged(self):
        e = self.slow_fast([297, 298, 299], [330, 331, 332])
        self.assertEqual(e["verdict"], "flag")
        e = self.slow_fast([297, 298, 299], [297, 298, 299])
        self.assertEqual(e["verdict"], "pass")

    def test_other_models_use_plain_bands(self):
        e = self.slow_fast([297, 298, 299], [330, 331, 332], slug="unsloth-Qwen3-8B-loom")
        self.assertEqual(e["verdict"], "improved")


class Ratchet(unittest.TestCase):
    def chk(self, *runs, **kw):
        return P.check("unsloth-Qwen3-8B-loom", "strix-halo", list(runs), None, METH, **kw)

    def test_promote_when_every_row_inside_the_band(self):
        self.assertEqual(self.chk(cmp({"decode": 1.01, "prefill_8k": 0.99}))["decision"], "promote")

    def test_single_reject_holds_at_once_single_regression_asks_for_a_rerun(self):
        self.assertEqual(self.chk(cmp({"decode_32k": 0.75}))["decision"], "hold")
        self.assertEqual(self.chk(cmp({"prefill_8k": 0.96}))["decision"], "rerun")

    def test_confirmed_regression_holds_noise_does_not(self):
        self.assertEqual(self.chk(cmp({"prefill_8k": 0.96}), cmp({"prefill_8k": 0.965}))["decision"], "hold")
        self.assertEqual(self.chk(cmp({"prefill_8k": 0.96}), cmp({"prefill_8k": 1.0}))["decision"], "promote")

    def test_improvement_needs_two_runs(self):
        r = self.chk(cmp({"prefill_8k": 1.05}))
        self.assertEqual(r["decision"], "improved-pending")
        self.assertEqual(self.chk(cmp({"prefill_8k": 1.05}), cmp({"prefill_8k": 1.04}))["decision"], "promote")

    def test_a_gain_does_not_pay_for_a_loss(self):
        self.assertEqual(self.chk(cmp({"decode": 1.2, "decode_32k": 0.8}))["decision"], "hold")

    def test_gate_failure_and_text(self):
        self.assertEqual(self.chk(cmp({"decode": 1.0}, gates={"gates:staging:dev": "FAIL"}))["decision"], "hold")
        self.assertEqual(self.chk(cmp({"decode": 1.0}, text="DIFF"))["decision"], "hold")
        self.assertEqual(self.chk(cmp({"decode": 1.0}, text="DIFF"), near_tie=True)["decision"], "promote")

    def test_row_restriction(self):
        c = cmp({"decode": 0.8, "prefill_8k": 1.0})
        self.assertEqual(P.check("m", "strix-halo", [c], None, METH, rows_only=["prefill_8k"])["decision"], "promote")

    def test_baseline_drift_is_informational(self):
        base = {"models": {"unsloth-Qwen3-8B-loom": {"strix-halo": {"decode": {"value": 130.0, "run": "r"}}}}}
        c = cmp({"decode": 1.0})            # new = 100 tok/s against a published 130: 23 % below the stored baseline
        r = P.check("unsloth-Qwen3-8B-loom", "strix-halo", [c], base, METH)
        self.assertEqual(r["decision"], "promote")
        self.assertEqual(r["baseline_drift"][0]["row"], "decode")

    def test_methodology_file_overrides_defaults(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"rows": {"decode": {"band": 0.10, "reject": 0.80}}, "bimodal_models": ["Foo-1B"]}, f)
        m = L.load_methodology(f.name)
        os.unlink(f.name)
        self.assertEqual(m["bands"]["decode"], (0.10, 0.80))
        self.assertEqual(m["bands"]["prefill_8k"], L.DEFAULT_BANDS["prefill_8k"])
        self.assertTrue(L.is_bimodal("x-Foo-1B-loom", m))
        self.assertFalse(L.is_bimodal("unsloth-Qwen3-0.6B-loom", m))


if __name__ == "__main__":
    unittest.main()
