import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "fleet"))
import sdk_delta_report as rep  # noqa: E402


def doc():
    return {"schema": "sdk-evidence/1", "generated": "t", "sdk": {"old": "0.2.0", "new": "0.3.0"}, "harness": {"commit": "a" * 40},
            "held": {"Held-loom": {"reason": "owner decision"}},
            "models": {
                "A-loom": {"kind": "text", "devices": {
                    "strix-halo": {"status": "qualified", "job_url": "https://x/job/1",
                                   "delta_pct": {"decode": 1.0, "prefill_8k": -1.0, "decode_32k": -2.6, "streams4": 0.5},
                                   "spec_acceptance": {"old": 0.35, "new": 0.34}, "regressions": []},
                    "radeon-r9700": {"status": "qualified", "job_url": "https://x/job/2",
                                     "delta_pct": {"decode": 1.0, "prefill_8k": -5.0, "decode_32k": -6.0, "streams4": 0.5},
                                     "regressions": [{"row": "prefill_8k", "pct": -5.0, "confirmed_abba_pct": -1.0,
                                                      "explanation": {"text": "ABBA within band", "url": "https://x/job/3"}},
                                                     {"row": "decode_32k", "pct": -6.0, "confirmed_abba_pct": None, "explanation": None}]},
                    "gfx1150": {"status": "no_runner", "reason": "no DevLab runner"}}},
                "Img-loom": {"kind": "image", "devices": {"strix-halo": {"status": "qualified", "image": {"pixels_match": True}}}}}}


class Report(unittest.TestCase):
    def test_verdict_and_band(self):
        rows = {(r["slug"], r["device"]): r for r in rep.collect(doc())}
        s = rows[("A-loom", "strix-halo")]
        self.assertEqual((s["ok"], s["n"], s["verdict"]), (4, 4, "pass"))   # -2.6 is inside the band
        r = rows[("A-loom", "radeon-r9700")]
        self.assertEqual((r["ok"], r["n"], r["verdict"]), (2, 4, "FAIL"))
        self.assertEqual(rows[("Img-loom", "strix-halo")]["verdict"], "pixels identical")

    def test_unexplained_regression_is_reported(self):
        text = rep.render(doc())
        self.assertIn("**UNEXPLAINED**", text)
        self.assertIn("[ABBA within band](https://x/job/3)", text)
        self.assertIn("no DevLab runner", text)
        self.assertIn("owner decision", text)
        bad = rep.problems(rep.collect(doc()))
        self.assertTrue(any("decode_32k" in b and "no explanation" in b for b in bad))
        self.assertTrue(any("two thirds" in b for b in bad))

    def test_clean_document_passes_strict(self):
        d = doc()
        del d["models"]["A-loom"]["devices"]["radeon-r9700"]
        self.assertEqual(rep.problems(rep.collect(d)), [])
        self.assertIn("all regressions explained", rep.render(d))

    def test_band_is_a_parameter(self):
        rows = rep.collect(doc(), band=1.0)
        s = [r for r in rows if r["device"] == "strix-halo" and r["slug"] == "A-loom"][0]
        self.assertEqual([g["row"] for g in s["regressions"]], ["decode_32k"])   # -2.6 > 1 band, -1.0 is not over it


if __name__ == "__main__":
    unittest.main()
