import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fleet"))
import baselines  # noqa: E402

CARDS = ROOT / "skills/loom-benchmark/scripts/card/test"


class Baselines(unittest.TestCase):
    def test_strix_from_a_loom_card_2_and_two_gpu_card(self):
        with tempfile.TemporaryDirectory() as t:
            for d, src in (("s/m1", CARDS / "legacy/flint/info.json"), ("r/m1", CARDS / "r9700_two/expected/info.json")):
                Path(t, d).mkdir(parents=True)
                shutil.copy(src, Path(t, d, "info.json"))
            doc = baselines.build({"strix-halo": f"{t}/s", "radeon-r9700": f"{t}/r"})
            m = doc["m1"]
            self.assertIn("decode", m["strix-halo"])
            e = m["strix-halo"]["decode"]
            self.assertGreater(e["value"], 0)
            self.assertTrue(e["run"])
            self.assertNotIn("spec_acceptance", m["strix-halo"])
            self.assertTrue(m["radeon-r9700"])       # rows or a state entry, never missing

    def test_ratchet_reads_the_store(self):
        sys.path.insert(0, str(ROOT / "fleet"))
        import sample_plan as SP
        with tempfile.TemporaryDirectory() as t:
            p = Path(t, "b.json")
            p.write_text(json.dumps({"models": {"x": {"strix-halo": {"decode": {"value": 1}, "_state": "x"}, "radeon-r9700": {"_state": "unsupported"}}}}))
            f = SP.Fleet(None, str(Path(t, "none.json")), str(p))
            self.assertEqual(f.rows_with_baseline("x", "strix-halo"), {"decode"})
            self.assertEqual(f.rows_with_baseline("x", "radeon-r9700"), set())
            self.assertIsNone(f.rows_with_baseline("y", "strix-halo"))


if __name__ == "__main__":
    unittest.main()
