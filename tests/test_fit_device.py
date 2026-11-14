import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "fleet"))
import fit_device  # noqa: E402
import sample_lib  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-kb"))
import registry  # noqa: E402


class FitDevice(unittest.TestCase):
    def rows(self):
        return json.load(open(os.path.join(ROOT, ".github", "devlab", "r9700_fit.json")))

    def test_reproduces_the_r9700_table(self):   # the generic formula gives the verdicts the R9700 table holds
        rows = self.rows()
        got = {v["slug"]: v["verdict"] for v in fit_device.verdicts("radeon-r9700", rows)}
        self.assertEqual(got, {r["slug"]: r["verdict"] for r in rows})

    def test_smaller_device_fits_less(self):
        rows = self.rows()
        n = lambda dev: sum(v["verdict"].startswith("fits") for v in fit_device.verdicts(dev, rows))
        self.assertLess(n("radeon-rx7900xtx"), n("radeon-r9700"))
        self.assertEqual(n("strix-halo"), len(rows))   # no memory_bytes: the package's own

    def test_targets_follow_the_registry(self):
        pinned = {"strix-halo": "gfx11", "radeon-rx7900xtx": "gfx11", "radeon-r9700": "gfx12"}
        self.assertEqual({k: v for k, v in sample_lib.GPUS.items() if k in pinned}, pinned)   # existing rows keep their target; new devices add rows
        self.assertEqual(list(sample_lib.GPUS), list(registry.devices()))


if __name__ == "__main__":
    unittest.main()
