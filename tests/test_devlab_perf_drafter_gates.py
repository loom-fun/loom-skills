"""devlab_perf.sh: a rebuilt variant takes the drafting gates (drafter ctx_max / occ_min) from the checkout's fleet/<slug>/llm_config.json."""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def snippet():
    s = open(os.path.join(ROOT, ".github", "scripts", "devlab_perf.sh")).read()
    m = re.search(r'"\$vroot/fleet/\$\{repo##\*/\}/llm_config\.json" <<\'EOF2\'\n(.*?)\nEOF2\n', s, re.S)
    return m.group(1)


class DrafterGates(unittest.TestCase):
    def run_it(self, card, fleet, override=None):
        d = tempfile.mkdtemp()
        a, b, f = (os.path.join(d, n) for n in ("cfg.json", "out.json", "fleet.json"))
        json.dump(card, open(a, "w"))
        if fleet is not None:
            json.dump(fleet, open(f, "w"))
        subprocess.run([sys.executable, "-c", snippet(), a, b, json.dumps(override or {}), f], check=True, capture_output=True)
        return json.load(open(b))

    def test_the_checkouts_gate_wins(self):
        out = self.run_it({"drafter": {"ctx_max": 3500, "occ_min": 2, "depth": 3}}, {"drafter": {"ctx_max": 0, "occ_min": 2}})
        self.assertEqual(out["drafter"], {"ctx_max": 0, "occ_min": 2, "depth": 3})

    def test_explicit_variant_gate_wins_and_unspecified_gate_uses_checkout(self):
        out = self.run_it(
            {"drafter": {"ctx_max": 3500, "occ_min": 4}},
            {"drafter": {"ctx_max": 2048, "occ_min": 2}},
            {"drafter": {"ctx_max": 0}},
        )
        self.assertEqual(out["drafter"], {"ctx_max": 0, "occ_min": 2})

    def test_no_fleet_config_or_no_drafter_keeps_the_card(self):
        self.assertEqual(self.run_it({"drafter": {"ctx_max": 3500}}, None)["drafter"], {"ctx_max": 3500})
        self.assertNotIn("drafter", self.run_it({"x": 1}, {"drafter": {"ctx_max": 0}}))

    def test_variant_config_override_still_applies(self):
        self.assertEqual(self.run_it({"layers": 40}, None, {"layers": 12})["layers"], 12)


if __name__ == "__main__":
    unittest.main()
