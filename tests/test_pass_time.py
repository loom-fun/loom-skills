"""Drafted decode rows normalised per verification pass (pass_time.py) and the per-kernel profile difference (profile_diff.py)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills/loom-benchmark/scripts'
sys.path.insert(0, str(SCRIPTS))
import pass_time  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data/pass_time'   # a same-job ABBA: one package on hrx-loom 0.2.0 (fun) and 0.3.0 (staging)


class PassTime(unittest.TestCase):
    def test_acceptance_moves_tps_not_pass_time(self):
        r = pass_time.compare(str(DATA))
        d2 = r['decode_tps_2k']
        self.assertLess(d2['tps_ratio'], 0.96)                        # -5.2 % tok/s ...
        self.assertAlmostEqual(d2['ms_per_pass_ratio'], 1.0, delta=0.005)   # ... at the same cost per verification pass
        self.assertLess(d2['tokens_per_pass_ratio'], 0.95)

    def test_both_counter_kinds(self):
        r = pass_time.compare(str(DATA))['decode_tps']
        self.assertEqual(r['new'][0]['passes'], 300 - 182)           # 0.3.0: tokens - accepted drafted tokens
        self.assertAlmostEqual(r['old'][0]['passes'], 300 / (1 + 0.523 * 3), places=0)   # 0.2.0: acceptance x depth


class ProfileDiff(unittest.TestCase):
    def test_per_pass_from_draft_counters(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            prof = lambda attn: {"scenario": "dec2k", "tokens": 256, "busy_ms": 2000 + attn, "steps": None,
                                 "families": {"attn": {"ms": attn}, "gemv": {"ms": 2000}}, "kernels": {"a": {"count": 9, "ms": attn}, "g": {"count": 9, "ms": 2000}}}
            req = lambda total: {"scenario": "dec2k", "requests": [{"total_s": total, "ttft_s": 1.0}], "draft": {"decode_tokens": 256, "draft_accepted_inputs": 156}}
            for name, obj in (("o.json", prof(1000)), ("n.json", prof(1300)), ("o.req", req(4.0)), ("n.req", req(4.3))):
                (d / name).write_text(json.dumps(obj))
            out = subprocess.run([sys.executable, str(SCRIPTS / 'profile_diff.py'), str(d / 'o.json'), str(d / 'n.json'),
                                  '--old-req', str(d / 'o.req'), '--new-req', str(d / 'n.req')], capture_output=True, text=True, check=True).stdout
        self.assertIn("'passes': 100", out)
        self.assertIn('busy per pass +10.0%', out)
        self.assertIn('wall per pass +10.0%', out)   # the decode part only: total minus the time to the first token
        self.assertIn('| attn | 10.0000 | 13.0000 | +3.0000 | +30.0 |', out)


if __name__ == '__main__':
    unittest.main()
