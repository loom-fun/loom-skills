"""The 0.3.0 drafted-token acceptance of a card (accepted tokens per verification pass, no proposal count) comes from the perf session's
decode requests, the same workload and aggregation as the earlier card's share of proposals, not from the server's cumulative /metrics."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/loom-benchmark/scripts/card'))
sys.path.insert(0, str(ROOT / 'skills/loom-benchmark/scripts'))
import make_info  # noqa: E402

# Qwen3.5-35B-A3B frozen production programs on hrx-loom 0.3.0, R9700 (the sweep's same-job ABBA, staging side): its 0.2.0 card has a
# session acceptance of 0.405 at depth 3 = 1.215 accepted tokens per pass; the 0.2.0 side of the same job measured 1.208.
DATA = Path(__file__).resolve().parent / 'data/acceptance_per_pass'


class AcceptancePerPass(unittest.TestCase):
    def test_decode_requests_match_the_earlier_card(self):
        rows = make_info.read_rows(str(DATA / 'perf_staging.jsonl'))
        self.assertAlmostEqual(make_info.session_per_pass(rows), 1.215, delta=0.02)

    def test_cumulative_metrics_undercount(self):
        # the /metrics dumps also count prefill rows, warm-ups and undrafted long decode as passes: 37 % lower, a false regression
        dumps = sorted(str(p) for p in DATA.glob('metrics_staging_*.txt'))
        self.assertLess(make_info.draft_metrics_acceptance(dumps)['accepted_per_pass'], 0.8)

    def test_no_rows_no_value(self):
        self.assertIsNone(make_info.session_per_pass([{"label": "x", "decode_tps": 10.0}]))


if __name__ == '__main__':
    unittest.main()
