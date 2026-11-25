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

    def test_repeat_rows_average_all_successful_per_pass_samples(self):
        # The short essay and three context repeats have separate request counters.
        import contextlib
        import io
        import json
        import runpy
        import tempfile
        from unittest.mock import patch
        import bench_common
        stream_result = {"status": 200, "prompt_tokens": 50, "ttft": 1, "decode_tps": 4, "tokens": 256}
        before = {"decode_tokens": 10, "draft_accepted_inputs": 0}
        snapshots = [value for accepted in (2, 1, 2, 3) for value in
                     (before, {"decode_tokens": 12 + accepted, "draft_accepted_inputs": accepted})]
        with tempfile.TemporaryDirectory() as tmp:
            script = ROOT / "skills/loom-benchmark/scripts/perf.py"
            log = Path(tmp) / "perf.jsonl"
            args = [str(script), "unused", "test", "--no-campaign", "--log", str(log), "--repeat-rows", "3"]
            with patch.object(sys, "argv", args), patch.object(bench_common, "stream", return_value=stream_result), \
                    patch.object(bench_common, "server_draft_counters", side_effect=snapshots), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(script), run_name="__main__")
            row = json.loads(log.read_text())
            self.assertEqual(row["accepted_per_pass"]["decode_tps_2k"], 1.0)

    def test_no_rows_no_value(self):
        self.assertIsNone(make_info.session_per_pass([{"label": "x", "decode_tps": 10.0}]))


if __name__ == '__main__':
    unittest.main()
