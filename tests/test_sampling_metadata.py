"""Published metadata failures cannot silently reduce DevLab sampling."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/loom-benchmark/scripts/card'))
sys.path.insert(0, str(ROOT / '.github/scripts'))
import card_lib
import ab_jobs
import device_jobs

SLUG = 'bartowski-Qwen2.5-0.5B-Instruct-loom'


def metadata(rate):
    return {'schema': card_lib.SCHEMA_ID, 'devices': {'strix-halo': {'performance': {'sessions': [{'metrics': {'decode': {'loom': rate}}}]}}}}


class SamplingMetadata(unittest.TestCase):
    def load(self, value):
        with mock.patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(value).encode())):
            return card_lib.published_decode(SLUG)

    def test_transport_and_json_failure_stop_before_plan_output(self):
        for failure in (URLError('offline'), HTTPError('url', 503, 'unavailable', {}, None), TimeoutError('timed out')):
            for main, args in ((ab_jobs.main, ['--device', 'strix-halo', '--preset', 'iterate', '--sequential', SLUG]),
                               (device_jobs.main, ['--device', 'radeon-r9700', '--preset', 'iterate', SLUG])):
                with self.subTest(failure=failure, generator=main.__module__):
                    output = io.StringIO()
                    with mock.patch('urllib.request.urlopen', side_effect=failure), contextlib.redirect_stdout(output):
                        with self.assertRaises(SystemExit) as caught:
                            main(args)
                    self.assertIn('refusing to infer normal-speed', str(caught.exception))
                    self.assertIn('huggingface.co/loom-fun/', str(caught.exception))
                    self.assertEqual(output.getvalue(), '')
        with mock.patch('urllib.request.urlopen', return_value=io.BytesIO(b'not json')):
            with self.assertRaises(card_lib.SamplingMetadataError):
                card_lib.published_decode(SLUG)

    def test_invalid_schema_and_numbers_fail_closed(self):
        for value in ([], {}, {'error': 'not a card'},
                      {'schema': card_lib.SCHEMA_ID, 'performance': []},
                      {'schema': card_lib.SCHEMA_ID, 'devices': None},
                      {'schema': card_lib.SCHEMA_ID, 'performance': {'sessions': {}}},
                      {'schema': card_lib.SCHEMA_ID, 'performance': {'sessions': [None]}},
                      *(metadata(rate) for rate in (True, 0, -1, '300', float('nan'), float('inf'), 10**400))):
            with self.subTest(value=value):
                with self.assertRaises(card_lib.SamplingMetadataError):
                    self.load(value)

    def test_genuine_absent_row_is_distinct_from_fetch_failure(self):
        for value in ({'schema': card_lib.SCHEMA_ID, 'devices': {}},
                      {'schema': card_lib.SCHEMA_ID, 'devices': {'strix-halo': {'performance': {'sessions': []}}}},
                      {'schema': card_lib.SCHEMA_ID, 'performance': {'sessions': [{'metrics': {'prefill_1k': {'loom': 1000}}}]}}):
            with self.subTest(value=value):
                self.assertIsNone(self.load(value))

    def test_legacy_and_current_rates_preserve_fast_classification(self):
        current = metadata(300.0)
        for value in (current, {'schema': card_lib.LEGACY_SCHEMA_IDS[0], 'performance': current['devices']['strix-halo']['performance']}):
            self.assertEqual(self.load(value), 300.0)
            self.assertEqual(card_lib.sampling_plan(self.load(value)), (6, 3))
        self.assertEqual(card_lib.sampling_plan(self.load(metadata(60.0))), (3, 1))

    def test_fast_generator_control_keeps_six_sessions_three_samples(self):
        output = io.StringIO()
        with mock.patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(metadata(300.0)).encode())), \
                contextlib.redirect_stdout(output):
            ab_jobs.main(['--device', 'strix-halo', '--preset', 'iterate', '--sequential', SLUG])
        job, = json.loads(output.getvalue())['jobs']
        self.assertEqual(job['repeat_rows'], 3)
        self.assertEqual(job['steps'].count('perf:fun:1'), 6)
        self.assertFalse(any(s.startswith('extend:') for s in job['steps']))
