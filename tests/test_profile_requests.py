"""The diagnostic profiler must reproduce the literal card prompt and retain stall evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'skills/loom-benchmark/scripts/profile_requests.py'
ESSAY = 'Write a detailed 300-word essay about the history of the printing press.'


class ProfileRequests(unittest.TestCase):
    def run_profile(self, *args):
        calls = []

        def stream(url, model, messages, tokens, **kwargs):
            calls.append((messages[0]['content'], tokens, kwargs['timeout']))
            if kwargs.get('on_token'):
                for arrival in (1.0, 1.02, 4.02):
                    kwargs['on_token'](arrival)
            return dict(status=200, prompt_tokens=32, tokens=tokens, ttft=.2,
                        decode_tps=40, total=8, text='fixture')

        counters = iter(({'draft_proposed_tokens': 3}, {'draft_proposed_tokens': 9}))
        bench = types.SimpleNamespace(doc=lambda n: 'doc', stream=stream,
                                      thinking_extra=lambda mode: {},
                                      server_draft_counters=lambda url: next(counters))
        output = io.StringIO()
        with patch.dict(sys.modules, {'bench_common': bench}), patch.object(sys, 'argv',
                [str(SCRIPT), 'http://fixture', 'model', 'dec', *args]), \
                patch.dict(os.environ, {k: '' for k in ('LOOM_PROFILE_LITERAL_DECODE',
                    'LOOM_PROFILE_TRACE_GAPS', 'LOOM_PROFILE_PROCESS_IO')}, clear=False), \
                contextlib.redirect_stdout(output):
            runpy.run_path(str(SCRIPT), run_name='__main__')
        return calls, json.loads(output.getvalue())

    def test_literal_reproduces_perf_request_and_records_stream_stall(self):
        calls, result = self.run_profile('--literal-decode', '--tokens', '300', '--trace-gaps', '--timeout', '120')
        self.assertTrue(calls[0][0].startswith('[warm-up '))
        self.assertEqual(calls[1], (ESSAY, 300, 120.0))
        self.assertEqual(result['prompt_mode'], 'literal-perf')
        self.assertEqual(result['requests'][0]['max_stream_chunk_gap_s'], 3.0)
        self.assertEqual(result['requests'][0]['stream_chunk_gaps_s'], [.02, 3.0])
        self.assertEqual(result['draft']['draft_proposed_tokens'], 6)

    def test_default_keeps_nonced_profile_prompt(self):
        calls, result = self.run_profile()
        self.assertTrue(calls[1][0].endswith(ESSAY))
        self.assertNotEqual(calls[1][0], ESSAY)
        self.assertEqual(calls[1][2], 300.0)
        self.assertNotIn('stream_chunk_gaps_s', result['requests'][0])


if __name__ == '__main__':
    unittest.main()
