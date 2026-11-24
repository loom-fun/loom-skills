"""The diagnostic profiler must reproduce the literal card prompt and retain stall evidence."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'skills/loom-benchmark/scripts/profile_requests.py'
ESSAY = 'Write a detailed 300-word essay about the history of the printing press.'


class ProfileRequests(unittest.TestCase):
    def run_profile(self, *args, scenario='dec'):
        calls = []

        def stream(url, model, messages, tokens, **kwargs):
            calls.append((messages[0]['content'], tokens, kwargs['timeout']))
            if kwargs.get('on_token'):
                for arrival in (1.0, 1.02, 4.02):
                    kwargs['on_token'](arrival)
            return dict(status=200, prompt_tokens=33406 if tokens == 128 else 32, tokens=tokens, ttft=.2,
                        decode_tps=40, total=8, text='fixture', finish='length')

        counters = iter({'draft_proposed_tokens': value} for value in (3, 9, 15))
        bench = types.SimpleNamespace(doc=lambda n: 'doc', stream=stream,
                                      thinking_extra=lambda mode: {},
                                      long_prompt=lambda metric, nonce: f'{metric}:{nonce}',
                                      server_draft_counters=lambda url: next(counters))
        fields = dict(completed_epochs=3, prefill_tokens=100, decode_tokens=8, draft_accepted_inputs=2)
        replay_metrics = iter((fields, dict(completed_epochs=102, prefill_tokens=33506,
                                          decode_tokens=136, draft_accepted_inputs=64)))

        def metrics(url, *, headers, timeout):
            self.assertEqual((url, headers, timeout), ('http://fixture/metrics', {'X-Loom-Model': 'model'}, 15))
            text = '\n'.join(f'loom_text{{field="{name}"}} {value}' for name, value in next(replay_metrics).items())
            return types.SimpleNamespace(text=text, raise_for_status=lambda: None)

        output = io.StringIO()
        modules = {'bench_common': bench, 'requests': types.SimpleNamespace(get=metrics)}
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv',
                [str(SCRIPT), 'http://fixture', 'model', scenario, *args]), \
                patch.dict(os.environ, {k: '' for k in ('LOOM_PROFILE_LITERAL_DECODE',
                    'LOOM_PROFILE_TRACE_GAPS', 'LOOM_PROFILE_PROCESS_IO', 'LOOM_PROFILE_PERF_WARMUP')}, clear=False), \
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
        self.assertEqual(result['requests'][0]['stream_chunk_arrivals_monotonic_ns'], [1000000000, 1020000000, 4020000000])
        self.assertEqual(result['draft']['draft_proposed_tokens'], 6)

    def test_perf_warmup_matches_essay_length_and_counts_its_work(self):
        calls, result = self.run_profile('--perf-warmup', '--literal-decode', '--tokens', '300')
        self.assertEqual(calls[0][1], 300)
        self.assertTrue(calls[0][0].endswith(ESSAY))
        self.assertNotEqual(calls[0][0], ESSAY)
        self.assertEqual(calls[1][0], ESSAY)
        self.assertEqual(result['warmup_mode'], 'perf-essay')
        self.assertEqual(result['warmup_draft']['draft_proposed_tokens'], 6)

    def test_default_keeps_nonced_profile_prompt(self):
        calls, result = self.run_profile()
        self.assertTrue(calls[1][0].endswith(ESSAY))
        self.assertNotEqual(calls[1][0], ESSAY)
        self.assertEqual(calls[1][2], 300.0)
        self.assertNotIn('stream_chunk_gaps_s', result['requests'][0])

    def test_32k_replay_keeps_paired_nonce_and_exact_epoch_deltas(self):
        nonce = '1795473830.4460852'
        calls, result = self.run_profile(scenario=f'replay32k-{nonce}')
        self.assertEqual(calls[0], (f'[warm-up {nonce}] {ESSAY}', 300, 300.0))
        self.assertEqual(calls[1], (f'decode_tps_32k:{nonce}', 128, 300.0))
        self.assertEqual(result['nonce'], nonce)
        request, = result['requests']
        self.assertEqual(request['prompt_sha256'], hashlib.sha256(calls[1][0].encode()).hexdigest())
        self.assertEqual(request['reply_sha256'], hashlib.sha256(b'fixture').hexdigest())
        self.assertEqual(request['counters'], dict(completed_epochs=99, prefill_tokens=33406,
                                                  decode_tokens=128, draft_accepted_inputs=62))


@unittest.skipIf(os.name == 'nt', 'DevLab Bash runs on Linux')
class CaptureFailure(unittest.TestCase):
    def test_server_start_failure_remains_an_error_instead_of_zero_captures(self):
        source = (SCRIPT.parents[3] / '.github/scripts/devlab_perf.sh').read_text()
        block = source[source.index('    profile|trace|observe)'):source.index('    probe)')]
        with tempfile.TemporaryDirectory() as work:
            code = ('declare -A res; kind=trace; p1=test; p2=dec; s=trace:test:dec; '
                    f'out={work}; serve_alone() {{ return 1; }}; '
                    'case "$kind" in\n' + block + '\nesac\nprintf "%s" "${res[$s]}"')
            result = subprocess.run(['bash', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'ERROR: diagnostic request failed')


if __name__ == '__main__':
    unittest.main()
