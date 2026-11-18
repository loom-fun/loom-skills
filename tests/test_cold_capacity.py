import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MODULE = Path(__file__).with_name('cold_capacity.py')
if not MODULE.exists():
    MODULE = Path(__file__).resolve().parents[1] / '.github/scripts/cold_capacity.py'
spec = importlib.util.spec_from_file_location('cold_capacity', MODULE)
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def status():
    return {'data': [{'id': 'test', 'context_length': 16, 'loom': {
        'registration': 'ready', 'device': 'gpu', 'pool': 'pool'}}],
        'server': {'resources': {'pools': [{'key': 'pool', 'memory': {
            'backing': 'fixed', 'storage': {'committed_bytes': 100},
            'workspace': {'committed_bytes': 20}}}]}}}


def answer(n):
    return {'choices': [{'finish_reason': 'length', 'message': {'content': 'yes'}}],
            'usage': {'prompt_tokens': n, 'completion_tokens': 1}}


OVER = {'error': {'type': 'loom_request_error', 'code': 'OUT_OF_RANGE',
                 'message': 'request exceeds the available context size of 16 tokens'}}


class CapacityTests(unittest.TestCase):
    def test_status_requires_selected_context_and_pool(self):
        got = c.status_contract(status(), 'test', {'context_capacity': 16})
        self.assertEqual(got['fixed_storage_total_bytes'], 100)
        self.assertEqual(got['workspace_subset_bytes'], 20)
        self.assertFalse(got['memory_is_peak'])
        for kind in ('context', 'id', 'backing', 'storage', 'duplicate'):
            bad = status()
            if kind == 'context': bad['data'][0]['context_length'] = 15
            if kind == 'id': bad['data'][0]['id'] = 'wrong'
            if kind == 'backing': bad['server']['resources']['pools'][0]['memory']['backing'] = 'elastic'
            if kind == 'storage': bad['server']['resources']['pools'][0]['memory']['storage']['committed_bytes'] = 0
            if kind == 'duplicate': bad['data'].append(copy.deepcopy(bad['data'][0]))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                c.status_contract(bad, 'test', {'context_capacity': 16})

    def test_completion_requires_actual_exact_counts(self):
        c.completed(200, answer(15), 15)
        for code, body in ((503, answer(15)), (200, answer(14)), (200, {}),
                           (200, {'error': 'failed'}), (200, dict(answer(15), usage={})),
                           (200, dict(answer(15), choices=[{'finish_reason': None}]))):
            with self.subTest(code=code, body=body), self.assertRaises(ValueError):
                c.completed(code, body, 15)

    def test_rejection_must_be_context_specific_400(self):
        c.rejected(400, OVER)
        wrong = copy.deepcopy(OVER); wrong['error']['message'] = 'out of memory'
        for code, body in ((200, OVER), (401, OVER), (503, OVER), (400, wrong),
                           (400, {}), (400, 'context exceeds'),
                           (400, {'error': {'message': OVER['error']['message']}})):
            with self.subTest(code=code, body=body), self.assertRaises(ValueError):
                c.rejected(code, body)

    def test_calibration_measures_template_overhead(self):
        calls = []
        def count(s):
            calls.append(s)
            return 7 + len(s) // 2
        prompt = c.calibrate(100, count)
        self.assertEqual(count(prompt), 100)
        self.assertLess(len(calls), 8)
        with self.assertRaises(ValueError):
            c.calibrate(100, lambda s: 3)

    def test_proc_identity_and_kib_units(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); p = root / '123'; p.mkdir()
            fields = ['S', '12', '34'] + ['0'] * 16 + ['567']
            (p / 'stat').write_text('123 (name with spaces) ' + ' '.join(fields))
            (p / 'cmdline').write_bytes(b'/bin/loom-serve\0--model=/pkg/model.json\0')
            (p / 'status').write_text('VmRSS:\t7 kB\nVmHWM: 11 kB\nRssFile: 2 kB\n')
            got = c.process(123, root)
            self.assertEqual((got['ppid'], got['pgrp'], got['start_ticks']), (12, 34, 567))
            self.assertEqual(got['memory']['VmHWM'], 11 * 1024)
            self.assertEqual(got['argv'][-1], '--model=/pkg/model.json')

    def test_probe_request_sequence_and_failed_overflow(self):
        for wrong_overflow in (False, True):
            calls = []
            class FakeRequests:
                def __init__(self, *args): pass
                def call(self, name, path, payload=None):
                    calls.append((name, payload))
                    if path == '/v1/status': return 200, status()
                    content = payload['messages'][0]['content']
                    n = 2 + len(content) // 2
                    if path.endswith('input_tokens'): return 200, {'input_tokens': n}
                    if name == 'over_limit':
                        self_outer.assertEqual(n, 16)
                        self_outer.assertEqual(payload['max_tokens'], 2)
                        return (503 if wrong_overflow else 400), OVER
                    return 200, answer(n)
            self_outer = self
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); pkg = root / 'pkg'; pkg.mkdir(); out = root / 'out'
                c.write(pkg / 'model.json', {'model': {'id': 'test'}, 'presets': {
                    'chat': {'execution': {'context_capacity': 16}}}})
                args = SimpleNamespace(package=str(pkg), url='http://test', out=str(out), preset=None, timeout=1)
                with patch.object(c, 'Requests', FakeRequests):
                    code = c.probe(args)
                self.assertEqual(code, int(wrong_overflow))
                summary = json.loads((out / 'summary.json').read_text())
                self.assertEqual(summary['passed'], not wrong_overflow)
                names = [n for n, _ in calls]
                self.assertIn('logical_full_limit', names)
                self.assertIn('sdk_stored_full_limit', names)
                self.assertEqual('health_after' in names, not wrong_overflow)


if __name__ == '__main__':
    unittest.main()
