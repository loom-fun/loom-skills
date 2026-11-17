"""Feature metadata must not pretend to be GPU measurement evidence."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / 'skills/loom-benchmark/scripts/card'
sys.path.insert(0, str(CARD))
try:
    import make_info
    import render_card
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f'needs {error.name}')


class FeatureEvidenceTests(unittest.TestCase):
    def build(self, evidence):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); package = root / 'pkg'
            shutil.copytree(CARD / 'test/flint/pkg', package)
            path = root / 'evidence.json'
            path.write_text(json.dumps({'evidence': 'loom-card-evidence/2', **evidence}))
            make_info.main(['build', str(package), '--repo', 'loom-fun-staging/ggml-org-Flint-loom',
                            '--gpus', 'strix-halo,radeon-r9700', '--device-results', 'none',
                            '--upstream', 'none', '--evidence', str(path), '--generator-commit', 'test'])
            return json.loads((package / 'info.json').read_text())

    def test_feature_and_serve_metadata_are_not_measurements(self):
        evidence = {'card': {'features': {'speculative_decoding': {'status': 'no', 'value': {
            'reason': 'The published QAT assistant is not available in this package.'}}}},
                    'versions': {'package': {'loom': '0.2.0'}},
                    'serve': {'served_context': 8192}}
        info = self.build(evidence)
        for device in ('strix-halo', 'radeon-r9700'):
            self.assertEqual(info['devices'][device]['state'], 'not_measured')
        self.assertEqual(info['devices']['strix-halo']['served_context'], 8192)
        self.assertEqual(info['card']['features']['speculative_decoding']['status'], 'no')

    def test_real_gate_marks_only_its_source_gpu(self):
        evidence = {'gates': [{'name': 'functional', 'status': 'pass', 'level': 'dev',
            'time': '2026-11-17T00:00:00Z', 'detail': 'actual functional evidence',
            'source': {'kind': 'devlab', 'run': '1', 'url': 'https://example.com/run/1',
                       'device': {'cpu': 'test host', 'gpu': 'AMD Radeon AI PRO R9700', 'os': 'Linux'}}}]}
        info = self.build(evidence)
        self.assertEqual(info['devices']['strix-halo']['state'], 'not_measured')
        self.assertNotIn('state', info['devices']['radeon-r9700'])
        self.assertEqual(info['devices']['radeon-r9700']['gates']['functional']['status'], 'pass')

    def test_unsupported_reason_and_legacy_text(self):
        old = 'not supported by Loom (the upstream model has it)'
        self.assertEqual(render_card.feature_detail('speculative_decoding', {'status': 'no', 'value': None}), old)
        self.assertEqual(render_card.feature_detail('image_input', {'status': 'no', 'value': 'legacy'}), old)
        reason = 'The published QAT assistant is not available in this package.'
        self.assertEqual(render_card.feature_detail('speculative_decoding', {'status': 'no', 'value': {'reason': reason}}), old + '; ' + reason)
        self.assertEqual(render_card.feature_detail('sampling', {'status': 'no', 'value': None}), 'greedy decoding only')


if __name__ == '__main__':
    unittest.main()
