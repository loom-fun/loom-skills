"""Strict negative controls accept comparator failures, never broken setup or XPASS."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / '.github/scripts/kbench_negative.py'
spec = importlib.util.spec_from_file_location('kbench_negative', SCRIPT)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def report(outcome='xfail'):
    return {'xfails': [{'record': '@wrong', 'diagnostic': 'EXPECT/003', 'outcome': outcome, 'policy': 'strict'}],
            'samples': [{'case': '@wrong', 'passed': False, 'expectations': {'failures': [{'detail': 'values differ'}]}}]}


class NegativeControls(unittest.TestCase):
    def test_raw_comparator_failure_required(self):
        self.assertEqual(M.validate(report(), ['@wrong=EXPECT/003']), 1)
        for mutation in ('compile', 'xpass', 'diagnostic', 'empty', 'passed', 'missing_passed', 'falsey_passed', 'duplicate_record', 'duplicate_sample'):
            r = report()
            if mutation == 'compile': r['samples'] = []
            elif mutation == 'xpass': r['xfails'][0]['outcome'] = 'xpass'
            elif mutation == 'diagnostic': r['xfails'][0]['diagnostic'] = 'TYPE/001'
            elif mutation == 'empty': r['samples'][0]['expectations']['failures'] = []
            elif mutation == 'passed': r['samples'][0]['passed'] = True
            elif mutation == 'missing_passed': del r['samples'][0]['passed']
            elif mutation == 'falsey_passed': r['samples'][0]['passed'] = None
            elif mutation == 'duplicate_record': r['xfails'].append(dict(r['xfails'][0]))
            else: r['samples'].append(dict(r['samples'][0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                M.validate(r, ['@wrong=EXPECT/003'])
        with self.assertRaises(ValueError):
            M.validate(report(), ['@wrong=EXPECT/003', '@wrong=EXPECT/003'])

    def test_configuration_validation_is_checked_before_any_tool(self):
        valid = {'file': 'negative.loom', 'xfail': ['@wrong=EXPECT/003']}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'config.json'
            for value, ok in (([], True), ([valid], True), (None, False), ({}, False),
                              ([dict(valid, file=9)], False), ([dict(valid, xfail=None)], False),
                              ([dict(valid, xfail=[42])], False),
                              ([dict(valid, xfail=['@wrong=EXPECT/003']*2)], False)):
                path.write_text(json.dumps({'suites': [{'suite': 's', 'negative_tests': value}]}))
                r = subprocess.run([sys.executable, str(SCRIPT), '--plan', str(path), 's'], capture_output=True)
                self.assertEqual(r.returncode, 0 if ok else 1, r.stderr)
            path.write_text(json.dumps({'suites': [{'suite': 's'}]}))
            self.assertEqual(M.plan(path, 's'), [])


    def test_cli_propagates_exit_and_xpass(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); tool = d/'fake.py'; tool.write_text('import os,json,sys\nopen(os.environ["ARGV"],"w").write(json.dumps(sys.argv))\nprint(os.environ["REPORT"])\nsys.exit(int(os.environ["EXIT"]))\n')
            for status, outcome, expected in ((0, 'xfail', 0), (0, 'xpass', 1), (9, 'xfail', 1)):
                env = dict(os.environ, ARGV=str(d/'args'), REPORT=json.dumps(report(outcome)), EXIT=str(status))
                p = subprocess.run([sys.executable, str(SCRIPT), sys.executable, str(tool), str(d/'raw.json'), '@wrong=EXPECT/003'], env=env, capture_output=True)
                self.assertEqual(p.returncode, expected, p.stderr)
                self.assertEqual(json.loads((d/'args').read_text()), [str(tool), '--device=amdgpu', '--xfail=@wrong=EXPECT/003'])
                self.assertEqual(json.loads((d/'raw.json').read_text()), report(outcome))



if __name__ == '__main__': unittest.main()
