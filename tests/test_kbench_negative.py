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
        for mutation in ('compile', 'xpass', 'diagnostic', 'empty', 'passed'):
            r = report()
            if mutation == 'compile': r['samples'] = []
            elif mutation == 'xpass': r['xfails'][0]['outcome'] = 'xpass'
            elif mutation == 'diagnostic': r['xfails'][0]['diagnostic'] = 'TYPE/001'
            elif mutation == 'empty': r['samples'][0]['expectations']['failures'] = []
            else: r['samples'][0]['passed'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                M.validate(r, ['@wrong=EXPECT/003'])

    def test_cli_propagates_exit_and_xpass(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); tool = d/'tool'; tool.write_text('#!/usr/bin/env python3\nimport os,json,sys\nopen(os.environ["ARGV"],"w").write(json.dumps(sys.argv[1:]))\nprint(os.environ["REPORT"])\nsys.exit(int(os.environ["EXIT"]))\n'); tool.chmod(0o755)
            for status, outcome, expected in ((0, 'xfail', 0), (0, 'xpass', 1), (9, 'xfail', 1)):
                env = dict(os.environ, ARGV=str(d/'args'), REPORT=json.dumps(report(outcome)), EXIT=str(status))
                p = subprocess.run([sys.executable, str(SCRIPT), str(tool), 'negative.loom', str(d/'raw.json'), '@wrong=EXPECT/003'], env=env, capture_output=True)
                self.assertEqual(p.returncode, expected, p.stderr)
                self.assertEqual(json.loads((d/'args').read_text()), ['negative.loom', '--device=amdgpu', '--xfail=@wrong=EXPECT/003'])
                self.assertEqual(json.loads((d/'raw.json').read_text()), report(outcome))

    def test_positive_path_has_no_new_arguments(self):
        script = (ROOT/'.github/scripts/devlab_kbench.sh').read_text()
        self.assertIn('"$root/.venv/bin/loom-test" "$t" --device=amdgpu >', script)
        self.assertIn("s.get('negative_tests', [])", script)


if __name__ == '__main__': unittest.main()
