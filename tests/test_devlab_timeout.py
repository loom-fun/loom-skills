import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch

WORKFLOW = Path(__file__).resolve().parents[1] / '.github/workflows/devlab-perf.yml'
SOURCE = WORKFLOW.read_text()
start = SOURCE.index("          python3 - <<'PY' >>")
code = SOURCE[SOURCE.index('\n', start) + 1:SOURCE.index('          PY', start)]
CODE = textwrap.dedent(code)


class DevlabTimeout(unittest.TestCase):
    def matrix(self, job):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work) / '.github/devlab'
            root.mkdir(parents=True)
            (root / 'perf.json').write_text(json.dumps({'jobs': [dict(name='bounded', **job)]}))
            output = io.StringIO()
            previous = os.getcwd()
            try:
                os.chdir(work)
                with patch.dict(sys.modules, {'registry': types.SimpleNamespace(runner_labels=lambda: {'strix': ['Linux']})}), \
                        patch.object(sys, 'path', sys.path.copy()), contextlib.redirect_stdout(output):
                    exec(CODE, {})
            finally:
                os.chdir(previous)
        value = next(line.split('=', 1)[1] for line in output.getvalue().splitlines() if line.startswith('matrix='))
        return json.loads(value)['include'][0]

    def test_explicit_job_and_step_caps_are_used(self):
        row = self.matrix({'timeout_minutes': 30, 'experiment_timeout_minutes': 27})
        self.assertEqual((row['timeout_minutes'], row['experiment_timeout_minutes']), (30, 27))
        self.assertIn('timeout-minutes: ${{ matrix.timeout_minutes }}', SOURCE)
        self.assertIn('timeout-minutes: ${{ matrix.experiment_timeout_minutes }}', SOURCE)
        self.assertIn('- name: upload results\n        if: always()', SOURCE)

    def test_backend_preflight_precedes_checkpoint_materialization(self):
        script = (WORKFLOW.parents[2] / '.github/scripts/devlab_perf.sh').read_text()
        self.assertLess(script.index('"$T" "$SV/no_uring_exec.py" --self-test'), script.index('fetch_card "$repo_ref"'))
        self.assertIn('--require-distinct-baseline', script)
        self.assertIn('results.partial.tsv', script)

    def test_existing_default_remains_240_minutes(self):
        row = self.matrix({})
        self.assertEqual((row['timeout_minutes'], row['experiment_timeout_minutes']), (240, 240))

    def test_step_cannot_exceed_job_and_boolean_cap_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.matrix({'timeout_minutes': 30, 'experiment_timeout_minutes': 31})
        with self.assertRaises(AssertionError):
            self.matrix({'timeout_minutes': True})


if __name__ == '__main__':
    unittest.main()
