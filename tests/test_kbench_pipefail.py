"""A failing benchmark child must fail its workflow step even when tee succeeds."""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(os.name == 'nt' or not shutil.which('bash'), 'GitHub Linux shell contract')
class KbenchPipeline(unittest.TestCase):
    def test_workflow_propagates_benchmark_failure_through_tee(self):
        source = (ROOT / '.github/workflows/devlab-kbench.yml').read_text()
        step = re.search(r'      - name: "kbench:.*?(?=\n      - name:)', source, re.S).group()
        self.assertIn('shell: bash\n', step)
        command = re.search(r'        run: (.*)', step).group(1)
        # Exercise the actual authored pipeline with only its external inputs
        # replaced. GitHub's explicit Bash template enables errexit+pipefail.
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'child.sh'
            script.write_text('exit 23\n')
            command = command.replace('bash .github/scripts/devlab_kbench.sh', f'bash "{script}"')
            command = re.sub(r'\$\{\{.*?\}\}', 'fixture', command)
            env = dict(os.environ, WORK=tmp)
            hidden = subprocess.run(['bash', '-e', '-c', command], env=env, capture_output=True, text=True)
            self.assertEqual(hidden.returncode, 0, 'default Bash hides the child failure behind tee')
            flags = ['bash', '--noprofile', '--norc', '-e', '-o', 'pipefail', '-c']
            failed = subprocess.run(flags + [command], env=env, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 23, failed.stderr)
            script.write_text('exit 0\n')
            passed = subprocess.run(flags + [command], env=env, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
