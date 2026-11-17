"""A rejected host-memory guard must prevent the AMD server from starting."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'skills/loom-setup/scripts/serve.sh'


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'POSIX shell fixture')
class ServeMemoryGuard(unittest.TestCase):
    def probe(self, device, active_wait='1', budget='none', bound='115000049320'):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            shutil.copyfile(SOURCE, work / 'serve.sh')
            for name in ('sdk/bin', 'bin', 'state'):
                (work / name).mkdir(parents=True)
            events = work / 'events'
            fixtures = {
                'memcheck.sh': '#!/bin/sh\necho guard >> "$REC"\nexit 23\n',
                'sdk/bin/loom-serve': '#!/bin/sh\necho binary >> "$REC"\nexit 1\n',
                'bin/ss': '#!/bin/sh\nexit 1\n',
                'bin/curl': '#!/bin/sh\nexit 7\n',
            }
            for name, text in fixtures.items():
                path = work / name
                path.write_text(text)
                path.chmod(0o755)
            (work / 'sdk/bin/python').symlink_to(sys.executable)
            env = dict(os.environ, VENV=str(work / 'sdk'), REC=str(events),
                       LOOM_SERVE_STATE=str(work / 'state'), LOOM_MEMNEED=bound,
                       LOOM_ACTIVE_WAIT=active_wait, PATH=str(work / 'bin') + ':' + os.environ['PATH'])
            result = subprocess.run(['bash', str(work / 'serve.sh'), 'fixture', '48196', device,
                                     budget, str(work / 'missing-model.json')], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            return result, events.read_text().splitlines() if events.exists() else []

    def test_active_wait_rewrite_still_checks_fixed_bound(self):
        result, events = self.probe('amdgpu')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, ['guard'], result.stdout + result.stderr)

    def test_explicit_uri_still_checks_elastic_budget(self):
        result, events = self.probe('amdgpu://?queue_family=0', budget='97000000000', bound='')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, ['guard'], result.stdout + result.stderr)

    def test_cpu_driver_keeps_its_existing_path(self):
        _, events = self.probe('local-sync')
        self.assertEqual(events, ['binary'])


if __name__ == '__main__':
    unittest.main()
