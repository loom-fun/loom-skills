"""A rejected host-memory guard must prevent the AMD server from starting."""
import os
import json
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
    def probe(self, device, active_wait='1', budget='none', bound='115000049320', sdk_version=None):
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
            if sdk_version:
                dist_info = work / f'hrx_loom-{sdk_version}.dist-info'
                dist_info.mkdir()
                (dist_info / 'METADATA').write_text(f'Name: hrx-loom\nVersion: {sdk_version}\n')
                (work / 'llm_config.json').write_text(json.dumps({'serve': {'args': ['--parameter_mode=preload', '--workers=4']}}))
                (work / 'sdk/bin/loom-serve').write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$REC"\nexit 1\n')
                env['PYTHONPATH'] = str(work)
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

    def test_package_preload_flag_depends_on_installed_sdk(self):
        for version in ('0.2.0', '0.3.0'):
            with self.subTest(version=version):
                _, events = self.probe('local-sync', sdk_version=version)
                self.assertIn('--workers=4', events)
                self.assertEqual('--parameter_mode=preload' in events, version == '0.2.0')
                self.assertFalse(any(arg.startswith('--memory_bytes=') for arg in events))


if __name__ == '__main__':
    unittest.main()
