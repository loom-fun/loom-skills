import ctypes.util
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'skills/loom-setup/scripts/no_uring_exec.py'
spec = importlib.util.spec_from_file_location('no_uring_exec', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@unittest.skipUnless(sys.platform.startswith('linux') and ctypes.util.find_library('seccomp'), 'Linux libseccomp')
class NoUringExec(unittest.TestCase):
    def test_filter_changes_only_requested_setup_to_enosys_and_preserves_exec_pid(self):
        probe = '''import ctypes,ctypes.util,json,os
s=ctypes.CDLL(ctypes.util.find_library('seccomp'));s.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];s.seccomp_syscall_resolve_name.restype=ctypes.c_int
nr=s.seccomp_syscall_resolve_name(b'io_uring_setup');c=ctypes.CDLL(None,use_errno=True);c.syscall.restype=ctypes.c_long;p=ctypes.create_string_buffer(256);ctypes.set_errno(0)
x=c.syscall(ctypes.c_long(nr),ctypes.c_uint(1),ctypes.byref(p));e=ctypes.get_errno()
if x>=0:os.close(x)
print(json.dumps({'pid':os.getpid(),'result':x,'errno':e,'read':open('/proc/self/status').read().startswith('Name:')}))
'''
        p = subprocess.Popen([sys.executable, str(SCRIPT), '--', sys.executable, '-c', probe],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = p.communicate(timeout=10)
        self.assertEqual(p.returncode, 0, stderr)
        result = json.loads(stdout)
        self.assertEqual((result['result'], result['errno']), (-1, 38))
        self.assertEqual(result['pid'], p.pid)
        self.assertTrue(result['read'])
        self.assertIn('ENOSYS', stderr)

    def test_preflight_records_exec_environment_and_fails_if_backend_not_distinct(self):
        import tempfile
        with tempfile.TemporaryDirectory() as work:
            output = Path(work) / 'preflight.json'
            result = subprocess.run([sys.executable, str(SCRIPT), '--self-test', str(output),
                                     '--require-distinct-baseline'], capture_output=True, text=True, timeout=10)
            data = json.loads(output.read_text())
        self.assertTrue(data['exec_and_filter_checks_pass'])
        self.assertTrue(data['restricted_child']['environment_equal'])
        self.assertEqual(data['restricted_child']['no_new_privs'], '1')
        self.assertEqual(result.returncode == 0, data['baseline_distinct'])
        if not data['baseline_distinct']:
            self.assertIn('no weights should be fetched', result.stderr)

    def test_timed_out_probe_is_killed_and_leaves_failure_receipt(self):
        import tempfile
        from unittest.mock import Mock
        child = Mock(pid=123)
        child.communicate.side_effect = [subprocess.TimeoutExpired('probe', 10), ('', 'stopped')]
        with tempfile.TemporaryDirectory() as work:
            output = Path(work) / 'partial.json'
            with patch.object(m, 'setup_probe', return_value={'available': True, 'syscall': 425}), \
                    patch.object(m.subprocess, 'Popen', return_value=child):
                with self.assertRaisesRegex(RuntimeError, 'child killed'):
                    m.self_test(output)
            data = json.loads(output.read_text())
        child.kill.assert_called_once()
        self.assertFalse(data['exec_and_filter_checks_pass'])
        self.assertEqual(data['error'], 'restricted child timed out')

    def test_missing_library_fails_closed_before_any_filter_install(self):
        with patch.object(m.ctypes.util, 'find_library', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                m.disable_io_uring()


if __name__ == '__main__':
    unittest.main()
