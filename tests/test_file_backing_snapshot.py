import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'skills/loom-benchmark/scripts/file_backing_snapshot.py'
spec = importlib.util.spec_from_file_location('file_backing_snapshot', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux proc metadata')
class FileBackingSnapshot(unittest.TestCase):
    def test_real_fd_has_exact_identity_flags_and_mount_without_payload_read(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / 'small-fixture.gguf'
            path.write_bytes(b'not model weights')
            with path.open('rb') as source:
                result = m.snapshot(os.getpid())
                record = next(x for x in result['files'] if x['fd'] == source.fileno())
                self.assertEqual(record['inode'], path.stat().st_ino)
                self.assertEqual(record['size_bytes'], 17)
                self.assertEqual(record['target'], str(path))
                self.assertFalse(record['direct'])
                self.assertIsNotNone(record['mount']['filesystem'])
                self.assertIn('VmRSS', result['process_memory_raw'])
                self.assertIn('MemAvailable', result['host_io']['meminfo_raw'])
                self.assertIsNotNone(result['locked_memory_limit_raw'])

    def test_unavailable_process_reports_unknown_instead_of_empty_success(self):
        result = m.snapshot(999999999)
        self.assertEqual(result['errors']['fd'], 'FileNotFoundError')
        self.assertEqual(result['errors']['mountinfo'], 'FileNotFoundError')
        self.assertIsNone(result['process_memory_raw'])


if __name__ == '__main__':
    unittest.main()
