import importlib.util
import json
import os
import tempfile
from pathlib import Path
import subprocess
import sys
import time
import unittest

spec = importlib.util.spec_from_file_location('thread_snapshot', Path(__file__).resolve().parents[1] / 'skills/loom-benchmark/scripts/thread_snapshot.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux proc diagnostics")
class SnapshotTest(unittest.TestCase):
    def child(self, code):
        p = subprocess.Popen([sys.executable, '-u', '-c', code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(self.stop, p)
        self.assertEqual(p.stdout.readline().strip(), 'ready')
        return p

    @staticmethod
    def stop(p):
        if p.poll() is None:
            p.terminate()
        p.wait(timeout=5)
        p.stdin.close()
        p.stdout.close()

    def test_blocked_child_exposes_mask_and_wait_without_invented_missing_values(self):
        p = self.child('import sys;print("ready");sys.stdin.read()')
        time.sleep(.02)
        row = m.snapshot(p.pid, include_stacks=True)
        self.assertEqual(row['event'], 'sample')
        self.assertTrue(row['process_status']['Cpus_allowed_list'])
        self.assertGreaterEqual(len(row['threads']), 1)
        thread = row['threads'][0]
        self.assertEqual(thread['stat']['state'], 'S')
        self.assertIsInstance(thread['wchan'], str)
        self.assertTrue(thread['kernel_stack'] is not None or thread['errors']['kernel_stack'])
        self.assertGreater(row['collection_ns'], 0)

    def test_busy_child_cpu_ticks_advance(self):
        p = self.child('print("ready");exec("while True: pass")')
        a = m.snapshot(p.pid)
        time.sleep(.2)
        b = m.snapshot(p.pid)
        ticks = lambda r: sum(r['process_stat'][k] for k in ('utime_ticks', 'stime_ticks'))
        self.assertGreater(ticks(b), ticks(a))
        self.assertGreater(b['monotonic_ns'], a['monotonic_ns'])

    def test_cli_stop_file_exits_gracefully_with_complete_json_rows(self):
        with tempfile.TemporaryDirectory() as work:
            out, stop = Path(work) / 'samples.jsonl', Path(work) / 'stop'
            p = subprocess.Popen([sys.executable, str(Path(m.__file__)), str(os.getpid()),
                                  '--output', str(out), '--duration', '10', '--interval', '0.1',
                                  '--stop-file', str(stop), '--host-io'])
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not (out.exists() and out.stat().st_size):
                    time.sleep(.02)
                stop.touch()
                self.assertEqual(p.wait(timeout=3), 0)
                rows = [json.loads(line) for line in out.read_text().splitlines()]
                self.assertTrue(rows)
                self.assertEqual(rows[0]['event'], 'sample')
                self.assertEqual(rows[0]['pid'], os.getpid())
                self.assertIn('MemTotal', rows[0]['host_io']['meminfo_raw'])
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()

    def test_missing_process_has_explicit_unavailable_event(self):
        row = m.snapshot(999999999)
        self.assertEqual(row['event'], 'process-unavailable')
        self.assertNotIn('process_io', row)
        self.assertEqual(row['error'], 'FileNotFoundError')


if __name__ == '__main__':
    unittest.main()
