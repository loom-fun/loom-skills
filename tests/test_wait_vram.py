"""wait_vram.sh: waits for the GPU's VRAM to be released, logs what was held, never blocks forever."""
import os
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "wait_vram.sh")


def run(files, *args, timeout=60):
    return subprocess.run(["bash", SCRIPT, *args], capture_output=True, text=True, timeout=timeout, env=dict(os.environ, VRAM_FILES=files))


@unittest.skipIf(os.name == "nt", "a bash script over a sysfs path: DevLab runners are Linux")
class WaitVram(unittest.TestCase):
    def test_free_gpu_returns_at_once_and_logs(self):
        with tempfile.TemporaryDirectory() as t:
            f = os.path.join(t, "used")
            open(f, "w").write(str(200 * 2**20))
            log = os.path.join(t, "log")
            r = run(f, "serve card", "1024", "30", log)
            self.assertIn("vram_used 200 MiB before serve card", r.stdout)
            self.assertIn("vram_used 200 MiB", open(log).read())

    def test_waits_while_the_previous_process_releases_its_memory(self):
        with tempfile.TemporaryDirectory() as t:
            f = os.path.join(t, "used")
            open(f, "w").write(str(20000 * 2**20))
            threading.Thread(target=lambda: (time.sleep(3), open(f, "w").write(str(300 * 2**20)))).start()
            t0 = time.time()
            r = run(f, "after vllm", "1024", "60")
            self.assertGreaterEqual(time.time() - t0, 2.5)
            self.assertIn("vram_used 300 MiB before after vllm", r.stdout)

    def test_a_gpu_that_holds_memory_by_itself_does_not_block_forever(self):
        with tempfile.TemporaryDirectory() as t:
            f = os.path.join(t, "used")
            open(f, "w").write(str(5000 * 2**20))
            t0 = time.time()
            r = run(f, "x", "1024", "120")
            self.assertLess(time.time() - t0, 60)   # constant for 20 s: stops waiting
            self.assertEqual(r.returncode, 0)
            self.assertIn("vram_used 5000 MiB", r.stdout)

    def test_no_sysfs_is_not_an_error(self):
        r = run("/nonexistent/none", "x")
        self.assertEqual(r.returncode, 0)
        self.assertIn("unknown", r.stdout)


class Wired(unittest.TestCase):
    def test_every_loom_serve_start_waits_for_the_vram(self):
        for rel in (".github/scripts/devlab_perf.sh", "skills/loom-benchmark/scripts/_lib.sh"):
            self.assertIn("wait_vram.sh", open(os.path.join(ROOT, rel)).read(), rel)


if __name__ == "__main__":
    unittest.main()
