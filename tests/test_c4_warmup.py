"""c4_steady.py warms up with untimed N-stream passes until two agree, logs them, and times the next pass."""
import json
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import requests  # noqa: F401
except ImportError:
    requests = None

SCRIPT = Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts/c4_steady.py"
TOKENS = 30


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, delays):
        super().__init__(("127.0.0.1", 0), Handler)
        self.delays, self.requests, self.lock = delays, 0, threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = min(body["max_tokens"], TOKENS)
        with self.server.lock:
            i = self.server.requests
            self.server.requests += 1
        # request 0 is the "Say hi." warm-up; then 4 streams per pass; pass k = (i - 1) // 4
        delay = 0.0 if n < TOKENS else self.server.delays[min((i - 1) // 4, len(self.server.delays) - 1)]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _ in range(n):
            time.sleep(delay)
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}).encode() + b"\n\n")
            self.wfile.flush()
        usage = {"usage": {"completion_tokens": n, "prompt_tokens": 5}, "choices": []}
        self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\n")


@unittest.skipIf(requests is None, "needs requests")
class WarmupTests(unittest.TestCase):
    def run_c4(self, delays, *args):
        srv = Server(delays)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            r = subprocess.run([sys.executable, str(SCRIPT), f"http://127.0.0.1:{srv.server_address[1]}", "4", *args],
                               capture_output=True, text=True, timeout=120, cwd=SCRIPT.parent)
        finally:
            srv.shutdown()
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_cold_passes_are_warm_up_not_the_result(self):
        # two slow passes that agree, then fast: the warm-up goes on (3 passes minimum, then two that agree), the
        # timed pass is a fast one, and the slow rates are logged.
        out = self.run_c4([0.04, 0.04, 0.004])
        self.assertEqual(len(out["warmup_agg_tps"]), 4, out)
        self.assertLess(out["warmup_agg_tps"][0], out["warmup_agg_tps"][2] / 3)
        self.assertGreater(out["agg_tps"], 3 * out["warmup_agg_tps"][0])

    def test_server_that_stays_slow_reports_slow(self):
        out = self.run_c4([0.03])
        self.assertEqual(len(out["warmup_agg_tps"]), 3, out)   # three agreeing slow passes end the warm-up
        self.assertLess(out["agg_tps"], 200)

    def test_warmup_off(self):
        out = self.run_c4([0.004], "--warmup-passes", "0")
        self.assertNotIn("warmup_agg_tps", out)


if __name__ == "__main__":
    unittest.main()
