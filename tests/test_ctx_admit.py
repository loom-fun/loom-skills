"""Exercise native-context admission with a streaming HTTP service."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import threading
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts/ctx_admit.py"


class ContextAdmissionTest(unittest.TestCase):
    def probe(self, mode, wait="3", *args):
        active = threading.Event()
        disconnected = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def send(self, status, body):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                if mode == "refuse":
                    self.send(503, b"insufficient context")
                    return
                if mode == "complete":
                    self.send(200, b'data: {"choices":[{"finish_reason":"length"}]}\n\n')
                    return
                if mode == "deferred":
                    active.set()
                try:
                    if not self.rfile.read(1):
                        disconnected.set()
                finally:
                    active.clear()

            def do_GET(self):
                body = json.dumps({"data": [{"loom": {"status": {"active_requests": int(active.is_set())}}}]}).encode()
                self.send(200, body)

            def log_message(self, *_args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as tmp:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                path = Path(tmp)
                (path / "model.json").write_text(json.dumps({"presets": {"chat": {"execution": {"context_capacity": 512}}}}))
                shim = """
import runpy
import sys
from types import SimpleNamespace

class Package:
    model_id = "fixture"
    chat = {}
    tok = SimpleNamespace(decode=lambda ids: " ".join(ids))
    def __init__(self, path):
        pass
    def ids(self, text):
        return text.split()
    def render_messages(self, *args):
        return "fixture"

sys.modules["bench_common"] = SimpleNamespace(Package=Package, doc=lambda _: "some text")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
                result = subprocess.run([sys.executable, "-c", shim, str(SCRIPT), tmp,
                                         f"http://127.0.0.1:{server.server_port}", "--wait", wait, "--tokens", "4",
                                         "--out", str(path / "admission.json"), *args],
                                        capture_output=True, text=True, timeout=7,
                                        env={**os.environ, "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"})
                if mode in ("deferred", "timeout"):
                    disconnected.wait(2)
                return result, json.loads((path / "admission.json").read_text()), disconnected.is_set()
            finally:
                server.shutdown()
                thread.join()

    def test_cancels_before_response_headers(self):
        result, evidence, disconnected = self.probe("deferred")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(disconnected)
        self.assertTrue(evidence["admitted"])
        self.assertIn("then cancelled", evidence["detail"])

    def test_completes_a_stream(self):
        result, evidence, _ = self.probe("complete", "3", "--complete", "3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(evidence["completed"])

    def test_reports_early_http_refusal(self):
        result, evidence, _ = self.probe("refuse")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertFalse(evidence["admitted"])
        self.assertIn("HTTP 503 insufficient context", evidence["detail"])

    def test_timeout_closes_the_request(self):
        result, evidence, disconnected = self.probe("timeout", "0.3")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertFalse(evidence["admitted"])
        self.assertTrue(disconnected)


if __name__ == "__main__":
    unittest.main()
