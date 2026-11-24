"""Exercise a cancellation before the streaming service sends response headers."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts/ctx_admit.py"


class ContextAdmissionTest(unittest.TestCase):
    def test_cancels_before_response_headers(self):
        active = threading.Event()
        disconnected = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                active.set()
                try:
                    if not self.rfile.read(1):
                        disconnected.set()
                finally:
                    active.clear()

            def do_GET(self):
                body = json.dumps({"data": [{"loom": {"status": {"active_requests": int(active.is_set())}}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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
                                         f"http://127.0.0.1:{server.server_port}", "--wait", "3", "--tokens", "4",
                                         "--out", str(path / "admission.json")], capture_output=True, text=True, timeout=7)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(disconnected.is_set())
                evidence = json.loads((path / "admission.json").read_text())
                self.assertTrue(evidence["admitted"])
                self.assertIn("then cancelled", evidence["detail"])
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
