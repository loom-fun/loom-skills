"""r9700_func.py waits for a model that is still loading (503 "model is unavailable") and reports any other failure with the server's body."""
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FUNC = os.path.join(ROOT, ".github", "scripts", "r9700_func.py")


def serve(handler):
    s = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def run(port, wait):
    with tempfile.TemporaryDirectory() as t:
        out = os.path.join(t, "o.json")
        r = subprocess.run([sys.executable, FUNC, f"http://127.0.0.1:{port}", "m", out], capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, R9700_FUNC_RETRY_S="0.05", R9700_FUNC_LOAD_WAIT=str(wait)))
        return r, json.load(open(out)) if os.path.exists(out) else None


class FuncWait(unittest.TestCase):
    def test_a_loading_model_is_waited_for(self):
        n = {"c": 0}

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n["c"] += 1
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if n["c"] < 4:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"model is unavailable; inspect /v1/status"}}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "Paris"}}]}).encode())

            def log_message(self, *a):
                pass
        s = serve(H)
        r, res = run(s.server_port, 30)
        self.assertEqual(res["checks"]["chat"]["ok"], True, r.stdout + r.stderr)
        self.assertEqual(n["c"], 4)

    def test_another_error_fails_with_the_servers_body(self):
        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"retained state leaves insufficient completion capacity"}}')

            def log_message(self, *a):
                pass
        s = serve(H)
        r, res = run(s.server_port, 30)
        self.assertFalse(res["checks"]["chat"]["ok"])
        self.assertIn("insufficient completion capacity", res["checks"]["chat"]["detail"])

    def test_a_model_that_never_loads_fails(self):
        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"model is unavailable; inspect /v1/status"}}')

            def log_message(self, *a):
                pass
        s = serve(H)
        r, res = run(s.server_port, 0.5)
        self.assertFalse(res["checks"]["chat"]["ok"])
        self.assertIn("unavailable", res["checks"]["chat"]["detail"])


if __name__ == "__main__":
    unittest.main()
