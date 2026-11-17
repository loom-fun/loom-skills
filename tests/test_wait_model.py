"""wait_model.py: a request refused while the model loads is retried until it is answered; a model that never loads, and other errors, fail."""
import http.server
import json
import os
import sys
import threading
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts"))
import wait_model  # noqa: E402


def server(refuse, code=503, body=b'{"error":{"message":"model is unavailable; inspect /v1/status"}}'):
    n = {"c": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "m"}]}).encode())

        def do_POST(self):
            n["c"] += 1
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if n["c"] <= refuse:
                self.send_response(code)
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass
    s = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, n


class WaitModel(unittest.TestCase):
    def test_waits_for_a_loading_model(self):
        s, n = server(3)
        self.assertEqual(wait_model.main([f"http://127.0.0.1:{s.server_port}", "--wait", "30", "--retry", "0.05"]), 0)
        self.assertEqual(n["c"], 4)

    def test_a_model_that_never_loads_fails(self):
        s, _ = server(10 ** 9)
        self.assertEqual(wait_model.main([f"http://127.0.0.1:{s.server_port}", "--wait", "0.5", "--retry", "0.05"]), 1)

    def test_another_error_is_not_a_loading_problem(self):
        s, _ = server(10 ** 9, code=400, body=b'{"error":"bad request"}')
        self.assertEqual(wait_model.main([f"http://127.0.0.1:{s.server_port}", "--wait", "30", "--retry", "0.05"]), 2)

    def test_a_closed_port_is_retried_until_the_wait_ends(self):
        self.assertEqual(wait_model.main(["http://127.0.0.1:1", "--wait", "0.3", "--retry", "0.05"]), 1)


if __name__ == "__main__":
    unittest.main()


class Wired(unittest.TestCase):
    def test_every_loom_serve_start_waits_for_the_first_answer(self):
        for rel in (".github/scripts/devlab_perf.sh", "skills/loom-benchmark/scripts/_lib.sh"):
            text = open(os.path.join(ROOT, rel)).read()
            self.assertIn("wait_model.py", text, rel)   # serve_alone (every step) and gates.sh's loom_up (the restarts after the reference fill)
