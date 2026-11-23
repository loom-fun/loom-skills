"""Greedy comparisons work with either deterministic or stochastic service defaults."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts/sampling_check.py"


class SamplingCheckTest(unittest.TestCase):
    def exercise(self, temperature_default, *, unstable_greedy=False, ignore_default_seed=False):
        requests = []

        def post(_url, *, timeout, json):
            requests.append(json)
            temperature = json.get("temperature", temperature_default)
            seed = json.get("seed")
            if temperature == 0 and unstable_greedy:
                reply = f"greedy-{len(requests)}"
            elif temperature == 0 or json.get("top_k") == 1:
                reply = "banana"
            elif seed is None or ("temperature" not in json and ignore_default_seed):
                reply = f"non-greedy-{len(requests)}"
            else:
                reply = f"sample-{seed}"
            return SimpleNamespace(json=lambda: {"choices": [{"message": {"content": reply}}]})

        package = SimpleNamespace(chat={"thinking": "always"}, model_id="sampling-fixture")
        dependencies = {
            "bench_common": SimpleNamespace(Package=lambda _: package, llama_greedy=None),
            "requests": SimpleNamespace(post=post),
        }
        entry = importlib.util.spec_from_file_location("sampling_check_fixture", SCRIPT)
        gate = importlib.util.module_from_spec(entry)
        output = io.StringIO()
        args = [str(SCRIPT), "fixture", "http://fixture", "--draws", "0"]
        with patch.dict(sys.modules, dependencies), patch.object(sys, "argv", args), contextlib.redirect_stdout(output):
            entry.loader.exec_module(gate)
            with self.assertRaises(SystemExit) as finished:
                gate.main()
        return finished.exception.code, output.getvalue()

    def test_default_greedy(self):
        self.assertEqual(self.exercise(0.0)[0], 0)

    def test_nonzero_default(self):
        code, output = self.exercise(1.0)
        self.assertEqual(code, 0, output)
        self.assertIn("OK   same seed reproduces with service defaults", output)

    def test_greedy_must_repeat(self):
        code, output = self.exercise(1.0, unstable_greedy=True)
        self.assertEqual(code, 1)
        self.assertIn("FAIL temperature 0 repeats", output)

    def test_service_defaults_must_honor_seed(self):
        code, output = self.exercise(1.0, ignore_default_seed=True)
        self.assertEqual(code, 1)
        self.assertIn("FAIL same seed reproduces with service defaults", output)


if __name__ == "__main__":
    unittest.main()
