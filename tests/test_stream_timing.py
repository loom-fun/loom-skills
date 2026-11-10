"""Exercise actual streaming timing logic with synthetic SSE events."""
import ast
import json
from pathlib import Path
import types
import unittest

REPO = Path(__file__).resolve().parents[1]
tree = ast.parse((REPO / "skills/loom-benchmark/scripts/bench_common.py").read_text(encoding="utf-8"))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "stream")


class Response:
    status_code = 200

    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        for event in self.events:
            yield b"data: " + json.dumps(event).encode()


class StreamTimingTests(unittest.TestCase):
    def run_stream(self, events):
        clock = iter([100.0, 101.0, 101.1])
        namespace = {"json": json, "time": types.SimpleNamespace(perf_counter=lambda: next(clock)),
                     "requests": types.SimpleNamespace(post=lambda *args, **kwargs: Response(events))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "bench_common.stream", "exec"), namespace)
        return namespace["stream"]("http://unused", "model", [], 4)

    def test_eos_only_reply_has_first_token_and_prefill_timing(self):
        result = self.run_stream([
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1, "prompt_tokens": 42}},
        ])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["ttft"], 1.0)
        self.assertEqual(result["prompt_tokens"], 42)
        self.assertIsNone(result["decode_tps"])

    def test_role_preface_does_not_count_as_a_generated_token(self):
        result = self.run_stream([
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1, "prompt_tokens": 42}},
        ])
        self.assertEqual(result["ttft"], 1.0)
        self.assertEqual(result["text"], "hello")


if __name__ == "__main__":
    unittest.main()
