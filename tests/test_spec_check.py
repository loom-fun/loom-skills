"""Speculative equivalence must handle incomplete reference evidence without accepting it."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts/spec_check.py"
SPEC = importlib.util.spec_from_file_location("spec_check", SCRIPT)
spec_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spec_check)


class SpeculativeReferenceTest(unittest.TestCase):
    def run_gate(self, reference):
        package = SimpleNamespace(render_messages=lambda messages, *_args, **_kwargs: messages[0]["content"],
                                  raw_from_chat=lambda message, *_args: message["content"], ids=list,
                                  tok=SimpleNamespace(decode="".join))
        judge = Mock(return_value=True)
        engine = Mock(return_value=reference)
        bench = SimpleNamespace(Package=lambda _pkg: package, llama_greedy=engine, near_tie_at=judge,
                                top_logprobs=lambda probs, k: {p["id"]: p["logprob"] for p in probs[k]["top_logprobs"]}
                                if k < len(probs) else {})

        def rows(content):
            return [json.dumps({"i": i, "request": {"messages": [{"content": str(i)}], "temperature": 0},
                                "message": {"content": content}, "finish": "stop", "n": 2,
                                "content": content, "reasoning": None, "calls": []}) for i in range(6)]

        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / "target.jsonl", Path(tmp) / "draft.jsonl"
            left.write_text("\n".join(rows("ax")))
            right.write_text("\n".join(rows("ay")))
            output = io.StringIO()
            with patch.dict(sys.modules, {"bench_common": bench}), contextlib.redirect_stdout(output):
                result = spec_check.compare(left, right, pkg="pkg", llama="ref")
        self.assertEqual(engine.call_count, 6)
        self.assertEqual([call.args[1] for call in engine.call_args_list], [[str(i), "a"] for i in range(6)])
        return result, judge.call_count, output.getvalue()

    def test_all_deferred_references_are_requested(self):
        result, judged, output = self.run_gate(({"_pending": True, "completion_probabilities": []}, [], []))
        self.assertEqual((result, judged), (1, 0))
        self.assertEqual(output.count("reference pending"), 6)

    def test_missing_aligned_probabilities_fail(self):
        result, judged, output = self.run_gate(({"completion_probabilities": []}, ["x"], []))
        self.assertEqual((result, judged), (1, 0))
        self.assertIn("SPEC_CHECK FAIL", output)

    def test_complete_reference_uses_existing_near_tie_judge(self):
        top = {"top_logprobs": [{"id": "x", "logprob": -0.3}, {"id": "y", "logprob": -0.4}]}
        result, judged, output = self.run_gate(({"completion_probabilities": [top]}, ["x"], [top]))
        self.assertEqual((result, judged), (0, 6))
        self.assertIn("SPEC_CHECK PASS: 0/6 completions identical, 6 near-tie", output)


if __name__ == "__main__":
    unittest.main()
