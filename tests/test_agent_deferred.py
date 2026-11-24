"""Verify deferred agent runs bind executable inputs without changing the no-reference gate."""

import contextlib
import io
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts"
if not find_spec("requests") or not find_spec("jinja2"):
    raise unittest.SkipTest("deferred agent fixtures require Requests and Jinja2")
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    import agent_golden as agent


class AgentDeferredTest(unittest.TestCase):
    def test_key_changes_with_nested_service_tokenizer_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(agent, "loom_runs_key", side_effect=lambda _pkg, _model, inputs: json.dumps(inputs)):
            root = Path(tmp)
            nested = root / "program/kernels/text.loom"
            nested.parent.mkdir(parents=True)
            files = [nested, root / "tokenizer.json", root / "model.json.lock"]
            for index, path in enumerate(files):
                path.write_text(f"old {index}")
            state = root / "saved.json"
            for path in files:
                key = agent.capture_key(tmp, "model", ["task"])
                state.write_text(json.dumps({"key": key, "captures": []}))
                self.assertIsNotNone(agent.kept_runs(state, key))
                path.write_text("new")
                self.assertIsNone(agent.kept_runs(state, agent.capture_key(tmp, "model", ["task"])))

    def test_without_llama_runs_live_even_if_defer_is_set(self):
        package = SimpleNamespace(model_id="fixture", render_messages=lambda *_args: "fixture", ids=list,
                                  raw_from_chat=lambda *_args: "Paris")
        replies = [{"choices": [{"finish_reason": "stop", "message": {"content": "Paris"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 5}}, {"input_tokens": 7}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.json"
            argv = ["agent_golden.py", tmp, "local", "--out", str(path)]
            with patch.object(agent, "Package", return_value=package), patch.object(agent, "TASKS", ["Question"]), \
                    patch.object(agent, "post", side_effect=replies) as post, \
                    patch.object(agent, "capture_key") as identity, patch.object(sys, "argv", argv), \
                    patch.dict(os.environ, {"LLAMA_DEFER": str(Path(tmp) / "pending.jsonl")}), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                agent.main()
            self.assertEqual(result.exception.code, 0)
            identity.assert_not_called()
            self.assertEqual(post.call_count, 2)
            self.assertNotIn("observed_trajectory_sha256", json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
