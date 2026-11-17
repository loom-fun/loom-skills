"""Card quickstarts send valid JSON to the selector advertised by model.json."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "skills/loom-benchmark/scripts/card"
sys.path.insert(0, str(CARD))
import render_card as renderer


def request_body(lines):
    command = "\n".join(lines).split("```bash\n", 1)[1].split("```", 1)[0]
    args = shlex.split(command.replace("\\\n", ""))
    assert args[:2] == ["curl", "http://127.0.0.1:8080/v1/chat/completions"]
    assert args[args.index("-H") + 1] == "Content-Type: application/json"
    return json.loads(args[args.index("-d") + 1])


class ChatRequestTests(unittest.TestCase):
    def test_exact_selector_and_shell_quoting(self):
        selector = "org/model's-name\\with-quote\""
        body = request_body(renderer.chat_request({"model": {"id": selector}}, {}))
        self.assertEqual(body, {"model": selector, "messages": [{"role": "user", "content": "Hello"}]})

    def test_optional_thinking_off(self):
        for default in (True, False):
            feats = {"thinking": {"status": "yes", "value": {"mode": "toggle", "default": default}}}
            body = request_body(renderer.chat_request({"model": {"id": "test"}}, feats))
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_always_thinking_uses_lowest_advertised_effort(self):
        feats = {"thinking": {"status": "yes", "value": {"mode": "always"}},
                 "reasoning_effort": {"status": "yes", "value": {"levels": ["high", "low", "medium"]}}}
        body = request_body(renderer.chat_request({"model": {"id": "test"}}, feats))
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertNotIn("chat_template_kwargs", body)

    def test_unknown_or_unsupported_controls_are_omitted(self):
        for status in ("unknown", "no", "n/a"):
            feats = {"thinking": {"status": status, "value": {"mode": "toggle"}},
                     "reasoning_effort": {"status": status, "value": {"levels": ["low"]}}}
            body = request_body(renderer.chat_request({"model": {"id": "test"}}, feats))
            self.assertEqual(set(body), {"model", "messages"})

    def test_single_and_multi_gpu_render_real_source_blocks(self):
        for fixture, package, requests in (("flint", "flint/pkg", 1), ("r9700_two", "r9700/pkg", 2), ("d1", "d1/repo", 0)):
            with self.subTest(fixture=fixture), tempfile.TemporaryDirectory() as temp:
                dest = Path(temp)
                shutil.copytree(CARD / "test" / package, dest, dirs_exist_ok=True)
                shutil.copy(CARD / "test" / fixture / "expected/info.json", dest / "info.json")
                with patch.dict(os.environ, {"LOOM_CARD_UPSTREAM": "none", "LOOM_DEVICE_RESULTS": "none"}):
                    text = renderer.render(str(dest))
                blocks = re.findall(r"```bash\n(curl .*?)```", text, re.S)
                self.assertEqual(len(blocks), requests)
                if requests:
                    model = json.loads((dest / "model.json").read_text())["model"]["id"]
                    for block in blocks:
                        self.assertEqual(request_body(["```bash", block, "```"])["model"], model)
                    if requests == 2:
                        self.assertEqual(len(re.findall(r"<details>.*?curl .*?</details>", text, re.S)), 2)


if __name__ == "__main__":
    unittest.main()
