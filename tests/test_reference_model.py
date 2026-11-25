"""Gates of a package whose GGUF llama.cpp cannot load: pinned reference GGUF + the inkling2 channel renderer."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills/loom-benchmark/scripts"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rm = load("reference_model", SCRIPTS / "reference_model.py")
probe = load("inkling2_chat_probe", SCRIPTS / "renderers/inkling2/chat_probe.py")

INKLING = ("unsloth/Inkling-2-Mini-GGUF", "ea0c79b2ecd73669e774c8519b79bfabfddbc027", "Inkling-2-Mini-UD-Q4_K_XL.gguf")


def package(tmp, repo, rev, gguf, call_format="inkling2"):
    p = Path(tmp) / "pkg"
    p.mkdir()
    (p / "model.json").write_text(json.dumps({"origins": {"checkpoint": {"kind": "hf", "repository": repo, "revision": rev}}}))
    (p / "llm_config.json").write_text(json.dumps({"gguf": gguf, "chat": {"call_format": call_format}}))
    return str(p)


class ReferenceModelTest(unittest.TestCase):
    def test_table_entry_is_pinned_and_same_revision(self):
        for e in rm.read_json(rm.TABLE)["models"]:
            self.assertRegex(e["reference"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(e["checkpoint"], f"{e['reference']['repo']}@{e['reference']['revision']}")   # same checkpoint, never another
            self.assertIsNotNone(rm.release_number(e["llamacpp_min"]))

    def test_lookup_by_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = rm.lookup(package(tmp, *INKLING))
            self.assertEqual(e["reference"]["file"], "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file")
        with tempfile.TemporaryDirectory() as tmp:   # another revision or another file of the repo: no entry
            self.assertIsNone(rm.lookup(package(tmp, INKLING[0], "0" * 40, INKLING[2])))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(rm.lookup(package(tmp, INKLING[0], INKLING[1], "Inkling-2-Mini-UD-Q8_K_XL.gguf")))

    def test_llamacpp_release_is_checked(self):
        e = {"llamacpp_min": "b1384", "why": "x"}
        self.assertIsNone(rm.check_tag(e, "b1384"))
        self.assertIsNone(rm.check_tag(e, "b1390"))
        for tag in ("b1383", "latest", "-", None):
            self.assertIsNotNone(rm.check_tag(e, tag))

    def test_fetch_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "hub/o/r/resolve/" / ("a" * 40) / "Mainline_Support"
            src.mkdir(parents=True)
            data = b"GGUF" + os.urandom(1000)
            (src / "m.gguf_file").write_bytes(data)
            entry = {"reference": {"repo": "o/r", "revision": "a" * 40, "file": "Mainline_Support/m.gguf_file", "size": len(data),
                                   "sha256": hashlib.sha256(data).hexdigest()}}
            base = (Path(tmp) / "hub").as_uri()
            got = rm.fetch(entry, os.path.join(tmp, "cache"), base)
            self.assertTrue(got.endswith("m.gguf"))
            self.assertEqual(Path(got).read_bytes(), data)
            self.assertEqual(rm.fetch(entry, os.path.join(tmp, "cache"), base), got)   # cached, re-verified
            Path(got).write_bytes(data[:-1] + b"x")   # corrupted cache: refused and removed
            with self.assertRaises(ValueError):
                rm.fetch(entry, os.path.join(tmp, "cache"), base)
            self.assertFalse(os.path.exists(got))
            entry["reference"]["sha256"] = "0" * 64   # wrong pin: refused
            with self.assertRaises(ValueError):
                rm.fetch(entry, os.path.join(tmp, "cache2"), base)

    def test_no_entry_prints_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(rm.main([package(tmp, "o/r", "b" * 40, "x.gguf"), "--cache", tmp, "--tag", "b1300"]), 0)

    def test_renderer_dir_matches_call_format(self):
        self.assertTrue((SCRIPTS / "renderers/inkling2/chat_probe.py").is_file())
        self.assertIn("renderers/", (SCRIPTS / "_lib.sh").read_text())


CHAT = {"reasoning_effort_default": "high", "reasoning_instructions": {"none": "0", "high": "0.9"}}


class Inkling2RendererTest(unittest.TestCase):
    def test_tool_conversation(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "d",
                                                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        msgs = [{"role": "user", "content": "Paris?"},
                {"role": "assistant", "content": "", "reasoning_content": "r",
                 "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "12C"}]
        got = probe.simulate(CHAT, msgs, tools, gen=True)
        want = ('<|message_system|>tool_declare<|content_xml|>[{"description":"d","name":"get_weather","parameters":'
                '{"properties":{"city":{"type":"string"}},"type":"object"},"type":"function"}]<|end_message|>'
                "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                "<|message_user|><|content_text|>Paris?<|end_message|>"
                "<|message_model|><|content_thinking|>r<|end_message|>"
                '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}<|end_message|>'
                "<|content_model_end_sampling|>"
                "<|message_tool|>get_weather<|content_text|>12C<|end_message|>")
        self.assertEqual(got, want)
        self.assertIn("Thinking effort level: 0<", probe.simulate(CHAT, msgs[:1], enable_thinking=False))

    def test_raw_reply(self):
        m = {"reasoning_content": "think", "content": "answer"}
        self.assertEqual(probe.raw_from_chat(CHAT, m, True),
                         "<|message_model|><|content_thinking|>think<|end_message|><|message_model|><|content_text|>answer<|end_message|>")
        self.assertEqual(probe.raw_from_chat(CHAT, {"content": "ans"}, False), "<|message_model|><|content_text|>ans")
        self.assertEqual(probe.raw_alternatives(CHAT, {"content": ""}, True), ["", "<|message_model|><|content_text|><|end_message|>"])


if __name__ == "__main__":
    unittest.main()
