import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_package", REPO / "skills/loom-benchmark/scripts/card/publish_package.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)


class PrepareKeepsDeclaredOrigins(unittest.TestCase):
    def test_catalog_package(self):
        if os.name == "nt":
            self.skipTest("symlinks")
        rev = "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            snap = tmp / "hub/models--unsloth--Q-GGUF/snapshots" / rev
            (snap / "UD").mkdir(parents=True)
            blob = tmp / "hub/models--unsloth--Q-GGUF/blobs/sha"
            blob.parent.mkdir(parents=True); blob.write_bytes(b"w")
            (snap / "UD/q-00001-of-00001.gguf").symlink_to(blob)
            pkg = tmp / "pkg"
            (pkg / "parameter").mkdir(parents=True); (pkg / "program").mkdir()
            (pkg / "parameter/q-00001-of-00001.gguf").symlink_to(snap / "UD/q-00001-of-00001.gguf")
            (pkg / "program/a.loom").write_text("x")
            (pkg / "tokenizer.json").write_text("{}")
            (pkg / "model.json").write_text(json.dumps({
                "origins": {"checkpoint": {"kind": "hf", "repository": "Qwen/Q", "revision": "a" * 40},
                            "quantized": {"kind": "hf", "repository": "unsloth/Q-GGUF", "revision": "old"}},
                "variants": {"v": {"services": {"chat": {"parameters": {"model": {"files": [
                    {"path": "parameter/q-00001-of-00001.gguf", "origin": "@quantized/UD/q-00001-of-00001.gguf"}]}},
                    "tokenizers": {"text": {"path": "tokenizer.json", "origin": "@checkpoint/tokenizer.json"}}}}}}}))
            (pkg / "llm_config.json").write_text(json.dumps({"gguf": str(pkg / "parameter/q-00001-of-00001.gguf")}))
            fetch = tmp / "fetch.sh"
            fetch.write_text("#!/bin/sh\necho '{\"files\": []}' > \"$(dirname \"$1\")/model.json.lock\"\n")
            fetch.chmod(0o755)
            out = tmp / "out"
            pp.main(["prepare", str(pkg), str(out), "--source", "unsloth/Q-GGUF", "--loom-fetch", str(fetch)])
            card = json.loads((out / "model.json").read_text())
            self.assertEqual(card["origins"]["checkpoint"]["repository"], "Qwen/Q")   # kept
            self.assertEqual(card["origins"]["quantized"]["revision"], rev)             # pinned to the cache
            entry = card["variants"]["v"]["services"]["chat"]["parameters"]["model"]["files"][0]
            self.assertEqual(entry["origin"], "@quantized/UD/q-00001-of-00001.gguf")
            self.assertFalse((out / "parameter").exists())


if __name__ == "__main__":
    unittest.main()
