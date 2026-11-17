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
    def make_pkg(self, tmp, rev):
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
        return pkg

    def test_catalog_package(self):
        if os.name == "nt":
            self.skipTest("symlinks")
        rev = "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pkg = self.make_pkg(tmp, rev)
            (pkg / "gen").mkdir()
            (pkg / "gen/flash2_cfg.json").write_text(json.dumps({
                "gguf": str(pkg / "parameter/q-00001-of-00001.gguf"), "head_dim": 512,
                "weights": {"format": "safetensors", "files": [str(tmp / "model.safetensors")]}}))
            fetch = tmp / "fetch.sh"
            fetch.write_text("#!/bin/sh\ncase \"$1\" in --help) exit 0;; esac\n"
                             "echo '{\"files\": []}' > \"$(dirname \"$1\")/model.json.lock\"\n")
            fetch.write_text(fetch.read_text() + 'touch "$(dirname "$1")/.loom-fetch.lock"\n')
            fetch.chmod(0o755)
            out = tmp / "out"
            pp.main(["prepare", str(pkg), str(out), "--source", "unsloth/Q-GGUF", "--loom-fetch", str(fetch)])
            card = json.loads((out / "model.json").read_text())
            self.assertEqual(card["origins"]["checkpoint"]["repository"], "Qwen/Q")   # kept
            self.assertEqual(card["origins"]["quantized"]["revision"], rev)             # pinned to the cache
            entry = card["variants"]["v"]["services"]["chat"]["parameters"]["model"]["files"][0]
            self.assertEqual(entry["origin"], "@quantized/UD/q-00001-of-00001.gguf")
            self.assertFalse((out / "parameter").exists())
            saved = json.loads((out / "gen/flash2_cfg.json").read_text())
            self.assertEqual(saved, {"gguf": "q-00001-of-00001.gguf", "head_dim": 512,
                                    "weights": {"format": "safetensors", "files": ["model.safetensors"]}})
            self.assertEqual((out / "program/a.loom").read_text(), "x")
            self.assertFalse((out / ".loom-fetch.lock").exists())

    def test_unexpected_fetch_lock_is_preserved_and_refused(self):
        if os.name == "nt":
            self.skipTest("POSIX mock fetch and symlinks")
        for command in ('echo unexpected > "$out/.loom-fetch.lock"',
                        'mkdir "$out/.loom-fetch.lock"',
                        'ln -s model.json.lock "$out/.loom-fetch.lock"'):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                pkg = self.make_pkg(tmp, "b" * 40)
                fetch = tmp / "fetch.sh"
                fetch.write_text('#!/bin/sh\ncase "$1" in --help) exit 0;; esac\n'
                                 'out=$(dirname "$1")\n'
                                 "echo '{\"files\": []}' > \"$out/model.json.lock\"\n" + command + '\n')
                fetch.chmod(0o755)
                out = tmp / "out"
                with self.assertRaisesRegex(SystemExit, "unexpected transient fetch lock"):
                    pp.main(["prepare", str(pkg), str(out), "--source", "unsloth/Q-GGUF", "--loom-fetch", str(fetch)])
                self.assertTrue((out / ".loom-fetch.lock").exists())

    def test_quiet_only_when_loom_fetch_lists_it(self):
        """hrx-loom 0.1.0's loom-fetch has no --quiet and exits 2 on it: prepare drops the flag there."""
        if os.name == "nt":
            self.skipTest("symlinks")
        rev = "b" * 40
        for lists_quiet in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                pkg = self.make_pkg(tmp, rev)
                fetch = Path(tmp) / "fetch.sh"
                fetch.write_text("#!/bin/sh\n"
                                 "case \"$1\" in --help) echo 'usage: loom-fetch [--no-input] %s'; exit 0;; esac\n"
                                 "for a in \"$@\"; do [ \"$a\" = --quiet ] && %s; done\n"
                                 "echo '{\"files\": []}' > \"$(dirname \"$1\")/model.json.lock\"\n"
                                 % ("[--quiet]" if lists_quiet else "", "echo seen-quiet >> \"$(dirname \"$1\")/quiet\"" if lists_quiet else "exit 2"))
                fetch.chmod(0o755)
                self.assertEqual(pp.fetch_supports(str(fetch), "--quiet"), lists_quiet)
                out = Path(tmp) / "out"
                pp.main(["prepare", str(pkg), str(out), "--source", "unsloth/Q-GGUF", "--loom-fetch", str(fetch)])
                self.assertEqual((out / "quiet").exists(), lists_quiet)
        self.assertFalse(pp.fetch_supports("/nonexistent/loom-fetch", "--quiet"))


if __name__ == "__main__":
    unittest.main()
