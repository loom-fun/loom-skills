"""Gemma4 shared-KV pruning must affect advance only and retain all verify rows."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "skills/loom-port-llm"


class SharedPruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.kit = Path(cls.tmp.name) / "kit"
        shutil.copytree(PORT / "starter", cls.kit)
        subprocess.run(["patch", "-s", "-p1", "-d", str(cls.kit)],
                       input=(PORT / "reference/gemma4-example/starter.patch").read_text(), text=True, check=True)
        tree = ast.parse((cls.kit / "gen_llm.py").read_text())
        picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prune_shared_rows"
                  or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TOKEN_INFO_TAIL" for t in n.targets)]
        cls.ns = {"CFG": {"kv_shared_from": 15}, "os": os, "re": __import__("re")}
        exec(compile(ast.Module(body=picked, type_ignores=[]), "pruning", "exec"), cls.ns)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.ns["CFG"] = {"kv_shared_from": 15}

    @staticmethod
    def module():
        prologue = "\n".join(("%info_size = 1", "%list_size = 1", "%count_size = 1", "%flash_tiles = buffer.alloca",
                              "%fallback_list = buffer.alloca", "%fallback_count = buffer.alloca",
                              "@g_attention_fallback_list[%t](%t, %info, %flash_tiles, %fallback_list, %fallback_count)"))
        return ("command.def public @llm_advance() {\n" + prologue + "\n  // ---- layer 14 owner\n%info %info_size\n"
                "  // ---- layer 15 shared\n%info %info_size %info_other %flash_tiles %fallback_list %fallback_count\n}\n"
                "command.def public @llm_verify() {\n%info %flash_tiles\n}\n")

    def test_only_shared_advance_tail_is_rewritten(self):
        source = self.module()
        with patch.dict(os.environ, {"PRUNE_SHARED": "1"}):
            out = self.ns["prune_shared_rows"](source)
        self.assertEqual(out.split("command.def public @llm_verify", 1)[1], source.split("command.def public @llm_verify", 1)[1])
        owner, tail = out.split("  // ---- layer 15 shared", 1)
        self.assertIn("layer 14 owner\n%info %info_size", owner)
        self.assertIn("%info_t %info_size %info_other %flash_tiles_t %fallback_list_t %fallback_count_t", tail)
        self.assertEqual(out.count('kernel.def export("g_token_info_tail")'), 1)
        self.assertIn("@g_attention_fallback_list[%t](%t, %info_t,", owner)

    def test_dflash_taps_keep_all_feature_rows(self):
        source = self.module()
        source = source.replace("\n}\ncommand.def public @llm_verify", "\n  // ---- layer 16 shared\n%info\n"
                                "  kernel.launch @g_dfl_tap[%t](%t, %slot, %x, %features)\n"
                                "  // ---- layer 17 shared\n%info\n}\ncommand.def public @llm_verify", 1)
        with patch.dict(os.environ, {"PRUNE_SHARED": "1"}):
            out = self.ns["prune_shared_rows"](source)
        kept, pruned = out.split("  // ---- layer 17 shared", 1)
        self.assertIn("layer 16 shared\n%info\n  kernel.launch @g_dfl_tap", kept)
        self.assertIn("layer 15 shared\n%info %info_size %info_other", kept)
        self.assertIn("\n%info_t\n}", pruned)
        self.assertLess(kept.index("@g_dfl_tap[%t]"), kept.index("// KV-shared layers:"))

    def test_final_layer_tap_leaves_program_unchanged(self):
        source = self.module().replace("\n}\ncommand.def public @llm_verify", "\n  kernel.launch @g_dfl_tap[%t](%t, %slot, %x, %features)\n}\ncommand.def public @llm_verify", 1)
        with patch.dict(os.environ, {"PRUNE_SHARED": "1"}):
            self.assertEqual(self.ns["prune_shared_rows"](source), source)

    def test_opt_out_and_unshared_models_are_identical(self):
        source = self.module()
        with patch.dict(os.environ, {"PRUNE_SHARED": "0"}):
            self.assertEqual(self.ns["prune_shared_rows"](source), source)
        for first in (None, 0):
            self.ns["CFG"] = {"kv_shared_from": first}
            with patch.dict(os.environ, {"PRUNE_SHARED": "1"}):
                self.assertEqual(self.ns["prune_shared_rows"](source), source)

    def test_lost_anchor_fails_closed(self):
        with patch.dict(os.environ, {"PRUNE_SHARED": "1"}):
            for source in ("command.def public @llm_verify() {}", self.module().replace("%count_size =", "%renamed =")):
                with self.assertRaisesRegex(ValueError, "shared-KV pruning"):
                    self.ns["prune_shared_rows"](source)

    def test_chunk_default_and_explicit_overrides(self):
        tree = ast.parse((PORT / "starter/build.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "a.prefill is None")
        for arch, shared, moe, explicit, configured, expected in (
            ("gemma4", 15, False, None, None, 2048), ("gemma4", None, False, None, None, 1024),
            ("qwen3", 15, False, None, None, 1024), ("gemma4", None, True, None, None, 4096),
            ("gemma4", 15, False, 1024, None, 1024), ("gemma4", 15, False, None, 4096, 4096)):
            args = type("Args", (), {"prefill": explicit})()
            ns = {"a": args, "cfg": {"arch": arch, "kv_shared_from": shared, "moe": moe, "prefill_capacity": configured}}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "build", "exec"), ns)
            self.assertEqual(args.prefill, expected)


if __name__ == "__main__":
    unittest.main()
