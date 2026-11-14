"""Router and structure lint of skills/loom-kb/kb.py (CPU only)."""
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kb", REPO / "skills/loom-kb/kb.py")
kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = kb.main(list(argv))
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class RouteTests(unittest.TestCase):
    def test_arch_chain_in_order(self):
        code, out, _ = run("route", "--arch", "qwen35moe", "--device", "strix-halo")
        self.assertEqual(code, 0)
        lines = [l for l in out.splitlines() if not l.startswith("#")]
        rel = [Path(l).relative_to(REPO / "skills/loom-kb").as_posix() for l in lines]
        chain = ["arch/llm/README.md", "arch/llm/moe/README.md", "arch/llm/moe/qwen35moe/README.md"]
        self.assertEqual([r for r in rel if r in chain], chain)
        self.assertIn("facet/moe/README.md", rel)
        self.assertIn("facet/hybrid-gdn/README.md", rel)

    def test_config_derives_facets(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "llm_config.json"
            cfg.write_text(json.dumps({"arch": "qwen35", "layers": 64, "kv_layers": 16, "kv_heads": 4,
                                       "vision": {"mmproj": "x"}, "draft": {"kind": "dflash"}}))
            code, out, _ = run("route", "--config", str(cfg), "--relative")
        self.assertEqual(code, 0)
        for node in ("facet/drafted/README.md", "facet/vision/README.md", "facet/hybrid-gdn/README.md",
                     "arch/vision/README.md", "arch/llm/hybrid/qwen35/README.md"):
            self.assertIn(node, out.splitlines())
        self.assertNotIn("facet/moe/README.md", out.splitlines())

    def test_topics_and_devices(self):
        code, out, _ = run("route", "--arch", "qwen3", "--device", "rdna4", "--topic", "decode", "--relative")
        self.assertEqual(code, 0)
        self.assertIn("topic/decode.md", out.splitlines())

    def test_unknown_values_fail_with_valid_list(self):
        for argv, word in ((("route", "--arch", "nope"), "qwen3"), (("route", "--arch", "qwen3", "--device", "x"), "strix-halo"),
                           (("route", "--arch", "qwen3", "--topic", "nope"), "decode"),
                           (("route", "--arch", "qwen3", "--facets", "nope"), "moe")):
            code, _, err = run(*argv)
            self.assertNotEqual(code, 0)
            self.assertIn(word, err)

    def test_gguf_header_reader(self):
        import struct
        def s(x):
            b = x.encode()
            return struct.pack("<Q", len(b)) + b
        blob = b"GGUF" + struct.pack("<IQQ", 3, 0, 3)
        blob += s("general.architecture") + struct.pack("<I", 8) + s("gemma4")
        blob += s("gemma4.attention.head_count_kv") + struct.pack("<II", 4, 2)
        blob += s("tokenizer.tokens") + struct.pack("<IIQ", 9, 8, 2) + s("a") + s("b")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf"
            p.write_bytes(blob)
            cfg = kb.cfg_from_gguf(p)
        self.assertEqual(cfg["arch"], "gemma4")
        self.assertEqual(cfg["kv_heads"], 2)


class MatchTests(unittest.TestCase):
    def test_rules(self):
        cfg = {"arch": "x", "kv_heads": 2, "sliding_window": 512, "layers": 40, "kv_layers": 10, "draft": None}
        self.assertTrue(kb.matches({"kv_heads": [None, 2]}, cfg))
        self.assertFalse(kb.matches({"kv_heads": [3, None]}, cfg))
        self.assertTrue(kb.matches({"sliding_window": [1, None]}, cfg))
        self.assertTrue(kb.matches({"recurrent_kv": True}, cfg))
        self.assertFalse(kb.matches({"has": ["draft"]}, cfg))
        self.assertFalse(kb.matches({}, cfg))


class LintTests(unittest.TestCase):
    def make(self, files):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return root

    SKILL = "---\nname: {n}\ndescription: d\n---\n# t\n"

    def test_clean_tree(self):
        root = self.make({"skills/a/SKILL.md": self.SKILL.format(n="a")})
        self.assertEqual(kb.lint(root)[0], [])

    def test_frontmatter_size_links_and_duplicates(self):
        para = "word " * 50
        root = self.make({
            "skills/a/SKILL.md": self.SKILL.format(n="a") + "\n" * 500,
            "skills/Bad/SKILL.md": "---\nname: Bad_Name\ndescription: d\n---\n[x](missing.md)\n",
            "skills/c/SKILL.md": self.SKILL.format(n="c") + para + "\n",
            "skills/d/SKILL.md": self.SKILL.format(n="d") + para + "\n",
        })
        text = "\n".join(kb.lint(root)[0])
        self.assertIn("lines (> 500)", text)
        self.assertIn("must be 1-64 chars", text)
        self.assertIn("broken link missing.md", text)
        self.assertIn("duplicate paragraph", text)

    def test_node_rules(self):
        scope = {"scope": "arch/llm/dense/f", "match": {"arch": ["f"]}, "owner": "t", "choices": []}
        root = self.make({
            "skills/loom-kb/SKILL.md": self.SKILL.format(n="loom-kb"),
            "skills/loom-kb/arch/llm/dense/f/README.md": "# f\n[o](../g/README.md)\n" + "x\n" * 120,
            "skills/loom-kb/arch/llm/dense/f/scope.json": json.dumps(scope),
            "skills/loom-kb/arch/llm/dense/g/README.md": "# g\n",
            "skills/loom-kb/arch/llm/dense/g/scope.json": json.dumps(dict(scope, scope="arch/llm/dense/g")),
        })
        text = "\n".join(kb.lint(root)[0])
        self.assertIn("no '## Contents'", text)
        self.assertIn("links to node", text)
        self.assertIn("claimed by 2 family nodes", text)

    def test_real_repo_lints_clean(self):
        errors, _ = kb.lint(REPO)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
