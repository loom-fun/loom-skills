"""make_kernel_tests.py: the big-pool attention case (a KV pool of > 2^20 positions) exists for every chain case of a kit.

The case used to be derived from `cases[-1]`, which gemma4's per-head-dim chains (`@attention_chain_d<D>`, followed by `@ple_prep`) are not: the
generation of unsloth-gemma-4-E2B-it's kernel tests died on an assert (0.2.0 sweep: ktest ERROR on both devices)."""
import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "fleet"
HAVE = all(importlib.util.find_spec(m) for m in ("gguf", "numpy", "jinja2", "yaml"))


class HeadSuffix(unittest.TestCase):
    def test_attention_child_preserves_suffixed_variant_data_paths(self):
        path = ROOT / "skills/loom-port-llm/starter/make_kernel_tests.py"
        body = ast.parse(path.read_text(encoding="utf-8")).body
        suffix_block = next(node for node in reversed(body)
                            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "KS")
        case = '''check.case public @attention_chain_listed {
  %q = check.file.read.npy path("q.npy") : tensor<1xf32>
  %k = check.file.read.npy path("klist_4_h48.npy") : tensor<1xi32>
}
'''
        context = {"cases": [case], "KS": "_h48", "re": re}
        program = ast.Module(body=[suffix_block], type_ignores=[])
        exec(compile(program, str(path), "exec"), context)
        (child,) = context["cases"]
        self.assertIn("@attention_chain_listed_h48", child)
        self.assertEqual(re.findall(r'path\("([^"]+)"\)', child), ["q_h48.npy", "klist_4_h48.npy"])


def bigpool_cases(slug):
    """build_fleet.py's stub package kit for SLUG, its kernel tests generated: the text of cases.loom-test and the generation's stdout."""
    with tempfile.TemporaryDirectory() as w:
        out = os.path.join(w, "fleet")
        r = subprocess.run([sys.executable, str(FLEET / "build_fleet.py"), "--tree", str(ROOT), "--out", out, "--slugs", slug, "--keep-work", "--jobs", "1"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
        work = os.path.join(out, "_work", slug)
        kt = os.path.join(w, "kt")
        env = dict(os.environ, LOOM_KB_ROOT=str(ROOT / "skills" / "loom-kb"))
        r = subprocess.run([sys.executable, "kit/make_kernel_tests.py", "cfg.json", kt], cwd=work, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-1200:]
        with open(os.path.join(kt, "cases.loom-test")) as f:
            text = f.read()
        files = set(os.listdir(kt))
        return text, files, r.stdout


@unittest.skipUnless(HAVE, "needs gguf numpy jinja2 pyyaml (fleet/preflight.py)")
class BigPool(unittest.TestCase):
    def check(self, slug, names):
        text, files, _ = bigpool_cases(slug)
        got = re.findall(r"check\.case public @(attention_bigpool\w*) \{", text)
        self.assertEqual(sorted(got), sorted(names))
        for npy in re.findall(r'path\("(a_orig_big\d+\.npy)"\)', text):
            self.assertIn(npy, files)
        for case in re.findall(r"check\.case public @attention_bigpool\w* \{.*?\n\}\n", text, re.S):
            self.assertGreater(int(re.search(r"%pool = check\.literal value\((\d+)\)", case)[1]), 1048576)

    def test_plain_chain(self):
        self.check("bartowski-Qwen2.5-0.5B-Instruct-loom", ["attention_bigpool"])

    def test_per_head_dim_chains_of_gemma4(self):
        self.check("unsloth-gemma-4-E2B-it-loom", ["attention_bigpool_d256", "attention_bigpool_d512"])


if __name__ == "__main__":
    unittest.main()
