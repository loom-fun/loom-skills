import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/loom-port-llm/starter"))
import dual_target  # noqa: E402
import ktest_select  # noqa: E402
import split_library  # noqa: E402

LIB = dual_target.apply((ROOT / "tests/data/dual_target_gemm.loom").read_text(), enable=True)
K, TWIN = "g_gemm_q4_k_store", "g_gemm_q4_k_store_rdna4"


def env(**kw):
    class Env:
        def __enter__(self):
            self.old = {k: os.environ.get(k) for k in kw}
            os.environ.update(kw)

        def __exit__(self, *a):
            for k, v in self.old.items():
                os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    return Env()


def deep_lib(kernels=4, values=20000):
    """A dual-target-shaped library whose encoding kernels alone are deeper than the parser resolves."""
    ks = "".join(f'kernel.def target(@rdna3) export("e{j}") @e{j}(%n: index) {{\n  kernel.return %n, %n, %n : index, index, index\n'
                 f'}} launch(%x: buffer) {{\n' + "".join(f"  %v{i} = index.constant 0 : index\n" for i in range(values))
                 + "  %l = encoding.layout.strided [1, 16] : encoding<layout>\n  %r = func.call @h() : () -> (index)\n  kernel.return\n}\n"
                 for j in range(kernels))
    return (dual_target.HEADER + "func.def inline @h() -> (index) {\n  func.return\n}\n" + ks
            + "command.def public target(@device) @p() launch(%x: buffer) {\n  kernel.launch @e0[%c](%x) : [index](buffer)\n}\n")


class SplitTests(unittest.TestCase):
    def test_auto_leaves_small_library(self):
        with env(LOOM_LIBRARY_SPLIT="auto"):
            self.assertFalse(split_library.needs_split(LIB))
        with env(LOOM_LIBRARY_SPLIT="always"):
            self.assertTrue(split_library.needs_split(LIB))
        with env(LOOM_LIBRARY_SPLIT="off"):
            self.assertFalse(split_library.needs_split(deep_lib()))

    def test_auto_splits_too_deep(self):
        text = deep_lib()
        with env(LOOM_LIBRARY_SPLIT="auto"):
            self.assertTrue(split_library.needs_split(text))
            prog, files = split_library.split_text(text)
        self.assertGreaterEqual(len(files), 2)
        for f in files:
            self.assertLessEqual(dual_target._enc_depth(f), dual_target.ENC_LIMIT)
            self.assertIn("func.def inline @h()", f)              # the helper rides with the kernels that call it
            self.assertTrue(f.startswith(dual_target.HEADER))
        self.assertEqual(sorted(re.findall(r"@e\d", "".join(files))), [f"@e{j}" for j in range(4)])
        self.assertEqual(len(re.findall(r"^kernel\.decl @e\d\(%n: index\) launch\(%x: buffer\)$", prog, re.M)), 4)
        self.assertNotIn("encoding<layout>", prog)
        self.assertIn("kernel.launch @e0[", prog)

    def test_twin_stays_with_kernel(self):
        with env(LOOM_LIBRARY_SPLIT="always"):
            prog, files = split_library.split_text(LIB)
        self.assertEqual(len(files), 1)
        self.assertIn(f"@{K}(", files[0])
        self.assertIn(f"@{TWIN}(", files[0])
        self.assertIn(f"kernel.decl @{TWIN}(", prog)
        self.assertEqual(prog.count("template.def<"), LIB.count("template.def<"))   # selection unchanged

    def test_dir_card_and_rerun(self):
        with tempfile.TemporaryDirectory() as d, env(LOOM_LIBRARY_SPLIT="always"):
            os.makedirs(os.path.join(d, "program"))
            Path(d, "program/llm.loom").write_text(LIB)
            card = {"programs": {"source": {"libraries": ["program/llm.loom", "program/vision.loom"]}}}
            self.assertEqual(split_library.split_dir(d, card), ["llm_kernels_1.loom"])
            self.assertEqual(card["programs"]["source"]["libraries"],
                             ["program/llm_kernels_1.loom", "program/llm.loom", "program/vision.loom"])
            self.assertIn(f"@{K}(", split_library.library_text(os.path.join(d, "program")))
            # dual_target over the split files: a no-op, and the program file's declared launches are not a cross-file error
            self.assertEqual(dual_target.apply_dir(os.path.join(d, "program")), [])
            # a rebuild that no longer needs the split removes the stale kernel file
            Path(d, "program/llm.loom").write_text(LIB)
            with env(LOOM_LIBRARY_SPLIT="off"):
                self.assertEqual(split_library.split_dir(d, {}), [])
            self.assertEqual(split_library.kernel_files(os.path.join(d, "program")), [])

    def test_ktest_groups(self):
        cases = (ROOT / "tests/data/dual_target_cases.loom-test").read_text()
        with tempfile.TemporaryDirectory() as d, env(LOOM_LIBRARY_SPLIT="always"):
            os.makedirs(os.path.join(d, "program"))
            Path(d, "program/llm.loom").write_text(LIB)
            split_library.split_dir(d, {})
            lib = split_library.library_text(os.path.join(d, "program"))
            sel, _ = ktest_select.select(lib, cases, "gfx1151")
            groups, skipped = ktest_select.split_groups(os.path.join(d, "program"), "llm.loom", sel)
            self.assertEqual(skipped, [])
            self.assertEqual(list(groups), ["llm_kernels_1.loom"])
            self.assertIn(f"kernel.launch @{K}[", groups["llm_kernels_1.loom"])

    @unittest.skipUnless(os.environ.get("LOOM_LINK") or shutil.which("loom-link"), "loom-link not found (set LOOM_LINK)")
    def test_links_through_the_decl(self):
        link = os.environ.get("LOOM_LINK") or shutil.which("loom-link")
        with tempfile.TemporaryDirectory() as d, env(LOOM_LIBRARY_SPLIT="always"):
            os.makedirs(os.path.join(d, "program"))
            Path(d, "program/llm.loom").write_text(LIB)
            card = {"programs": {"source": {"libraries": ["program/llm.loom"]}}}
            split_library.split_dir(d, card)
            libs = [f"--library={os.path.join(d, p)}" for p in card["programs"]["source"]["libraries"]]
            for arch, want in (("gfx1151", K), ("gfx1201", TWIN)):
                r = subprocess.run([link, *libs, "--root=@dual_probe", f"--target=amdgpu:{arch}"], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(f'export("{want}") @{want}(', r.stdout, arch)   # the definition, pulled from the kernel file


if __name__ == "__main__":
    unittest.main()
