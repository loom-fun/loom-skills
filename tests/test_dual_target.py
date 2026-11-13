import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))
import dual_target  # noqa: E402
import ktest_select  # noqa: E402

# A q4_k GEMM (g_gemm_q4_k_store, from a generated llm.loom) with its gfx12 twin g_gemm_q4_k_store_rdna4, and a
# command program that launches it twice with one signature.
FIXTURE = (ROOT / "tests/data/dual_target_gemm.loom").read_text()
K, TWIN = "g_gemm_q4_k_store", "g_gemm_q4_k_store_rdna4"
CASES = (ROOT / "tests/data/dual_target_cases.loom-test").read_text()   # a GEMM case, and one of a kernel not in the build


def loom_compile():
    for c in (os.environ.get("LOOM_COMPILE"), shutil.which("loom-compile"), "/home/jfowers/lt/loom010/.venv/bin/loom-compile"):
        if c and os.path.exists(c):
            return c


class DualTargetTests(unittest.TestCase):
    def apply(self, text, **kw):
        return dual_target.apply(text, enable=True, **kw)

    def test_twin_detected(self):
        self.assertEqual(dual_target.twins(FIXTURE), [K])

    def test_rewrite(self):
        out = self.apply(FIXTURE)
        self.assertTrue(out.startswith("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\namdgpu.target<gfx12-generic> @rdna4\n"))
        self.assertIn(f'kernel.def target(@rdna3) export("{K}") @{K}(', out)
        self.assertIn(f'kernel.def target(@rdna4) export("{TWIN}") @{TWIN}(', out)
        self.assertNotIn("kernel.launch @" + K + "[%t", out)           # the program launches go through the template
        self.assertEqual(out.count("template.apply<@sel1_" + K + ">"), 2)
        self.assertEqual(out.count("template.decl"), 1)                 # one template per launch signature
        self.assertEqual(len(re.findall(r"^template\.def<@sel1_\w+> target\(@rdna[34]\)", out, re.M)), 2)
        self.assertIn("command.program.def public target(@device) @dual_probe", out)

    def test_gfx11_kernel_text_only_gains_the_tag(self):
        out = self.apply(FIXTURE)
        self.assertEqual(out.replace("kernel.def target(@rdna3) export", "kernel.def export", 1).count(
            FIXTURE[FIXTURE.index(f'kernel.def export("{K}")'):FIXTURE.index(f'kernel.def target(@rdna4) export("{TWIN}")')]), 1)

    def test_idempotent(self):
        once = self.apply(FIXTURE)
        self.assertEqual(self.apply(once), once)

    def test_no_twin_untouched(self):
        plain = FIXTURE.replace(TWIN, "g_other")
        plain = plain[:plain.index('kernel.def target(@rdna4)')] + plain[plain.index("command.program.def") - 1:]
        self.assertEqual(dual_target.twins(plain), [])
        self.assertEqual(self.apply(plain), plain)

    def test_env_off(self):
        old = os.environ.get("LOOM_RDNA4")
        try:
            os.environ["LOOM_RDNA4"] = "0"
            self.assertFalse(dual_target.enabled())
            self.assertEqual(dual_target.apply(FIXTURE), FIXTURE)
            os.environ["LOOM_RDNA4"] = "1"
            self.assertNotEqual(dual_target.apply(FIXTURE), FIXTURE)
        finally:
            if old is None:
                os.environ.pop("LOOM_RDNA4", None)
            else:
                os.environ["LOOM_RDNA4"] = old

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.loom"
            p.write_text(FIXTURE)
            r = subprocess.run([sys.executable, str(STARTER / "dual_target.py"), str(p)], capture_output=True, text=True,
                               env={**os.environ, "LOOM_RDNA4": "1"})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(p.read_text(), self.apply(FIXTURE))

    @unittest.skipUnless(loom_compile(), "loom-compile not found (set LOOM_COMPILE)")
    def test_compiles_for_both_targets(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dual.loom"
            p.write_text(self.apply(FIXTURE))
            for arch, root in (("gfx1151", K), ("gfx1201", TWIN)):
                out = Path(d) / f"{arch}.hsaco"
                r = subprocess.run([loom_compile(), str(p), f"--target=amdgpu:{arch}", f"--root={root}",
                                    "--format=amdgpu-hsaco", f"--output={out}"], capture_output=True, text=True)
                self.assertTrue(out.exists() and out.stat().st_size > 0, f"{arch}: {r.stdout}{r.stderr}")


class KernelTestSelectTests(unittest.TestCase):
    lib = dual_target.apply(FIXTURE, enable=True)

    def test_gfx12_launches_the_twin(self):
        text, skipped = ktest_select.select(self.lib, CASES, "gfx1201", rdna4=True)
        self.assertIn(f"kernel.launch @{TWIN}[", text)
        self.assertNotIn(f"kernel.launch @{K}[", text)
        self.assertEqual(len(skipped), 1)   # the case of the kernel that is not in the build

    def test_gfx11_and_rdna4_off_launch_the_original(self):
        for arch, on in (("gfx1151", True), ("gfx1201", False), ("", True)):
            text, _ = ktest_select.select(self.lib, CASES, arch, rdna4=on)
            self.assertIn(f"kernel.launch @{K}[", text)
            self.assertNotIn(TWIN + "[", text)

    def test_gfx11_only_kernel_is_skipped_on_gfx12(self):
        lib = self.lib.replace(f'kernel.def target(@rdna4) export("{TWIN}") @{TWIN}(', 'kernel.def target(@rdna4) export("g_other_rdna4") @g_other_rdna4(')
        text, skipped = ktest_select.select(lib, CASES, "gfx1201", rdna4=True)
        self.assertNotIn("check.case public @gemm_q4_k_store", text)
        self.assertTrue(any("gfx11-only" in m for m in skipped), skipped)
        text, _ = ktest_select.select(lib, CASES, "gfx1151", rdna4=True)
        self.assertIn("check.case public @gemm_q4_k_store", text)

    @unittest.skipUnless(loom_compile(), "loom-compile not found (set LOOM_COMPILE)")
    def test_combined_files_compile(self):
        with tempfile.TemporaryDirectory() as d:
            for arch, root in (("gfx1151", K), ("gfx1201", TWIN)):
                text, _ = ktest_select.select(self.lib, CASES, arch, rdna4=True)
                p = Path(d) / f"{arch}.loom"
                p.write_text(self.lib + text)
                out = Path(d) / f"{arch}.hsaco"
                r = subprocess.run([loom_compile(), str(p), f"--target=amdgpu:{arch}", f"--root={root}",
                                    "--format=amdgpu-hsaco", f"--output={out}"], capture_output=True, text=True)
                self.assertTrue(out.exists() and out.stat().st_size > 0, f"{arch}: {r.stdout}{r.stderr}")


if __name__ == "__main__":
    unittest.main()
