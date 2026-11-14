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


class ProgramSelectTests(unittest.TestCase):
    """target.dual.select=program: one template.apply per command program."""
    prog = dual_target.apply(FIXTURE, enable=True, select="program")

    def test_launch_is_the_default_and_unchanged(self):
        self.assertEqual(dual_target.apply(FIXTURE, enable=True, select="launch"), dual_target.apply(FIXTURE, enable=True))

    def test_program_is_one_apply(self):
        out = self.prog
        self.assertEqual(out.count("template.apply<@dual_probe_body>"), 1)
        self.assertEqual(out.count("template.apply<@sel"), 0)
        self.assertIn("command.program.def public target(@device) @dual_probe() launch(", out)
        self.assertEqual(len(re.findall(r"^template\.def<@dual_probe_body> target\(@rdna[34]\)", out, re.M)), 2)
        m3 = re.search(r"template\.def<@dual_probe_body> target\(@rdna3\).*?\n}\n", out, re.S).group(0)
        m4 = re.search(r"template\.def<@dual_probe_body> target\(@rdna4\).*?\n}\n", out, re.S).group(0)
        self.assertEqual(m3.count(f"kernel.launch @{K}["), 2)
        self.assertEqual(m4.count(f"kernel.launch @{TWIN}["), 2)
        self.assertNotIn(f"@{K}[", m4)

    def test_kernels_unchanged(self):
        a, b = dual_target.apply(FIXTURE, enable=True), self.prog
        self.assertEqual(a[:a.index("template.decl")], b[:b.index("template.decl")])

    def test_idempotent(self):
        self.assertEqual(dual_target.apply(self.prog, enable=True, select="program"), self.prog)

    def test_group_in_program(self):
        # a group in a program: gfx11 launches kept in the rdna3 provider, the single rdna4 launch in the rdna4 one
        lines = FIXTURE.split("\n")
        i = next(n for n, l in enumerate(lines) if l.startswith("  kernel.launch @" + K))
        alt = lines[i].replace("@" + K, "@" + TWIN)
        lines[i:i + 1] = ["  // dual-group: rdna4 " + alt.strip(), lines[i], "  // dual-group: end"]
        out = dual_target.apply("\n".join(lines), enable=True, select="program")
        m4 = re.search(r"template\.def<@dual_probe_body> target\(@rdna4\).*?\n}\n", out, re.S).group(0)
        self.assertEqual(m4.count(f"kernel.launch @{TWIN}["), 2)
        self.assertNotIn("dual-group", out)

    def test_bad_mode(self):
        old = os.environ.get("LOOM_DUAL_SELECT")
        try:
            os.environ["LOOM_DUAL_SELECT"] = "bogus"
            with self.assertRaises(ValueError):
                dual_target.apply(FIXTURE, enable=True)
        finally:
            os.environ.pop("LOOM_DUAL_SELECT") if old is None else os.environ.__setitem__("LOOM_DUAL_SELECT", old)

    @unittest.skipUnless(shutil.which("loom-link") or os.path.exists("/home/jfowers/lt/loom010/.venv/bin/loom-link"), "loom-link not found")
    def test_link_keeps_one_provider_per_target(self):
        link = shutil.which("loom-link") or "/home/jfowers/lt/loom010/.venv/bin/loom-link"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dual.loom"
            p.write_text(self.prog)
            for arch, want, not_want in (("gfx1151", K + "[", TWIN), ("gfx1201", TWIN + "[", "@" + K + "[")):
                r = subprocess.run([link, str(p), "--root=@dual_probe", f"--target=amdgpu:{arch}"], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                body = r.stdout[r.stdout.index("template.def<@dual_probe_body>"):]
                self.assertEqual(body.count("kernel.launch @" + want), 2, arch)
                self.assertNotIn(not_want, body, arch)


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


class ApplyDirTests(unittest.TestCase):
    def test_cross_file_launch_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lib.loom").write_text(FIXTURE[:FIXTURE.index("command.program.def")])
            (Path(d) / "prog.loom").write_text(FIXTURE[FIXTURE.index("command.program.def"):])
            with self.assertRaisesRegex(ValueError, "defined in program/lib.loom"):
                dual_target.apply_dir(d)

    def test_dir_rewrites_defining_file_only(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "llm.loom").write_text(FIXTURE)
            (Path(d) / "control.loom").write_text("func.def @f() {\n  func.return\n}\n")
            self.assertEqual(dual_target.apply_dir(d), ["llm.loom"])
            self.assertEqual((Path(d) / "control.loom").read_text(), "func.def @f() {\n  func.return\n}\n")


class EncodingsFirstTests(unittest.TestCase):
    def test_small_file_unchanged(self):
        self.assertEqual(dual_target.encodings_first(FIXTURE), FIXTURE)

    def test_deep_file_reordered_stably(self):
        filler = "kernel.def export(\"f\") @f() {\n" + "".join(f"  %v{i} = index.constant 0 : index\n" for i in range(dual_target.ENC_LIMIT + 10)) + "}\n"
        enc = "kernel.def export(\"e\") @e() {\n  %l = encoding.layout.strided [%a, %b] : encoding<layout>\n}\n"
        text = "func.def @h() {\n}\n" + filler + enc + "command.program.def public target(@device) @p() {\n}\n"
        out = dual_target.encodings_first(text)
        self.assertEqual(sorted(out.split("\n")), sorted(text.split("\n")))
        self.assertLess(out.index('@e()'), out.index('@f()'))
        self.assertLess(out.index('@h()'), out.index('@e()'))
        self.assertTrue(out.rstrip().endswith("command.program.def public target(@device) @p() {\n}".rstrip()))
