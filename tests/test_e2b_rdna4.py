"""gfx12 fusions and flash twins (f16o_fuse.py, dual_target launch groups, flash2_*.py, ktest_select)."""
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))
import dual_target  # noqa: E402
import f16o_fuse  # noqa: E402
import flash2_trim  # noqa: E402
import gen_gemm_rdna4 as R  # noqa: E402
import ktest_select  # noqa: E402

GROUP_LIB = '''kernel.def export("A") @A(%n: index) {
}
kernel.def target(@rdna4) export("A_rdna4") @A_rdna4(%n: index) {
}
kernel.def target(@rdna4) export("F_rdna4") @F_rdna4(%n: index) {
}
command.program.def public @p() launch() {
  // dual-group: rdna4 kernel.launch @F_rdna4[%t](%t, %x, %y) : [index](index, buffer, buffer)
  kernel.launch @A[%t](%t, %x) : [index](index, buffer)
  kernel.launch @B[%t](%t, %y, %x) : [index](index, buffer, buffer)
  // dual-group: end
}
'''


class LaunchGroups(unittest.TestCase):
    def test_group_becomes_a_template_with_two_providers(self):
        out = dual_target.apply(GROUP_LIB, enable=True)
        self.assertIn("template.apply<@grp1_F_rdna4>(%t, %x, %y) : (index, buffer, buffer)", out)
        g3 = re.search(r"template\.def<@grp1_F_rdna4> target\(@rdna3\)[^\n]*\{\n(.*?)  template\.return", out, re.S).group(1)
        self.assertEqual(g3.count("kernel.launch"), 2)                      # the gfx11 provider launches both kernels, in order
        self.assertLess(g3.index("@A["), g3.index("@B["))
        g4 = re.search(r"template\.def<@grp1_F_rdna4> target\(@rdna4\)[^\n]*\{\n(.*?)  template\.return", out, re.S).group(1)
        self.assertEqual(g4.count("kernel.launch"), 1)
        self.assertIn("@F_rdna4[", g4)
        self.assertEqual(dual_target.apply(out, enable=True), out)           # idempotent
        self.assertEqual(dual_target.apply(GROUP_LIB, enable=False), GROUP_LIB)

    def test_templates_follow_the_gfx12_only_kernels(self):
        out = dual_target.apply(GROUP_LIB, enable=True)
        self.assertLess(out.index('export("F_rdna4")'), out.index("template.decl @grp1_F_rdna4"))


class Ktest(unittest.TestCase):
    LIB = ('kernel.def target(@rdna3) export("A") @A(%n: index) {\n}\nkernel.def target(@rdna4) export("A_rdna4") @A_rdna4(%n: index) {\n}\n'
           'kernel.def target(@rdna4) export("F_rdna4") @F_rdna4(%n: index) {\n}\n')
    CASES = 'check.case public @c1 {\n kernel.launch @A[%x](%x)\n}\ncheck.case public @c2 {\n kernel.launch @F_rdna4[%x](%x)\n}\n'

    def test_gfx12_only_kernel_cases_run_on_gfx12_only(self):
        t12, s12 = ktest_select.select(self.LIB, self.CASES, "gfx1201", rdna4=True)
        t11, s11 = ktest_select.select(self.LIB, self.CASES, "gfx1151", rdna4=True)
        self.assertEqual((t12.count("check.case"), s12), (2, []))
        self.assertEqual(t11.count("check.case"), 1)
        self.assertTrue(any("gfx12-only" in s for s in s11))


class F16OutKernel(unittest.TestCase):
    def test_swiglu_twin_stores_f16_and_writes_the_scales(self):
        fixture = (ROOT / "tests/data/dual_target_gemm.loom").read_text()
        env = dict(os.environ, GEMM_FORMATS="q4_k:swiglu", LOOM_RDNA4="1", GEMM_F16O_ON="swiglu", GEMM_SKIP_PADDING="1", GEMM_N_FASTEST="1")
        gen = subprocess.run([sys.executable, str(STARTER / "gen_gemm.py")], capture_output=True, text=True, check=True, env=env).stdout
        ks = [k for k in re.split(r"(?m)^(?=kernel\.def )", gen) if "_f16o_rdna4" in k.split("\n", 1)[0]]
        self.assertEqual(len(ks), 3)                                          # the plain, _m128 and _m256 tile kernels
        for k in ks:
            self.assertIn("%scales_out: buffer", k)
            self.assertIn("view.store %so_val, %so_view[%tile_row, %c0]", k)
            self.assertNotIn("xf32, %y_layout>", k)                          # the result view is f16
            self.assertIn("xf16, %y_layout>", k)
        off = subprocess.run([sys.executable, str(STARTER / "gen_gemm.py")], capture_output=True, text=True, check=True,
                             env=dict(os.environ, GEMM_FORMATS="q4_k:swiglu", LOOM_RDNA4="1", GEMM_SKIP_PADDING="1", GEMM_N_FASTEST="1")).stdout
        self.assertNotIn("f16o", off)                                         # off by default
        self.assertTrue(fixture)

    def test_switch(self):
        os.environ.pop("GEMM_F16O", None)
        self.assertFalse(R.f16o_swiglu({}))
        self.assertTrue(R.f16o_swiglu({"rdna4_f16_ffn": True}))
        os.environ["GEMM_F16O"] = "0"
        try:
            self.assertFalse(R.f16o_swiglu({"rdna4_f16_ffn": True}))
            os.environ["GEMM_F16O"] = "swiglu"
            self.assertTrue(R.f16o_swiglu({}))
        finally:
            os.environ.pop("GEMM_F16O", None)


class FuseProgram(unittest.TestCase):
    PROGRAM = '''command.program.def public @llm_advance() launch() {
  scf.if %mm {
    scf.if %big {
      scf.if %huge {
        kernel.launch @g_gemm_q4_k_swiglu_m256[%tp, %nF](%tp, %nH, %nF, %h16, %s_h, %wg, %wu, %f) : [index, index](index, index, index, buffer, buffer, view<8xi8>, view<8xi8>, buffer)
      } else {
        kernel.launch @g_gemm_q4_k_swiglu_m128[%tp, %nF](%tp, %nH, %nF, %h16, %s_h, %wg, %wu, %f) : [index, index](index, index, index, buffer, buffer, view<8xi8>, view<8xi8>, buffer)
      }
    } else {
      kernel.launch @g_gemm_q4_k_swiglu[%tp, %nF](%tp, %nH, %nF, %h16, %s_h, %wg, %wu, %f) : [index, index](index, index, index, buffer, buffer, view<8xi8>, view<8xi8>, buffer)
    }
  } else {
    kernel.launch @g_mv_q4_k_swiglu[%nF, %t](%nF, %t, %h16q, %wg, %wu, %f) : [index, index](index, index, buffer, view<8xi8>, view<8xi8>, buffer)
  }
  scf.if %mm {
    kernel.launch @g_row_scale_f16_live_lv[%t](%t, %info, %nF, %f, %s_f, %f16) : [index](index, buffer, index, buffer, buffer, buffer)
  } else {
    kernel.launch @g_quant_q8[%t, %nF](%t, %nF, %f, %f16q) : [index, index](index, index, buffer, buffer)
  }
}
'''

    def test_every_gemm_leaf_gets_the_row_scale_in_a_group(self):
        out, n = f16o_fuse.fuse(self.PROGRAM)
        self.assertEqual(n, 1)
        self.assertEqual(out.count("// dual-group: rdna4 kernel.launch @g_gemm_q4_k_swiglu"), 3)
        self.assertEqual(out.count("// dual-group: end"), 3)
        self.assertEqual(out.count("@g_row_scale_f16_live_lv"), 3)           # one per leaf; none left in the second scf.if
        self.assertIn("@g_gemm_q4_k_swiglu_m256_f16o_rdna4[%tp, %nF](%tp, %nH, %nF, %h16, %s_h, %wg, %wu, %f16, %s_f)", out)
        self.assertIn("scf.if %mm {\n  } else {\n    kernel.launch @g_quant_q8", out)
        g = dual_target.apply('kernel.def export("g_gemm_q4_k_swiglu_m256") @g_gemm_q4_k_swiglu_m256(%n: index) {\n}\n'
                              'kernel.def target(@rdna4) export("g_gemm_q4_k_swiglu_m256_rdna4") @g_gemm_q4_k_swiglu_m256_rdna4(%n: index) {\n}\n' + out, enable=True)
        self.assertIn("template.apply<@grp1_g_gemm_q4_k_swiglu_m256_f16o_rdna4>", g)

    def test_no_match_no_change(self):
        text = self.PROGRAM.replace("%f, %s_f, %f16)", "%other, %s_f, %f16)")
        self.assertEqual(f16o_fuse.fuse(text), (text, 0))


class FlashTrims(unittest.TestCase):
    def test_lazy_rescale_is_skipped_only_when_no_column_changed(self):
        text = "".join(f"    %os{d} = vector.mulf %a{d}, %alpha_v : vector<8xf32>\n    %vr{d} = index.add %dim_base, %qd{d}_o : index\n" for d in range(2))
        out = flash2_trim.lazy("  %x = 1\n" + text)
        self.assertIn("kernel.subgroup.vote.any %lz_chg", out)
        self.assertIn("scf.if %lz_any -> (vector<8xf32>, vector<8xf32>)", out)
        self.assertIn("scf.yield %a0, %a1", out)                              # the else branch keeps the accumulators
        self.assertEqual(out.count("%vr0 ="), 1)

    def test_hot_copy_trims(self):
        lines = ["    %key_valid = index.cmp ule, %key_pos, %pos_last : index", "    %masked0 = vector.select %valid0, %sc0, %neg_v : vector<8xf32>",
                 "    %p0 = vector.select %valid0, %pe0, %zero8 : vector<8xf32>"]
        self.assertEqual(flash2_trim.hot(lines, False)[1:], lines[1:])
        out = flash2_trim.hot(lines, True)
        self.assertEqual(out[0], "    %key_valid = scalar.constant true : i1")
        self.assertTrue(all("select" not in o for o in out))


if __name__ == "__main__":
    unittest.main()
