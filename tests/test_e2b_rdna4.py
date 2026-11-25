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
command.def public @p() launch() {
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
    PROGRAM = '''command.def public @llm_advance() launch() {
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


class FlashKeySplitD256(unittest.TestCase):
    """gen_flash2s.py FLASH2_B=2 / 3 at head_dim 256 with 6 query heads per KV head (Qwen3.5-class 27B): the key-split loop of flash2_b2.py."""
    CFG = ROOT / "fleet/amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom/llm_config.json"

    def gen(self, tmp, **env):
        out = Path(tmp) / "flash2.loom"
        e = dict(os.environ, LLM_CONFIG=str(self.CFG), FLASH2_TARGET="llm", FLASH2_OUT=str(out), **env)
        for k in ("FLASH2_B", "FLASH2_TRIM"):
            if k not in env:
                e.pop(k, None)
        subprocess.run([sys.executable, str(STARTER / "gen_flash2.py")], check=True, env=e, capture_output=True)
        return out.read_text()

    def test_default_keeps_the_staged_twin(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotIn("%ql_view", self.gen(tmp))

    def test_two_and_three_heads_per_wave(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            t2 = self.gen(tmp, FLASH2_B="2")
            t3 = self.gen(tmp, FLASH2_B="3")
        for t in (t2, t3):
            self.assertEqual(t.count('export("g_attention_flash2_rdna4")'), 1)
            self.assertEqual(t.count('export("g_attention_flash2_partial_rdna4")'), 1)
            self.assertIn("%clanes = index.constant 128 : index", t)               # the four waves of a head group
            self.assertIn("%head1 = index.assume", t)
        self.assertIn("workgroups(%tiles, %ckv, %c1)", t2)
        self.assertIn("%ckv = index.constant 12 : index", t2)                       # 4 KV heads x 3 head pairs
        self.assertIn("view<2x32x136xf16>", t2)
        self.assertNotIn("%head2 =", t2)
        self.assertIn("%ckv = index.constant 8 : index", t3)                        # 4 KV heads x 2 head triples
        self.assertIn("view<3x32x136xf16>", t3)
        self.assertIn("%head2 = index.assume", t3)

    def test_fixed_reference(self):
        """FLASH2_B2_FM=1: both gfx12 kernels (whole and partial tiles: cut invariance) get the fixed-reference loops (unmasked + masked tail)
        and the per-lane second pass; the gfx11 kernel is unchanged."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            t2 = self.gen(tmp, FLASH2_B="2")
            fm = self.gen(tmp, FLASH2_B="2", FLASH2_B2_FM="1")
        r4 = 'export("g_attention_flash2_rdna4")'
        self.assertEqual(t2[:t2.index(r4)], fm[:fm.index(r4)])
        cut = fm.index('export("g_attention_flash2_partial_rdna4")')
        for w in (fm[fm.index(r4):cut], fm[cut:]):
            self.assertIn("%sinitA = vector.fragment<init> %fm_nmv", w)              # S^T accumulated onto -m0
            self.assertIn("%sinit2A = vector.fragment<init> %fm2_nmv", w)            # second pass: overflowed lanes at their true max
            self.assertIn("[%c0 to %fm_nfull step %c1]", w)                           # unmasked loop below the diagonal
            self.assertIn("[%fm_nfull to %nblk step %c1]", w)
            self.assertIn("scf.if %fm_any", w)
            self.assertNotIn("vector.expf<afn>", w)


class RopeQg(unittest.TestCase):
    """rope_qg.fuse: g_qk_rope_store_qg reads q|gate rows and copies the gate; trunk launches lose g_split_qgate, the MTP launch keeps both."""
    PROG = '''kernel.def export("g_qk_rope_store") @g_qk_rope_store(%tokens: index) {
} launch(%tokens_arg: index, %pool_arg: index, %q: buffer, %k: buffer) {
  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x24x256xf32>
  %x0 = view.load %q_view[%token, %qh, %d0] : view<[%tokens]x24x256xf32> -> f32
  view.store %r0, %q_view[%token, %qh, %d0] : f32, view<[%tokens]x24x256xf32>
  kernel.return
}
command.def public @p() launch() {
  kernel.launch @g_split_qgate[%t](%t, %qg, %q, %gate) : [index](index, buffer, buffer, buffer)
  kernel.launch @g_qk_rope_store[%t](%t, %layer_3, %rope_3, %context, %pool, %q, %k, %v, %qn, %kn) : [index](index, index, i32, index, index, buffer, buffer, buffer, view<256xf32>, view<256xf32>)
  kernel.launch @g_split_qgate[%t](%t, %qg, %q, %gate) : [index](index, buffer, buffer, buffer)
  kernel.launch @g_qk_rope_store[%t](%t, %mtp_layer, %rope_on, %context, %pool, %q, %k, %v, %qn, %kn) : [index](index, index, i32, index, index, buffer, buffer, buffer, view<256xf32>, view<256xf32>)
}
'''

    def test_fuse(self):
        import rope_qg
        out, n = rope_qg.fuse(self.PROG)
        self.assertEqual(n, 1)
        self.assertIn("%pool_arg: index, %qg: buffer, %q: buffer, %gate: buffer, %k: buffer", out)
        self.assertIn("view.load %qg_view[%token, %qh, 0, %d0] : view<[%tokens]x24x2x256xf32> -> f32", out)
        self.assertIn("view.load %qg_view[%token, %qh, 1, %d0]", out)
        self.assertIn("view.store %r0_gate, %gate_view[%token, %qh, %d0]", out)
        self.assertIn("@g_qk_rope_store_qg[%t](%t, %layer_3, %rope_3, %context, %pool, %qg, %q, %gate, %k,", out)
        self.assertIn("index, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, view<256xf32>", out)
        self.assertEqual(out.count("@g_split_qgate["), 1)                          # the MTP pair keeps its split
        self.assertIn("@g_qk_rope_store[%t](%t, %mtp_layer", out)


class FlashSwaTwinTwoHeads(unittest.TestCase):
    """FLASH2_SWA_TWIN=s at head_dim 256 with 2 query heads per KV head (gemma4 12B/26B/31B sliding layers): the gfx12 twin is gen_flash2s.py's
    split-dims kernel only with its 16-key tiles (FLASH2_KT256__RDNA4=16: four waves stage a tile); 32-key tiles keep the generic twin; the gfx11
    kernel is the generic one either way."""
    CFG = ROOT / "fleet/unsloth-gemma-4-12B-it-loom/llm_config.json"

    def gen(self, tmp, **env):
        import json
        cfg = json.loads(self.CFG.read_text())
        cfg["head_dim"] = cfg["head_dim_swa"]                      # the sliding layers' flash kernel (head_dim 256, 16 heads / 8 KV heads)
        cp = Path(tmp) / "cfg.json"; cp.write_text(json.dumps(cfg))
        out = Path(tmp) / "flash2.loom"
        e = {k: v for k, v in os.environ.items() if not k.startswith("FLASH2_")}
        e.update(LLM_CONFIG=str(cp), FLASH2_TARGET="llm", FLASH2_OUT=str(out), **env)
        subprocess.run([sys.executable, str(STARTER / "gen_flash2.py")], check=True, env=e, capture_output=True)
        t = out.read_text()
        a = t.index('export("g_attention_flash2_rdna4")')
        return t[:a], t[a:]

    def test_sixteen_key_tiles_take_the_split_dims_twin(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            g11_0, r4_0 = self.gen(tmp)
            g11_s, r4_s = self.gen(tmp, FLASH2_SWA_TWIN="s")
            g11_16, r4_16 = self.gen(tmp, FLASH2_SWA_TWIN="s", FLASH2_KT256__RDNA4="16")
        self.assertEqual(r4_s, r4_0)                                # 32-key tiles: gen_flash2s.py needs 8 waves, the generic twin stays
        self.assertIn("%clanes = index.constant 64 : index", r4_0)
        self.assertIn("%clanes = index.constant 128 : index", r4_16)   # 2 heads x 2 waves of 128 dims
        self.assertNotEqual(r4_16, r4_0)
        self.assertEqual(g11_16, g11_0)                             # the gfx11 kernel is unchanged


if __name__ == "__main__":
    unittest.main()
