"""gfx12 fused front of the gated delta-net chain (gen_gdn_front.py, choice hybrid.gdn.front): the kernel text, its launch group in the programs, the legacy default."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/loom-port-llm/starter"))
import dual_target  # noqa: E402
import gen_gdn  # noqa: E402
import gen_gdn_front  # noqa: E402

CFG = {"gdn": {"k_heads": 16, "v_heads": 48, "head_k": 128, "head_v": 128, "conv_kernel": 4}, "recurrent_layers": list(range(48))}
GUARD = ("    %publish = func.call @g_gdn_publish(%metadata, %spans, %span) : (buffer, index, index) -> (i1)\n    scf.if %publish {\n", "\n    }")
LOOM_COMPILE = Path(os.environ.get("LOOM_COMPILE", Path.home() / "loom/.venv/bin/loom-compile"))


def front_text(**kw):
    G = gen_gdn.gdn_geometry(CFG)
    return gen_gdn_front.front_kernel(G, 48, G["layer_bytes"], 1e-6, *GUARD, **kw)


class FrontMode(unittest.TestCase):
    G = gen_gdn.gdn_geometry(CFG)

    def test_legacy_default_is_off(self):
        env = {k: v for k, v in os.environ.items() if k not in ("GDN_FRONT",)}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(gen_gdn.front_mode(self.G))

    def test_on_needs_the_env_and_the_gfx12_twins(self):
        with unittest.mock.patch.dict(os.environ, {"GDN_FRONT": "1", "LOOM_RDNA4": "1"}):
            self.assertTrue(gen_gdn.front_mode(self.G))
        with unittest.mock.patch.dict(os.environ, {"GDN_FRONT": "1", "LOOM_RDNA4": "0"}):
            self.assertFalse(gen_gdn.front_mode(self.G))

    def test_narrow_heads_are_refused(self):
        G = dict(self.G, DK=64)
        with unittest.mock.patch.dict(os.environ, {"GDN_FRONT": "1"}):
            self.assertFalse(gen_gdn.front_mode(G))


class FrontKernel(unittest.TestCase):
    def test_geometry_and_history_carry(self):
        t = front_text()
        self.assertIn('kernel.def target(@rdna4) export("g_gdn_front_rdna4") @g_gdn_front_rdna4(%spans: index)', t)
        self.assertIn("workgroups(%spans, %nblocks, %c1) workgroup_size(%cblock, %c1, %c1)", t)
        self.assertIn("%nblocks = index.constant 80 : index", t)                    # 10240 channels / 128
        self.assertIn("%f_0, %f_1, %f_2 = scf.for %base = [%c0 to %length step %cS](%p0 = %h0 : f32, %p1 = %h1 : f32, %p2 = %h2 : f32)", t)
        self.assertEqual(len(re.findall(r"scalar\.fmaf %w\d, ", t)), 8 * 4)         # 8 steps x 4 taps
        self.assertEqual(len(re.findall(r"view\.store %y\d, %out_view", t)), 8)
        self.assertEqual(t.count("kernel.workgroup.reduce<addf> %sqv : vector<8xf32>"), 1)   # one reduction per tile of 8 steps

    def test_commit_is_guarded_by_publish(self):
        t = front_text()
        i = t.index("%publish = func.call @g_gdn_publish")
        self.assertGreater(t.index("view.store %f_0, %new_view[0, %ch]"), i)
        self.assertLess(t.index("scf.yield"), i)                                    # the stores follow the tile loop
        G = gen_gdn.gdn_geometry(CFG)
        self.assertNotIn("g_gdn_publish", gen_gdn_front.front_kernel(G, 48, G["layer_bytes"], 1e-6))

    def test_wrong_head_width_is_refused(self):
        G = gen_gdn.gdn_geometry(dict(CFG, gdn=dict(CFG["gdn"], head_k=64)))
        with self.assertRaises(AssertionError):
            gen_gdn_front.front_kernel(G, 48, G["layer_bytes"], 1e-6)

    @unittest.skipUnless(LOOM_COMPILE.exists(), "loom-compile not installed")
    def test_compiles_for_gfx12(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "front.loom"
            src.write_text(gen_gdn.SLOT + front_text(target=False))
            r = subprocess.run([str(LOOM_COMPILE), str(src), "--format=amdgpu-hsaco", "--target=amdgpu:gfx12-generic", f"--output={d}/front.hsaco"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertTrue((Path(d) / "front.hsaco").stat().st_size > 1000)


class FrontGroup(unittest.TestCase):
    def test_group_in_the_decode_branch_becomes_one_template(self):
        launch = gen_gdn.front_launch(0).replace("@CK@", "40960")
        text = f"""kernel.def export("g_dummy") @g_dummy(%t: index) {{
}} launch(%a: index) {{
}}
kernel.def target(@rdna4) export("g_dummy_rdna4") @g_dummy_rdna4(%t: index) {{
}} launch(%a: index) {{
}}
command.program.def public @llm_advance() launch(%t: index) {{
  scf.if %gemm {{
    kernel.launch @g_gdn_conv[%t](%t, %spans, %layer_0, %metadata, %info, %origins, %arena, %gq, %conv_0, %gs) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer, view<40960xf32>, buffer)
  }} else {{
    // dual-group: rdna4 {launch}
    kernel.launch @g_gdn_conv[%t](%t, %spans, %layer_0, %metadata, %info, %origins, %arena, %gq, %conv_0, %gs) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer, view<40960xf32>, buffer)
    kernel.launch @g_gdn_conv_commit[%spans](%t, %spans, %layer_0, %metadata, %origins, %arena, %gq) : [index](index, index, index, buffer, buffer, buffer, buffer)
    kernel.launch @g_gdn_qk_norm[%t](%t, %gs, %ginv) : [index](index, buffer, buffer)
    // dual-group: end
  }}
  command.return
}}
"""
        out = dual_target.apply(text, enable=True, select="launch")
        self.assertEqual(out.count("template.apply<@grp1_g_gdn_front_rdna4>"), 1)
        rdna4 = out[out.index("target(@rdna4) @grp1_g_gdn_front_rdna4_rdna4"):]
        rdna4 = rdna4[:rdna4.index("template.return")]
        self.assertEqual(rdna4.count("kernel.launch"), 1)
        self.assertIn("@g_gdn_front_rdna4[", rdna4)
        rdna3 = out[out.index("target(@rdna3) @grp1_g_gdn_front_rdna4_rdna3"):]
        rdna3 = rdna3[:rdna3.index("template.return")]
        self.assertEqual([m for m in re.findall(r"@(g_gdn_\w+)\[", rdna3)], ["g_gdn_conv", "g_gdn_conv_commit", "g_gdn_qk_norm"])


if __name__ == "__main__":
    unittest.main()
