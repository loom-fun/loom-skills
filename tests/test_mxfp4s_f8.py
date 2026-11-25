"""MXFP4 prefill GEMMs on fp8 WMMA (gen_mxfp4s: gemm_mxfp4s_f8_m256, f8_producer, fuse_mxfp4s_f8; choice prefill.gemm.mxfp4s_f8)."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:   # the decode checks need numpy (CI without it skips them)
    np = None

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))
import gen_mxfp4s as R  # noqa: E402

HEADER = "target.decl @device\namdgpu.target<gfx11-generic> @rdna3\namdgpu.target<gfx12-generic> @rdna4\n\n"
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float64) if np is not None else None


def loom_compile():
    for c in (os.environ.get("LOOM_COMPILE"), shutil.which("loom-compile"), str(Path.home() / "loom/.venv/bin/loom-compile"), "/home/jfowers/lt/loom010/.venv/bin/loom-compile"):
        if c and os.path.exists(c):
            return c


def e5m2(b):
    b = b.astype(np.int64)
    s, e, m = (b >> 7) & 1, (b >> 2) & 31, b & 3
    v = np.where(e == 0, m * 2.0 ** -16, (1 + m / 4) * 2.0 ** (e.astype(np.float64) - 15))
    return np.where(s == 1, -v, v)


def decode(e, q):
    """The kernel's arithmetic on [n] block exponent bytes e and [n, 16] nibble bytes q: [n, 32] bf8 bytes (table + perm + sign)."""
    a = ((e.astype(np.int64) - 107) << 2)
    a4 = (a * 0x01010101) & 0xFFFFFFFF
    tlo, thi = (0x0E0C0800 + (a4 & 0xFFFFFF00)) & 0xFFFFFFFF, (0x16141210 + a4) & 0xFFFFFFFF
    tab = np.zeros((len(e), 8), np.uint8)
    for i in range(4):
        tab[:, i], tab[:, 4 + i] = (tlo >> (8 * i)) & 255, (thi >> (8 * i)) & 255
    lo = np.take_along_axis(tab, (q & 7).astype(np.int64), 1) | ((q << 4) & 0x80).astype(np.uint8)
    hi = np.take_along_axis(tab, ((q >> 4) & 7).astype(np.int64), 1) | (q & 0x80).astype(np.uint8)
    return np.concatenate([lo, hi], 1)


@unittest.skipIf(np is None, "numpy not installed")
class DecodeTests(unittest.TestCase):
    def test_every_exponent_of_the_exact_range_decodes_exactly(self):
        rng = np.random.default_rng(3)
        e = np.repeat(np.arange(R.MXFP4_F8_E_MIN, R.MXFP4_F8_E_MAX + 1), 4000).astype(np.uint8)
        q = rng.integers(0, 256, (len(e), 16)).astype(np.uint8)
        got = e5m2(decode(e, q))
        mag = FP4[np.concatenate([q & 7, (q >> 4) & 7], 1)]
        sgn = np.where(np.concatenate([(q >> 3) & 1, (q >> 7) & 1], 1) == 1, -1.0, 1.0)
        want = mag * sgn * 2.0 ** (e.astype(np.float64)[:, None] - 127 + R.MXFP4_F8_SHIFT)
        np.testing.assert_array_equal(got, want)

    def test_range_helper(self):
        self.assertEqual(R.mxfp4_f8_range(np.array([107, 125], np.uint8)), (107, 125, True))
        self.assertFalse(R.mxfp4_f8_range(np.array([106, 125], np.uint8))[2])
        self.assertFalse(R.mxfp4_f8_range(np.array([107, 133], np.uint8))[2])


class KernelTests(unittest.TestCase):
    def test_enabled_follows_the_choice_and_the_twins(self):
        old = dict(os.environ)
        try:
            os.environ.pop("GEMM_MXFP4S_F8", None)
            self.assertFalse(R.f8_enabled())
            os.environ["GEMM_MXFP4S_F8"] = "1"
            self.assertTrue(R.f8_enabled())
            os.environ["LOOM_RDNA4"] = "0"
            self.assertFalse(R.f8_enabled())
        finally:
            os.environ.clear(); os.environ.update(old)

    @unittest.skipUnless(loom_compile(), "loom-compile not available")
    def test_gemm_kernels_compile_for_gfx1201_without_spills(self):
        for mode, order in (("store", "pairs"), ("swiglu", "pairs"), ("store", "split16"), ("swiglu", "split16")):
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "t.loom"
                f.write_text(HEADER + R.gemm_mxfp4s_f8_m256(mode, order=order))
                out = Path(d) / "o.hsaco"
                r = subprocess.run([loom_compile(), str(f), "--format=amdgpu-hsaco", "--target=amdgpu:gfx1201", f"--output={out}"], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[:2000])
                blob = out.read_bytes()
                i = blob.find(b".private_segment_fixed_size")
                self.assertGreaterEqual(i, 0)
                self.assertEqual(blob[i + len(b".private_segment_fixed_size")], 0, "the kernel spills to scratch")

    @unittest.skipUnless(loom_compile(), "loom-compile not available")
    def test_sweep_options_compile_for_gfx1201_without_spills(self):
        """the tile shape / pipelining / grid / prefetch / output / pitch options of the generator (device/rdna4/prefill.md: all measured, none a win)"""
        for mode, kw in (("store", dict(bm=128, tn=256)), ("store", dict(bm=256, tn=256)), ("swiglu", dict(bm=256, tn=128)), ("store", dict(swp=True)),
                         ("store", dict(rf=2, swp=True)), ("swiglu", dict(swz=True)), ("store", dict(pfa=2)), ("swiglu", dict(yf16=True)), ("store", dict(pad=32))):
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "t.loom"
                f.write_text(HEADER + R.gemm_mxfp4s_f8_m256(mode, **kw))
                out = Path(d) / "o.hsaco"
                r = subprocess.run([loom_compile(), str(f), "--format=amdgpu-hsaco", "--target=amdgpu:gfx1201", f"--output={out}"], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, (mode, kw, r.stderr[:1500]))
                blob = out.read_bytes()
                i = blob.find(b".private_segment_fixed_size")
                self.assertEqual(blob[i + len(b".private_segment_fixed_size")], 0, f"{mode} {kw} spills to scratch")

    @unittest.skipUnless(loom_compile(), "loom-compile not available")
    def test_kernel_designs_compile_for_gfx1201_without_spills(self):
        """Check the exact scratch allocation of every design, mode and nibble order."""
        compiler = Path(loom_compile()).resolve()
        # The compiler can come from a different SDK environment than the test runner.
        sdk_python = compiler.with_name("python.exe" if os.name == "nt" else "python")
        sdk_version = subprocess.check_output(
            [sdk_python, "-c", "from importlib.metadata import version; print(version('hrx-loom'))"],
            text=True,
        ).strip()
        lvs_store_bytes = 4 if sdk_version.startswith("0.2.") else 0
        for name, kws in R.F8_KERNELS.items():
            for mode, kw in zip(("store", "swiglu"), kws):
                for order in ("pairs", "split16"):
                    with tempfile.TemporaryDirectory() as d:
                        f = Path(d) / "t.loom"
                        f.write_text(HEADER + R.gemm_mxfp4s_f8_m256(mode, order=order, **kw))
                        out = Path(d) / "o.hsaco"
                        r = subprocess.run([compiler, str(f), "--format=amdgpu-hsaco", "--target=amdgpu:gfx1201", f"--output={out}"], capture_output=True, text=True)
                        self.assertEqual(r.returncode, 0, (name, mode, r.stderr[:1500]))
                        blob = out.read_bytes()
                        i = blob.find(b".private_segment_fixed_size")
                        self.assertGreaterEqual(i, 0, f"{name} {mode} missing scratch metadata")
                        want = lvs_store_bytes if (name, mode) == ("lvs", "store") else 0
                        self.assertEqual(blob[i + len(b".private_segment_fixed_size")], want, f"{name} {mode} scratch bytes")

    def test_kernel_design_choice(self):
        """GEMM_MXFP4S_F8_KERNEL selects the generator kwargs; unset = v1 = the #879 kernel; lv keeps the 128 WMMA per step and loads 16 bytes per lane"""
        old = os.environ.pop("GEMM_MXFP4S_F8_KERNEL", None)
        try:
            self.assertEqual(R.f8_kernel_kwargs("store"), {})
            os.environ["GEMM_MXFP4S_F8_KERNEL"] = "lv"
            self.assertEqual(R.f8_kernel_kwargs("swiglu"), {"lv": True, "rf": 4})
            for mode in ("store", "swiglu"):
                t = R.gemm_mxfp4s_f8_m256(mode, **R.f8_kernel_kwargs(mode))
                self.assertEqual(t.count("vector.mma "), 16 * 8)
                self.assertNotIn("vector.fragment.load<", t.split("scf.for")[1].split("scf.yield")[0])
        finally:
            os.environ.pop("GEMM_MXFP4S_F8_KERNEL", None)
            if old is not None:
                os.environ["GEMM_MXFP4S_F8_KERNEL"] = old

    def test_default_kernel_is_independent_of_the_sweep_options(self):
        """the options default to the kernel of #879: the explicit defaults produce the same text"""
        for mode in ("store", "swiglu"):
            self.assertEqual(R.gemm_mxfp4s_f8_m256(mode), R.gemm_mxfp4s_f8_m256(mode, swp=False, swz=False, pfa=0, yf16=False, pad=16, bm=256))

    def test_gemm_kernel_text_contract(self):
        for mode in ("store", "swiglu"):
            t = R.gemm_mxfp4s_f8_m256(mode)
            self.assertEqual(t.count("vector.mma "), 16 * 8)   # 16 accumulators per wave, 8 K substeps of 16 per 128-value step
            self.assertIn("f8e4m3", t)
            self.assertIn("f8e5m2", t)
            self.assertIn("%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, %w: buffer, %ws: buffer", t)   # the f16 kernels' leading arguments, then the two planes
            self.assertIn("%s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>", t)   # gen_gemm's skip-padding anchor
            self.assertTrue(t.endswith("  kernel.return\n}\n"))
            self.assertIn("%row_origin = index.mul %row_group, %c256 : index", t)


class IcodesTests(unittest.TestCase):
    WRAP = """
kernel.def target(@rdna4) export("t_icodes") @t_icodes() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%w: buffer, %ws: buffer, %y: buffer, %d: buffer) {
  %o = index.constant 0 : offset
  %g = index.constant 3 : index
  %codes, %s = func.call @g_mxfp4s_icodes(%w, %ws, %o, %o, %g) : (buffer, buffer, offset, offset, index) -> (vector<32xi8>, f32)
  %yv = buffer.view %y[%o] : buffer -> view<32xi8>
  vector.store %codes, %yv[0] : vector<32xi8>, view<32xi8>
  %dv = buffer.view %d[%o] : buffer -> view<1xf32>
  view.store %s, %dv[0] : f32, view<1xf32>
  kernel.return
}
"""

    @unittest.skipUnless(loom_compile(), "loom-compile not available")
    def test_icodes_compile_for_gfx1201(self):
        for order in ("pairs", "split16"):
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "t.loom"
                f.write_text(HEADER + R.mxfp4s_icodes(order) + self.WRAP)
                r = subprocess.run([loom_compile(), str(f), "--format=amdgpu-hsaco", "--target=amdgpu:gfx1201", f"--output={d}/o.hsaco"], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[:2000])

    def test_orders_differ_only_in_the_join(self):
        a, b = R.mxfp4s_icodes("pairs"), R.mxfp4s_icodes("split16")
        self.assertIn("vector.interleave<0>", a)
        self.assertIn("vector.concat<0>", b)
        self.assertEqual(a.replace("vector.interleave<0>", "J"), b.replace("vector.concat<0>", "J"))


class ProducerTests(unittest.TestCase):
    FIXTURE = (ROOT / "tests/data/f8_producers.loom").read_text()

    def kernel(self, name):
        a = self.FIXTURE.index(f'kernel.def export("{name}")')
        end = "  kernel.return\n}\n"
        return self.FIXTURE[a:self.FIXTURE.index(end, a) + len(end)]

    def test_twins_extend_the_signature_and_keep_the_f16_outputs(self):
        for name in ("g_row_scale_f16_live_lv", "g_post_add_norm_rs_live_lv"):
            base = self.kernel(name)
            t = R.f8_producer(base)
            self.assertIn(f'export("{name}_f8_rdna4")', t)
            self.assertIn("%x8: buffer, %s8: buffer, %w16: index) {", t)
            self.assertEqual(t.count("scf.if %w16_on {"), 1)   # the f16 operand store is guarded by the launch flag
            self.assertIn("f8E4M3", t)
            self.assertEqual(t.count("view.store %f8_scale"), 1)   # the live row's scales ...
            self.assertEqual(t.count("view.store %lv_zf, %lv_s8"), 2)   # ... and the padding row's (0, 0)
            for line in base.split("\n"):   # every line of the f16 kernel is still there
                if "launch(" in line or "kernel.def" in line:
                    continue
                self.assertIn(line, t, line)

    @unittest.skipUnless(loom_compile(), "loom-compile not available")
    def test_twins_compile_for_gfx1201(self):
        for name in ("g_row_scale_f16_live_lv", "g_post_add_norm_rs_live_lv"):
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "t.loom"
                f.write_text(HEADER + R.f8_producer(self.kernel(name)))
                r = subprocess.run([loom_compile(), str(f), "--format=amdgpu-hsaco", "--target=amdgpu:gfx1201", f"--output={d}/o.hsaco"], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[:2000])


class FuseTests(unittest.TestCase):
    @staticmethod
    def store(k, n, x, sx, w, y, big=True):
        mv = f"""    scf.if %single {{
      kernel.launch @g_mv_mxfp4s_k{k}_r4[{n}, %t]({n}, %t, %zero_i32, %h, {w}, {w}_sc, {y}) : [index, index](index, index, i32, buffer, view<8xi8>, view<8xi8>, buffer)
    }} else {{
      kernel.launch @g_mv_mxfp4s_k{k}_t4_r4[{n}, %t]({n}, %t, %zero_i32, %h, {w}, {w}_sc, {y}) : [index, index](index, index, i32, buffer, view<8xi8>, view<8xi8>, buffer)
    }}"""
        args = f"(%tp, {'%nH' if k == 5120 else '%nFF'}, {n}, {x}, {sx}, {w}, {w}_sc, {y}) : [index, index](index, index, index, buffer, buffer, view<8xi8>, view<8xi8>, buffer)"
        return f"""  scf.if %mm {{
    scf.if %big {{
      kernel.launch @g_gemm_mxfp4s_store_m128[%tp, {n}]{args}
    }} else {{
      kernel.launch @g_gemm_mxfp4s_store[%tp, {n}]{args}
    }}
    }} else {{
{mv}
    }}"""

    @property
    def program(self):
        return ProducerTests.FIXTURE + f"""
command.def public @llm_advance() launch(%weights: buffer) {{
  %nH = index.constant 5120 : index
  %nFF = index.constant 17408 : index
  %half_row = index.constant 10240 : offset
  %half_wide = index.constant 12288 : offset
  %half_ffn = index.constant 34816 : offset
  %f16 = buffer.alloca<global> align(256) %f16_size : buffer
  scf.if %mm {{
    kernel.launch @g_post_add_norm_rs_live_lv[%t](%t, %info, %y, %wn, %x, %wn, %h, %s_h, %h16) : [index](index, buffer, buffer, view<5120xf32>, buffer, view<5120xf32>, buffer, buffer, buffer)
  }} else {{
    kernel.launch @g_post_add_norm[%t](%t, %y, %wn, %x, %wn, %h) : [index](index, buffer, view<5120xf32>, buffer, view<5120xf32>, buffer)
  }}
{self.store(5120, "%nFF", "%h16", "%s_h", "%wg", "%f")}
{self.store(5120, "%nFF", "%h16", "%s_h", "%wu", "%fu")}
  scf.if %gemm {{
    kernel.launch @g_silu_mul_lv[%t](%t, %nFF, %f, %fu, %info) : [index](index, index, buffer, buffer, buffer)
  }} else {{
    kernel.launch @g_silu_mul[%t](%t, %nFF, %f, %fu) : [index](index, index, buffer, buffer)
  }}
  scf.if %mm {{
    kernel.launch @g_row_scale_f16_live_lv[%t](%t, %info, %nFF, %f, %s_f, %f16) : [index](index, buffer, index, buffer, buffer, buffer)
  }}
{self.store(17408, "%nH", "%f16", "%s_f", "%wd", "%y")}
  command.return
}}
"""

    def test_swiglu_pair_and_store_leaves_are_grouped(self):
        text, n = R.fuse_mxfp4s_f8(self.program)
        self.assertEqual(n, 3)   # gate + up (one fused group) and down
        self.assertEqual(text.count("// dual-group: rdna4 kernel.launch @g_gemm_mxfp4s_swiglu_f8_m256_rdna4[%tp, %nFF](%tp, %nH, %nFF, %h8, %s_h8, %wg, %wg_sc, %wu, %wu_sc, %f)"), 1)
        self.assertEqual(text.count("// dual-group: rdna4 kernel.launch @g_gemm_mxfp4s_store_f8_m256_rdna4[%tp, %nH](%tp, %nFF, %nH, %f8, %s_f8, %wd, %wd_sc, %y)"), 1)
        self.assertIn("// dual-group: rdna4 kernel.launch @g_row_scale_f16_live_lv_f8_rdna4[%t](%t, %info, %nFF, %f, %s_f, %f16, %f8, %s_f8, %w16_flag)", text)
        self.assertIn("// dual-group: rdna4 kernel.launch @g_post_add_norm_rs_live_lv_f8_rdna4[%t](%t, %info, %y, %wn, %x, %wn, %h, %s_h, %h16, %h8, %s_h8, %w16_flag)", text)
        self.assertEqual(text.count("scf.if %huge {"), 2)
        for buf, width in (("%h8", 5120), ("%f8", 17408)):
            self.assertRegex(text, rf"{buf}_row = index\.constant {width} : offset")
        self.assertNotIn("%o8 =", text)   # no reader of %o16: its fp8 copy is not made
        self.assertLess(text.index('export("g_row_scale_f16_live_lv_f8_rdna4")'), text.index("command.def"))

    def test_the_gfx11_program_keeps_every_launch(self):
        import dual_target
        dummy = 'kernel.def export("g_dummy") @g_dummy(%a: index) {\n}\n\nkernel.def target(@rdna4) export("g_dummy_rdna4") @g_dummy_rdna4(%a: index) {\n}\n\n'
        text, _ = R.fuse_mxfp4s_f8(dummy + self.program)
        dual = dual_target.apply(text, enable=True, select="launch")
        self.assertIn("template.apply<@grp", dual)
        base = re.findall(r"(?s)template\.def<@grp\d+_(\w+)> target\(@rdna3\) @\w+\([^)]*\) \{\n(.*?)template\.return", dual)
        twin = re.findall(r"(?s)template\.def<@grp\d+_(\w+)> target\(@rdna4\) @\w+\([^)]*\) \{\n(.*?)template\.return", dual)
        self.assertEqual(len(base), 4)   # two producers, the swiglu pair, the down GEMM
        self.assertEqual(sorted(k for k, _ in base), sorted(k for k, _ in twin))
        for _, body in base:
            self.assertNotIn("_f8_", body)   # the gfx11 provider is the launches the f16 program had
        launches = sorted(body.count("kernel.launch") for _, body in base)
        self.assertEqual(launches, [1, 1, 1, 3])   # the swiglu group: gate, up, silu_mul on gfx11
        for _, body in twin:
            self.assertEqual(body.count("kernel.launch"), 1)
            self.assertIn("_f8_", body)

    def test_an_unknown_reader_leaves_the_f16_path(self):
        prog = self.program.replace("  command.return\n", "  kernel.launch @g_mystery[%t](%t, %f16) : [index](index, buffer)\n  command.return\n")
        text, n = R.fuse_mxfp4s_f8(prog)
        self.assertEqual(n, 2)   # %f16 has an unknown user: the down GEMM stays; %h16's pair is still converted
        self.assertNotIn("%f8 =", text)

    def test_a_k_the_step_does_not_divide_keeps_the_f16_path(self):
        text, n = R.fuse_mxfp4s_f8(self.program.replace("%nH = index.constant 5120 : index", "%nH = index.constant 2880 : index"))
        self.assertEqual(n, 1)   # the down GEMM (K = 17408) only

    def test_no_llm_advance_no_change(self):
        t = "kernel.def export(\"x\") @x() {\n}\n"
        self.assertEqual(R.fuse_mxfp4s_f8(t), (t, 0))


if __name__ == "__main__":
    unittest.main()
