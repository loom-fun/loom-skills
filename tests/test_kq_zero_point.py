"""prefill.gemm.kq_zero_point (GEMM_KQ_ZP): the Q4_K / Q5_K dequant into the WMMA f16 operand rounded once per weight (zero-point form)."""
import importlib
import os
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

STARTER = Path(__file__).resolve().parents[1] / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))


def gen(on):
    os.environ["GEMM_KQ_ZP"] = "1" if on else "0"
    try:
        import gen_gemm
        return importlib.reload(gen_gemm)
    finally:
        os.environ.pop("GEMM_KQ_ZP", None)


class KqZeroPoint(unittest.TestCase):
    def tearDown(self):
        gen(False)

    def test_off_is_the_f16_fused_dequant(self):
        g = gen(False)
        for fmt in ("q4_k", "q5_k"):
            text = g.dequant16(fmt, "%w", None, "deq0")
            self.assertIn("%deq0_neg_min_h = scalar.fptrunc %deq0_neg_min : f32 to f16", text)
            self.assertIn("%deq0 = vector.fmaf %deq0_codes, %deq0_scales, %deq0_minima : vector<16xf16>", text)
            self.assertNotIn("xf32>", text)

    def test_on_is_the_zero_point_form(self):
        g = gen(True)
        for fmt, top in (("q4_k", 15), ("q5_k", 31)):
            text = g.dequant16(fmt, "%w", None, "deq0")
            self.assertNotIn("fptrunc %deq0_scale", text)
            self.assertNotIn("_neg_min_h", text)
            self.assertNotIn("xf32>", text)   # no f32 vector work
            self.assertIn("%deq0_zsc, %deq0_zmn = func.call @g_k_scale_minimum(%deq0_zpk, %deq0_group)", text)
            self.assertIn(f"%deq0_zhi = scalar.constant {top} : i32", text)
            self.assertIn("%deq0_zn = vector.fmaf %deq0_codes, %deq0_zscs, %deq0_zn0s : vector<16xf16>", text)
            self.assertIn("%deq0 = vector.fmaf %deq0_zds, %deq0_zn, %deq0_zks : vector<16xf16>", text)
        lines, _, _ = g._g32_dequant("q4_k", "%w", "g", "%c0")
        self.assertIn("      %g = vector.fmaf %g_zds, %g_zn, %g_zks : vector<32xf16>", lines)
        # other formats are untouched
        self.assertEqual(g.dequant16("q6_k", "%w", None, "deq0"), gen(False).dequant16("q6_k", "%w", None, "deq0"))

    def test_on_without_a_block_is_an_error(self):
        g = gen(True)
        with self.assertRaises(AssertionError):
            g.kq_zero_point("%x = vector.splati 1 : vector<4xi32>", 15)

    @unittest.skipUnless(find_spec("numpy") and find_spec("gguf"), "numpy and gguf are required for the numerical check")
    def test_zero_point_form_is_ggml_rounded_once(self):
        """The kernel's arithmetic in numpy (f16 operands, one rounding per f16 FMA, c0 from the bit-trick reciprocal + one Newton
        step) on Q4_K blocks with f16-subnormal d and a large minimum (Ornith-1.5-122B layer 12): every weight within one f16 ulp
        of ggml's f32 dequant rounded to f16 (98.9 % equal on these random blocks, 99.97 % on the real layer); the f16 default: 51 %."""
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import Q4_K, dequantize
        rng = np.random.default_rng(7)
        blocks = rng.integers(0, 256, (256, 144), dtype=np.uint8)
        blocks[:, 0:2] = np.frombuffer(np.float16(3.1e-5).tobytes(), np.uint8)    # d: f16 subnormal
        blocks[:, 2:4] = np.frombuffer(np.float16(4.4e-4).tobytes(), np.uint8)    # dmin
        ideal = dequantize(blocks, GGMLQuantizationType.Q4_K).reshape(-1, 8, 32).astype(np.float16)
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        dm = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
        sc, mn = Q4_K.get_scale_min(blocks[:, 4:16])
        sc, mn = sc.reshape(-1, 8).astype(np.float32), mn.reshape(-1, 8).astype(np.float32)
        p = blocks[:, 16:].reshape(-1, 4, 32)
        c = np.empty((len(blocks), 8, 32), np.uint8)
        c[:, 0::2], c[:, 1::2] = p & 15, p >> 4
        s, m = d * sc, dm * mn                                                   # @g_k_metadata
        r0 = (np.int32(2129859011) - s.view(np.int32)).view(np.float32)
        r = r0 * (np.float32(2) - s * r0)
        with np.errstate(invalid="ignore", over="ignore"):
            c0 = np.clip(np.nan_to_num(m * r + np.float32(0.5)).astype(np.int64), 0, 15).astype(np.float32)
        k = (c0 * s - m).astype(np.float16).astype(np.float64)
        f = lambda a: a.astype(np.float16).astype(np.float64)
        n = (c * f(sc)[:, :, None] + f(-(sc * c0))[:, :, None]).astype(np.float16)
        w = (f(d)[:, :, None] * n.astype(np.float64) + k[:, :, None]).astype(np.float16)
        ref = dequantize(blocks, GGMLQuantizationType.Q4_K).reshape(-1, 8, 32).astype(np.float64)
        ulp = np.spacing(np.abs(ideal)).astype(np.float64)
        inside = (m <= 15.5 * s)[:, :, None] & np.ones_like(c, bool)    # the minimum inside the code range (every real group of layer 12)
        self.assertGreater(inside.mean(), 0.2)
        err = np.abs(w.astype(np.float64) - ideal.astype(np.float64))
        self.assertTrue(np.all(err[inside] <= ulp[inside]))
        self.assertGreater((w == ideal)[inside].mean(), 0.98)     # the rest: one ulp away at ties of the f32 value
        s16, m16 = s.astype(np.float16), m.astype(np.float16)
        f16 = (c * f(s16)[:, :, None] - f(m16)[:, :, None]).astype(np.float16)
        self.assertLess((f16 == ideal)[inside].mean(), 0.6)       # the default: shared rounding of scale and minimum
        # outside the code range (clamped zero point) it is never worse than the default
        self.assertLessEqual(np.abs(w.astype(np.float64) - ref).sum(), np.abs(f16.astype(np.float64) - ref).sum())

if __name__ == "__main__":
    unittest.main()
