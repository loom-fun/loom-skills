"""prefill.gemm.kq_deq_f32 (GEMM_KQ_DEQ_F32): the Q4_K / Q5_K dequant into the WMMA f16 operand in f32 with one rounding."""
import importlib
import os
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

STARTER = Path(__file__).resolve().parents[1] / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))


def gen(on):
    os.environ["GEMM_KQ_DEQ_F32"] = "1" if on else "0"
    try:
        import gen_gemm
        return importlib.reload(gen_gemm)
    finally:
        os.environ.pop("GEMM_KQ_DEQ_F32", None)


class KqDeqF32(unittest.TestCase):
    def tearDown(self):
        gen(False)

    def test_off_is_the_f16_fused_dequant(self):
        g = gen(False)
        for fmt in ("q4_k", "q5_k"):
            text = g.dequant16(fmt, "%w", None, "deq0")
            self.assertIn("%deq0_neg_min_h = scalar.fptrunc %deq0_neg_min : f32 to f16", text)
            self.assertIn("%deq0 = vector.fmaf %deq0_codes, %deq0_scales, %deq0_minima : vector<16xf16>", text)
            self.assertNotIn("xf32>", text)

    def test_on_rounds_once_from_f32(self):
        g = gen(True)
        for fmt in ("q4_k", "q5_k"):
            text = g.dequant16(fmt, "%w", None, "deq0")
            self.assertNotIn("fptrunc %deq0_scale", text)
            self.assertNotIn("_neg_min_h", text)
            self.assertIn("%deq0_codes = vector.uitofp %deq0_bytes : vector<16xi8> to vector<16xf32>", text)
            self.assertIn("%deq0_w32 = vector.fmaf %deq0_codes, %deq0_scales, %deq0_minima : vector<16xf32>", text)
            self.assertIn("%deq0 = vector.fptrunc %deq0_w32 : vector<16xf32> to vector<16xf16>", text)
        lines, _, _ = g._g32_dequant("q4_k", "%w", "g", "%c0")
        self.assertIn("      %g = vector.fptrunc %g_w32 : vector<32xf32> to vector<32xf16>", lines)
        # other formats are untouched
        self.assertEqual(g.dequant16("q6_k", "%w", None, "deq0"), gen(False).dequant16("q6_k", "%w", None, "deq0"))

    def test_on_without_a_block_is_an_error(self):
        g = gen(True)
        with self.assertRaises(AssertionError):
            g.kq_deq_f32("%x = vector.splati 1 : vector<4xi32>")

    @unittest.skipUnless(find_spec("numpy") and find_spec("gguf"), "numpy and gguf are required for the numerical check")
    def test_f32_dequant_is_ggml_rounded_once_where_f16_scales_are_not(self):
        """A Q4_K block with an f16-subnormal d (Ornith-1.5-122B layer 12): the f32 path equals ggml's f32 dequant rounded to f16
        for every weight; the f16 path (scale and minimum rounded first) does not."""
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import Q4_K, dequantize
        rng = np.random.default_rng(7)
        blocks = rng.integers(0, 256, (64, 144), dtype=np.uint8)
        blocks[:, 0:2] = np.frombuffer(np.float16(3.1e-5).tobytes(), np.uint8)    # d: f16 subnormal
        blocks[:, 2:4] = np.frombuffer(np.float16(4.4e-4).tobytes(), np.uint8)    # dmin
        ref = dequantize(blocks, GGMLQuantizationType.Q4_K).reshape(-1, 8, 32).astype(np.float64)
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        dm = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
        sc, mn = Q4_K.get_scale_min(blocks[:, 4:16])
        sc, mn = sc.reshape(-1, 8).astype(np.float32), mn.reshape(-1, 8).astype(np.float32)
        p = blocks[:, 16:].reshape(-1, 4, 32)
        c = np.empty((len(blocks), 8, 32), np.uint8)
        c[:, 0::2], c[:, 1::2] = p & 15, p >> 4
        f32 = (c.astype(np.float32) * (d * sc)[:, :, None] - (dm * mn)[:, :, None]).astype(np.float16)
        np.testing.assert_array_equal(f32, ref.astype(np.float16))
        s16, m16 = (d * sc).astype(np.float16), (dm * mn).astype(np.float16)
        f16 = (c * s16.astype(np.float64)[:, :, None] - m16.astype(np.float64)[:, :, None]).astype(np.float16)
        self.assertGreater(np.abs(f16.astype(np.float64) - ref).max(), 2 * np.abs(f32.astype(np.float64) - ref).max())


if __name__ == "__main__":
    unittest.main()
