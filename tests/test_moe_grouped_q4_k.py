"""Packed Q4_K numerical reference for the grouped Ornith 122B kernel fixture."""

from importlib.util import find_spec
from pathlib import Path
import struct
import sys
import unittest


STARTER = Path(__file__).resolve().parents[1] / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))


class GroupedQ4K(unittest.TestCase):
    @unittest.skipUnless(find_spec("numpy") and find_spec("gguf"), "numpy and gguf are required for numerical checks")
    def test_packed_q4_k_rounds_coefficients_then_fuses_the_product(self):
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from ktest_more import grouped_q4_k_f16

        unit = 1 / 1024
        header = bytearray(12)
        # Groups 0 and 7 encode scale 3 and minimum 9 in the two header layouts.
        header[0], header[4], header[11] = 3, 9, 0x93
        packed = bytearray(128)
        packed[0], packed[127] = 0x03, 0x30
        block = struct.pack("<ee", 1 + unit, 1 + unit) + header + packed
        data = np.frombuffer(block, dtype=np.uint8).reshape(1, 144)

        def half(value):
            return struct.unpack("<e", struct.pack("<e", value))[0]

        # Both stored coefficients are 1 + 1/1024. The packed multiplier 3 rounds to 3 + 4/1024
        # and the minimum multiplier 9 rounds to 9 + 8/1024 before the fused operation.
        scale = half(3 * (1 + unit))
        minimum = half(9 * (1 + unit))
        expected_value = half(3 * scale - minimum)
        self.assertEqual(expected_value, 4 * unit)
        self.assertEqual(half(half(3 * scale) - minimum), 8 * unit)
        legacy = dequantize(data, GGMLQuantizationType.Q4_K).reshape(-1)
        self.assertEqual((legacy[0], legacy[255]), (0.0, 0.0))

        expected = np.zeros(256, dtype=np.float16)
        expected[:32] = expected[224:] = -minimum
        expected[0] = expected[255] = expected_value
        np.testing.assert_array_equal(grouped_q4_k_f16(data).reshape(-1), expected)

    def test_ornith_grouped_kernels_and_fixture_keep_their_contract(self):
        import gen_gemm
        import gen_moe

        for mode, n, k, symbol in (("swiglu", 1024, 3072, "g_moe_ggu_q4_k_k3072"),
                                   ("store", 3072, 1024, "g_moe_gdn_q4_k_k1024")):
            with self.subTest(kernel=symbol):
                ir = gen_moe.grouped_gemm("q4_k", mode, n, k, 256, 8, gen_gemm.gemm)
                self.assertIn(f'export("{symbol}")', ir)
                self.assertIn("scalar.fptrunc %deq0_scale : f32 to f16", ir)
                self.assertIn("scalar.fptrunc %deq0_neg_min : f32 to f16", ir)
                self.assertIn("vector.fmaf %deq0_codes, %deq0_scales, %deq0_minima : vector<16xf16>", ir)

        fixture = (STARTER / "make_kernel_tests.py").read_text(encoding="utf-8").split("@moe_gemm_{li}", 1)[1]
        self.assertIn("actual(%act) expected(%eact) atol({0.15 * float(np.abs(act_ref).mean()):.3g}) rtol(5.0e-2)", fixture)
        self.assertIn("actual(%ys) expected(%eys) atol({0.05 * float(np.abs(ys_ref).mean()):.3g}) rtol(5.0e-2)", fixture)


if __name__ == "__main__":
    unittest.main()
