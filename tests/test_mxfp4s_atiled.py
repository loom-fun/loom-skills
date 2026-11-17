"""The FP8 activation layout has one bijective byte mapping and unchanged math."""
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/loom-port-llm/starter'))
import gen_mxfp4s as G
import mxfp4s_atiled as A


class ActivationTiles(unittest.TestCase):
    def test_producer_mapping_matches_independent_fragment_lane_layout(self):
        # Construct the consumer's lane-major tile independently of offset().
        for rows, width in ((16, 32), (32, 128), (256, 5120)):
            seen = set()
            for rt in range(rows // 16):
                for kt in range(width // 16):
                    start = (rt * (width // 16) + kt) * 256
                    for lane in range(32):
                        row = rt * 16 + lane % 16
                        for byte in range(8):
                            col = kt * 16 + (lane // 16) * 8 + byte
                            self.assertEqual(A.offset(row, col, width), start + lane * 8 + byte)
                            seen.add(A.offset(row, col, width))
            self.assertEqual(len(seen), rows * width)
            self.assertEqual(max(seen), rows * width - 1)
        # A wrong half/lane formula must be detected, not accepted as a permutation.
        self.assertNotEqual(A.offset(1, 8, 32), 1 * 16 + 8)

    def test_gemm_keeps_mma_order_and_weight_fragment_loads(self):
        for mode in ('store', 'swiglu'):
            with patch.dict(os.environ, {'GEMM_MXFP4S_A_TILED': '0'}):
                base = G.gemm_mxfp4s_f8_m256(mode)
            with patch.dict(os.environ, {'GEMM_MXFP4S_A_TILED': '1'}):
                tiled = G.gemm_mxfp4s_f8_m256(mode)
            instructions = lambda t: [x.strip() for x in t.splitlines() if 'vector.mma ' in x or 'fragment.load<rhs>' in x]
            self.assertEqual(instructions(base), instructions(tiled))
            self.assertNotIn('vector.fragment.load<lhs> %x_g8', tiled)
            self.assertIn('view<8xi8> -> vector<8xi8>', tiled)

    def test_incompatible_permuted_k_design_fails_closed(self):
        with patch.dict(os.environ, {'GEMM_MXFP4S_A_TILED': '1'}):
            with self.assertRaises(AssertionError):
                G.gemm_mxfp4s_f8_m256('store', lv=True)


if __name__ == '__main__':
    unittest.main()
