"""Q4_0 expert addressing and one-round half-block staging contracts."""
import os
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

STARTER = Path(__file__).resolve().parents[1] / 'skills/loom-port-llm/starter'
sys.path.insert(0, str(STARTER))
import gen_gemm
import gen_moe
import moe_q4_0


class Q40Experts(unittest.TestCase):
    def test_half_multiply_matches_reference_for_every_finite_scale(self):
        # Independent exhaustive numerical check: every finite half scale and all
        # 16 centered codes. The reference widens before multiply, then rounds.
        scales = np.arange(65536, dtype=np.uint16).view(np.float16)
        scales = scales[np.isfinite(scales)][:, None]
        codes = np.arange(-8, 8, dtype=np.float16)[None, :]
        with np.errstate(over='ignore', invalid='ignore'):
            half = (scales * codes).astype(np.float16)
            reference = (scales.astype(np.float32) * codes.astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(half.view(np.uint16), reference.view(np.uint16))

    def test_expert_stride_and_fused_up_offset(self):
        self.assertEqual(gen_moe.row_bytes('q4_0', 2816), 1584)
        for mode, k, n, fused in [('gu', 2816, 704, True), ('dn', 704, 2816, False)]:
            text = gen_moe.expert_mv('q4_0', k, n, 128, 8, 2, mode, fused=fused)
            self.assertIn('@g_q4_0_values', text)
            self.assertIn(f'%expert_bytes = index.constant {k // 32 * 18 * n * (2 if fused else 1)} : offset', text)
            if fused:
                self.assertIn(f'%up_bytes = index.constant {k // 32 * 18 * n} : offset', text)
            # K=704 is 22 groups: the remaining lanes must never read weights.
            self.assertIn('%wv0 = scf.if %ok', text)

    def test_grouped_fused_and_mixed_formats_both_targets(self):
        with patch.dict(os.environ, {'LOOM_RDNA4': '1'}):
            for fmt, mode, k, n, fused in [('q4_0', 'swiglu', 2816, 704, True),
                                         ('q4_0', 'store', 704, 2816, False),
                                         ('q4_0+q5_0', 'swiglu', 2816, 704, False),
                                         ('q5_0+q4_0', 'swiglu', 2816, 704, False)]:
                text = gen_moe.grouped_gemm(fmt, mode, n, k, 128, 8, gen_gemm.gemm, fused=fused)
                self.assertNotIn('func.call @g_q4_0_values', text)
                self.assertIn('vector.subi', text)
                self.assertIn('target(@rdna4)', text)
                self.assertIn('%tile_used', text)
                if fused:
                    self.assertIn('index.add %w2_row_origin, %deq1_block_add', text)
                if mode == 'swiglu':
                    self.assertIn('%gx_valid', text)

    def test_non_q40_templates_are_identical(self):
        for fmt in ['q4_k', 'q5_k', 'q6_k', 'q8_0', 'q5_0', 'mxfp4']:
            for mode in ['store', 'swiglu']:
                base = gen_gemm.gemm(fmt, mode)
                self.assertEqual(base, moe_q4_0.specialize_grouped(base, fmt, mode))

    def test_independent_reference_matches_ggml_bytes(self):
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        path = STARTER.parents[2] / '.github/devlab/kbench/moe_q4_0/gen_bench.py'
        spec = importlib.util.spec_from_file_location('q40_bench', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(47)
        raw, expected = module.weights(rng, (3, 64, 704))
        actual = dequantize(raw.reshape(-1, 18), GGMLQuantizationType.Q4_0).reshape(expected.shape)
        np.testing.assert_array_equal(expected, actual.astype(np.float64))
        # Deliberate wrong layouts cannot pass the independent reference.
        changed = raw.view(np.uint8).reshape(-1, 18).copy()
        changed[:, 2:] = (changed[:, 2:] << 4) | (changed[:, 2:] >> 4)
        wrong = module.dequant_reference(changed, expected.shape)
        self.assertGreater(np.max(np.abs(wrong - expected)), .01)

    def test_template_drift_fails_closed(self):
        with self.assertRaisesRegex(AssertionError, 'Q4_0 expert staging'):
            moe_q4_0.specialize_grouped('changed template', 'q4_0', 'store')


if __name__ == '__main__':
    unittest.main()
