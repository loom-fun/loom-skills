"""The FP8 activation layout has one bijective byte mapping and unchanged math."""
import os
import json
import tempfile
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

    def test_resolved_rdna4_rule_activates_matching_consumers_and_producer(self):
        # Use the device-scoped exports from the actual build resolver.
        kb = Path(__file__).resolve().parents[1] / 'skills/loom-kb'
        sys.path.insert(0, str(kb))
        import resolve
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / 'model' / 'activation-test'
            node.mkdir(parents=True)
            (node / 'scope.json').write_text(json.dumps({
                'scope': 'model/activation-test', 'match': {'slug': 'activation-test'}, 'owner': 'test',
                'choices': [{'when': {'devices': ['rdna4']},
                             'set': {'prefill.gemm.mxfp4s_a_tiled': True},
                             'evidence': 'test: device-scoped activation'}]}))
            cfg = {'name': 'activation-test', 'arch': 'qwen3', 'hidden': 5120,
                   'layers': 64, 'heads': 32, 'kv_heads': 8, 'head_dim': 128,
                   'ffn': 17408, 'tensors': {}}
            resolved = resolve.resolve_full(cfg, ('rdna3', 'rdna4'), tmp,
                                            draft=None, slug='activation-test')
        self.assertEqual(resolved['env'], {'GEMM_MXFP4S_A_TILED__RDNA4': '1'})
        with patch.dict(os.environ, resolved['env'], clear=True):
            self.assertTrue(A.enabled())
            tiled = G.gemm_mxfp4s_f8_m256('store')
            self.assertNotIn('vector.fragment.load<lhs> %x_g8', tiled)
            import ast
            module = ast.parse((kb.parent / 'loom-port-llm/starter/gen_llm.py').read_text())
            source = next(ast.literal_eval(n.value) for n in module.body
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                          and t.id == 'ROW_SCALE' for t in n.targets))
            producer = G.f8_producer(source)
            self.assertIn('%at_byte', producer)
            self.assertNotIn('vector.store %q8, %x8_view', producer)
        # An explicit disabled twin overrides an enabled base variable.
        with patch.dict(os.environ, {'GEMM_MXFP4S_A_TILED': '1',
                                     'GEMM_MXFP4S_A_TILED__RDNA4': ''}, clear=True):
            self.assertFalse(A.enabled())

    def test_incompatible_permuted_k_design_fails_closed(self):
        with patch.dict(os.environ, {'GEMM_MXFP4S_A_TILED': '1'}):
            with self.assertRaises(AssertionError):
                G.gemm_mxfp4s_f8_m256('store', lv=True)


if __name__ == '__main__':
    unittest.main()
