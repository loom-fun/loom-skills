"""Full-block predicates and exact transformation guards, CPU only."""
import importlib.util
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'skills/loom-port-llm/starter/attention_masks.py'
sys.path.insert(0, str(PATH.parent))
spec = importlib.util.spec_from_file_location('attention_masks_1145', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(u=2):
    lines = ['kernel.def export("g_attention_split") @g_attention_split() {',
             '    %m_final, %l_final, %acc_final = scf.for %j = [%lo_al to %hi_abs step %cU]() {',
             '      %kb_off = func.call @g_kv_offset']
    for i in range(u):
        lines += [f'      %okl{i} = index.cmp uge, %key{i}_raw, %lo_abs : index',
                  f'      %okh{i} = index.cmp ult, %key{i}_raw, %hi_abs : index',
                  f'      %ok{i} = scalar.andi %okl{i}, %okh{i} : i1']
    lines += [f'      %s{i} = scf.select %ok{i}, %sr{i}, %neg_big : f32' for i in range(u)]
    lines += ['    }', '  kernel.return', '}']
    return '\n'.join(lines) + '\n'


class FullBlockMasks(unittest.TestCase):
    def test_aligned_interior_predicate(self):
        rng = random.Random(11452)
        for u in (1, 2, 4, 8, 16):
            for _ in range(10000):
                lo = rng.randrange(262145)
                hi = rng.randrange(lo, 262145)
                full_lo = ((lo + u - 1) // u) * u
                full_hi = max((hi // u) * u, full_lo)
                for raw in (lo - 1, lo, hi - 1, hi):
                    j = max(0, raw // u * u)
                    for bits in (32, 64):
                        pred = ((j - full_lo) % (1 << bits)) < full_hi - full_lo
                        self.assertEqual(pred, lo <= j and j + u <= hi, (u, lo, hi, j))

    def test_default_preserves_text_exactly(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.full_blocks(fixture(), 'g_attention_split', 2), fixture())

    def test_target_override_only_changes_gfx12_twin(self):
        with patch.dict(os.environ, {'ATTN_FULL_BLOCKS__RDNA4': '1'}, clear=True):
            self.assertEqual(module.full_blocks(fixture(), 'g_attention_split', 2), fixture())
            self.assertIn('scf.if %full', module.full_blocks(fixture(), 'g_attention_split', 2, twin=True))

    def test_branch_uses_raw_scores_and_keeps_edge_masks(self):
        with patch.dict(os.environ, {'ATTN_FULL_BLOCKS': '1'}, clear=True):
            text = module.full_blocks(fixture(), 'g_attention_split', 2)
            self.assertIn('%full_less = index.constant 1', text)
            self.assertIn('scf.yield %sr0, %sr1 : f32, f32', text)
            self.assertIn('%edge_s0 = scf.select %ok0, %sr0, %neg_big', text)
            self.assertEqual(text.count('%ok0_width ='), 1)
            self.assertLess(text.index('} else {'), text.index('%ok0_width ='))
            for bad in (fixture().replace('%okl0 =', '%wrong ='), fixture().replace('%s0 =', '%wrong =')):
                with self.assertRaises(AssertionError):
                    module.full_blocks(bad, 'g_attention_split', 2)
