"""Unsigned split masks: exact interval equivalence, including inactive spans."""
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AttentionRangeMask(unittest.TestCase):
    def test_modular_equivalence(self):
        for bits in (4, 32, 64):
            modulus = 1 << bits
            points = range(modulus) if bits == 4 else (0, 1, 15, 16, 262143, 262144, modulus // 2, modulus - 2, modulus - 1)
            for lo in points:
                for hi in points:
                    if lo > hi:
                        continue
                    for key in points:
                        self.assertEqual(lo <= key < hi, ((key - lo) % modulus) < ((hi - lo) % modulus), (bits, key, lo, hi))

    def test_reversed_sentinels_cannot_use_the_ordered_identity(self):
        # Unmeasured span/segment masks stay legacy; the identity requires lo <= hi.
        for bits in (32, 64):
            lo, hi, key = 1 << 30, 0, 1 << 30
            self.assertFalse(lo <= key < hi)
            self.assertTrue(((key - lo) % (1 << bits)) < ((hi - lo) % (1 << bits)))

    def test_serving_intervals(self):
        rng = random.Random(1145)
        for _ in range(10000):
            position = rng.randrange(262144)
            low = max(0, position + 1 - rng.choice((16, 4096, 262144)))
            visible = position + 1 - low
            splits = rng.choice((1, 16, 64))
            split = rng.randrange(splits)
            count = (visible + splits - 1) // splits
            lo = low + min(split * count, visible)
            hi = low + min((split + 1) * count, visible)
            for key in (lo - lo % 16, max(0, lo - 1), lo, hi - 1, hi, hi + 15):
                for bits in (32, 64):
                    self.assertEqual(lo <= key < hi, ((key - lo) % (1 << bits)) < hi - lo)

    def test_scope_only_enables_representative(self):
        import json
        import sys
        sys.path.insert(0, str(ROOT / 'skills/loom-kb'))
        import resolve
        target = 'bartowski-Qwen2.5-0.5B-Instruct-loom'
        for path in (ROOT / 'fleet').glob('*/llm_config.json'):
            with path.open() as stream:
                cfg = json.load(stream)
            result = resolve.resolve_full(cfg, slug=path.parent.name)
            key = 'decode.attn.full_blocks'
            if path.parent.name == target:
                self.assertEqual(result['choices'][key]['value'], {'rdna3': False, 'rdna4': True})
                self.assertEqual(result['env'].get('ATTN_FULL_BLOCKS__RDNA4'), '1')
                self.assertNotIn('ATTN_FULL_BLOCKS', result['env'])
            else:
                self.assertFalse(result['values'][key], path.parent.name)
                self.assertNotIn('ATTN_FULL_BLOCKS__RDNA4', result['env'])
