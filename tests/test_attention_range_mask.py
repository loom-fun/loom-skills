"""Unsigned split masks: exact interval equivalence, including inactive spans."""
import ast
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'skills/loom-port-llm/starter/gen_llm.py'


def generator_helper():
    # gen_llm has command-line generation side effects. Load this pure emitter alone.
    tree = ast.parse(SOURCE.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('attention_range_mask', 'attention_paged_range_masks')]
    ns = {'os': os}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


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

    def test_inactive_sentinels_clamped_empty(self):
        for bits in (32, 64):
            modulus = 1 << bits
            for key in (0, 1, 16, 262143, 262144, (1 << 30) - 1, 1 << 30):
                lo, hi = 1 << 30, 0
                self.assertFalse(((key - lo) % modulus) < ((max(hi, lo) - lo) % modulus))
            self.assertTrue(((lo - lo) % modulus) < ((hi - lo) % modulus))

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

    def test_emitter_legacy_and_unordered(self):
        emit = generator_helper()['attention_range_mask']
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(emit('%key', '%lo', '%hi', '%ok', '%l', '%h'),
                             '      %l = index.cmp uge, %key, %lo : index\n'
                             '      %h = index.cmp ult, %key, %hi : index\n'
                             '      %ok = scalar.andi %l, %h : i1')
        with patch.dict(os.environ, {'ATTN_RANGE_MASK': '1'}, clear=True):
            ordered = emit('%key', '%lo', '%hi', '%ok', '%l', '%h')
            inactive = emit('%key', '%lo', '%hi', '%ok', '%l', '%h', ordered=False)
            self.assertNotIn('index.max', ordered)
            self.assertIn('%ok_top = index.max %hi, %lo', inactive)
            self.assertIn('%ok_width = index.sub %ok_top, %lo', inactive)
            self.assertEqual(ordered.count('index.cmp'), 1)

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
            self.assertEqual(result['values']['decode.attn.range_mask'], path.parent.name == target, path.parent.name)

    def test_paged_rewrite_matches_only_exact_mask(self):
        helpers = generator_helper()
        emit, rewrite = helpers['attention_range_mask'], helpers['attention_paged_range_masks']
        with patch.dict(os.environ, {}, clear=True):
            before = emit('%key12_raw', '%lo_abs', '%hi_abs', '%ok12', '%okl12', '%okh12')
            self.assertEqual(rewrite(before), before)
        with patch.dict(os.environ, {'ATTN_RANGE_MASK': '1'}, clear=True):
            after = rewrite(before)
            self.assertEqual(after, emit('%key12_raw', '%lo_abs', '%hi_abs', '%ok12', '%okl12', '%okh12'))
            self.assertEqual(rewrite(after), after)
            unrelated = before.replace('%hi_abs', '%different_hi')
            self.assertEqual(rewrite(unrelated), unrelated)
