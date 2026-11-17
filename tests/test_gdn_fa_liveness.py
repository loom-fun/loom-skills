"""The shared LDS lifetimes are disjoint and preserve every arithmetic step."""
import os
import re
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/loom-port-llm/starter'))
import gen_gdn_fused as G
import gdn_fa_liveness as L


def original():
    geometry = dict(HK=16, HV=48, DK=128, DV=128, C=10240, K=4, state_bytes=3145728)
    with patch.dict(os.environ, {'GDN_FA_SHARE_LDS': '0'}, clear=True):
        return G.phase_a(geometry, 48, 3268608, 1e-6, True, 3)


class SharedLifetimes(unittest.TestCase):
    def test_arithmetic_and_solve_order_are_unchanged(self):
        base = original(); shared = L.rewrite(base)
        ops = lambda t: Counter(l.strip() for l in t.splitlines() if re.search(
            r'(?:scalar\.(?:fmaf|rsqrtf|logisticf|softplusf|expf)|vector\.mma|reduce<)', l))
        self.assertEqual(ops(base), ops(shared))
        solve = lambda t: [l.strip() for l in t.splitlines() if re.match(r'\s*%(?:tf\d+_|T\d+ =)', l)]
        self.assertEqual(solve(base), solve(shared))
        self.assertEqual(base.count('kernel.barrier<workgroup>'), shared.count('kernel.barrier<workgroup>'))

    def test_aliases_have_disjoint_access_lifetimes_and_fit(self):
        text = L.rewrite(original())
        q_reads = [m.start() for m in re.finditer(r'(?:vector\.load|fragment\.load<lhs>) %q_view\[', text)]
        v_writes = [m.start() for m in re.finditer(r'view.store %bvh\d+, %bv_view\[', text)]
        a_reads = [m.start() for m in re.finditer(r'(?:vector|view)\.load %a_view\[', text)]
        t_writes = [m.start() for m in re.finditer(r'view.store %T\d+h, %t_view\[', text)]
        self.assertTrue(q_reads and v_writes and a_reads and len(t_writes) == 32)
        self.assertLess(max(q_reads), min(v_writes))
        self.assertIn('kernel.barrier<workgroup>', text[max(q_reads):min(v_writes)])
        self.assertLess(max(a_reads), min(t_writes))
        self.assertIn('%bv_view = buffer.view %q_buf[', text)
        self.assertIn('%t_view = buffer.view %a_buf[', text)
        def view_bytes(name):
            m = re.search(r'%' + name + r' = buffer.view[^\n]*view<(\d+)x(\d+)xf(\d+)>', text)
            self.assertIsNotNone(m, name)
            rows, cols, bits = map(int, m.groups())
            return rows * cols * bits // 8
        self.assertLessEqual(view_bytes('bv_view'), view_bytes('q_view'))
        self.assertLessEqual(view_bytes('t_view'), view_bytes('a_view'))

    def test_changed_solve_structure_fails_closed(self):
        malformed = original().replace('    %T31h = scalar.fptrunc', '    %changed = scalar.fptrunc')
        with self.assertRaises(AssertionError):
            L.rewrite(malformed)


if __name__ == '__main__':
    unittest.main()
