"""Independent reference/guard checks for the experimental producer kernel suite."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np=None

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/loom-port-llm/starter'))
import mxfp4_producers as P


def suite():
    spec=importlib.util.spec_from_file_location('producer_bench',ROOT/'.github/devlab/kbench/mxfp4_producers/gen_bench.py')
    module=importlib.util.module_from_spec(spec)
    with patch.object(sys,'argv',['generator',str(ROOT)]):spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None,'numpy unavailable')
class ProducerReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.B=suite()

    def test_independent_fp8_codebook_and_rne_ties(self):
        values=np.array([0,2**-9,2**-6,.5,1,1.5,128,192,256,448],np.float32)
        bits=np.array([0,1,8,48,56,60,112,116,120,126],np.uint8)
        np.testing.assert_array_equal(self.B.fp8_bits(values).view(np.uint8),bits)
        np.testing.assert_array_equal(self.B.fp8_bits(-values).view(np.uint8),bits|128)
        # Midpoints alternate toward the even code; values are exactly binary.
        mid=np.array([1+1/16,1+3/16,2**-10,3*2**-10],np.float32)
        np.testing.assert_array_equal(self.B.fp8_bits(mid).view(np.uint8),[56,58,0,2])

    def test_row_oracle_preserves_dead_zero_tiny_and_large_groups(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            text=self.B.row_case(out,5120,8,1,self.B.VARIANTS,True)
            q=np.load(out/'row_5120_8_1_eq.npy');h=np.load(out/'row_5120_8_1_eh.npy');s=np.load(out/'row_5120_8_1_es.npy');s8=np.load(out/'row_5120_8_1_es8.npy')
            self.assertTrue(np.all(q[0]==31));self.assertTrue(np.all(h[0]==37));self.assertTrue(np.all(s[0]==0));self.assertTrue(np.all(s8[0]==0))
            self.assertTrue(np.all(q[1]==0));self.assertTrue(np.all(h[1]==0))
            self.assertGreater(s8[2,0],s8[4,0]);self.assertLess(s[3,0],s[4,0])
            self.assertEqual(text.count('kernel.launch'),5)
            self.assertEqual(text.count('check.expect.equal'),20)

    def test_postnorm_mutable_inputs_are_distinct_and_h32_tail_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            full=self.B.post_case(out,256,0,True,self.B.VARIANTS,True)
            tail=self.B.post_case(out,257,0,True,self.B.VARIANTS,True)
            self.assertIn('actual(%h_h32) expected(%dead_h)',full)
            self.assertIn('actual(%h_h32) expected(%h_base)',tail)
            self.assertNotIn('%dead_h',tail)
            for arm in self.B.VARIANTS:
                self.assertIn(f'%x_{arm}, %wn',tail)
                self.assertTrue((out/f'post_257_0_1_x_{arm}.npy').exists())
            expected=np.load(out/'post_257_0_1_eh.npy')
            self.assertTrue(np.all(expected[0]==37));self.assertGreater(np.std(expected[1]),.5)

    def test_fail_closed_reader_width_and_source_contracts(self):
        row,post=self.B.baseline_producers(ROOT)
        with self.assertRaises(ValueError):P.hold_row_inputs(row,4096)
        with self.assertRaises(ValueError):P.postnorm_dead_h32(post,paired_ab_fp8=False)
        with self.assertRaises(ValueError):P.postnorm_wg160(post.replace('x160x32xf32','x128x32xf32'))
        with self.assertRaises(ValueError):P.guard_half_arithmetic(row.replace('%hv = vector.fptrunc','%changed = vector.fptrunc'))
        # Every variant comes from current templates; no committed generated IR.
        defs=self.B.kernels(ROOT)
        self.assertEqual(set(defs),set(self.B.VARIANTS))
        self.assertNotIn('%h32_needed',defs['base']['post'])
        self.assertIn('%h32_needed',defs['h32']['post'])
        self.assertNotIn('%held_v',defs['fixed']['row_17408'])
        self.assertIn('%held_v16',defs['held']['row_17408'])


if __name__=='__main__':unittest.main()
