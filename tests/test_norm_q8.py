"""The fused producer preserves Q8's ABI and only rewrites exact dataflow pairs."""
import sys
import re
import runpy
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/loom-port-llm/starter'))
import gen_norm_q8


def source_pair(hidden):
    s = (ROOT / 'skills/loom-port-llm/starter/gen_llm.py').read_text()
    a = s.index('kernel.def export("g_post_add_norm")')
    norm = eval('f\'\'\'' + s[a:s.index('\n"""', a)] + '\'\'\'',
                dict(HG=hidden // 32, H=hidden, POST_NORM=False, POST_EPS=1e-6, EPS=1e-6))
    a = s.index('kernel.def export("g_quant_q8")')
    return norm + '\n', s[a:s.index('\n"""', a)] + '\n'


class NormQ8(unittest.TestCase):
    def test_quantization_sequence_is_the_original(self):
        for h in (1024, 2048, 2560, 5120):
            fused = gen_norm_q8.kernel(*source_pair(h), h)
            self.assertIn('%hv, %h_view', fused)
            self.assertIn('vector.roundf %q8_sc', fused)
            self.assertIn('scalar.fptrunc %q8_d : f32 to f16', fused)
            self.assertIn(f'index.constant {h} : offset', fused)
            self.assertIn(f'view<[%tokens]x{h // 32}x2xf32>', fused)
            self.assertNotIn('%q8_rows', fused)
            self.assertNotIn('%q8_groups', fused)

    def test_only_adjacent_matching_producer_is_fused(self):
        pair = ('    kernel.launch @g_post_add_norm[%t](%t, %y, %w, %x, %w2, %h) : [index](index, buffer, view<1024xf32>, buffer, view<1024xf32>, buffer)\n'
                '    kernel.launch @g_quant_q8[%t, %nH](%t, %nH, %h, %hq) : [index, index](index, index, buffer, buffer)')
        result, n = gen_norm_q8.wire(pair)
        self.assertEqual(n, 1)
        self.assertIn('@g_post_add_norm_q8[%t]', result)
        self.assertNotIn('@g_quant_q8[', result)
        for bad in (pair.replace('%nH', '%nFF'), pair.replace('%nH, %h,', '%nH, %other,'),
                    pair.replace('    kernel.launch @g_quant', '    command.launch @other() : ()\n    kernel.launch @g_quant')):
            self.assertEqual(gen_norm_q8.wire(bad), (bad, 0))

    def test_generated_cases_use_testbench_value_sources(self):
        # loom-compile checks kernel IR, but does not execute/plan check.case.
        # This is the testbench operation contract, not kernel scalar syntax.
        generator = ROOT / '.github/devlab/kbench/norm_q8/gen_bench.py'
        with tempfile.TemporaryDirectory() as out, patch.object(sys, 'argv', [str(generator), str(ROOT), out]), patch.dict(sys.modules, {'numpy': None}):
            bench = runpy.run_path(str(generator))
            for rows in (1, 4, 63):
                for correctness in (False, True):
                    case = bench['case_ir'](1024, rows, True, test=correctness)
                    ops = re.findall(r'^  (?:%\w+ = )?([a-z]\w*\.[\w.]+)', case, re.M)
                    supported = {'check.literal', 'check.file.read.npy', 'check.generate.fill',
                                 'kernel.launch', 'check.expect.equal', 'check.expect.approximate', 'check.return'}
                    self.assertTrue(set(ops) <= supported, set(ops) - supported)
                    self.assertEqual(ops.count('check.literal'), 2)
                    if correctness:
                        self.assertEqual(ops.count('check.expect.equal'), 3)
                        self.assertEqual(ops.count('check.expect.approximate'), 1)

    def test_prefill_branch_unchanged(self):
        text = '  scf.if %gemm {\n    kernel.launch @g_post_add_norm_rs_live[%t](...)\n  } else {\n  }'
        self.assertEqual(gen_norm_q8.wire(text), (text, 0))
