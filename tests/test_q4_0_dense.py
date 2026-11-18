"""Q4_0 dense choices: prefill.gemm.q4_0_native (GEMM_Q4_0_NATIVE) and decode.q4_0.act_q8 (Q4_0_Q8).

Both read the environment at import, so each case runs the generator in a subprocess."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / 'skills/loom-port-llm/starter'


def run_py(code, **env):
    e = dict(os.environ, LOOM_RDNA4='1', **env)
    for k in ('GEMM_Q4_0_NATIVE', 'Q4_0_Q8'):
        if k not in env:
            e.pop(k, None)
    return subprocess.run([sys.executable, '-c', code], cwd=STARTER, env=e, check=True, capture_output=True, text=True).stdout


def gemm_text(**env):
    e = dict(os.environ, LOOM_RDNA4='1', GEMM_FORMATS='q4_0:store+swiglu', GEMM_SKIP_PADDING='1', GEMM_N_FASTEST='1', **env)
    if 'GEMM_Q4_0_NATIVE' not in env:
        e.pop('GEMM_Q4_0_NATIVE', None)
    return subprocess.run([sys.executable, str(STARTER / 'gen_gemm.py')], cwd=STARTER, env=e, check=True, capture_output=True, text=True).stdout


class Q40Dense(unittest.TestCase):
    def test_legacy_gemm_is_helper_64_row_only(self):
        text = gemm_text()
        self.assertIn('func.call @g_q4_0_values', text)
        self.assertNotIn('@g_gemm_q4_0_store_m256', text)
        # the 128-row name is the 64-row kernel under another name
        self.assertIn('@g_gemm_q4_0_store_m128', text)

    def test_native_gemm_has_big_kernels_and_moe_snippet(self):
        text = gemm_text(GEMM_Q4_0_NATIVE='1')
        self.assertNotIn('func.call @g_q4_0_values', text)
        for k in ('store_m128', 'store_m256', 'swiglu_m128', 'swiglu_m256'):
            self.assertIn(f'@g_gemm_q4_0_{k}(', text)
            self.assertIn(f'@g_gemm_q4_0_{k}_rdna4(', text)
        # the one-rounding F16 multiply of the expert staging (#1244), centered codes
        self.assertIn('vector.subi', text)
        self.assertIn('vector<16xf16>', text)

    def test_native_snippet_is_the_expert_one(self):
        out = run_py('import gen_gemm, moe_q4_0; print(gen_gemm.dequant16("q4_0", "w", None, "d") == moe_q4_0.dequant16("w", "d"))',
                     GEMM_Q4_0_NATIVE='1')
        self.assertEqual(out.strip(), 'True')
        out = run_py('import gen_gemm; print("q4_0" in gen_gemm.HELPER)', GEMM_Q4_0_NATIVE='1')
        self.assertEqual(out.strip(), 'False')
        out = run_py('import gen_gemm; print("q4_0" in gen_gemm.HELPER)')
        self.assertEqual(out.strip(), 'True')

    def test_grouped_expert_staging_unchanged_by_native(self):
        code = ('import gen_gemm, gen_moe; print(gen_moe.grouped_gemm("q4_0", "swiglu", 704, 2816, 128, 8, gen_gemm.gemm, fused=True))')
        self.assertEqual(run_py(code), run_py(code, GEMM_Q4_0_NATIVE='1'))

    def test_act_q8_adds_q4_0_icodes(self):
        self.assertEqual(run_py('import formats; print("q4_0" in formats.ICODES)').strip(), 'False')
        self.assertEqual(run_py('import formats; print("q4_0" in formats.ICODES)', Q4_0_Q8='1').strip(), 'True')

    def test_scope_node_turns_both_on_for_q4_0_models_only(self):
        sys.path.insert(0, str(ROOT / 'skills/loom-kb'))
        import resolve
        kb = str(ROOT / 'skills/loom-kb')
        q40 = {'arch': 'gemma4', 'tensors': {'blk.0.attn_q.weight': {'type': 'q4_0'}}}
        q4k = {'arch': 'gemma4', 'tensors': {'blk.0.attn_q.weight': {'type': 'q4_k'}}}
        on = resolve.resolve_full(q40, ('rdna3', 'rdna4'), kb, None, None, None)['values']
        off = resolve.resolve_full(q4k, ('rdna3', 'rdna4'), kb, None, None, None)['values']
        for key in ('prefill.gemm.q4_0_native', 'decode.q4_0.act_q8'):
            self.assertIs(on[key], True, key)
            self.assertIs(off[key], False, key)


class KernelLimit(unittest.TestCase):
    def k(self, name):
        return (f'kernel.def export("{name}") @{name}(%n: index) {{\n  kernel.launch.config workgroups(%n) : index\n'
                f'}} launch(%x: buffer) {{\n  kernel.return\n}}\n')

    def test_drops_only_unreferenced_above_the_limit(self):
        sys.path.insert(0, str(STARTER))
        import dual_target
        used = "".join(self.k(f"k{i}") for i in range(4))
        prog = "command.def public @p() {\n" + "".join(f"  kernel.launch @k{i}[%t](%t) : [index](index)\n" for i in range(4)) + "}\n"
        text = used + self.k("dead") + prog
        self.assertEqual(dual_target.limit_kernels(text, limit=5), (text, []))   # at the limit: untouched
        new, dropped = dual_target.limit_kernels(text, limit=4)
        self.assertEqual(dropped, ["dead"])
        self.assertNotIn("@dead(", new)
        self.assertEqual(new.count("kernel.def"), 4)
        with self.assertRaises(ValueError):
            dual_target.limit_kernels(text, limit=3)


class LintSeesTensorTypes(unittest.TestCase):
    def test_load_cfg_overlays_target_tensor_types(self):
        sys.path.insert(0, str(ROOT / 'fleet'))
        import scope_lint
        qat = scope_lint.load_cfg(str(ROOT), 'unsloth-gemma-4-31B-it-qat-loom')
        self.assertEqual(sorted({t['type'] for t in qat['tensors'].values()}), ['f32', 'q4_0'])
        # the target file only: the DFlash drafter's GGUF dump sits in the same directory
        plain = scope_lint.load_cfg(str(ROOT), 'unsloth-gemma-4-31B-it-loom')
        self.assertNotIn('q4_0', {t['type'] for t in plain['tensors'].values()})
        self.assertNotIn('dflash', ''.join(plain['tensors']).lower())


if __name__ == '__main__':
    unittest.main()
