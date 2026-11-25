"""moe.prefill.cw_q4k (gen_moe_cw_q4k.py): drop-in column-wave grouped Q4_K MoE GEMMs."""
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

STARTER = Path(__file__).resolve().parents[1] / 'skills/loom-port-llm/starter'
sys.path.insert(0, str(STARTER))
import gen_gemm
import gen_moe
import gen_moe_cw_q4k

SHAPES = [('swiglu', 512, 2048), ('store', 2048, 512)]   # (mode, N, K) of the qwen35moe gate/up and down GEMMs
E, KU = 256, 8


def build(env, mode, N, K):
    with patch.dict(os.environ, env):
        return gen_moe.grouped_gemm('q4_k', mode, N, K, E, KU, gen_gemm.gemm)


def signature(text, name):
    m = re.search(r'kernel\.def (?:target\(@\w+\) )?export\("%s"\) @%s\(([^)]*)\).*?\} launch\(([^)]*)\)' % (name, name), text, re.S)
    assert m, name
    return m.group(1), m.group(2)


class CwQ4K(unittest.TestCase):
    def test_off_is_the_64_row_kernel(self):
        for mode, N, K in SHAPES:
            base = build({'MOE_CW_Q4K': '0'}, mode, N, K)
            with patch.dict(os.environ, {}):
                os.environ.pop('MOE_CW_Q4K', None)
                unset = gen_moe.grouped_gemm('q4_k', mode, N, K, E, KU, gen_gemm.gemm)
            self.assertEqual(base, unset)
            self.assertNotIn('%cq_m63', base)

    def test_same_names_and_launch_signatures_both_targets(self):
        for store16 in ('0', '1'):
            for mode, N, K in SHAPES:
                env = {'LOOM_RDNA4': '1', 'MOE_STORE16': store16}
                base = build({**env, 'MOE_CW_Q4K': '0'}, mode, N, K)
                cw = build({**env, 'MOE_CW_Q4K': '1'}, mode, N, K)
                self.assertIn('%cq_m63', cw)
                names = lambda t: re.findall(r'export\("(\w+)"\)', t)
                self.assertEqual(names(base), names(cw))
                for n in names(base):
                    self.assertEqual(signature(base, n)[1], signature(cw, n)[1], n)
                self.assertIn('kernel.def target(@rdna4) export("%s_rdna4")' % names(base)[0], cw)
                # gfx12 fragments are 8 values per lane, gfx11 16
                tw = cw[cw.index('target(@rdna4)'):]
                self.assertNotIn('vector<16xf16>, vector<16xf16>, vector<8xf32>', tw)
                self.assertIn('vector<16xf16>, vector<16xf16>, vector<8xf32>', cw[:cw.index('target(@rdna4)')])
                yt = 'f16' if (store16 == '1' and mode == 'store') else 'f32'
                self.assertEqual(f'x{yt}, %y_layout>' in cw, True)
                self.assertNotIn('x%s, %%y_layout>' % ('f32' if yt == 'f16' else 'f16'), cw)

    def test_modes_and_scope(self):
        with patch.dict(os.environ, {'MOE_CW_Q4K': 'store'}):
            self.assertEqual(gen_moe_cw_q4k.modes(), {'store'})
            self.assertTrue(gen_moe_cw_q4k.applies('q4_k', 'store', 2048, 512))
            self.assertFalse(gen_moe_cw_q4k.applies('q4_k', 'swiglu', 512, 2048))
            self.assertFalse(gen_moe_cw_q4k.applies('q5_k', 'store', 2048, 512))
        with patch.dict(os.environ, {'MOE_CW_Q4K': '1,xch'}):
            self.assertEqual(gen_moe_cw_q4k.modes(), {'swiglu', 'store'})
            self.assertTrue(gen_moe_cw_q4k.xch())
            self.assertFalse(gen_moe_cw_q4k.applies('q4_k', 'swiglu', 512, 2048, fused=True))
            self.assertFalse(gen_moe_cw_q4k.applies('q4_k+q6_k', 'swiglu', 512, 2048))

    def test_live_fragment_bodies(self):
        text = gen_moe_cw_q4k.grouped_gemm_cw_q4k('swiglu', 512, 2048, E, KU)
        for nf in range(1, 5):   # body nf has 2 weights x nf fragments x 8 K16 MMAs per chunk
            self.assertEqual(len(re.findall(r'= vector\.mma [^\n]*_n%d\b' % nf, text)) // 1, 2 * nf * 8)

    @unittest.skipIf(np is None, 'numpy not installed')
    def test_word_wide_scale_decode_and_f16_products_match_the_reference(self):
        """_meta's word-wide 6-bit scale/min decode (both chunk halves) equals llama.cpp's get_scale_min_k4, and the f16
        products d * sc, -(dmin * m) equal fptrunc of the f32 products (what @g_k_metadata + fptrunc give the 64-row kernel)."""
        rng = np.random.default_rng(5)
        q = rng.integers(0, 256, (4096, 12), dtype=np.uint8)
        w = q.view('<u4').astype(np.uint64)   # (n, 3) words w1, w2, w3
        w1, w2, w3 = w[:, 0], w[:, 1], w[:, 2]
        M = lambda v: np.uint64(v)
        lo_sc, lo_m = w1 & M(0x3F3F3F3F), w2 & M(0x3F3F3F3F)
        hi_sc = (w3 & M(0x0F0F0F0F)) | ((w1 >> M(2)) & M(0x30303030))
        hi_m = ((w3 >> M(4)) & M(0x0F0F0F0F)) | ((w2 >> M(2)) & M(0x30303030))
        byte = lambda v, j: ((v >> M(8 * j)) & M(255)).astype(np.int64)
        for j in range(8):
            if j < 4:
                sc, mn = q[:, j] & 63, q[:, j + 4] & 63
                gs, gm = byte(lo_sc, j), byte(lo_m, j)
            else:
                sc = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
                mn = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
                gs, gm = byte(hi_sc, j - 4), byte(hi_m, j - 4)
            np.testing.assert_array_equal(gs, sc)
            np.testing.assert_array_equal(gm, mn)
        d = np.arange(65536, dtype=np.uint16).view(np.float16)
        d = d[np.isfinite(d)][:, None]
        codes = np.arange(64, dtype=np.float16)[None, :]
        with np.errstate(over='ignore', invalid='ignore'):
            # an f16 multiply rounds the exact product once (IEEE); the 64-row kernel rounds the f32 product, which is exact
            exact = (d.astype(np.float64) * codes.astype(np.float64)).astype(np.float16)
            ref = (d.astype(np.float32) * codes.astype(np.float32)).astype(np.float16)
            nexact = ((-d).astype(np.float64) * codes.astype(np.float64)).astype(np.float16)
            nref = (-(d.astype(np.float32) * codes.astype(np.float32))).astype(np.float16)
        np.testing.assert_array_equal(exact.view(np.uint16), ref.view(np.uint16))
        np.testing.assert_array_equal(nexact.view(np.uint16), nref.view(np.uint16))

if __name__ == '__main__':
    unittest.main()
