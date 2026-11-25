"""moe.prefill.cw_q4k (gen_moe_cw_q4k.py): drop-in column-wave grouped Q4_K / Q5_K MoE GEMMs."""
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


FMTS = ('q4_k', 'q5_k')


def build(env, mode, N, K, fmt='q4_k'):
    with patch.dict(os.environ, env):
        return gen_moe.grouped_gemm(fmt, mode, N, K, E, KU, gen_gemm.gemm)


def signature(text, name):
    m = re.search(r'kernel\.def (?:target\(@\w+\) )?export\("%s"\) @%s\(([^)]*)\).*?\} launch\(([^)]*)\)' % (name, name), text, re.S)
    assert m, name
    return m.group(1), m.group(2)


class CwQ4K(unittest.TestCase):
    def test_off_is_the_64_row_kernel(self):
        for fmt in FMTS:
            for mode, N, K in SHAPES:
                base = build({'MOE_CW_Q4K': '0'}, mode, N, K, fmt)
                with patch.dict(os.environ, {}):
                    os.environ.pop('MOE_CW_Q4K', None)
                    unset = gen_moe.grouped_gemm(fmt, mode, N, K, E, KU, gen_gemm.gemm)
                self.assertEqual(base, unset)
                self.assertNotIn('%cq_m63', base)

    def test_same_names_and_launch_signatures_both_targets(self):
        for fmt, store16, (mode, N, K) in [(f, s, sh) for f in FMTS for s in ('0', '1') for sh in SHAPES]:
                env = {'LOOM_RDNA4': '1', 'MOE_STORE16': store16}
                base = build({**env, 'MOE_CW_Q4K': '0'}, mode, N, K, fmt)
                cw = build({**env, 'MOE_CW_Q4K': '1'}, mode, N, K, fmt)
                self.assertIn('%cq_m63', cw)
                self.assertEqual('%cq_m16' in cw, fmt == 'q5_k')
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
            self.assertTrue(gen_moe_cw_q4k.applies('q5_k', 'store', 2048, 512))
            self.assertFalse(gen_moe_cw_q4k.applies('q6_k', 'store', 2048, 512))
            self.assertFalse(gen_moe_cw_q4k.applies('q4_k+q5_k', 'swiglu', 512, 2048))
        with patch.dict(os.environ, {'MOE_CW_Q4K': '1,xch'}):
            self.assertEqual(gen_moe_cw_q4k.modes(), {'swiglu', 'store'})
            self.assertTrue(gen_moe_cw_q4k.xch())
            self.assertFalse(gen_moe_cw_q4k.applies('q4_k', 'swiglu', 512, 2048, fused=True))
            self.assertFalse(gen_moe_cw_q4k.applies('q4_k+q6_k', 'swiglu', 512, 2048))

    def test_live_fragment_bodies(self):
        for fmt in FMTS:
            for mode, N, K in SHAPES:
                text = gen_moe_cw_q4k.grouped_gemm_cw_q4k(mode, N, K, E, KU, fmt=fmt)
                for nf in range(1, 5):   # body nf has 2 weight fragments x nf row fragments x 8 K16 MMAs per chunk
                    self.assertEqual(len(re.findall(r'= vector\.mma [^\n]*_n%d\b' % nf, text)), 2 * nf * 8)
                    if fmt == 'q5_k':    # every decoded K16 piece takes its fifth bits
                        self.assertEqual(len(re.findall(r'%%d\d\d\d_mq_n%d = vector\.ori' % nf, text)), 2 * 8)

    def test_xch_is_q4_k_only(self):
        a = gen_moe_cw_q4k.grouped_gemm_cw_q4k('store', 2048, 512, E, KU, fmt='q5_k', xch=True)
        self.assertEqual(a, gen_moe_cw_q4k.grouped_gemm_cw_q4k('store', 2048, 512, E, KU, fmt='q5_k'))

    @unittest.skipIf(np is None, 'numpy not installed')
    def test_q5_k_codes_match_the_reference_layout(self):
        """The column-wave Q5_K code path (code words at byte 48 + 64 * half + 32 * (g // 2) + 16 * kk, nibble g % 2; qh words
        at byte 16 + 16 * kk, shifted right 4 for the upper chunk half, then (qh << (4 - g)) & 0x10101010) gives llama.cpp's
        dequantize_row_q5_K codes ((ql & 0xF or ql >> 4) + 16 * qh bit) for every value of a superblock, gfx11 (16 values per
        lane) and gfx12 (lanes 16-31: bytes +8 of every 16)."""
        rng = np.random.default_rng(7)
        blocks = rng.integers(0, 256, (512, 176), dtype=np.uint8)
        qh, ql = blocks[:, 16:48], blocks[:, 48:176]
        ref = np.zeros((len(blocks), 256), np.int64)
        for j in range(4):   # llama.cpp: 64 values per step, u1 = 1 << 2j, u2 = 2 << 2j
            for l in range(32):
                ref[:, 64 * j + l] = (ql[:, 32 * j + l] & 0xF) + 16 * ((qh[:, l] >> (2 * j)) & 1)
                ref[:, 64 * j + 32 + l] = (ql[:, 32 * j + l] >> 4) + 16 * ((qh[:, l] >> (2 * j + 1)) & 1)
        u32 = lambda b: np.ascontiguousarray(b).view('<u4').astype(np.uint64)
        M = np.uint64
        for g12 in (False, True):
            for hi in ((0, 1) if g12 else (0,)):
                nb = 8 if g12 else 16
                got = np.full_like(ref, -1)
                for half in range(2):
                    for g in range(4):
                        for kk in range(2):
                            c0 = 48 + 64 * half + 32 * (g // 2) + 16 * kk + 8 * hi
                            h0 = 16 + 16 * kk + 8 * hi
                            w = u32(blocks[:, c0:c0 + nb])
                            w = (w >> M(4)) if g % 2 else w
                            mk = w & M(0x0F0F0F0F)
                            h = u32(blocks[:, h0:h0 + nb])
                            h = (h >> M(4)) if half else h
                            code = mk | (((h << M(4 - g)) & M(0xFFFFFFFF)) & M(0x10101010))
                            vals = code.astype('<u4').view(np.uint8).reshape(len(blocks), nb)
                            v0 = 128 * half + 32 * g + 16 * kk + 8 * hi
                            got[:, v0:v0 + nb] = vals
                if g12:   # each half-wave covers its 8 of every 16
                    mask = ((np.arange(256) // 8) % 2) == hi
                    np.testing.assert_array_equal(got[:, mask], ref[:, mask])
                else:
                    np.testing.assert_array_equal(got, ref)

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
