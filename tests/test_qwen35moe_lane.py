"""Generator switches of the Qwen3.5/3.6-35B-A3B lane: each is off by default (the text is unchanged) and does what it says when on.

moe.prefill.store16 (MOE_STORE16), moe.route.tokens_per_wg past the 4-token LDS budget (ROUTE_TR=8), draft.dflash.split_window
(DFL_SPLIT_WIN) and hybrid.gdn.chunk11_cn in the pf mode of the chunk kernel. CPU only."""
import importlib
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

STARTER = Path(__file__).resolve().parents[1] / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "draft"))
import gen_gemm  # noqa: E402
import gen_moe  # noqa: E402

H, E, KU = 2048, 256, 8


def down_gemm():
    return gen_moe.grouped_gemm("q4_k", "store", H, 512, E, KU, gen_gemm.gemm)


class Store16(unittest.TestCase):
    def test_off_by_default(self):
        with patch.dict(os.environ, {"MOE_STORE16": ""}):
            self.assertIn("view<64x64xf32, %y_layout>", down_gemm())
            self.assertIn(f"view<[%rows]x{H}xf32>", gen_moe.combine_sorted(H, KU))

    def test_f16_rows_written_and_read(self):
        with patch.dict(os.environ, {"MOE_STORE16": "1"}):
            g, c = down_gemm(), gen_moe.combine_sorted(H, KU)
        self.assertNotIn("view<64x64xf32, %y_layout>", g)
        self.assertIn("view<64x64xf16, %y_layout>", g)
        self.assertIn("%tile_offset = index.scale %tile_first, %two_y16", g)
        self.assertNotIn(f"view<[%rows]x{H}xf32>", c)
        self.assertEqual(c.count("scalar.extf %vh"), KU)

    def test_gate_up_gemm_untouched(self):
        a = gen_moe.grouped_gemm("q4_k", "swiglu", 512, H, E, KU, gen_gemm.gemm)
        with patch.dict(os.environ, {"MOE_STORE16": "1"}):
            b = gen_moe.grouped_gemm("q4_k", "swiglu", 512, H, E, KU, gen_gemm.gemm)
        self.assertEqual(a, b)


class Router8(unittest.TestCase):
    def route(self, tr):
        with patch.object(gen_moe, "ROUTE_TR", tr):
            return gen_moe.route_prefill_live(H, E, KU, 1, True, False)

    def test_default_is_four_tokens_with_logits_in_lds(self):
        t = self.route(0)
        self.assertIn("%cTR = index.constant 4 : index", t)
        self.assertIn("%lg = buffer.alloca<workgroup>", t)
        self.assertNotIn("%kept0", t)

    def test_eight_tokens_keep_logits_in_registers(self):
        t = self.route(8)
        self.assertIn("%cTR = index.constant 8 : index", t)
        self.assertNotIn("%lg = buffer.alloca<workgroup>", t)   # the logits reuse the x rows' LDS
        self.assertIn(f"%xs_bytes = index.constant {8 * H * 4} : offset", t)
        self.assertIn("%lg_view = buffer.view %xs[%zero]", t)
        self.assertEqual(len(re.findall(r"view\.store %kept\d, %lg_view", t)), 8)
        self.assertIn("scf.for %bq = [%c0 to %cHB step %cu4]", t)          # four 128-value blocks per step
        self.assertIn("%store_lane = scalar.andi %store_lt, %gl_lt : i1", t)   # padding rows stay unwritten
        # the sums keep their order: block u + 1 adds onto block u's partial
        self.assertIn("%n1_0 = scalar.addf %n0_0, %d1_0 : f32", t)

    def test_too_many_tokens_refused(self):
        with self.assertRaises(AssertionError):
            self.route(16)   # 16 x 2048 f32 rows do not fit 64 KiB


class DflashSplitWindow(unittest.TestCase):
    F = {"window": 4096, "conv": None, "layers": 6, "heads": 32, "kv_heads": 8, "head_dim": 128, "block_size": 16,
         "target_layers": [0] * 8, "rms_eps": 1e-6, "rope_theta": 1e7}

    def test_windowed_drafter_keeps_the_per_token_kernel_by_default(self):
        import gen_dflash
        with patch.dict(os.environ, {"DFL_SPLIT_WIN": ""}):
            self.assertFalse(gen_dflash.split_enabled(self.F))
            self.assertTrue(gen_dflash.split_enabled(dict(self.F, window=0)))

    def test_split_window_bounds_every_split_to_the_window(self):
        import gen_dflash
        with patch.dict(os.environ, {"DFL_SPLIT_WIN": "1"}):
            self.assertTrue(gen_dflash.split_enabled(self.F))
        g = {"ALLRED_SETUP": "", "allreduce": lambda parts, tag, sums: ""}
        w = gen_dflash.ctx_split_kernel(self.F, 32, 8, 128, 8 * 128 * 2, g)
        u = gen_dflash.ctx_split_kernel(dict(self.F, window=0), 32, 8, 128, 8 * 128 * 2, g)
        self.assertIn("%cwt = index.constant 4095 : index", w)
        self.assertIn("%lo_r = index.add %ctx_lo, %lo_rr : index", w)
        self.assertNotIn("%ctx_lo", u)
        self.assertNotIn("@@CTX_RANGE@@", w + u)


class Chunk11Cn(unittest.TestCase):
    def test_cn_reaches_the_pf_mode(self):
        import gen_gdn
        G = {"DK": 128, "DV": 128, "HV": 32, "HK": 16}
        with patch.object(gen_gdn, "recurrence_pf", lambda g: True), patch.object(gen_gdn, "_row_lanes", lambda a, b: (16, 2)):
            with patch.dict(os.environ, {"GDN_CHUNK11": "1", "GDN_CHUNK11_CN": "16"}):
                self.assertEqual(gen_gdn.chunk11_mode(G), "pf")
                self.assertEqual(gen_gdn.chunk11_cn(G), 16)
            with patch.dict(os.environ, {"GDN_CHUNK11": "", "GDN_CHUNK11_CN": "16"}):
                self.assertIsNone(gen_gdn.chunk11_cn(G))


if __name__ == "__main__":
    unittest.main()
