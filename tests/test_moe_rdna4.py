"""MoE grouped-GEMM generators: the gfx12 twin K_rdna4 beside K (gen_gemm_rdna4.py), K's text unchanged, LOOM_RDNA4=0 = no twin."""
import os
import re
import sys
import unittest

STARTER = os.path.join(os.path.dirname(__file__), "..", "skills", "loom-port-llm", "starter")
LFM = os.path.join(os.path.dirname(__file__), "..", "skills", "loom-port-llm", "reference", "lfm2moe-example")
sys.path[:0] = [STARTER, LFM]
import gen_gemm  # noqa: E402
import gen_moe  # noqa: E402
import gen_moe_lfm  # noqa: E402

CASES = [("q4_k", "swiglu", {}), ("q6_k", "store", {}), ("mxfp4", "swiglu", dict(fused=True, bias=True)), ("q4_k+q6_k", "swiglu", {}),
         ("q8_0", "store", {}), ("q5_0", "swiglu", {})]


def gen(fn, off):
    old = os.environ.get("LOOM_RDNA4")
    os.environ["LOOM_RDNA4"] = "0" if off else "1"
    try:
        return fn()
    finally:
        os.environ.pop("LOOM_RDNA4") if old is None else os.environ.__setitem__("LOOM_RDNA4", old)


class MoeTwins(unittest.TestCase):
    def test_twin_beside_unchanged_kernel(self):
        for fmt, mode, kw in CASES:
            fn = lambda: gen_moe.grouped_gemm(fmt, mode, 768, 2048, 128, 8, gen_gemm.gemm, **kw)
            off, on = gen(fn, True), gen(fn, False)
            self.assertNotIn("rdna4", off)
            self.assertTrue(on.startswith(off), (fmt, mode))
            twin = on[len(off):]
            k = re.search(r'export\("(\w+)"\)', off).group(1)
            self.assertIn(f'kernel.def target(@rdna4) export("{k}_rdna4") @{k}_rdna4(', twin)
            self.assertEqual(re.search(r"\} launch\((.*)\) \{", off).group(1), re.search(r"\} launch\((.*)\) \{", twin).group(1))
            self.assertNotIn("vector<16xf16>, vector<16xf16>, vector<8xf32>", twin)
            self.assertIn("vector.fragment.load<rhs>", twin)

    def test_lfm_kit_bounds_are_tight_in_both_kernels(self):
        fn = lambda: gen_moe_lfm.grouped_gemm("q4_k", "store", 2048, 1536, 64, 4, gen_gemm.gemm)
        off, on = gen(fn, True), gen(fn, False)
        self.assertTrue(on.startswith(off))
        for text in (off, on[len(off):]):   # the 2^20 x 2^18 dynamic-stride extent is rejected by the fragment store
            self.assertNotIn("1048576), multiple_of(%rows_arg, 64)", text)
            self.assertIn(f"range(%rows_arg, 64, {8192 * 4 + 64 * 64})", text)
            self.assertIn("range(%n_arg, 64, 2048)", text)


if __name__ == "__main__":
    unittest.main()
