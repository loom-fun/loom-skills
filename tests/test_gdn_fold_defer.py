"""decode.spec.gdn_fold_defer (GDN_FOLD_DEFER): gen_gdn.defer_kernels / _defer_rows + draft/gen_draft's flush and pend launches."""
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/loom-port-llm/starter/draft"))
sys.path.insert(0, str(ROOT / "skills/loom-port-llm/starter"))
import gen_gdn  # noqa: E402
import gen_draft  # noqa: E402

CFG = {"gdn": {"conv_kernel": 4, "k_heads": 16, "v_heads": 48, "head_k": 128, "head_v": 128, "inner": 6144, "v_order": "grouped"},
       "recurrent_layers": [i for i in range(64) if i % 4 != 3]}
ENV = {"GDN_ROW_LANES": "8", "GDN_COLS": "4", "LLM_DRAFT": '{"kind": "mtp"}', "GDN_CHUNK": "1"}
KEYS = list(ENV) + ["GDN_FOLD", "GDN_FOLD_DEFER"]


def with_env(fn, **extra):
    old = {k: os.environ.get(k) for k in KEYS}
    try:
        os.environ.update(ENV)
        for k in ("GDN_FOLD", "GDN_FOLD_DEFER"):
            os.environ.pop(k, None)
        os.environ.update(extra)
        return fn()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def kernels(**extra):
    return with_env(lambda: gen_gdn.gdn_kernels(CFG, 1e-6), **extra)


def geometry(**extra):
    return with_env(lambda: gen_gdn.gdn_geometry(CFG), **extra)


def kernel(text, name):
    a = text.index(f'export("{name}")')
    a = text.rindex("kernel.def", 0, a)
    return text[a:text.index("  kernel.return\n}", a)]


class GdnFoldDefer(unittest.TestCase):
    def test_off_changes_nothing(self):
        # without the choice (or without GDN_FOLD) the slots and the fold kernels are those of decode.spec.gdn_fold
        self.assertEqual(geometry(GDN_FOLD="1"), geometry(GDN_FOLD="1", GDN_FOLD_DEFER="0"))
        self.assertEqual(geometry(), geometry(GDN_FOLD_DEFER="1"))
        self.assertEqual(kernels(GDN_FOLD="1"), kernels(GDN_FOLD="1", GDN_FOLD_DEFER="0"))
        self.assertNotIn("g_gdn_pend", kernels(GDN_FOLD="1"))

    def test_slot_holds_the_pending_fold(self):
        off = geometry(GDN_FOLD="1")
        NG = len(CFG["recurrent_layers"])
        block = gen_gdn.PEND_HEAD + gen_gdn.FOLD_W * (16 * 128 + 48 * 128 + 48) * 4
        self.assertEqual(off["pend_stride"], 0)
        tail = geometry(GDN_FOLD="1", GDN_FOLD_DEFER="1")   # default: every layer's pending block after the states, layer stride unchanged
        self.assertEqual(tail["layer_bytes"], off["layer_bytes"])
        self.assertEqual(tail["pend_base"], off["layer_bytes"] * NG)
        self.assertGreaterEqual(tail["pend_stride"], block)
        self.assertEqual(tail["slot_bytes"], tail["pend_base"] + tail["pend_stride"] * NG)
        self.assertEqual(tail["pend_base"] % 256, 0)
        self.assertEqual(tail["pend_stride"] % 256, 0)

    def test_on_emits_the_defer_kernels(self):
        t = kernels(GDN_FOLD="1", GDN_FOLD_DEFER="1")
        for k in ("g_gdn_pend", "g_gdn_flush", "g_gdn_pend_clear", "g_gdn_fold", "g_gdn_recurrence_rec_rdna4"):
            self.assertIn(f'export("{k}")', t)
        # every other kernel is unchanged (the layer stride too)
        off = kernels(GDN_FOLD="1")
        for k in ("g_gdn_conv", "g_gdn_recurrence", "g_gdn_recurrence_rdna4", "g_gdn_fold", "g_gdn_fold_hist", "g_gdn_conv_bank"):
            self.assertEqual(kernel(t, k), kernel(off, k), k)

    def test_pending_fold_uses_the_fold_order(self):
        t = kernels(GDN_FOLD="1", GDN_FOLD_DEFER="1")
        for name in ("g_gdn_recurrence_rec", "g_gdn_recurrence_rec_rdna4"):
            r = kernel(t, name)
            # S = fma(k, dk, S * decay) per pending step, then the speculative recurrence starts from the folded state
            self.assertIn("%dfr_sd0 = vector.mulf %dfr_s0, %dfr_decv : vector<16xf32>", r)
            self.assertIn("%dfr_n0 = vector.fmaf %dfr_kv, %dfr_dkv0, %dfr_sd0 : vector<16xf32>", r)
            self.assertIn("%s0 = %pinit0 : vector<16xf32>", r)
            self.assertNotIn("%s0 = %init0 :", r)
            # a speculative span writes the folded state to its writer slot
            self.assertIn("scf.if %rank_spec {\n    vector.store %pinit0, %next_view[%si0] {cache_scope = device, cache_temporal = non_temporal}", r)
        rec_on, rec_off = kernel(t, "g_gdn_recurrence_rec_rdna4"), kernel(kernels(GDN_FOLD="1"), "g_gdn_recurrence_rec_rdna4")
        ops = lambda s: [l.strip() for l in s.split("\n") if re.search(r"= (vector|scalar)\.(dotf|expf|softplusf|logisticf|subf)", l)]
        self.assertEqual(ops(rec_on), ops(rec_off))
        f = kernel(t, "g_gdn_flush")
        self.assertIn("%sd = vector.mulf %s, %decay_vector : vector<4xf32>", f)
        self.assertIn("%nx = vector.fmaf %kv, %dkv, %sd : vector<4xf32>", f)

    def test_flush_and_clear_only_publishing_spans(self):
        t = kernels(GDN_FOLD="1", GDN_FOLD_DEFER="1")
        self.assertIn("%mine = scalar.andi %active, %publish : i1", kernel(t, "g_gdn_flush"))
        self.assertIn("%live = scalar.andi %active, %publish : i1", kernel(t, "g_gdn_pend_clear"))
        p = kernel(t, "g_gdn_pend")
        self.assertIn("%mine = scalar.andi %active, %spec : i1", p)
        self.assertIn("func.call @g_gdn_span(%committed", p)

    def test_programs(self):
        sig = gen_draft.ADVANCE_SIG
        prog = (sig + "\n" + gen_draft.H_ALLOCA + "  %t = config.get @llm.tokens : index\n"
                "  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)\n}\n")
        g = {"GDN": geometry(GDN_FOLD="1", GDN_FOLD_DEFER="1"), "NG": 48}
        on = with_env(lambda: gen_draft.advance_program(prog, g), GDN_FOLD="1", GDN_FOLD_DEFER="1")
        self.assertLess(on.index("kernel.launch @g_gdn_flush["), on.index("kernel.launch @g_gdn_pend_clear["))
        self.assertLess(on.index("kernel.launch @g_gdn_pend_clear["), on.index("kernel.launch @g_token_info["))
        g_off = {"GDN": geometry(GDN_FOLD="1"), "NG": 48}
        off = with_env(lambda: gen_draft.advance_program(prog, g_off), GDN_FOLD="1")
        self.assertNotIn("g_gdn_flush", off)


if __name__ == "__main__":
    unittest.main()
