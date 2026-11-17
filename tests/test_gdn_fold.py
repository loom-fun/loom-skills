"""decode.spec.gdn_fold (GDN_FOLD): gen_gdn.fold_kernels + draft/gen_draft.verify_program's fold path."""
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


def kernels(**extra):
    old = {k: os.environ.get(k) for k in list(ENV) + ["GDN_FOLD"]}
    try:
        os.environ.update(ENV)
        os.environ.pop("GDN_FOLD", None)
        os.environ.update(extra)
        return gen_gdn.gdn_kernels(CFG, 1e-6)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def kernel(text, name):
    a = text.index(f'export("{name}")')
    a = text.rindex("kernel.def", 0, a)
    return text[a:text.index("  kernel.return\n}", a)]


class GdnFold(unittest.TestCase):
    def test_off_emits_nothing_new(self):
        t = kernels()
        for k in ("g_gdn_conv_bank", "g_gdn_recurrence_rec", "g_gdn_fold"):
            self.assertNotIn(k, t)

    def test_on_emits_the_fold_kernels(self):
        t = kernels(GDN_FOLD="1")
        for k in ("g_gdn_conv_bank", "g_gdn_recurrence_rec", "g_gdn_recurrence_rec_rdna4", "g_gdn_fold", "g_gdn_fold_hist"):
            self.assertIn(f'export("{k}")', t)
        self.assertIn('kernel.def target(@rdna4) export("g_gdn_recurrence_rec_rdna4")', t)
        # the unchanged kernels stay byte-identical
        off = kernels()
        for k in ("g_gdn_conv", "g_gdn_recurrence", "g_gdn_recurrence_rdna4"):
            self.assertEqual(kernel(t, k), kernel(off, k))

    def test_recording_keeps_the_recurrence_arithmetic(self):
        t = kernels(GDN_FOLD="1")
        base, rec = kernel(t, "g_gdn_recurrence_rdna4"), kernel(t, "g_gdn_recurrence_rec_rdna4")
        ops = lambda s: [l.strip() for l in s.split("\n") if re.search(r"= (vector|scalar)\.(mulf|fmaf|subf|dotf|expf|softplusf|logisticf)", l)]
        self.assertEqual(ops(base), ops(rec))
        self.assertIn("%bank: buffer) {", rec)
        self.assertIn("vector.store %db_next, %bd_view[%rb_row, %out_channel]", rec)

    def test_fold_update_matches_the_recurrence_order(self):
        f = kernel(kernels(GDN_FOLD="1"), "g_gdn_fold")
        # S = fma(k, dk, S * decay): the recurrence's mulf then fmaf, nothing reassociated
        self.assertIn("%sd0 = vector.mulf %s0, %decay_vector : vector<4xf32>", f)
        self.assertIn("%n0 = vector.fmaf %kv, %dkv0, %sd0 : vector<4xf32>", f)
        self.assertIn("workgroups(%blocks, %layers, %spans)", f)
        self.assertIn("func.call @g_gdn_span(%committed", f)


if __name__ == "__main__":
    unittest.main()
