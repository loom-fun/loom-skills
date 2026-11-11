import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "skills/loom-benchmark/scripts"))
spec = importlib.util.spec_from_file_location("roofline", REPO / "skills/loom-roofline/roofline.py")
roofline = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(roofline)
except ModuleNotFoundError as e:   # roofline.py reads GGUFs (gguf, requests): not installed in the unit-test job
    raise unittest.SkipTest(f"roofline.py needs {e.name}")

L, ROWS, TOP_K = 8, 16, 2048
META = {"general.architecture": "qwen4exp", "qwen4exp.block_count": L, "qwen4exp.embedding_length": 256,
        "qwen4exp.attention.head_count": 4, "qwen4exp.attention.head_count_kv": 2, "qwen4exp.attention.key_length": 64,
        "qwen4exp.expert_count": 8, "qwen4exp.expert_used_count": 2, "qwen4exp.attention.indexer.top_k": TOP_K,
        "qwen4exp.attention.indexer.head_count": 4, "qwen4exp.attention.indexer.key_length": 32,
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4] * (L // 4), "qwen4exp.ple.head_vocab_sizes": [7] * ROWS}
PLE = (1 << 20) * 90   # 2^20 rows of 90 bytes
TENSORS = {"token_embd.weight": (256 * 1000, [256, 1000]), "output.weight": (256 * 1000, [256, 1000]),
           "per_layer_token_embd.weight": (PLE, [160, 1 << 20])}
for i in range(L):
    if i % 4 == 3:
        TENSORS[f"blk.{i}.attn_q.weight"] = (1000, [256, 100])
    else:
        TENSORS[f"blk.{i}.ssm_a"] = (4, [4])
    TENSORS[f"blk.{i}.ffn_down_exps.weight"] = (8000, [10, 10, 8])


class Qwen4expRoofline(unittest.TestCase):
    def setUp(self):
        self.m = roofline.shape_of(META, TENSORS)

    def test_ple_table_costs_one_row_per_hash_head(self):
        per_token = self.m["detail"]["embed/head"]
        self.assertLess(per_token, 256 * 1000 + 256 + ROWS * 90 + 1)   # head + a token_embd row + 16 PLE rows
        self.assertGreater(per_token, 256 * 1000 + ROWS * 90 - 1)

    def test_only_the_gdn_free_layers_attend_and_sparsely(self):
        self.assertEqual(self.m["geo"]["attn"], {3, 7})
        short, long_ = roofline.kv_read_bytes(self.m["geo"], 1000), roofline.kv_read_bytes(self.m["geo"], 100000)
        self.assertGreater(long_, short)
        # past top_k the K+V read stops growing; only the (compressed) indexer keys do
        grow = roofline.kv_read_bytes(self.m["geo"], 200000) - long_
        self.assertEqual(grow, 2 * (100000 // 4) * 32 * 2)

    def test_dense_attention_for_other_archs_is_unchanged(self):
        meta = {k: v for k, v in META.items() if "indexer" not in k and "compress" not in k}
        geo = roofline.shape_of(meta, TENSORS)["geo"]
        self.assertEqual(roofline.kv_read_bytes(geo, 4000), 2 * 2 * 2 * 64 * 4000 * 2)


if __name__ == "__main__":
    unittest.main()
