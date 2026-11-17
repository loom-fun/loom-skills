"""A second safetensors family (Llama, Quark MXFP4) is configuration: a name map + the inspector's generic paths (no weights)."""
import gzip
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "skills" / "loom-port-llm" / "starter"
sys.path.insert(0, str(STARTER))
import inspect_safetensors as I  # noqa: E402

DUMP = ROOT / "tests" / "data" / "llama31_8b_quark_mxfp4_header.json.gz"   # header + config dump of amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ (no weights)


class LlamaFamily(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = I.Source.from_dump(json.load(gzip.open(DUMP, "rt")))
        cls.cfg = I.build_config(cls.src, ["/nonexistent/model.safetensors"], drop_chat=True)

    def test_geometry(self):
        c = self.cfg
        self.assertEqual((c["arch"], c["layers"], c["hidden"], c["heads"], c["kv_heads"], c["head_dim"], c["ffn"], c["vocab"]),
                         ("llama", 32, 4096, 32, 8, 128, 14336, 128256))
        self.assertEqual((c["rope_type"], c["rope_dims"], c["qk_norm"], c["tied_embeddings"]), ("neox", 128, False, False))
        for k in ("gdn", "recurrent_layers", "mtp", "kv_layers"):
            self.assertNotIn(k, c)           # a plain transformer carries no hybrid keys

    def test_formats(self):
        t = self.cfg["tensors"]
        self.assertEqual(self.cfg["formats"], ["bf16", "mxfp4s"])
        self.assertEqual(t["blk.5.attn_k.weight"]["type"], "mxfp4s")
        self.assertEqual(t["blk.5.attn_k.weight"]["hf_scale"], "model.layers.5.self_attn.k_proj.weight_scale")
        self.assertEqual(t["output.weight"]["type"], "bf16")
        self.assertEqual(t["blk.0.attn_norm.weight"]["type"], "bf16")    # plain RMSNorm: read as stored, no prepared copy
        self.assertFalse(any("prep" in e for e in t.values()))

    def test_bos_from_the_chat_template(self):
        try:
            import jinja2  # noqa: F401
        except ImportError:
            self.skipTest("jinja2")
        cfg = I.build_config(self.src, ["/nonexistent/model.safetensors"])   # tokenizer_config has no add_bos_token: the template's bos decides
        self.assertEqual(cfg["bos_id"], 128000)

    def test_llama3_rope_factors_match_the_gguf_converter(self):
        f = self.cfg["rope_freq_factors"]
        self.assertEqual(len(f), 64)
        self.assertEqual((f[0], f[-1]), (1.0, 8.0))
        # Llama-3.2 (factor 32) values the GGUF carries as rope_freqs.weight (fleet/unsloth-Llama-3.2-3B-Instruct-loom)
        g = I.rope_freq_factors({"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                 "original_max_position_embeddings": 8192, "rope_theta": 500000.0}, 128)
        want = json.load(open(ROOT / "fleet" / "unsloth-Llama-3.2-3B-Instruct-loom" / "llm_config.json"))["rope_freq_factors"]
        self.assertEqual(len(g), len(want))
        for a, b in zip(g, want):
            self.assertTrue(math.isclose(a, b, rel_tol=1e-5), (a, b))

    def test_no_scaling_gives_no_factors(self):
        self.assertIsNone(I.rope_freq_factors({"rope_type": "default", "rope_theta": 1e4}, 128))
        self.assertIsNone(I.rope_freq_factors(None, 128))


if __name__ == "__main__":
    unittest.main()
