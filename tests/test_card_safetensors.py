"""A safetensors MXFP4 card: roofline from the llm_config tensor table, the vLLM baseline wording, a device the format is not a target of."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "skills/loom-benchmark/scripts"))
sys.path.insert(0, str(REPO / "skills/loom-benchmark/scripts/card"))
sys.path.insert(0, str(REPO / "skills/loom-port-llm/starter"))
sys.path.insert(0, str(REPO / "fleet"))
try:
    import render_card
    import card_lib
    import inspect_safetensors
    import st_stub
    spec = importlib.util.spec_from_file_location("roofline", REPO / "skills/loom-roofline/roofline.py")
    roofline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roofline)
except ModuleNotFoundError as e:   # the card scripts need requests / gguf: not installed in the unit-test job
    raise unittest.SkipTest(f"needs {e.name}")

SLUG = "amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom"


class SafetensorsCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = inspect_safetensors.build_config(inspect_safetensors.Source.from_dump(st_stub.load(str(REPO / "fleet" / SLUG / "safetensors" / "model.safetensors.json.gz"))), drop_chat=True)

    def test_roofline_counts_the_trunk_weights_at_4_25_bits(self):
        m = roofline.model_shape_from_config(self.cfg)
        gb = m["wb"] / 1e9
        # trunk MXFP4 linears 12.9 GB + bf16 head 2.54 GB + the gathered embedding row; the MTP block (blk.64) is the drafter and is not counted
        self.assertGreater(gb, 15.0)
        self.assertLess(gb, 15.7)
        self.assertAlmostEqual(roofline.ceilings(m, bw=613e9)["decode_tps"], 613 / gb, delta=0.5)
        self.assertEqual(m["geo"]["L"], 64)
        self.assertEqual(len(m["geo"]["attn"]), 16)   # the full-attention layers; the 48 gated-delta-net layers carry ssm tensors

    def test_other_baseline_engine_text(self):
        base = {"name": "vLLM", "version": "vllm/0.18.0", "args": "--speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 3}'"}
        lines = "\n".join(render_card.methodology(base, {}, ("mtp", 3), None, "Values are the median of 3 sessions", 3))
        self.assertIn("vLLM ROCm `vllm/0.18.0`", lines)
        self.assertIn("on both engines", lines)
        self.assertNotIn("llama.cpp ROCm", lines)
        self.assertNotIn("GGUF", lines)

    def test_no_baseline_rows_say_so(self):
        base = {"name": "vLLM", "version": None, "args": None, "differs": ["not measured on this card yet"]}
        lines = "\n".join(render_card.methodology(base, {}, ("mtp", 3), None, "Values are the median of 3 sessions", 3, measured=False))
        self.assertIn("No baseline engine is measured", lines)
        self.assertIn("vLLM ROCm: not measured on this card yet", lines)

    def test_vllm_draft_flags(self):
        self.assertEqual(render_card.vllm_draft("--speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 3}'"), ("MTP", 3))
        self.assertEqual(render_card.vllm_draft("--max-model-len 65536"), (None, None))

    def test_a_card_validates_with_a_device_that_is_not_a_target(self):
        info = {"schema": card_lib.SCHEMA_ID, "repo": "loom-fun-staging/x", "card": {}, "versions": {"card_generator": "unknown"}, "sources": {},
                "devices": {"strix-halo": {"state": "not_a_target", "reason": "no fp8 wmma"}}}
        self.assertEqual([e for e in card_lib.validate(info) if "strix-halo" in e], [])
        info["devices"]["strix-halo"].pop("reason")
        self.assertTrue([e for e in card_lib.validate(info) if "not_a_target" in e])


if __name__ == "__main__":
    unittest.main()
