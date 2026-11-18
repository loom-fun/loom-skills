"""Gemma-4 assistants with per-layer KV heads (the exact-QAT 26B-A4B / 31B assistants) and the no-drafting card reason."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "skills/loom-port-llm/starter/draft"
CARD = ROOT / "skills/loom-benchmark/scripts/card"
sys.path.insert(0, str(DRAFT))
sys.path.insert(0, str(CARD))
import find_drafter  # noqa: E402
import gen_assist  # noqa: E402
import render_card  # noqa: E402

SWA = [1, 1, 1, 0]


def header(kv_heads, hidden=5376, fmt="Q4_0"):
    kv = {"general.architecture": "gemma4-assistant", "gemma4-assistant.embedding_length_out": hidden,
          "gemma4-assistant.attention.head_count_kv": kv_heads, "gemma4-assistant.attention.sliding_window_pattern": SWA,
          "gemma4-assistant.attention.key_length": 512, "gemma4-assistant.attention.key_length_swa": 256}
    return kv, {"token_embd.weight": (fmt, [1024, 262144]), "blk.0.attn_q.weight": (fmt, [1024, 8192])}


class PerLayerKV(unittest.TestCase):
    def test_kv_per_kind(self):
        self.assertEqual(gen_assist.kv_per_kind([16, 16, 16, 4], SWA), (16, 4))
        self.assertEqual(gen_assist.kv_per_kind(2, SWA), (2, 2))   # E2B/E4B: one number
        with self.assertRaises(AssertionError):
            gen_assist.kv_per_kind([16, 8, 16, 4], SWA)

    def test_check_assistant_matches_per_kind(self):
        cfg = json.load(open(ROOT / "fleet/unsloth-gemma-4-31B-it-qat-loom/llm_config.json"))
        self.assertIsNone(find_drafter.check_assistant(cfg, *header([16, 16, 16, 4])))
        self.assertIn("KV heads", find_drafter.check_assistant(cfg, *header([16, 16, 16, 16])))
        self.assertIn("embedding_length_out", find_drafter.check_assistant(cfg, *header([16, 16, 16, 4], hidden=2816)))

    def test_check_assistant_scalar_kv(self):
        cfg = {"hidden": 2560, "kv_heads": 2, "head_dim": 512, "head_dim_swa": 256, "vocab": 262144}
        kv, t = header(2, hidden=2560, fmt="Q8_0")
        self.assertIsNone(find_drafter.check_assistant(cfg, kv, t))


class NoDraftReason(unittest.TestCase):
    def test_generic_and_specific(self):
        self.assertEqual(render_card.no_draft_reason(None), render_card.NO_DRAFTER)
        self.assertEqual(render_card.no_draft_reason("loom-fun/not-a-classified-repo"), render_card.NO_DRAFTER)
        data = json.load(open(ROOT / "skills/loom-benchmark/data/drafting_classification.json"))
        for repo, c in data.items():
            reason = render_card.no_draft_reason(repo)
            if c["choice"] != "none":
                self.assertIn(f"classified {c['choice']} drafter", reason, repo)
            elif c["evidence"].startswith("no MTP head in the GGUF; no drafter from the model maker"):
                self.assertEqual(reason, render_card.NO_DRAFTER, repo)
            else:
                self.assertNotEqual(reason, render_card.NO_DRAFTER, repo)
                self.assertNotIn("no drafter published for this exact model", reason, repo)


if __name__ == "__main__":
    unittest.main()
