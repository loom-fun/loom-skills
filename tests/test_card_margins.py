"""Card margins select measured backends per row/GPU without changing evidence."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/loom-benchmark/scripts/card"))
import render_card  # noqa: E402


def performance(rocm, vulkan, loom=319.6, engine="llama.cpp"):
    return {"baseline": {"name": engine, "version": "test", "args": None},
            "sessions": [{"source": "rocm", "date": "2026-11-17", "runs": {"decode_prefill": 3, "streams4": 3},
                          "metrics": {k: {"loom": loom, "baseline": v} for k, v in rocm.items()}}],
            "baselines": {"vulkan": {"baseline": {"name": "llama.cpp", "version": "test", "args": None},
                                     "session": {"source": "vulkan", "date": "2026-11-16",
                                                 "metrics": {k: {"baseline": v} for k, v in vulkan.items()}}}}}


def section(info, gpu):
    lines = []
    feats = {"context": {"value": {"native": 32768}}, "speculative_decoding": {"status": "no"}}
    render_card.perf_section(lines, info, {"kind": "chat", "drafter": None, "spec": None}, feats,
                             None, gpu, "a" * 64, 32768)
    return "\n".join(lines)


class CardMargins(unittest.TestCase):
    def test_per_gpu_row_selection_preserves_values_and_provenance(self):
        info = {"sources": {k: {"kind": "devlab", "run": k, "device": None, "hrx_loom": None}
                            for k in ("rocm", "vulkan")},
                "devices": {"strix-halo": {"performance": performance({"decode": 293.4, "prefill_1k": 400},
                                                                        {"decode": 324.1, "prefill_1k": 200})},
                            "radeon-r9700": {"performance": performance({"decode": 200}, {"decode": 100})}}}
        before = copy.deepcopy(info)
        text = section(info, "strix-halo")
        self.assertIn("| decode tok/s, short prompt | 293.4 | 319.6 | -1 % |", text)
        self.assertIn("| prefill tok/s, 1k document | 400.0 | 319.6 | -20 % |", text)
        self.assertIn("| 324.1 |", text)
        self.assertIn("margin vs best llama.cpp", text)
        self.assertIn("max(ROCm, Vulkan)", text)
        self.assertIn("each row on this GPU", text)
        self.assertIn("DevLab Dispatch run rocm", text)
        self.assertIn("measured 2026-11-16; source: DevLab Dispatch run vulkan", text)
        self.assertIn("| decode tok/s, short prompt | 200.0 | 319.6 | +60 % |", section(info, "radeon-r9700"))
        self.assertEqual(info, before)

    def test_missing_or_invalid_backends_use_valid_measurement(self):
        base = {"name": "llama.cpp"}
        invalid = (None, 0, -1, float("nan"), float("inf"), True)
        for bad in invalid:
            for rocm, vk, expected in ((bad, 300, 300), (300, bad, 300), (bad, bad, None)):
                with self.subTest(rocm=rocm, vk=vk):
                    extra = performance({}, {"decode": vk})["baselines"]
                    self.assertEqual(render_card.margin_baseline("decode", {"baseline": rocm}, base, extra), expected)
        self.assertEqual(render_card.margin_baseline("decode", {"baseline": 300}, base, {}), 300)
        self.assertIsNone(render_card.margin_baseline("decode_2k", {}, base,
                                                     performance({}, {"decode": 300})["baselines"]))

    def test_vulkan_only_row_and_no_baselines_render(self):
        perf = performance({"decode": None, "decode_2k": None}, {"decode": 324.1})
        info = {"sources": {"rocm": {"kind": "devlab", "device": None, "hrx_loom": None},
                            "vulkan": {"kind": "devlab"}}, "devices": {"strix-halo": {"performance": perf}}}
        text = section(info, "strix-halo")
        self.assertIn("| decode tok/s, short prompt | — | 319.6 | -1 % |", text)
        self.assertIn("| decode tok/s, 2k context | — | 319.6 | — |", text)

    def test_safetensors_retains_vllm_margin(self):
        perf = performance({"decode": 293.4}, {"decode": 324.1}, engine="vLLM")
        info = {"sources": {"rocm": {"kind": "devlab", "device": None, "hrx_loom": None},
                            "vulkan": {"kind": "devlab"}}, "devices": {"radeon-r9700": {"performance": perf}}}
        text = section(info, "radeon-r9700")
        self.assertIn("| metric | vLLM | Loom | margin |", text)
        self.assertIn("| decode tok/s, short prompt | 293.4 | 319.6 | +9 % |", text)
        self.assertNotIn("margin vs best llama.cpp", text)


if __name__ == "__main__":
    unittest.main()
