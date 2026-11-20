"""A card's Strix rows measured on a local machine at a named power mode (methodology rule 18), with Gufo as a further
engine running its own model: accepted as performance only with the power mode, rendered with the machine, the power per
engine and the Gufo model, and Gufo never becomes the margin's baseline."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "skills/loom-benchmark/scripts/card"
sys.path.insert(0, str(CARD))
import card_lib  # noqa: E402
import render_card  # noqa: E402

POWER = {"mode": "123w", "sampler": "hwmon", "busy_w": 40, "band": [120, 125], "flags": [],
         "engines": {"loom": {"median_w": 123.2, "median_ghz": 2.55}, "llamacpp-rocm": {"median_w": 122.8, "median_ghz": 2.51},
                     "gufo": {"median_w": 121.9, "median_ghz": 2.5}}}


def info():
    src = {"kind": "local", "run": "q38-123w", "url": None, "device": {"cpu": "AMD RYZEN AI MAX+ 395 (GMKtec EVO-X2)", "os": "Linux"},
           "hrx_loom": None, "power_mode": "123w", "power": POWER}
    return {"sources": {"local-q38-123w": src, "local-q38-123w-2": dict(src)},
            "devices": {"strix-halo": {"performance": {
                "baseline": {"name": "llama.cpp", "version": "llamacpp-rocm/b1374", "args": None},
                "sessions": [{"source": "local-q38-123w", "date": "2026-11-20", "runs": {"decode_prefill": 3, "streams4": 3},
                              "aggregate": "median", "metrics": {"prefill_8k": {"loom": 660.0, "baseline": 300.0}}}],
                "baselines": {"gufo": {"baseline": {"name": "Gufo", "version": "0.35.0 (0260e11)", "args": "--speculative dflash2",
                                                    "model": {"file": "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf",
                                                              "sha256": "2a5f" + "0" * 60, "label": "Gufo (UD-Q4_K_XL)"},
                                                    "differs": ["drafting: DFlash2, not MTP"]},
                                       "session": {"source": "local-q38-123w-2", "date": "2026-11-20",
                                                   "metrics": {"prefill_8k": {"baseline": 900.0}}}}}}}}}


class LocalPowerCard(unittest.TestCase):
    def test_perf_source_ok(self):
        self.assertTrue(card_lib.perf_source_ok({"kind": "devlab"}))
        self.assertFalse(card_lib.perf_source_ok({"kind": "local"}))
        self.assertTrue(card_lib.perf_source_ok({"kind": "local", "power_mode": "123w"}))
        self.assertFalse(card_lib.perf_source_ok(None))

    def test_render(self):
        lines = []
        feats = {"context": {"value": {"native": 262144}}, "speculative_decoding": {"status": "no"}}
        render_card.perf_section(lines, info(), {"kind": "chat", "drafter": None, "spec": None}, feats, None, "strix-halo", "a" * 64, 262144)
        text = "\n".join(lines)
        self.assertIn("| Gufo (UD-Q4_K_XL) |", text)
        self.assertIn("| 900.0 |", text)
        self.assertIn("+120 %", text)   # 660 / 300: the margin is against llama.cpp, never Gufo's own model
        self.assertIn("a local machine (not an AMD DevLab Dispatch runner), in its 123 W power mode", text)
        self.assertIn("Loom 123 W / 2.55 GHz", text)
        self.assertIn("Gufo 122 W / 2.50 GHz", text)
        self.assertIn("same machine, power mode and job", text)
        self.assertIn("running its OWN model file `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`", text)
        self.assertIn("not in the margin or the win / tie / loss counts", text)
        self.assertIn("drafting: DFlash2, not MTP", text)
        self.assertEqual(render_card.local_power_gpus(info(), ["strix-halo"]), ["strix-halo"])

    def test_perf_local_requires_power(self):
        with tempfile.TemporaryDirectory() as t:
            log = os.path.join(t, "loom.jsonl")
            with open(log, "w") as f:
                f.write(json.dumps({"label": "card", "kind": "loom", "time": "2026-11-20 10:00:00", "decode_tps": 28.0}) + "\n")
            base = [sys.executable, "-I", str(CARD / "make_info.py"), "perf", "--loom-jsonl", log, "--source-kind", "local",
                    "--device", "none", "--hrx-loom", "none", "--run", "j", "-o", os.path.join(t, "ev.json")]
            r = subprocess.run(base, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("power mode", r.stderr + r.stdout)
            pe = os.path.join(t, "power_steps.json")
            with open(pe, "w") as f:
                json.dump({"source": "hwmon", "busy_w": 40, "band": [120, 125], "engines": POWER["engines"],
                           "steps": [{"step": "perf:card:1", "flag": "median 99 W outside 120-125 W"}]}, f)
            r = subprocess.run(base + ["--power-mode", "123w", "--power-evidence", pe], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            src = json.load(open(os.path.join(t, "ev.json")))["performance"]["session"]["source"]
            self.assertEqual(src["power_mode"], "123w")
            self.assertEqual(src["power"]["flags"], ["perf:card:1: median 99 W outside 120-125 W"])
            r = subprocess.run([sys.executable, "-I", str(CARD / "make_info.py"), "perf", "--loom-jsonl", log, "--device", "none",
                                "--hrx-loom", "none", "--power-mode", "123w", "-o", os.path.join(t, "x.json")], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)   # DevLab sessions carry no power mode


if __name__ == "__main__":
    unittest.main()
