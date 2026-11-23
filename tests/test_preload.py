"""--parameter_mode=preload on a card's serve line (device.json card.preload_max_weight_bytes): a GPU without the field is unchanged, the R9700
preloads every package, Strix Halo preloads packages up to 40 GB of weights and says how to start on a stock host above that."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CARD = os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card")
sys.path.insert(0, CARD)
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-kb"))
import card_lib  # noqa: E402

FLINT = os.path.join(CARD, "test", "flint")
PRE = "--parameter_mode=preload"
STOCK = "systemd-run --user --pty --same-dir -p SystemCallFilter=~io_uring_setup -p SystemCallErrorNumber=ENOSYS loom-serve ..."


class PreloadRule(unittest.TestCase):
    def test_rule_per_gpu(self):
        self.assertEqual(card_lib.card_serve_args("radeon-r9700", 200_000_000_000), ["--pool_backing=fixed", PRE])
        self.assertEqual(card_lib.card_serve_args("radeon-r9700", None), ["--pool_backing=fixed", PRE])
        self.assertEqual(card_lib.card_serve_args("strix-halo", 40_000_000_000), ["--pool_backing=fixed", PRE])
        self.assertEqual(card_lib.card_serve_args("strix-halo", 40_000_000_001), ["--pool_backing=fixed"])
        self.assertEqual(card_lib.card_serve_args("strix-halo", None), ["--pool_backing=fixed"])   # unknown weights: unchanged
        self.assertEqual(card_lib.preload_mode("strix-halo", 70_000_000_000), "default")
        self.assertEqual(card_lib.preload_mode("gfx1201-16gb", 1), "preload")   # the 16 GB class: no weight limit, like the R9700
        self.assertEqual(card_lib.card_serve_args("gfx1201-16gb", 14_000_000_000),
                         ["--memory_bytes=15000000000", "--state_slab_bytes=4mib", "--weight_slab_bytes=64mib", PRE])
        for g in ("gfx1150", "radeon-rx7900xtx"):   # no field: unchanged
            self.assertIsNone(card_lib.preload_mode(g, 1), g)
            self.assertEqual(card_lib.card_serve_args(g, 1), card_lib.gpu_serve_args(g))

    def test_data_file_serve_args_win(self):
        info = {"devices": {"gfx1150": {"serve_args": ["--memory_bytes=1"]}}}
        self.assertEqual(card_lib.info_serve_args(info, "gfx1150", 1), ["--memory_bytes=1"])

    def test_weight_bytes_sums_every_parameter_file_once(self):
        pf = card_lib.load_package(os.path.join(FLINT, "pkg"))
        self.assertEqual(card_lib.package_weight_bytes(pf["model"], pf["lock"]), 23387911136 + 239808192)   # not the tokenizer
        self.assertIsNone(card_lib.package_weight_bytes(pf["model"], {}))


class Render(unittest.TestCase):
    def render(self, weights, sdk=None):
        with tempfile.TemporaryDirectory() as w:
            shutil.copytree(os.path.join(FLINT, "pkg"), w, dirs_exist_ok=True)
            info = json.load(open(os.path.join(FLINT, "expected", "info.json")))
            for d in info["devices"].values():
                d.pop("quick_start", None)   # a card of these weights: make_info.py build would record the Quick start from them
            if sdk:
                pin_file = os.path.join(w, "llm_config.json")
                cfg = card_lib.load_json(pin_file)
                cfg["sdk"] = {"hrx_loom": sdk, "source": "pypi"}
                with open(pin_file, "w", encoding="utf-8") as f:
                    json.dump(cfg, f)
                for dev in info["devices"].values():
                    dev["serve_args"] = card_lib.gpu_serve_args("strix-halo") + [PRE]
            json.dump(info, open(os.path.join(w, "info.json"), "w"))
            lock = json.load(open(os.path.join(w, "model.json.lock")))
            lock["files"][0]["size"] = weights
            json.dump(lock, open(os.path.join(w, "model.json.lock"), "w"))
            subprocess.run([sys.executable, "-I", os.path.join(CARD, "render_card.py"), w], check=True, capture_output=True)
            return open(os.path.join(w, "README.md"), encoding="utf-8").read()

    def test_fitting_strix_card_preloads(self):
        md = self.render(30_000_000_000)
        self.assertIn("--pool_backing=fixed " + PRE, md)
        self.assertIn("performance rows were measured without it unless their job served with it", md)
        self.assertNotIn(STOCK, md)

    def test_sdk_030_uses_default_file_mode_even_with_legacy_device_flags(self):
        readme = self.render(30_000_000_000, "0.3.0")
        self.assertIn('--pool_backing=fixed', readme)
        self.assertNotIn('--parameter_mode=', readme)
        self.assertNotIn(STOCK, readme)
        self.assertEqual(card_lib.card_serve_args("radeon-r9700", None, {"hrx_loom": "0.3.0"}),
                         ["--pool_backing=fixed"])
        legacy = {"args": ["--pool_backing=fixed", PRE], "device": card_lib.QUICK_START_DEVICE, "preset": "chat"}
        info = {"devices": {"strix-halo": {"quick_start": legacy}}}
        self.assertEqual(card_lib.recorded_quick_start(info, {"sdk": {"hrx_loom": "0.3.0"}}, "strix-halo")["args"],
                         ["--pool_backing=fixed"])
        self.assertEqual(card_lib.recorded_quick_start(info, {"sdk": {"hrx_loom": "0.2.0"}}, "strix-halo"), legacy)

    def test_big_strix_card_keeps_default_and_says_how_to_start(self):
        md = self.render(70_000_000_000)
        self.assertNotIn(PRE + "\n", md)
        self.assertIn(STOCK, md)
        self.assertEqual(md.count("performance rows were measured without it"), 0)


if __name__ == "__main__":
    unittest.main()
