"""The device / quant / methodology registries (skills/loom-kb): consistency, and that every reader builds the table it replaced."""
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "skills/loom-kb"
STARTER = ROOT / "skills/loom-port-llm/starter"
CARD = ROOT / "skills/loom-benchmark/scripts/card"
for p in (KB, STARTER, CARD, ROOT / ".github/scripts"):
    sys.path.insert(0, str(p))
import registry  # noqa: E402
import card_lib  # noqa: E402
import formats  # noqa: E402
import dual_target  # noqa: E402
import matched_spec  # noqa: E402


class DeviceRegistry(unittest.TestCase):
    def test_tree_is_consistent(self):
        fams, devs = registry.families(), registry.devices()
        self.assertEqual(list(fams), ["rdna3", "rdna4", "cdna3", "xdna2"])
        old = ["strix-halo", "radeon-rx7900xtx", "radeon-r9700"]
        self.assertEqual([d for d in devs if d in old], old)   # the first rows keep their relative order; a new device row may follow (gfx1201-16gb, gfx1150)
        for name, f in fams.items():
            self.assertTrue((KB / "device" / name / "README.md").exists(), name)
            for d in f["devices"]:
                self.assertEqual(devs[d]["family"], name)
        for key, d in devs.items():
            self.assertTrue((KB / d["path"] / "README.md").exists(), key)
            self.assertIn(d["roofline"]["default"], d["roofline"]["presets"])
            self.assertIn(key, fams[d["family"]]["devices"])
            if d["runner"]["labels"]:   # a device with no DevLab runner (spec-only row) has none
                self.assertIn("Linux" if key == "strix-halo" else "devlab-dispatch", d["runner"]["labels"])   # a Strix job without Linux can land on Windows
        self.assertEqual({f["status"] for f in fams.values()}, {"supported", "planned"})

    def test_lookup_by_alias(self):
        self.assertEqual(registry.device("gfx1201")["device"], "radeon-r9700")
        self.assertEqual(registry.device("radeon-r9700s")["device"], "radeon-r9700")   # the DevLab R9700S roofline preset
        self.assertEqual(registry.device("gfx1151")["device"], "strix-halo")
        self.assertEqual(registry.family_of("radeon-r9700")["twin_suffix"], "_rdna4")
        with self.assertRaises(KeyError):
            registry.device("gfx9999")

    def test_runner_labels(self):   # a Strix job without Linux can land on Windows; the workflows' inline tables must agree
        d = registry.devices()
        self.assertEqual(d["strix-halo"]["runner"]["labels"], ["strix-halo", "devlab-dispatch", "Linux", "self-hosted"])
        self.assertEqual(d["radeon-r9700"]["runner"]["labels"], ["self-hosted", "devlab-dispatch", "radeon-r9700"])
        for wf in ("devlab-perf.yml", "devlab-kbench.yml"):   # the workflows read the registry (no inline copy to drift)
            txt = (ROOT / ".github/workflows" / wf).read_text(encoding="utf-8")
            self.assertIn("registry.runner_labels()", txt, wf)
            self.assertNotIn('"devlab-dispatch"', txt, f"{wf}: an inline runner label table")
        self.assertEqual(registry.runner_labels(), {dev["runner"]["key"]: dev["runner"]["labels"] for dev in d.values() if dev["runner"]["labels"]})
        out = subprocess.run([sys.executable, str(ROOT / ".github/scripts/runner_device.py"), "--labels"], capture_output=True, text=True, check=True).stdout
        self.assertEqual(json.loads(out), registry.runner_labels())
        prefixes = [(dev["runner"].get("ab_prefix") or dev["runner"]["key"] + "ab-") for dev in d.values() if dev["runner"]["labels"]]
        self.assertEqual(len(prefixes), len(set(prefixes)), "A/B job name prefixes are unique per runner")

    def test_info_schema_devices(self):   # the card schema's device enum is the registry's device list (a device row adds one entry)
        schema = json.loads((CARD / "info_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["devices"]["propertyNames"]["enum"], list(registry.devices()))

    def test_card_gpus(self):   # card_lib.GPUS as it was hard-coded
        g = card_lib.GPUS
        old = ["strix-halo", "radeon-rx7900xtx", "radeon-r9700"]
        self.assertEqual([d for d in g if d in old], old)
        self.assertEqual(list(g), list(registry.devices()))
        # the card shows its GPUs in this order (device.json card.order, else the registry's): Strix Halo, R9700, gfx1201 16 GB, gfx1150
        self.assertEqual([d for d in card_lib.GPU_ORDER if d != "radeon-rx7900xtx"],
                         ["strix-halo", "radeon-r9700", "gfx1201-16gb", "gfx1150"])
        self.assertEqual({k: v for k, v in g["strix-halo"].items() if k != "detect"},
                         {"name": 'AMD Ryzen AI Max+ 395 "Strix Halo"', "short": "Strix Halo", "arch": "gfx1151", "memory": None,
                          "roofline": "strix-halo", "memory_bytes": None, "runner": "Strix Halo", "tag": "strix-halo",
                          "serve_args": ["--pool_backing=fixed"],
                          "preload_max_weight_bytes": 40_000_000_000})   # Strix Halo is served like the R9700 (hrx-loom 0.2.0's elastic backing costs its prefill)
        self.assertEqual({k: v for k, v in g["radeon-r9700"].items() if k not in ("detect", "preset_suffix")},
                         {"name": "AMD Radeon AI PRO R9700", "short": "Radeon AI PRO R9700", "arch": "gfx1201", "memory": "32 GB",
                          "roofline": "radeon-r9700s", "memory_bytes": 34_000_000_000, "runner": "R9700", "tag": "radeon-r9700",
                          "serve_args": ["--pool_backing=fixed"], "preload_max_weight_bytes": None})
        self.assertEqual(g["radeon-r9700"]["preset_suffix"], "-r9700")
        self.assertEqual(card_lib.gpu_preset_name(["chat"]), "chat-r9700")
        self.assertEqual(card_lib.gpu_preset({"presets": ["chat", "chat-r9700"], "preset": "chat"}, "radeon-r9700"), "chat-r9700")
        self.assertEqual(card_lib.gpu_preset({"presets": ["chat", "chat-r9700"], "preset": "chat"}, "strix-halo"), "chat")
        for text, want in (("AMD Strix Halo (DevLab Dispatch runner X)", "strix-halo"), ("AMD Radeon AI PRO R9700S (gfx1201)", "radeon-r9700"),
                           ("gfx1151", "strix-halo"), ("nothing", None)):
            self.assertEqual(card_lib.gpu_of({"cpu": text}) if hasattr(card_lib, "gpu_of") else registry.detect(text), want)

    def test_roofline_presets(self):
        p = registry.roofline_presets()
        pinned = {"strix-halo": (216e9, 59e12), "radeon-rx7900xtx": (960e9, 123e12), "radeon-r9700": (613e9, 191.4e12), "radeon-r9700s": (613e9, 214e12)}
        self.assertEqual({k: (v["bw"], v["flops"]) for k, v in p.items() if k in pinned}, pinned)   # existing rows keep their values; new devices add presets
        for g in card_lib.GPUS.values():   # every card GPU names a roofline preset that exists
            self.assertIn(g["roofline"], p)

    def test_roofline_module_builds_its_table(self):
        sys.path.insert(0, str(ROOT / "skills/loom-benchmark/scripts"))
        spec = importlib.util.spec_from_file_location("roofline", ROOT / "skills/loom-roofline/roofline.py")
        roofline = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(roofline)
        except ModuleNotFoundError as e:   # roofline.py reads GGUFs (gguf, requests): not installed in the unit-test job
            self.skipTest(f"roofline.py needs {e.name}")
        self.assertEqual(roofline.DEVICES["radeon-r9700s"]["name"], "AMD Radeon AI PRO R9700S (gfx1201, 32 GB, 2.35 GHz)")
        self.assertEqual((roofline.BW, roofline.FLOPS), (216e9, 59e12))
        self.assertEqual(roofline.AT_CEILING, 0.03)

    def test_dual_target_rows(self):
        r = registry.dual_target()
        self.assertEqual((r["base"], r["twins"], r["suffix"]), ("rdna3", ["rdna4"], "_rdna4"))
        self.assertEqual(r["header"], "target.decl @device\namdgpu.target<gfx11-generic> @rdna3\namdgpu.target<gfx12-generic> @rdna4\n\n")
        self.assertEqual((dual_target.BASE, dual_target.TWIN, dual_target.SUFFIX, dual_target.HEADER, dual_target.R9700_PRESET_SUFFIX),
                         (r["base"], r["twins"][0], r["suffix"], r["header"], "-r9700"))

    def test_dual_target_fallback_equals_registry(self):
        """A kit copied outside the skills tree (no loom-kb next to it) uses built-in rows: they must equal the registry's."""
        code = ("import sys, dual_target as d; print(repr((d.BASE, d.TWIN, d.SUFFIX, d.HEADER, d.R9700_PRESET_SUFFIX)))")
        out = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(STARTER)!r}); {code}"],
                             capture_output=True, text=True, env={**os.environ, "LOOM_SKILLS": "/nonexistent"}, check=True).stdout
        r = registry.dual_target()
        self.assertEqual(out.strip(), repr((r["base"], r["twins"][0], r["suffix"], r["header"], "-r9700")))


class QuantRegistry(unittest.TestCase):
    def test_formats_agree_with_code(self):
        q = registry.quants()
        self.assertEqual(set(q), set(formats.FMT))
        for f, d in q.items():
            self.assertTrue((KB / "quant" / f / "README.md").exists(), f)
            self.assertEqual((d["block_size"], d["bytes_per_block"]), (formats.SB[f], formats.FMT[f]), f)
            self.assertEqual(d["low_bit"], f in formats.LOWBIT, f)
            self.assertEqual(d["int8_path"], f in formats.ICODES, f)
            self.assertEqual(d["ktest"]["gemm_default"], f in formats.ICODES, f)
        self.assertEqual({f for f, d in q.items() if d["int8_path"]}, set(formats.ICODES))
        self.assertEqual(set(q), set(formats.DECODES))

    def test_moe_support_agrees_with_gen_moe(self):
        src = (STARTER / "gen_moe.py").read_text(encoding="utf-8")
        moe = set(eval(re.search(r"^FMT = (\{.*?\})", src, re.M).group(1)))
        self.assertEqual({f for f, d in registry.quants().items() if d["moe"]}, moe)

    def test_factory_quants_come_from_the_registry(self):
        src = (ROOT / ".github/scripts/loom_factory.py").read_text(encoding="utf-8")   # parses its arguments at import: read the source
        self.assertIn("QUANTS = registry.file_quants()", src)
        self.assertEqual(registry.file_quants(), ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"])


class Methodology(unittest.TestCase):
    M = registry.methodology()
    MD = (KB / "method/methodology.md").read_text(encoding="utf-8")

    def test_rows_are_the_card_rows(self):
        self.assertEqual(list(self.M["rows"]), card_lib.METRIC_KEYS)

    def test_markdown_states_the_json_numbers(self):
        block = re.search(r"<!-- bands -->(.*?)<!-- /bands -->", self.MD, re.S).group(1)
        stated = {}
        for line in re.findall(r"- ([\w, ]+): \+-([\d.]+) %, reject below ([\d.]+)", block):
            for row in (r.strip() for r in line[0].split(",")):
                stated[row] = (round(float(line[1]) / 100, 4), float(line[2]))
        self.assertEqual(stated, {k: (v["band"], v["reject"]) for k, v in self.M["rows"].items() if v["reject"] is not None})
        tie = int(self.M["tie_band"] * 100)
        self.assertIn(f"win > +{tie} %, tie within +-{tie} %", self.MD)
        s = self.M["sessions"]
        self.assertIn(f"above ~{s['fast_decode_tps']}\ntok/s) gets {s['fast']} sessions", self.MD.replace("\n   ", "\n"))
        self.assertIn(f"`perf.py --repeat-rows {s['fast_repeat_rows']}`", self.MD)
        self.assertIn(f"every other model, and every other row, {s['normal']} sessions", self.MD.replace("\n   ", " "))
        self.assertEqual(s["repeat_rows"], ["decode_2k", "decode_32k", "prefill_8k"])
        self.assertIn("decode_2k, decode_32k and prefill_8k three times", self.MD.replace("\n   ", " "))
        self.assertIn(f"{self.M['sequential']['start_sessions_per_side']}+{self.M['sequential']['start_sessions_per_side']} sessions to "
                      f"{self.M['sequential']['extend_to']}+{self.M['sequential']['extend_to']}", self.MD)

    def test_old_path_is_a_pointer(self):
        old = (ROOT / "skills/loom-benchmark/reference/methodology.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(old.strip().splitlines()), 5)
        self.assertIn("skills/loom-kb/method/methodology.md", old)

    def test_readers(self):
        self.assertEqual(matched_spec.TIE_BAND, 0.03)
        self.assertEqual((card_lib.FAST_DECODE_TPS, card_lib.SESSIONS, card_lib.SESSIONS_FAST, card_lib.REPEAT_FAST), (100, 3, 6, 3))
        self.assertEqual(card_lib.sampling_plan(150), (6, 3))
        self.assertEqual(card_lib.sampling_plan(50), (3, 1))
        self.assertEqual(registry.band("c4"), registry.band("streams4"))
        self.assertEqual(registry.band("prefill_32k"), 0.02)
        self.assertEqual(registry.band("something_new"), self.M["default_band"])

    def test_gate_levels_exist_in_gates_sh(self):
        sh = (ROOT / "skills/loom-benchmark/scripts/gates.sh").read_text(encoding="utf-8")
        self.assertIn("case $level in " + "|".join(["dev", "fast", "quick", "full"]) + ")", sh)
        self.assertEqual(set(self.M["gates"]["levels"]), {"dev", "fast", "quick", "full"})
        names = set(card_lib.GATE_NAMES)
        for lvl, d in self.M["gates"]["levels"].items():
            self.assertLessEqual(set(d.get("gates", [])), names, lvl)
        self.assertEqual(self.M["gates"]["levels"]["full"]["alias_of"], "quick")

    def test_baselines_match_the_job_writers(self):
        b = self.M["baselines"]
        for k in ("strix-halo", "radeon-r9700"):
            self.assertIn(k, registry.devices())
        self.assertEqual(b["radeon-r9700"]["rocm"]["build"], "b1375")
        self.assertEqual(b["strix-halo"]["rocm"]["fallback"], "b1374")
        self.assertEqual({b[k]["vulkan"]["build"] for k in b if k in registry.devices() and "vulkan" in b[k]}, {"b12211"})
        for k, d in registry.devices().items():   # a device with a runner names its llama.cpp builds; "none: ..." (or no entry) says it has no runner
            if d["runner"]["labels"]:
                self.assertTrue({"rocm", "vulkan"} <= set(b.get(k, {})), f"{k}: a runner needs llama.cpp builds in methodology.json baselines")
            else:
                self.assertNotIn("rocm", b.get(k, {}), f"{k}: builds without a runner")


class ABCompare(unittest.TestCase):
    """strix_ab_compare.py: per-row bands by default, --band forces a flat one (streams4: 4 % per row, 3 % flat)."""

    def run_compare(self, ratio, *extra):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            Path(d, "c4_funnd.txt").write_text("100\n100\n100\n")
            Path(d, "c4_stagingnd.txt").write_text(f"{100 * ratio}\n" * 3)
            out = Path(d) / "cmp.json"
            res = subprocess.run([sys.executable, str(ROOT / ".github/scripts/strix_ab_compare.py"), d, "--json", str(out), *extra],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr[-400:])
            return json.loads(out.read_text(encoding="utf-8"))

    def test_per_row_band_by_default(self):
        j = self.run_compare(1.035)
        self.assertEqual(j["metrics"]["streams4"]["status"], "PASS")
        self.assertEqual((j["band"], j["bands"], j["verdict"]), (0.04, {"streams4": 0.04}, "pass"))

    def test_flat_band_forces_one_band(self):
        j = self.run_compare(1.035, "--band", "0.03")
        self.assertEqual(j["metrics"]["streams4"]["status"], "OUT")
        self.assertEqual((j["band"], j["bands"], j["verdict"]), (0.03, {"streams4": 0.03}, "rerun"))

    def test_outside_the_row_band(self):
        self.assertEqual(self.run_compare(1.045)["metrics"]["streams4"]["status"], "OUT")


if __name__ == "__main__":
    unittest.main()
