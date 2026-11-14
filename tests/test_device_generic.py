"""Device-generic campaign tooling: device_jobs.py / fit.py / make_info --fit / build.py presets / the card of a device without a runner read the
device registry; the R9700 entry points (r9700_jobs.py, r9700_fit.py) give what they always gave."""
import contextlib
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, ".github", "scripts")
CARD = os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card")
sys.path.insert(0, CARD)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-port-llm", "starter"))
import card_lib  # noqa: E402
import device_jobs  # noqa: E402
import fit  # noqa: E402
import make_info  # noqa: E402

registry = card_lib.registry
RX, R9, STRIX = "radeon-rx7900xtx", "radeon-r9700", "strix-halo"


def run(name, args):
    old = (card_lib.published_decode, sys.argv)
    card_lib.published_decode = lambda slug, org="loom-fun": 300.0 if "0.6B" in slug else 60.0
    sys.argv = [name] + args
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(SCRIPTS, name), run_name="__main__")
    finally:
        card_lib.published_decode, sys.argv = old
    return buf.getvalue()


class DeviceJobs(unittest.TestCase):
    def test_wrapper_is_the_generic_tool(self):   # r9700_jobs.py == device_jobs.py --device radeon-r9700 (the _comment names the entry point)
        for args in (["--all"], ["--all", "--gates", "dev", "--reps", "4", "--card-org", "loom-fun"], ["unsloth-Qwen3-0.6B-loom"]):
            a = json.loads(run("r9700_jobs.py", args))
            b = json.loads(run("device_jobs.py", ["--device", R9, *args]))
            a["_comment"] = a["_comment"].replace("r9700_jobs.py", "device_jobs.py")
            self.assertEqual(a, b)

    def test_only_measurable_devices(self):   # a runner, a pool and baselines
        jd = device_jobs.job_devices()
        self.assertIn(R9, jd)
        self.assertNotIn(RX, jd)      # no runner
        self.assertNotIn(STRIX, jd)   # no memory pool (ab_jobs.py covers it)
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "device_jobs.py"), "--device", RX, "x"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)   # argparse: RX is not a choice (no runner, no pool fit table, no baselines)

    def test_field_names_follow_the_runner_key(self):
        row = {"slug": "s", "w": 3e9, "kv": 40e9, "kvpp": 4e5, "ctx": 1e5, "verdict": "fits (reduced ctx)", "rx7900xtx_execution": {"prefill_capacity": 2048}}
        job, ex_ctx, old, ctx, preset = device_jobs.device_fields(RX, row, {"memory_bytes": 20e9}, 24e9)
        self.assertEqual(preset, {"context_capacity": ctx, "pool_capacity": ctx, "prefill_capacity": 2048})
        self.assertIn("require_preset_rx7900xtx", job)
        self.assertIn("memory_bytes_rx7900xtx", job)
        self.assertNotIn("memory_bytes_r9700", job)
        row.pop("rx7900xtx_execution")
        self.assertIsNone(card_lib.gpu_preset_execution(row, 1000, RX))
        self.assertEqual(card_lib.gpu_preset_name(["chat"], RX), "chat-rx7900xtx")
        self.assertEqual(card_lib.gpu_preset_name(["chat"]), "chat-r9700")

    def test_ab_devices_are_the_registry_devices_with_a_runner(self):
        import ab_jobs
        b = registry.methodology()["baselines"]
        self.assertEqual(sorted(ab_jobs.DEVICES), sorted(k for k, d in registry.devices().items() if d["runner"]["labels"] and "rocm" in b.get(k, {})))
        self.assertEqual({k: ab_jobs.DEVICES[k]["prefix"] for k in (STRIX, R9)}, {STRIX: "sab-", R9: "rab-"})   # grandfathered job names


class FitTable(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(registry.fit_table(R9), ".github/devlab/r9700_fit.json")
        self.assertEqual(registry.fit_table(RX), ".github/devlab/radeon-rx7900xtx_fit.json")
        self.assertEqual(registry.fit_execution_key(R9), "r9700_execution")
        self.assertEqual(registry.fit_execution_key(RX), "rx7900xtx_execution")

    def test_approx_is_the_fit_device_tool(self):
        sys.path.insert(0, os.path.join(ROOT, "fleet"))
        import fit_device
        self.assertIs(fit_device.verdicts, fit.verdicts)
        out = subprocess.run([sys.executable, os.path.join(SCRIPTS, "fit.py"), "--device", RX, "--approx", "--json"], capture_output=True, text=True, check=True).stdout
        self.assertEqual(json.loads(out), fit.verdicts(RX, json.load(open(os.path.join(ROOT, ".github/devlab/r9700_fit.json")))))

    def test_strix_has_no_table(self):
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "fit.py"), "--device", STRIX], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)


class MakeInfoFit(unittest.TestCase):
    def test_fit_arguments(self):
        r9 = os.path.join(ROOT, ".github", "devlab", "r9700_fit.json")
        self.assertEqual(make_info.fit_tables(["x/r9700_fit.json"]), [(R9, "x/r9700_fit.json")])      # a path: the device whose table has that name
        self.assertEqual(make_info.fit_tables(["x/other.json"]), [(R9, "x/other.json")])               # the legacy single table
        self.assertEqual(make_info.fit_tables([f"{RX}=y.json"]), [(RX, "y.json")])
        self.assertEqual([(g, os.path.normpath(p)) for g, p in make_info.fit_tables(["auto"])], [(R9, os.path.normpath(r9))])                                    # tables that exist in this checkout
        self.assertEqual(make_info.fit_tables(None), [])


class Presets(unittest.TestCase):
    def test_dual_target_suffixes(self):
        import dual_target
        self.assertEqual(dual_target.R9700_PRESET_SUFFIX, "-r9700")
        self.assertEqual(dual_target.PRESET_SUFFIXES, {k: d["preset_suffix"] for k, d in registry.devices().items() if d.get("preset_suffix")})
        self.assertEqual(dual_target.PRESET_SUFFIXES[RX], "-rx7900xtx")


class UnmeasuredDevice(unittest.TestCase):
    def test_no_quick_start_without_a_runner(self):   # a device with no runner and no results: "not measured", no Quick start
        src = os.path.join(CARD, "test", "r9700")
        two = os.path.join(CARD, "test", "r9700_two", "expected", "info.json")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as t:
            shutil.copytree(os.path.join(src, "pkg"), t, dirs_exist_ok=True)
            info = json.load(open(two))
            info["devices"][RX] = {"state": "not_measured"}
            json.dump(info, open(os.path.join(t, "info.json"), "w"), indent=1)
            subprocess.run([sys.executable, os.path.join(CARD, "render_card.py"), t], check=True, capture_output=True)
            txt = open(os.path.join(t, "README.md"), encoding="utf-8").read()
        self.assertNotIn("Serve on the AMD Radeon RX 7900 XTX", txt)
        self.assertIn("Serve on the AMD Radeon AI PRO R9700", txt)
        self.assertIn("not measured (no runner for this GPU yet)", txt)
        self.assertNotIn("not yet re-measured", txt.split("### Radeon RX 7900 XTX")[1].split("###")[0])
        self.assertIn("— not measured |", txt)


if __name__ == "__main__":
    unittest.main()
