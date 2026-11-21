"""The XSJ evidence path works offline with a registry opt-in fixture."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

CARD = Path(__file__).resolve().parents[1] / "skills" / "loom-benchmark" / "scripts" / "card"
sys.path.insert(0, str(CARD))
import card_lib  # noqa: E402
import make_info  # noqa: E402
import publish_package  # noqa: E402
import render_card  # noqa: E402

GPU = "gfx1152"
SLUG = "unsloth-Qwen3-0.6B-loom"
RUN = "20261120-160817-s2j9"
SID = f"xsj-{RUN}"
DAY = "2026-11-20"
HARDWARE = {"host": "amd-krk-1", "arch": GPU, "cpu": "AMD Ryzen AI 7 350",
            "gpu": "AMD Radeon 860M", "os": "Windows 11", "driver": "32.0.31041.1004"}
SDK = {"hrx_loom": "0.2.0", "source": "pypi", "fingerprint": "baf14a1234567890"}


def source(**overrides):
    return {"kind": "xsj", "run": RUN, "url": None, "device": HARDWARE, "hrx_loom": SDK, **overrides}


class XsjEvidence(unittest.TestCase):
    def setUp(self):
        # The production registry and device enum are owned by the device PR.
        rows = card_lib.registry.devices()
        krackan = copy.deepcopy(rows["gfx1150"])
        krackan.update(device=GPU, gfx=GPU, name="AMD Radeon 860M", short="Radeon 860M",
                       preset_suffix=f"-{GPU}", runner={"key": GPU, "labels": [], "backend": "xsj"})
        krackan["card"].update(tag=GPU, detect={"pattern": "gfx1152|radeon 860m|krackan", "flags": "i"})
        self.enterContext(patch.object(card_lib.registry, "devices", return_value={**rows, GPU: krackan}))
        self.enterContext(patch.dict(card_lib.GPUS, {GPU: card_lib._gpu_row(krackan)}))
        if GPU not in card_lib.GPU_ORDER:
            self.enterContext(patch.object(card_lib, "GPU_ORDER", [*card_lib.GPU_ORDER, GPU]))
        schema_path = card_lib.SCHEMA_PATH
        schema = card_lib.load_json(schema_path)
        names = schema["properties"]["devices"]["propertyNames"]["enum"]
        if GPU not in names:
            names.append(GPU)
        load_json = card_lib.load_json
        self.enterContext(patch.object(card_lib, "load_json", side_effect=lambda path: schema if path == schema_path else load_json(path)))
        self.enterContext(patch.dict(os.environ, {"LOOM_CARD_UPSTREAM": "none"}))
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.pkg = self.tmp / "pkg"
        shutil.copytree(CARD / "test" / "r9700" / "pkg", self.pkg)
        self.digest = card_lib.programs_sha256(self.pkg)
        self.devices = self.tmp / "devices"
        self.devices.mkdir()
        self.device_path = self.write("device.json", HARDWARE)
        self.sdk_path = self.write("sdk.json", SDK)
        self.loom = self.write("loom.jsonl", {"time": f"{DAY}T16:10:00", "decode_tps": 27})
        self.llama = self.write("llama.jsonl", {"time": f"{DAY}T16:11:00", "decode_tps": 23})
        self.gates = self.write("gates.json", {"ts": f"{DAY}T16:12:00", "level": "full",
                                              "gates": {"consistency": {"pass": True}, "golden": {"pass": False},
                                                        "sampling": {"pass": True}}})

    def write(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        return path

    def identity_args(self, **overrides):
        fields = {"source-kind": "xsj", "gpu": GPU, "run": RUN, "device": self.device_path,
                  "hrx-loom": self.sdk_path, "package": self.pkg, "fetched-programs": self.digest, **overrides}
        return [arg for k, v in fields.items() if v is not None for arg in (f"--{k}", str(v))]

    def perf(self, backend="rocm", **overrides):
        path = self.tmp / f"{backend}.json"
        main = ["--loom-jsonl", str(self.loom)] if backend == "rocm" else []
        make_info.main(["perf", "--backend", backend, *main, "--llama-jsonl", str(self.llama),
                        "--baseline-version", "b1382" if main else "b12211",
                        "--differs", "prefill_32k exceeds the tested context capacity",
                        *self.identity_args(**overrides), "-o", str(path)])
        return card_lib.load_json(path)

    def gate_evidence(self, **overrides):
        path = self.tmp / "gates-evidence.json"
        make_info.main(["gates", str(self.gates), *self.identity_args(**overrides), "-o", str(path)])
        return card_lib.load_json(path)

    def device_results(self, *evidence):
        return {"schema": card_lib.DEVICE_RESULTS_ID, "device": GPU, "gpu": HARDWARE["gpu"],
                "models": {SLUG: {"fits": True, "date": DAY, "max_context": 32768, "serve_args": [],
                                  "execution": {"prefill_capacity": 512}, "evidence": evidence and list(evidence) or []}}}

    def build(self, *args):
        make_info.main(["build", str(self.pkg), "--repo", f"loom-fun/{SLUG}", "--upstream", "none",
                        "--generator-commit", "0" * 40, "--report", "generated",
                        "--device-results", str(self.devices), *map(str, args)])
        return card_lib.load_json(self.pkg / "info.json")

    def test_perf_vulkan_gates_and_regeneration_preserve_sources(self):
        gate_run = "20261120-170817-b5x9"
        doc = self.device_results(self.perf(), self.perf("vulkan"), self.gate_evidence(run=gate_run))
        self.write("devices/gfx1152.json", doc)
        info = self.build()
        dev = info["devices"][GPU]
        self.assertEqual(info["sources"][SID], source())
        self.assertEqual(dev["performance"]["sessions"][0]["metrics"]["decode"]["loom"], 27)
        self.assertEqual(dev["performance"]["baselines"]["vulkan"]["session"]["source"], SID)
        self.assertEqual(dev["gates"]["consistency"]["source"], f"xsj-{gate_run}")
        self.assertEqual(dev["gates"]["golden"]["status"], "fail")
        self.assertEqual(dev["gates"]["golden"]["level"], "full")
        for output in (render_card.render(self.pkg), render_card.render_report(self.pkg)):
            self.assertIn(f"XSJ lab run {RUN}", output)
            self.assertIn(f"XSJ lab run {gate_run}", output)
            self.assertIn("`amd-krk-1` (gfx1152)", output)
            self.assertIn("prefill_32k exceeds the tested context capacity", output)
            self.assertNotIn("AMD DevLab Dispatch Radeon", output)
        inherited = self.build("--from-info", self.pkg / "info.json", "--device-results", "none")
        self.assertEqual(inherited["sources"], info["sources"])
        self.assertEqual(inherited["devices"][GPU]["performance"], dev["performance"])
        imported = self.build("--perf-from", f"{GPU}={self.pkg / 'info.json'}", "--device-results", "none")
        self.assertEqual(imported["devices"][GPU]["performance"], dev["performance"])

    def test_comparisons_cannot_mix_xsj_runs(self):
        other = self.perf("vulkan", run="20261120-170817-b5x9")
        self.write("devices/gfx1152.json", self.device_results(self.perf(), other))
        model = self.pkg / "model.json"
        before = model.read_bytes()
        with self.assertRaisesRegex(SystemExit, "Loom and Vulkan values must come from the same XSJ run"):
            self.build()
        self.assertEqual(model.read_bytes(), before)
        self.assertFalse((self.pkg / "info.json").exists())

    def test_measured_backends_show_their_own_four_stream_flags(self):
        rocm, vulkan = self.perf(), self.perf("vulkan")
        rocm_flags = "-ngl 99 -cram 0 --device ROCm0 -c 40960 -kvu -np 4"
        vulkan_flags = "-ngl 99 -cram 0 --device Vulkan0 -c 147456 -np 4"
        rocm["performance"]["baseline"]["differs"] = [f"Four-stream flags: {rocm_flags}."]
        vulkan["performance"]["baseline"]["differs"] = [f"Four-stream flags: {vulkan_flags}"]
        self.write("devices/gfx1152.json", self.device_results(rocm, vulkan))
        self.build()
        for output in (render_card.render(self.pkg), render_card.render_report(self.pkg)):
            self.assertIn(f"- llama.cpp ROCm four-stream server arguments: `{rocm_flags}`.", output)
            self.assertIn(f"- llama.cpp Vulkan four-stream server arguments: `{vulkan_flags}`.", output)
            self.assertNotIn("Not supported / not measured:", output)

    def test_xsj_notes_keep_coverage_reasons_and_punctuation(self):
        rocm, vulkan = self.perf(), self.perf("vulkan")
        rocm["performance"]["baseline"]["differs"] = ["Loom: not measured: a longer request exceeded the preset."]
        vulkan["performance"]["baseline"]["differs"] = ["not supported: this build could not load the model."]
        vulkan["performance"]["session"]["metrics"] = {}
        self.write("devices/gfx1152.json", self.device_results(rocm, vulkan))
        self.build()
        for output in (render_card.render(self.pkg), render_card.render_report(self.pkg)):
            self.assertIn("- Loom: not measured: a longer request exceeded the preset.\n", output)
            self.assertIn("- llama.cpp Vulkan: not supported: this build could not load the model.\n", output)

    def test_vulkan_only_is_retained(self):
        self.write("devices/gfx1152.json", self.device_results(self.perf("vulkan")))
        perf = self.build()["devices"][GPU]["performance"]
        self.assertEqual(perf["sessions"], [])
        self.assertEqual(perf["baselines"]["vulkan"]["session"]["metrics"]["decode"]["baseline"], 23)
        self.assertIn(f"XSJ lab run {RUN}", render_card.render(self.pkg))

    def test_cli_refuses_missing_identity_stale_digest_and_ineligible_targets(self):
        cases = [
            ({"run": "20261131-160817-ab12"}, "lab run ID"),
            ({"device": "auto"}, "JSON files captured"),
            ({"hrx-loom": None}, "JSON files captured"),
            ({"gpu": None}, "requires --gpu"),
            ({"gpu": "gfx1150"}, "XSJ-enabled GPU without DevLab runner"),
            ({"gpu": "strix-halo"}, "XSJ-enabled GPU without DevLab runner"),
            ({"package": None}, "requires --package"),
            ({"fetched-programs": None}, "digest captured on XSJ"),
            ({"fetched-programs": "a" * 64}, "differs from the package digest"),
        ]
        for overrides, expected in cases:
            for build_evidence in (self.perf, self.gate_evidence):
                with self.subTest(overrides=overrides, command=build_evidence.__name__):
                    with self.assertRaisesRegex(SystemExit, expected):
                        build_evidence(**overrides)
        with self.assertRaisesRegex(SystemExit, "requires a source of kind xsj"):
            self.perf(**{"source-kind": "machine"})

    def test_reported_gpu_matches_registry_after_windows_trademark_cleanup(self):
        actual = self.write("actual-gpu.json", dict(HARDWARE, gpu="AMD Radeon(TM) 860M Graphics"))
        self.perf(device=actual)
        wrong = self.write("wrong-gpu.json", dict(HARDWARE, gpu="AMD Radeon RX 7900 XTX"))
        with self.assertRaisesRegex(SystemExit, "recorded GPU does not match the registry"):
            self.perf(device=wrong)

    def test_backend_marker_and_empty_devlab_labels_both_required(self):
        dev = card_lib.registry.device(GPU)
        for runner in ({"backend": "xsj", "labels": ["devlab"]}, {"labels": []}, {"backend": "other", "labels": []}):
            with self.subTest(runner=runner), patch.dict(card_lib.GPUS, {GPU: card_lib._gpu_row(dict(dev, runner=runner))}):
                self.assertFalse(card_lib.xsj_enabled(GPU))
                with self.assertRaisesRegex(SystemExit, "XSJ-enabled GPU without DevLab runner"):
                    self.perf()

    def test_device_results_reject_relabelled_rerouted_and_incomplete_evidence(self):
        example = self.perf()
        mutations = [
            (dict(example, gpu="gfx1150"), "explicitly name"),
            ({k: v for k, v in example.items() if k != "gpu"}, "explicitly name"),
            (self.with_source(example, source(kind="devlab")), "requires a source of kind xsj"),
            (self.with_source(example, source(device=dict(HARDWARE, arch="gfx1150"))), "expected gfx1152"),
            (self.with_source(example, source(device=dict(HARDWARE, host=""))), "device.host"),
            (self.with_source(example, source(hrx_loom={"hrx_loom": None})), "recorded SDK version"),
            (self.with_source(example, None), "requires a source of kind xsj"),
        ]
        for evidence, expected in mutations:
            with self.subTest(expected=expected):
                doc = dict(self.device_results(evidence), provenance=source())
                self.assertRegex("\n".join(card_lib.validate_device_results(doc)), expected)
        gate = self.gate_evidence()
        gate["gates"][0]["gpu"] = "strix-halo"
        self.assertRegex("\n".join(card_lib.validate_device_results(self.device_results(gate))), "explicitly name")
        other = dict(self.device_results(self.with_source(dict(example, gpu="gfx1150"), source(kind="machine"))),
                     device="gfx1150")
        self.assertRegex("\n".join(card_lib.validate_device_results(other)), "run ID and hardware require")

    def test_xsj_gpus_cannot_share_performance_sections_in_either_direction(self):
        self.write("devices/gfx1152.json", self.device_results())
        info = self.build("--gpus", "strix-halo")
        for gpu, other in ((GPU, "strix-halo"), ("strix-halo", GPU)):
            with self.subTest(gpu=gpu):
                invalid = copy.deepcopy(info)
                invalid["devices"][gpu]["shares_section_with"] = other
                self.assertRegex("\n".join(card_lib.validate(invalid)), "XSJ GPU must have its own performance section")
                doc = dict(self.device_results(), device=gpu, share_section_with=other)
                self.assertRegex("\n".join(card_lib.validate_device_results(doc)), "XSJ GPU must have its own performance section")

    @staticmethod
    def with_source(example, src):
        ev = copy.deepcopy(example)
        ev["performance"]["session"]["source"] = src
        return ev

    def test_run_identity_conflicts_and_stale_evidence_are_refused(self):
        perf, gate = self.perf(), self.gate_evidence()
        stale = copy.deepcopy(gate)
        stale["gates"][0]["programs_sha256"] = "a" * 64
        self.write("devices/gfx1152.json", self.device_results(perf, stale))
        with self.assertRaisesRegex(SystemExit, "differs from the package digest"):
            self.build()
        clash = copy.deepcopy(gate)
        for row in clash["gates"]:
            row["source"]["device"]["host"] = "amd-krk-2"
        self.write("devices/gfx1152.json", self.device_results(perf, clash))
        with self.assertRaisesRegex(SystemExit, "conflicting source records"):
            self.build()
        self.write("devices/gfx1152.json", self.device_results(perf, gate))
        info = self.build()
        info["sources"][SID]["kind"] = "factory"
        self.assertRegex("\n".join(card_lib.validate(info)), "must have kind xsj")
        program = next((self.pkg / "program").glob("*.loom"))
        program.write_bytes(program.read_bytes() + b"\n// changed on ingest\n")
        for renderer in (render_card.render, render_card.render_report):
            with self.assertRaisesRegex(SystemExit, "differs from the package digest"):
                renderer(self.pkg)
        with self.assertRaisesRegex(SystemExit, "differs from the package digest"):
            self.build("--from-info", self.pkg / "info.json", "--device-results", "none")

    def test_updated_package_keeps_historical_gates_without_reusing_performance(self):
        self.write("devices/gfx1152.json", self.device_results(self.perf(), self.perf("vulkan"), self.gate_evidence()))
        old = self.build()
        self.assertEqual(old["card"]["features"]["sampling"]["status"], "yes")
        program = self.pkg / "program" / "service.loom"
        program.write_bytes(program.read_bytes() + b"\n// new package revision\n")

        inherited = self.build("--from-info", self.pkg / "info.json", "--device-results", "none", "--reset-performance")
        self.assertNotIn("performance", inherited["devices"][GPU])
        self.assertEqual(inherited["devices"][GPU]["gates"], old["devices"][GPU]["gates"])
        self.assertEqual(inherited["sources"], old["sources"])
        facts = card_lib.package_facts(card_lib.load_package(self.pkg))
        self.assertEqual(inherited["card"]["features"]["sampling"], facts["features"]["sampling"])
        readme = render_card.render(self.pkg)
        self.assertIn("not re-run on this package (the last pass was on a previous build", readme)
        self.assertIn("fail on a previous build", readme)
        self.assertIn("Loom timings are not recorded for this package", readme)
        self.assertNotIn("the nightly gates re-run it", readme)
        self.assertIn(f"XSJ lab run {RUN}", readme)
        self.assertIn("not the same as this repository's programs", render_card.render_report(self.pkg))
        with self.assertRaisesRegex(SystemExit, "differs from the package digest"):
            self.build("--from-info", self.pkg / "info.json", "--reset-performance")

    def test_upload_dry_run_checks_programs_without_using_the_hub(self):
        self.write("devices/gfx1152.json", self.device_results(self.perf()))
        self.build()
        render_card.main([str(self.pkg)])
        args = ["upload", str(self.pkg), "--repo", SLUG, "--card-only", "--dry-run"]
        program = self.pkg / "program" / "service.loom"
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            publish_package.main(args)
            program.write_bytes(program.read_bytes() + b"\n// stale before upload\n")
            with self.assertRaisesRegex(SystemExit, "differs from the package digest"):
                publish_package.main(args)
            program.unlink()
            with self.assertRaisesRegex(SystemExit, "XSJ publication requires the local package"):
                publish_package.main(args)

    def test_card_only_upload_checks_the_immutable_parent_programs(self):
        self.write("devices/gfx1152.json", self.device_results(self.perf()))
        self.build()
        render_card.main([str(self.pkg)])
        remote = self.tmp / "remote"
        shutil.copytree(self.pkg, remote)
        program = remote / "program" / "service.loom"
        original = program.read_bytes()
        program.write_bytes(original + b"\n// different parent program\n")
        parent, commit, repo = "a" * 40, "b" * 40, f"loom-fun/{SLUG}"
        api = Mock(endpoint="https://huggingface.co")
        api.model_info.return_value = SimpleNamespace(sha=parent, siblings=[])
        api.upload_folder.return_value = SimpleNamespace(oid=commit, commit_url=f"{api.endpoint}/{repo}/commit/{commit}")
        hub = SimpleNamespace(HfApi=Mock(return_value=api), snapshot_download=Mock(return_value=remote),
                              hf_hub_download=Mock(return_value=remote / "model.json"))
        args = ["upload", str(self.pkg), "--repo", SLUG, "--card-only", "--parent-commit", parent]
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            with self.assertRaisesRegex(SystemExit, "target programs differ"):
                publish_package.main(args)
            api.upload_folder.assert_not_called()
            hub.snapshot_download.assert_called_once_with(repo, revision=parent, allow_patterns=["program/*.loom"])
            program.write_bytes(original)
            publish_package.main(args)
        api.upload_folder.assert_called_once()
        self.assertEqual(api.upload_folder.call_args.kwargs["parent_commit"], parent)
        self.assertEqual(api.upload_folder.call_args.kwargs["allow_patterns"], list(publish_package.CARD))


if __name__ == "__main__":
    unittest.main()
