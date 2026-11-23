"""fleet/devices data files: schema, the card's per-GPU entries, shared / own performance sections, GPU order, 0.2.0 PyPI Quick start."""
import copy
import json
import os
import shlex
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

TEST = os.path.join(CARD, "test")
SLUG = "unsloth-Qwen3-0.6B-loom"


def fixture(name):
    with open(os.path.join(TEST, "devices", name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def make_info(*args, env=None):
    return subprocess.run([sys.executable, "-I", os.path.join(CARD, "make_info.py"), *args], capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


class Schema(unittest.TestCase):
    def test_fixtures_validate(self):
        for n in ("gfx1201-16gb", "gfx1150"):
            self.assertEqual(card_lib.validate_device_results(fixture(n), n), [])

    def test_the_schema_file_is_the_one_cards_use(self):
        self.assertTrue(os.path.isfile(card_lib.DEVICE_SCHEMA_PATH))

    def test_gfx1201_vram_classes(self):   # the class is the device: 32 GB radeon-r9700, 16 GB gfx1201-16gb; no product-named device
        g = card_lib.GPUS
        self.assertNotIn("radeon-rx9070xt", g)
        self.assertEqual((g["radeon-r9700"]["memory"], g["gfx1201-16gb"]["memory"]), ("32 GB", "16 GB"))
        self.assertEqual(g["radeon-r9700"]["arch"], g["gfx1201-16gb"]["arch"])
        self.assertEqual(g["gfx1201-16gb"]["preset_suffix"], "-16gb")
        self.assertEqual(g["radeon-r9700"]["preset_suffix"], "-r9700")
        self.assertEqual(g["gfx1201-16gb"]["title"], "Radeon gfx1201, 16 GB (RX 9070 XT, RX 9070)")
        self.assertEqual(g["gfx1201-16gb"]["serve_args"], ["--memory_bytes=15000000000", "--state_slab_bytes=4mib", "--weight_slab_bytes=64mib"])
        self.assertEqual(g["gfx1201-16gb"]["preload_max_weight_bytes"], None)   # --parameter_mode=preload on every package
        self.assertEqual(card_lib.gpu_from_device({"gpu": "AMD Radeon RX 9070 XT (gfx1201)"}), "gfx1201-16gb")   # not the R9700 that also says gfx1201
        self.assertEqual(card_lib.gpu_from_device({"gpu": "AMD Radeon AI PRO R9700 (gfx1201)"}), "radeon-r9700")

    def test_legacy_rx9070xt_file_is_read_as_the_16gb_class(self):
        old = dict(fixture("gfx1201-16gb"), device="radeon-rx9070xt")
        self.assertEqual(card_lib.validate_device_results(old, "rx9070xt.json"), [])
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "rx9070xt.json"), "w", encoding="utf-8") as f:
                json.dump(old, f)
            self.assertEqual(list(card_lib.load_device_results(d)), ["gfx1201-16gb"])

    def test_refusals(self):
        d = fixture("gfx1201-16gb")
        bad = copy.deepcopy(d)
        bad["models"][SLUG]["max_context"] = None
        self.assertTrue(any("fits without max_context" in e for e in card_lib.validate_device_results(bad)))
        bad = copy.deepcopy(d)
        bad["device"] = "radeon-nothing"
        self.assertTrue(any("unknown device" in e for e in card_lib.validate_device_results(bad)))
        bad = copy.deepcopy(d)
        bad["models"]["other-Model-loom"]["serve_args"] = ["--x"]
        self.assertTrue(any("does not fit has no" in e for e in card_lib.validate_device_results(bad)))
        bad = copy.deepcopy(d)
        bad["models"][SLUG]["evidence"] = [{"evidence": card_lib.EVIDENCE_ID}]
        self.assertTrue(any("shares" in e for e in card_lib.validate_device_results(bad)))   # a shared GPU has no evidence of its own
        bad = copy.deepcopy(fixture("gfx1150"))
        src = {"kind": "devlab", "run": "1", "url": None, "device": None, "hrx_loom": None}
        bad["models"][SLUG]["evidence"] = [{"evidence": card_lib.EVIDENCE_ID, "performance": {"session": {"source": src}}}]
        self.assertTrue(any("kind machine" in e for e in card_lib.validate_device_results(bad)))   # a DevLab-less GPU's performance is a named machine's
        bad = copy.deepcopy(fixture("gfx1150"))
        bad["models"][SLUG]["execution"] = {"pool_capacity": 1}
        self.assertTrue(card_lib.validate_device_results(bad))


class Order(unittest.TestCase):
    def test_card_order(self):
        self.assertEqual([g for g in card_lib.GPU_ORDER if g != "radeon-rx7900xtx"], ["strix-halo", "radeon-r9700", "gfx1201-16gb", "gfx1150", "gfx1152"])
        info = {"devices": {g: {"state": "not_measured"} for g in ("gfx1150", "gfx1201-16gb", "strix-halo", "radeon-r9700")}}
        self.assertEqual(card_lib.card_gpus(info), ["strix-halo", "radeon-r9700", "gfx1201-16gb", "gfx1150"])
        self.assertEqual(card_lib.derived_tags({"speculative_decoding": {"status": "no", "value": None}, "image_input": {"status": "no"}},
                                               ("gfx1150", "gfx1201-16gb", "strix-halo"))[-3:], ["strix-halo", "gfx1201-16gb", "gfx1150"])

    def test_serves(self):
        self.assertTrue(card_lib.serves({"devices": {"gfx1150": {"state": "not_measured", "serve_args": []}}}, "gfx1150"))
        self.assertFalse(card_lib.serves({"devices": {"gfx1150": {"state": "not_measured"}}}, "gfx1150"))   # no runner, nothing recorded
        self.assertFalse(card_lib.serves({"devices": {"radeon-r9700": {"state": "unsupported", "required_bytes": 1}}}, "radeon-r9700"))


class Cards(unittest.TestCase):
    """build + render of the r9700 fixture package with a device file directory."""

    @staticmethod
    def model(pkg):
        with open(os.path.join(pkg, "model.json"), encoding="utf-8") as f:
            return json.load(f)

    def build(self, files, repo=f"loom-fun/{SLUG}", gpus="strix-halo,radeon-r9700", sdk=None):
        tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR") or None)
        self.addCleanup(shutil.rmtree, tmp, True)
        pkg, dev = os.path.join(tmp, "pkg"), os.path.join(tmp, "dev")
        shutil.copytree(os.path.join(TEST, "r9700", "pkg"), pkg)
        os.makedirs(dev)
        for n, doc in files.items():
            with open(os.path.join(dev, n + ".json"), "w", encoding="utf-8") as f:
                json.dump(doc, f)
        if sdk:
            p = os.path.join(pkg, "llm_config.json")
            cfg = json.load(open(p))
            cfg["sdk"] = sdk
            json.dump(cfg, open(p, "w"))
        r = make_info("build", pkg, "--repo", repo, "--org", "loom-fun-staging", "--generator-commit", "0" * 40, "--report", "generated", "--upstream", "none",
                      "--memory-bytes", "17000000000", "--gpus", gpus, "--device-results", dev)
        self.assertEqual(r.returncode, 0, r.stderr)
        subprocess.run([sys.executable, "-I", os.path.join(CARD, "render_card.py"), pkg], check=True, capture_output=True)
        with open(os.path.join(pkg, "info.json"), encoding="utf-8") as f, open(os.path.join(pkg, "README.md"), encoding="utf-8") as g:
            return pkg, json.load(f), g.read()

    def test_no_entry_no_gpu(self):   # a model the file does not list, or lists as not fitting, shows no trace of the GPU
        for repo in ("loom-fun/someone-Else-loom", f"loom-fun/{SLUG}"):
            files = {"gfx1201-16gb": fixture("gfx1201-16gb")}
            if repo.endswith("Else-loom"):
                files["gfx1201-16gb"]["models"]["someone-Else-loom"] = {"fits": False, "date": "2026-11-16", "reason": "too big"}
            else:
                files["gfx1201-16gb"]["models"][SLUG] = {"fits": False, "date": "2026-11-16", "reason": "too big"}
            pkg, info, readme = self.build(files, repo=repo)
            self.assertNotIn("gfx1201-16gb", info["devices"])
            self.assertNotIn("9070", readme)
            self.assertEqual(list(self.model(pkg)["presets"]), ["chat"])

    def test_serve_only_gpu(self):   # fits, no evidence: a Quick start block, a line in the performance part, no table
        d = fixture("gfx1150")
        pkg, info, readme = self.build({"gfx1150": d})
        self.assertEqual(info["devices"]["gfx1150"]["state"], "not_measured")
        self.assertIn("<summary>AMD Radeon 890M (gfx1150)</summary>", readme)
        self.assertIn("the package serves here (see Quick start); its performance is not measured yet.", readme)
        self.assertIn("### Radeon 890M (gfx1150)", readme)
        self.assertEqual(list(self.model(pkg)["presets"]), ["chat", "chat-gfx1150"])
        qs = info["devices"]["gfx1150"]["quick_start"]   # the Quick start as data: the preset and every loom-serve flag the README prints
        self.assertEqual(qs["preset"], "chat-gfx1150")
        self.assertEqual(qs["device"], "amdgpu://?completion_spin_ns=4294967295")
        self.assertFalse(any(a.startswith("--device") for a in qs["args"]))
        self.assertIn("chat-gfx1150 --materialize=copy", readme)
        self.assertIn(" ".join(shlex.quote(a) for a in [f"--device={qs['device']}"] + qs["args"]), readme)

    def test_from_info_without_device_files_preserves_unmeasured_serve_metadata(self):
        pkg, original, _ = self.build({"gfx1150": fixture("gfx1150")})
        prior = os.path.join(pkg, "info.json")
        result = make_info("build", pkg, "--from-info", prior, "--device-results", "none",
                           "--upstream", "none", "--memory-bytes", "17000000000")
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(prior, encoding="utf-8") as stream:
            rebuilt = json.load(stream)
        self.assertEqual(rebuilt["devices"]["gfx1150"], original["devices"]["gfx1150"])
        self.assertEqual(rebuilt["sources"], original["sources"])

    def test_bare_quick_start_requires_the_same_qualified_package_and_device(self):
        gpu = "radeon-r9700"
        sdk = {"hrx_loom": "0.3.0", "source": "pypi", "hrx_system_sha": "a" * 40, "wheel_sha256": "b" * 64}
        pkg, before, _ = self.build({}, sdk=sdk)
        package = card_lib.load_package(pkg)
        control = before["devices"][gpu]["quick_start"]
        preset = control["preset"]
        programs = card_lib.programs_sha256(pkg)
        context = package["model"]["presets"][preset]["execution"]["context_capacity"]
        proof = {"gpu": gpu, "preset": preset, "staging_repo": before["repo"], "programs_sha256": programs, "sdk": sdk,
                 "package_sha256": card_lib.quick_start_package_sha256(pkg, package, preset),
                 "run": 1000, "job_url": "https://github.com/loom-fun/loom-skills/actions/runs/1000/job/2000",
                 "artifact_id": 3000, "artifact_sha256": "e" * 64,
                 "default_delta_pct": dict.fromkeys(card_lib.ROOFLINE_METRIC, 0),
                 "launch": {"tuned": [f"--device={control['device']}", *control["args"]], "default": ["--device=amdgpu"]},
                 "sessions": [{"mode": mode, "answers_pass": True, "metrics": dict.fromkeys(card_lib.ROOFLINE_METRIC, 100),
                               "admission": {"admitted": True, "context": context, "tokens": context - 64}}
                              for mode in ("tuned", "default", "default", "tuned")]}
        proof_path = os.path.join(pkg, card_lib.QUICK_START_QUALIFICATION)
        card_lib.write_json(proof_path, proof)
        prior = os.path.join(pkg, "info.json")

        def rebuild():
            result = make_info("build", pkg, "--from-info", prior, "--device-results", "none", "--upstream", "none")
            self.assertEqual(result.returncode, 0, result.stderr)
            return card_lib.load_json(prior)

        after = rebuild()
        self.assertEqual(after["devices"][gpu]["quick_start"], {"preset": preset, "device": "amdgpu", "args": []})
        self.assertEqual(after["devices"]["strix-halo"]["quick_start"], before["devices"]["strix-halo"]["quick_start"])
        subprocess.run([sys.executable, "-I", os.path.join(CARD, "render_card.py"), pkg], check=True, capture_output=True)
        with open(os.path.join(pkg, "README.md"), encoding="utf-8") as readme:
            self.assertIn("--device=amdgpu\n", readme.read())

        for change in ({"gpu": "strix-halo"}, {"preset": "other"}, {"programs_sha256": "c" * 64},
                       {"sdk": dict(sdk, wheel_sha256="d" * 64)}, {"staging_repo": "loom-fun-staging/Other-loom"},
                       {"default_delta_pct": {"decode": -9}}, {"launch": {"tuned": [], "default": ["--device=amdgpu"]}}):
            with self.subTest(change=change):
                self.assertIsNone(card_lib.qualified_bare_quick_start(pkg, package, before["repo"], gpu, control, programs, proof | change))
        for bad_session in ({"admission": {"admitted": True, "context": context, "tokens": context // 2}},
                            {"metrics": {"decode": 100}}, {"metrics": dict.fromkeys(card_lib.ROOFLINE_METRIC, 90)}):
            sessions = list(proof["sessions"])
            sessions[1] = sessions[1] | bad_session
            with self.subTest(bad_session=bad_session):
                self.assertIsNone(card_lib.qualified_bare_quick_start(pkg, package, before["repo"], gpu, control, programs, proof | {"sessions": sessions}))
        model_path = os.path.join(pkg, "model.json")
        package["model"]["presets"][preset]["execution"]["context_capacity"] += 1024
        card_lib.write_json(model_path, package["model"])
        self.assertEqual(rebuild()["devices"][gpu]["quick_start"], control)

    def test_default_preset_reused_without_overrides(self):   # gfx1150 data that needs no other preset: `chat`
        d = fixture("gfx1150")
        d["models"][SLUG].pop("execution")
        d["models"][SLUG]["max_context"] = 40960   # the native context: nothing to cap
        pkg, info, readme = self.build({"gfx1150": d})
        self.assertEqual(list(self.model(pkg)["presets"]), ["chat"])
        self.assertNotIn("chat-gfx1150", readme)
        self.assertNotIn("served_context", info["devices"]["gfx1150"])

    def test_pypi_pin_quick_start(self):
        d = fixture("gfx1201-16gb")
        d["share_section_with"] = None
        pkg, info, readme = self.build({"gfx1201-16gb": d}, sdk={"hrx_loom": "0.2.0", "source": "pypi"})
        qs = readme.split("## Quick start")[1].split("## Performance")[0]
        self.assertEqual(qs.count('pipx install "hrx-loom[hf]==0.2.0"'), 3)   # Strix Halo, R9700, gfx1201 16 GB
        self.assertIn("Open your GPU and copy its commands.", qs)
        self.assertNotIn("pre-release", qs)
        self.assertNotIn("releases/download", qs)
        self.assertIn(f"loom-fetch huggingface:loom-fun-staging/{SLUG}:chat-16gb --materialize=copy --output=Qwen3-0.6B", qs)
        self.assertNotIn("/model.json", qs)
        self.assertEqual(qs.count("<details>"), 3)
        self.assertEqual(qs.count("</details>"), 3)
        for block in qs.split("<details>")[1:]:   # blank lines around fenced code inside <details>, so HF renders the blocks
            self.assertIn("</summary>\n\n```bash\n", block)
            self.assertIn("\n```\n\n", block)

    def test_sections(self):   # a GPU without a host section (or share_section_with absent) has its own section; the 4-GPU golden has the shared one
        shared = fixture("gfx1201-16gb")
        own = copy.deepcopy(shared)
        own["share_section_with"] = None
        for d in (shared, own):   # the R9700 has no results on this card: nothing to share
            _, info, readme = self.build({"gfx1201-16gb": d})
            self.assertIn("### Radeon gfx1201, 16 GB (RX 9070 XT, RX 9070)", readme)
        with open(os.path.join(TEST, "gpus4", "expected", "README.md"), encoding="utf-8") as f:
            golden = f.read()
        self.assertIn("### Radeon AI PRO R9700 / gfx1201 16 GB (gfx1201, 32 GB / 16 GB)", golden)
        self.assertNotIn("### Radeon gfx1201, 16 GB", golden)
        self.assertIn("| gate | description | Strix Halo | Radeon AI PRO R9700 / gfx1201 16 GB | Radeon 890M |", golden)

    def test_shared_without_a_host_section_is_own(self):   # the R9700 is not on the card: nothing to share
        _, info, readme = self.build({"gfx1201-16gb": fixture("gfx1201-16gb")}, gpus="strix-halo")
        self.assertNotIn("shares_section_with", info["devices"]["gfx1201-16gb"])
        self.assertIn("### Radeon gfx1201, 16 GB (RX 9070 XT, RX 9070)", readme)


if __name__ == "__main__":
    unittest.main()
