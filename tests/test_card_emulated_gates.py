"""DevLab gates at the 16 GB budget keep their runner provenance and never create perf rows."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts/card"))
import card_lib
import make_info
import render_card


class EmulatedGates(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.gpu = "gfx1201-16gb"
        self.quick = {"preset": "chat-16gb", "device": "amdgpu", "args": ["--memory_bytes=15000000000"]}
        self.save("info.json", {"devices": {self.gpu: {"quick_start": self.quick}}})
        self.model = {"model": {"id": "example"}, "presets": {"chat-16gb": {"execution": {}}}}
        self.save("model.json", self.model)
        self.save("model_trial.json", self.model)
        self.save("sdk_trial.json", {"hrx_loom": "0.3.0"})
        self.save("llm_config.json", {"sdk": {"hrx_loom": "0.3.0"}})
        (self.root / "program").mkdir()
        (self.root / "program/test.loom").write_text("program")
        (self.root / "programs_sha256_trial.txt").write_text(card_lib.programs_sha256(self.root))
        (self.root / "runner.txt").write_text("r9700-worker")
        self.save("summary.json", {"job": "g16", "runner": "r9700-worker"})
        self.save("job.json", {"name": "g16", "runner_name": "r9700-worker", "labels": ["radeon-r9700"],
                               "run_id": 11, "html_url": card_lib.SKILLS_URL + "/actions/runs/11/job/22", "conclusion": "success"})
        self.server = {"runner": "r9700", "run": "11", "flags": ["--preset=chat-16gb", "--device=amdgpu", *self.quick["args"]]}
        for step in ("admit", "gates"):
            self.save(f"{step}_serve_trial.json", self.server)
        self.save("gates_trial/gates.json", {"ts": "2026-11-25T04:00:00", "level": "dev",
                                             "gates": {"agent": {"pass": False}, "agent_think_off": {"pass": True}}})
        self.save("admit_trial.json", {"admitted": False, "context": 128, "tokens": 120, "detail": "refused"})
        self.args = SimpleNamespace(artifact=self.root, variant="trial", job=self.root / "job.json",
                                    package=self.root, o=self.root / "ev.json")

    def save(self, name, obj):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    def test_mixed_results(self):
        make_info.cmd_emulated_gates(self.args)
        ev = json.loads(self.args.o.read_text())
        self.assertNotIn("performance", ev)
        self.assertEqual(ev["gpu"], self.gpu)
        self.assertFalse(ev["admission"]["admitted"])
        rows = {g["name"]: g for g in ev["gates"]}
        self.assertEqual(rows["agent"]["status"], "fail")
        self.assertEqual(rows["agent_think_off"]["status"], "pass")
        row = rows["agent"]
        self.assertEqual(card_lib.emulated_gate_errors(self.gpu, row, row["source"]), [])
        for bad in (None, {"device": "radeon-r9700", "memory_bytes": 34000000000}):
            self.assertTrue(card_lib.emulated_gate_errors(self.gpu, dict(row, emulated_on=bad), row["source"]))
        info = {"sources": {"s": row["source"]}, "devices": {self.gpu: {"gates": {"agent": dict(row, source="s")}}}}
        self.assertFalse(card_lib.requalified(info, self.gpu, "0.3.0"))

    def test_build_render_and_lint(self):
        make_info.cmd_emulated_gates(self.args)
        pkg = self.root / "fixture"
        shutil.copytree(Path(make_info.__file__).parent / "test/r9700/pkg", pkg)
        ev = json.loads(self.args.o.read_text())
        for row in [*ev["gates"], ev["admission"]]:
            row["programs_sha256"] = card_lib.programs_sha256(pkg)
        self.save("ev.json", ev)
        cfg = json.loads((pkg / "llm_config.json").read_text())
        self.save("fixture/llm_config.json", dict(cfg, sdk={"hrx_loom": "0.3.0"}))
        make_info.main(["build", str(pkg), "--repo", "example/model", "--upstream", "none",
                        "--device-results", "none", "--evidence", str(self.args.o)])
        info = card_lib.load_info(pkg / "info.json")
        rows = info["devices"][self.gpu]["gates"]
        self.assertEqual(rows["agent"]["status"], "fail")
        readme = render_card.render(str(pkg))
        for name, verdict in (("agent", "fail"), ("agent_think_off", "pass")):
            line = next(line for line in readme.splitlines() if line.startswith(f"| `{name}`"))
            self.assertIn(verdict, line)
            self.assertIn("measured on Radeon AI PRO R9700 limited to 16 GB (SDK 0.3.0)", line)
            self.assertIn("/actions/runs/11/job/22", line)
        del rows["agent"]["emulated_on"]
        self.assertTrue(any("emulated_on" in error for error in card_lib.validate(info)))

    def test_admission_budget_must_match_gates(self):
        self.save("admit_serve_trial.json", dict(self.server, flags=["--memory_bytes=34000000000"]))
        with self.assertRaisesRegex(SystemExit, "admit: live server"):
            make_info.cmd_emulated_gates(self.args)

    def test_programs_must_match(self):
        (self.root / "program/test.loom").write_text("other")
        with self.assertRaisesRegex(SystemExit, "programs differ"):
            make_info.cmd_emulated_gates(self.args)
