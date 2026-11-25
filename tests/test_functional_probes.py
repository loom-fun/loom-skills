"""Specialized card gates require recorded observations on the measured package."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fleet"))
import functional_probes as probes


class ProbeEvidence(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.art = Path(t.name)
        self.package = self.art / "pkg"
        self.package.mkdir()

    def read(self, names, *steps):
        return probes.read_probes(self.art, self.package, "v", names, dict.fromkeys(steps))

    def test_features_and_consistency(self):
        (self.art / "func_v.json").write_text(json.dumps({"checks": {"chat": {"ok": True}, "tool_calling": {"ok": False}}}))
        log = ("identical when concurrent: [True, True, True, True] identical when repeated alone (same path): "
               "[True, True, True, True]\nPASS\n")
        for trial in range(1, 6):
            (self.art / f"consistency_v_{trial}.log").write_text(log)
        def check():
            return self.read(["feature_smoke", "final_strict_consistency"], "func:v", "consistency:v:5")["final-stress"]
        self.assertFalse(check()["feature_smoke"]["pass"])
        self.assertTrue(check()["final_strict_consistency"]["pass"])
        (self.art / "consistency_v_3.log").write_text(log.replace("True", "False", 1))
        self.assertFalse(check()["final_strict_consistency"]["pass"])
        (self.art / "consistency_v_3.log").write_text("PASS\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            check()

    def test_cross_mode_checks_each_pair_once(self):
        records = [dict(rep=r, prompt=p, text=str(p)) for r in range(5) for p in range(4)]
        def write(conc):
            (self.art / f"determinism_v_c{conc}.jsonl").write_text("\n".join(map(json.dumps, records)))
        write(1)
        write(4)
        def check():
            return self.read(["final_cross_mode_determinism"], "determinism:v:1:5", "determinism:v:4:5")["final-stress"]["final_cross_mode_determinism"]["pass"]
        self.assertTrue(check())
        records[3]["text"] = "different"
        write(4)
        self.assertFalse(check())
        records.append(records[0])
        write(4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check()

    def test_cold_measurements(self):
        cold = self.art / "coldcapacity_v"
        cold.mkdir()
        (self.package / "program").mkdir()
        (self.package / "program/test.loom").write_bytes(b"sample")
        exec_cfg = {"context_capacity": 64}
        (self.package / "model.json").write_text(json.dumps({"presets": {"chat": {"execution": exec_cfg}}, "model": {"id": "example"}}))
        doc = {"passed": True, "phase": "complete", "preset": "chat", "execution": exec_cfg, "model_id": "example",
               "program_sha256": {"program/test.loom": hashlib.sha256(b"sample").hexdigest()},
               "over_limit": {"error": {"code": "OUT_OF_RANGE", "type": "loom_request_error"}},
               "active": {"fixed_storage_total_bytes": 1024, "workspace_subset_bytes": 512},
               "logical_full_limit_usage": {"prompt_tokens": 63}, "sdk_stored_full_limit_usage": {"prompt_tokens": 64},
               "health_after_usage": {"completion_tokens": 1}}
        (cold / "summary.json").write_text(json.dumps(doc))
        phases = {p: {"card0.mem_info_vram_used": 20, "card0.mem_info_gtt_used": 5, "host.MemAvailable": 5, "1:2.VmHWM": 2}
                  for p in ("cold", "loaded", "full_context")}
        mem = {"samples": 3, "identity_ok": True, "stopped_by_marker": True, "amd_cards_observed": True,
               "driver_used_counters_observed": True, "maxima_by_phase": phases, "minima_by_phase": phases}
        (cold / "memory_summary.json").write_text(json.dumps(mem))
        (cold / "monitor_metadata.json").write_text(json.dumps({"cards": [{"card": "card0"}], "interval_seconds": 0.1}))
        results = self.read(probes.COLD, "coldcapacity:v")
        self.assertTrue(results["quick"]["over_capacity"]["pass"])
        self.assertEqual(set(results["cold-load-observation"]), probes.COLD - {"over_capacity"})
        self.assertTrue(all(r["pass"] for r in results["cold-load-observation"].values()))
        doc["over_limit"]["error"]["code"] = "OTHER"
        (cold / "summary.json").write_text(json.dumps(doc))
        self.assertFalse(self.read(["over_capacity"], "coldcapacity:v")["quick"]["over_capacity"]["pass"])
        with self.assertRaisesRegex(ValueError, "runner summary"):
            self.read(["over_capacity"])
        (self.package / "program/test.loom").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "different package"):
            self.read(["cold_capacity"], "coldcapacity:v")
