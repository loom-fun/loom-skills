"""fleet/final_pass.py helpers (no network, no GPU) and regen.py's logits mode."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "fleet"))
import final_pass as fp  # noqa: E402
import regen  # noqa: E402

BASE = "a\nb\n  view.store %value, %view[2] : i32, view<4xi32>\nc\n"
NEW = ("a\n// logits:begin\nconfig.def @llm.logits = false : i1\n// logits:end\n\nb\n"
       "  // logits:begin\n  %x = 1\n  view.store %m, %view[2] : i32, view<4xi32>\n  // logits:end\nc\n")


class Helpers(unittest.TestCase):
    def test_strip_logits_blocks(self):
        self.assertEqual(fp.strip_logits(NEW), "a\nb\nc\n")

    def test_residual_allows_only_the_top_k_store(self):
        r = fp.residual(BASE, NEW)
        self.assertEqual([x[1:].strip() for x in r], ["view.store %value, %view[2] : i32, view<4xi32>"])
        self.assertTrue(fp.residual_ok(r))
        self.assertFalse(fp.residual_ok(fp.residual(BASE, NEW.replace("c\n", "c\nd\n"))))   # a line outside any block

    def test_measured_programs_prefers_the_remeasure_block_then_the_newest_gate(self):
        info = {"devices": {"g": {"performance": {"sessions": []}, "gates": {"x": {"programs_sha256": "old", "date": "2026-11-01"}, "y": {"programs_sha256": "new", "date": "2026-11-09"}}},
                            "h": {"performance": {"sessions": []}, "remeasure": {"programs_sha256": "rm"}, "gates": {"x": {"programs_sha256": "g"}}},
                            "n": {}}}
        self.assertEqual(fp.measured_programs(info, "g"), "new")
        self.assertEqual(fp.measured_programs(info, "h"), "rm")
        self.assertIsNone(fp.measured_programs(info, "n"))

    def model(self, logits):
        cfg = {"llm.card": True, **({"llm.logits": False} if logits else {})}
        return {"variants": {"gguf": {"services": {"chat": {"config": cfg}}}}, "origins": {"checkpoint": {"kind": "hf"}}}

    def test_model_json_gets_only_the_logits_key(self):
        out, problems = fp.patch_model_json(self.model(False), self.model(True))
        self.assertEqual(out["variants"]["gguf"]["services"]["chat"]["config"]["llm.logits"], False)
        self.assertIn("origins", out)
        self.assertEqual(problems, [])
        moved = self.model(True)
        moved["variants"]["gguf"]["services"]["chat"]["config"]["llm.context"] = 4
        self.assertTrue(fp.patch_model_json(self.model(False), moved)[1])

    def test_llm_config_merge_stamps_the_generator_and_flags_other_changes(self):
        old = {"chat": {"a": 1}, "generator": {"skills_commit": "old", "dirty": True}}
        out, problems = fp.merge_llm_config(old, {"chat": {"a": 1}, "resolved": {"scopes": []}}, "REF")
        self.assertEqual((out["generator"]["skills_commit"], out["generator"]["dirty"], out["resolved"]), ("REF", False, {"scopes": []}))
        self.assertEqual(problems, [])
        self.assertTrue(fp.merge_llm_config(old, {"chat": {"a": 2}}, "REF")[1])
        d_old = {"chat": 1, "draft": {"kind": "assist", "gguf": "/home/x/w/d.gguf"}}
        out, problems = fp.merge_llm_config(d_old, {"chat": 1, "draft": {"kind": "assist", "gguf": "d.gguf"}}, "REF")
        self.assertEqual((out["draft"]["gguf"], problems), ("d.gguf", []))
        self.assertTrue(fp.merge_llm_config(d_old, {"chat": 1, "draft": {"kind": "assist", "gguf": "e.gguf"}}, "REF")[1])


class JobOf(unittest.TestCase):
    def test_latest_finished_job_whatever_its_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            def job(name, completed, stage, decision):
                p = os.path.join(d, name)
                os.makedirs(p)
                open(os.path.join(p, "logits_staging.json"), "w").write("{}")
                return {"slug": "m", "device": "g", "stage": stage, "decision": decision, "completed": completed, "dir": p}
            state = {"jobs": {"a": job("a", "2026-11-15T09:00:00Z", "needs_decision", "hold"), "b": job("b", "2026-11-15T10:00:00Z", "needs_decision", "hold"),
                              "c": job("c", "2026-11-15T11:00:00Z", "failed", "rerun")}}
            self.assertEqual(fp.job_of(state, "m", "g")[0], "b")          # the held job counts; a failed job and an older one do not
            self.assertEqual(fp.job_of(state, "m", "other"), (None, None))
            self.assertEqual(fp.job_of(None, "m", "g"), (None, None))
            os.remove(os.path.join(d, "b", "logits_staging.json"))         # no logits result: no job
            self.assertEqual(fp.job_of(state, "m", "g")[0], "a")


class LogitsMode(unittest.TestCase):
    def test_steps_are_cut_to_the_functional_job(self):
        doc = {"jobs": [{"steps": ["perf:fun:1", "perf:staging:1", "c4:funnd", "textmatch:fun", "func:staging", "ktest:staging", "gates:staging:dev"],
                         "variants": [{"name": "fun"}, {"name": "staging"}]}]}
        regen.logits_steps(doc, ["http://b", "abc"])
        j = doc["jobs"][0]
        self.assertEqual(j["steps"], ["func:staging", "ktest:staging", "logits:staging"])
        self.assertEqual([v["name"] for v in j["variants"]], ["staging"])
        self.assertEqual(j["bundle"], {"url": "http://b", "sha256": "abc"})
        regen.logits_steps(doc, {"build": {"extension": "loom-fun/loom-gufo-extension@x", "gufo": "loom-fun/gufo@y"}})
        self.assertEqual(doc["jobs"][0]["bundle"]["build"]["gufo"], "loom-fun/gufo@y")

    def test_plan_defaults_to_the_pinned_bundle_and_can_reuse_a_run(self):
        import argparse
        pins = json.load(open(os.path.join(ROOT, ".github", "devlab", "gufo_bundle.json")))
        ns = argparse.Namespace(bundle=None, bundle_build=None, bundle_run=None)
        with tempfile.TemporaryDirectory() as d:
            doc = {"jobs": []}
            with mock.patch.object(regen, "gen_jobs", lambda st, dev, chunk: (doc, [], list(chunk))):
                for run, want in ((None, {"build": {"extension": pins["extension"], "gufo": pins["gufo"]}}), ("43700000000", {"run": "43700000000"})):
                    ns = argparse.Namespace(slugs=["x"], all=False, device=["strix-halo"], mode="logits", old=None, new_org="loom-fun-staging", gates="dev",
                                            batch=8, no_pin=True, work=d, state=os.path.join(d, "s.json"), force=True, replicas=1, sequential=False,
                                            no_baselines=False, priority=False, bundle=None, bundle_build=None, bundle_run=run)
                    with contextlib.redirect_stdout(io.StringIO()):
                        regen.cmd_plan(ns)
                    self.assertEqual(json.load(open(ns.state))["opts"]["bundle"], want)

    def verdict(self, results):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"results": results}, open(os.path.join(d, "summary.json"), "w"))
            return regen.logits_verdict(d, {})[0]

    def test_verdict_needs_all_three_steps_and_no_failure(self):
        ok = {"func:staging": "PASS", "ktest:staging": "PASS (62/62)", "logits:staging": "LOGITS PASS"}
        self.assertEqual(self.verdict(ok), "promote")
        self.assertEqual(self.verdict(dict(ok, **{"logits:staging": "not supported (no gufo bundle)"})), "hold")
        self.assertEqual(self.verdict(dict(ok, **{"logits:staging": "FAIL argmax"})), "hold")
        self.assertEqual(self.verdict({k: v for k, v in ok.items() if k != "logits:staging"}), "rerun")


class CarriedPrograms(unittest.TestCase):
    def test_sessions_of_a_covered_build_are_not_stale_after_an_identical_proof(self):
        sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
        import card_lib
        info = {"performance": {"sessions": [{"programs_sha256": "table", "metrics": {"decode": {"loom": 1.0}}}]},
                "remeasure": {"programs_sha256": "here", "compared_programs": "measured", "carried_programs": ["measured", "table"], "result": "identical"}}
        with mock.patch.object(card_lib, "card_metrics", lambda i: {"decode": ({}, 0)}), mock.patch.object(card_lib, "single", lambda i: i):
            self.assertEqual(card_lib.stale_metrics(info, "here"), {})
            info["remeasure"]["carried_programs"] = ["measured"]
            self.assertEqual(card_lib.stale_metrics(info, "here"), {"decode": "table"})




class LogitsCapability(unittest.TestCase):
    def test_a_package_declares_the_capability_by_the_key_in_any_service_config(self):
        sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
        import card_lib
        self.assertTrue(card_lib.declares_logits({"variants": {"gguf": {"services": {"chat": {"config": {"llm.logits": False}}}}}}))
        self.assertFalse(card_lib.declares_logits({"variants": {"gguf": {"services": {"chat": {"config": {"llm.card": True}}}}}}))
        self.assertTrue(card_lib.gate_applies("logits", "chat", {}, True))
        self.assertFalse(card_lib.gate_applies("logits", "chat", {}, False))
        self.assertNotIn("gufo", card_lib.GATE_NAMES)      # a card claims the logits readback, not Gufo's conformance suite
        self.assertIn("gufo", card_lib.RETIRED_GATES)      # old evidence carrying it is dropped


if __name__ == "__main__":
    unittest.main()
