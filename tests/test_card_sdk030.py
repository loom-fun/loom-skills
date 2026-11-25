"""SDK 0.3.0 card rules: drafted-token telemetry (#1401), the card-format file check and the card lint (#1398-#1404, #1413).

Telemetry fixtures (skills/loom-benchmark/scripts/card/test/sdk03/telemetry):
- metrics_030_abi1.txt: loom-serve 0.3.0 /metrics after a perf session of unsloth-Qwen3.5-4B-loom (service ABI 1, MTP depth 3),
  DevLab run 45527446282 (metrics_staging_001.txt without histogram buckets). There is no draft_proposed_tokens: the ABI 1
  publisher passes no proposal counts, so hrx-system v0.3.0 service/text/statistics.c emits topic schema 2 without it.
- metrics_020_abi1.txt: the same counters in the hrx-loom 0.2.0 form (schema 1, draft_proposed_tokens = verifications x depth,
  as hrx-system v0.2.0 serve/service/text/telemetry.c counts it: (1393 - 736) x 3 = 1971), constructed from that source.
- perf_020_sage.jsonl / perf_030_sage.jsonl: perf.py rows of InternScience-Sage-A1.5-loom on Strix Halo, hrx-loom 0.2.0
  ("fun": per-row acceptance recorded) and 0.3.0 ("staging": none), DevLab run 45267567358.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills/loom-benchmark/scripts"
CARD = SCRIPTS / "card"
TEL = CARD / "test/sdk03/telemetry"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(CARD))
import draft_telemetry as bench_common  # noqa: E402  (bench_common re-exports these; this module needs no third-party package)
import card_files  # noqa: E402
import card_lib  # noqa: E402
import lint_card  # noqa: E402
import make_info  # noqa: E402


class DraftTelemetry(unittest.TestCase):
    def test_030_abi1_reports_accepted_without_proposals(self):
        v = bench_common.draft_counters((TEL / "metrics_030_abi1.txt").read_text(encoding="utf-8"))
        self.assertEqual(v, {"draft_accepted_inputs": 736.0, "decode_tokens": 1393.0})
        self.assertTrue(bench_common.proposals_unreported(v))
        self.assertIsNone(bench_common.acceptance_between({}, v))   # never from the configured depth
        self.assertEqual(bench_common.per_pass_between({}, v), round(736 / (1393 - 736), 3))

    def test_020_reports_the_share_of_proposals(self):
        v = bench_common.draft_counters((TEL / "metrics_020_abi1.txt").read_text(encoding="utf-8"))
        self.assertFalse(bench_common.proposals_unreported(v))
        self.assertEqual(bench_common.acceptance_between({}, v), round(736 / 1971, 3))
        # SDK 0.2 counted every verification as depth proposals: its share x depth is the accepted tokens per pass
        self.assertAlmostEqual(bench_common.acceptance_between({}, v) * 3, bench_common.per_pass_between({}, v), places=2)

    def test_perf_session_dumps(self):
        self.assertEqual(make_info.draft_metrics_acceptance([str(TEL / "metrics_030_abi1.txt")]),
                         {"loom": None, "unreported": "proposals", "accepted_per_pass": round(736 / 657, 3)})
        self.assertEqual(make_info.draft_metrics_acceptance([str(TEL / "metrics_020_abi1.txt")]), {"loom": round(736 / 1971, 3)})

    def test_no_drafting_counters(self):
        self.assertEqual(bench_common.draft_counters('x{field="decode_tokens"} 5\n'), {})

    def test_spec_gate_lines(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t, "gates_spec.log")
            p.write_text("SPEC_CHECK PASS: 16/16 completions identical, 0 near-tie, acceptance not reported (736 drafted tokens "
                         "accepted in 657 verification passes, 1.120 per pass; the server counts no proposals)\n", encoding="utf-8")
            self.assertEqual(make_info.acceptance_metric([str(p)]),
                             {"loom": None, "unreported": "proposals", "accepted_per_pass": 1.12})
            self.assertIsNone(make_info.acceptance([str(p)]))
            p.write_text("SPEC_CHECK PASS: 13/13 completions identical, 0 near-tie, acceptance 68% of 2709 proposals\n", encoding="utf-8")
            self.assertEqual(make_info.acceptance_metric([str(p)]), {"loom": 0.68})

    def test_recorded_perf_rows(self):
        def rows(name):
            return [json.loads(x) for x in (TEL / name).read_text(encoding="utf-8").splitlines() if x.strip()]
        old = make_info.acceptance_samples([r for r in rows("perf_020_sage.jsonl") if r.get("kind") == "loom"])
        new = make_info.acceptance_samples([r for r in rows("perf_030_sage.jsonl") if r.get("kind") == "loom"])
        self.assertTrue(old["decode"] and old["decode_2k"])
        self.assertEqual(new, {})   # 0.3.0 rows record no share of proposals (the harness now adds accepted_per_pass)


class CardFiles(unittest.TestCase):
    def package(self, t, extra=()):
        for rel in ("README.md", "model.json", "model.json.lock", "info.json", "llm_config.json", "tokenizer.json",
                    "weights.loom", "head_weights.loom", "program/llm.loom", "gen/flash2_cfg.json", "library/motif/x.loom",
                    ".gitattributes", *extra):
            Path(t, rel).parent.mkdir(parents=True, exist_ok=True)
            Path(t, rel).write_text("{}" if rel.endswith(".json") else "x\n", encoding="utf-8")

    def test_card_format_passes(self):
        with tempfile.TemporaryDirectory() as t:
            self.package(t)
            self.assertEqual(card_files.check_package(t), [])

    def test_foreign_files_and_staging_text(self):
        with tempfile.TemporaryDirectory() as t:
            self.package(t, ("sdk-qualification.json", "quick-start-qualification.json", "dev/bench/a.json", "dev/b.json"))
            Path(t, "REPORT.md").write_text("The staging package passed correctness checks.\n", encoding="utf-8")
            Path(t, "info.json").write_text(json.dumps({"repo": "loom-fun-staging/x-loom", "promotion_hold": {}}), encoding="utf-8")
            got = "\n".join(card_files.check_package(t))
            for want in ("sdk-qualification.json: not a file", "quick-start-qualification.json: not a file",
                         "dev/ (2 files)", "REPORT.md:1: staging wording", "info.json:1: a loom-fun-staging reference",
                         "promotion_hold"):
                self.assertIn(want, got)
            self.assertNotIn("info.json:1: staging wording", got)   # the org name is one finding, not two

    def test_program_identifiers_are_code(self):
        with tempfile.TemporaryDirectory() as t:
            self.package(t)
            Path(t, "program/llm.loom").write_text("%staging = ...\n", encoding="utf-8")
            self.assertEqual(card_files.check_package(t), [])


class Requalification(unittest.TestCase):
    def info(self):
        return {"sources": {"a": {"hrx_loom": {"hrx_loom": "0.3.0"}}, "b": {"hrx_loom": {"hrx_loom": "0.2.0"}}, "c": {"hrx_loom": None}},
                "card": {}, "devices": {"strix-halo": {"gates": {"golden": {"source": "a", "status": "pass"}},
                                                         "performance": {"sessions": [{"source": "a", "metrics": {"decode": {"loom": 9.0}}}]}},
                                        "gfx1201-16gb": {"gates": {"golden": {"source": "b", "status": "pass"}},
                                                         "performance": {"sessions": [{"source": "b", "metrics": {"decode": {"loom": 5.0}}}]}}}}

    def test_carried(self):
        i = self.info()
        pin = {"hrx_loom": "0.3.0"}
        self.assertIsNone(card_lib.carried_sdk(i, "a", pin))
        self.assertEqual(card_lib.carried_sdk(i, "b", pin), "0.2.0")
        self.assertEqual(card_lib.carried_sdk(i, "c", pin), card_lib.UNRECORDED_SDK)
        self.assertTrue(card_lib.requalified(i, "strix-halo", pin))
        self.assertFalse(card_lib.requalified(i, "gfx1201-16gb", pin))
        dropped = card_lib.drop_carried(i, pin)
        self.assertEqual(dropped["devices"]["gfx1201-16gb"], {"state": "not_measured"})
        self.assertEqual(dropped["devices"]["strix-halo"], i["devices"]["strix-halo"])

    def test_version_ranges(self):
        self.assertTrue(card_lib.sdk_uses_file_parameters({"hrx_loom": "0.3.0"}))
        self.assertTrue(card_lib.sdk_uses_file_parameters({"hrx_loom": "0.4.1"}))
        self.assertTrue(card_lib.sdk_uses_file_parameters({"hrx_loom": "1.0.0"}))
        self.assertFalse(card_lib.sdk_uses_file_parameters({"hrx_loom": "0.2.0"}))

    def test_render_has_no_single_release_check(self):
        text = (CARD / "render_card.py").read_text(encoding="utf-8")
        self.assertNotIn('== "0.3.0"', text)


class Lint(unittest.TestCase):
    """lint_card.py over the sdk03 golden card (flint's package, re-measured on 0.3.0), offline."""

    def card(self, t):
        d = Path(t, "card")
        shutil.copytree(CARD / "test/flint/pkg", d)
        cfg = json.loads((d / "llm_config.json").read_text(encoding="utf-8"))
        cfg["sdk"] = {k: v for k, v in json.loads((CARD / "test/sdk03/evidence/hrx_loom.json").read_text(encoding="utf-8")).items()
                      if k not in ("artifact_id", "branch")}
        (d / "llm_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        for f in ("info.json", "README.md", "REPORT.md"):
            shutil.copy(CARD / "test/sdk03/expected" / f, d / f)
        return d

    def test_sdk03_golden(self):
        with tempfile.TemporaryDirectory() as t:
            d = self.card(t)
            res = lint_card.lint(str(d), runs=lint_card.Runs(offline=True))
            self.assertEqual(res["files"], [])
            self.assertEqual(res["render"], [])
            self.assertEqual(res["versions"], [])
            self.assertEqual(res["acceptance"], [])
            self.assertEqual(res["admission"], [])
            self.assertEqual(res["delta"], ["card: strix-halo prefill_2k -4.0 % unexplained (beyond -3 %)"])

    def test_hand_edit_failed_run_and_staging_text(self):
        with tempfile.TemporaryDirectory() as t:
            d = self.card(t)
            readme = (d / "README.md").read_text(encoding="utf-8")
            (d / "README.md").write_text(readme.replace("## Features", "The staging package passed correctness checks.\n\n## Features"), encoding="utf-8")
            info = json.loads((d / "info.json").read_text(encoding="utf-8"))
            sid = next(iter(info["sources"]))
            info["sources"][sid]["conclusion"] = "failure"
            (d / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
            res = lint_card.lint(str(d), runs=lint_card.Runs(offline=True))
            self.assertTrue(any("staging wording" in p for p in res["files"]), res["files"])
            self.assertTrue(any("concluded failure" in p for p in res["runs"]), res["runs"])
            self.assertTrue(res["render"], "a hand edit or a failed run must not pass the render check")

    def test_dropped_acceptance_and_missing_delta(self):
        with tempfile.TemporaryDirectory() as t:
            d = self.card(t)
            info = json.loads((d / "info.json").read_text(encoding="utf-8"))
            info.pop("sdk_delta")
            for s in info["devices"]["strix-halo"]["performance"]["sessions"]:
                s["metrics"].pop("spec_acceptance", None)
            info["devices"]["strix-halo"].pop("admission")
            (d / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
            res = lint_card.lint(str(d), runs=lint_card.Runs(offline=True))
            self.assertTrue(res["delta"] and "no delta line" in res["delta"][0], res["delta"])
            self.assertTrue(res["acceptance"], res)
            self.assertTrue(res["admission"], res)


class AcceptanceDelta(unittest.TestCase):
    """The acceptance delta compares one quantity from one workload (perf-session acceptance) on both sides."""
    def info(self):
        return json.loads((CARD / "test/sdk03/expected/info.json").read_text(encoding="utf-8"))

    def row(self, info, depth=3):
        pin = {"hrx_loom": "0.3.0"}
        rows = card_lib.delta_rows(info, pin, info["sdk_delta"]["comparisons"][0], depth)["strix-halo"]
        return next(r for r in rows if r["metric"] == "spec_acceptance")

    def test_perf_session_old_side(self):
        info = self.info()
        self.assertEqual(info["sdk_delta"]["comparisons"][0]["devices"]["strix-halo"]["metrics"]["spec_acceptance_session"], 0.325)
        r = self.row(info)
        self.assertEqual(r["per_pass"], {"old": 0.975, "new": 1.249})   # not the spec gate's 68 % x 3 = 2.04
        self.assertFalse(r["different_workloads"])
        self.assertEqual(r["change"], round(1.249 / 0.975 - 1, 4))

    def test_spec_gate_only_old_card_has_no_delta(self):
        info = self.info()
        del info["sdk_delta"]["comparisons"][0]["devices"]["strix-halo"]["metrics"]["spec_acceptance_session"]
        r = self.row(info)
        self.assertTrue(r["different_workloads"])
        self.assertIsNone(r["change"])
        self.assertIsNone(r["per_pass"])
        self.assertEqual((r["old"], r["accepted_per_pass"]), (0.68, 1.249))
        import render_card
        text = "\n".join(render_card.delta_lines(info, {"hrx_loom": "0.3.0"}, True, 3))
        self.assertIn("68 % of proposals in its speculative-decoding gate, this card 1.25 per verification pass in the performance "
                      "session (different workloads, no delta)", text)
        self.assertNotIn("-38.8", text)


if __name__ == "__main__":
    unittest.main()


class NoGenerator(unittest.TestCase):
    """A hand-written package (llm_config "generator" says unknown, no commit recoverable) states it has no generator."""
    GEN = {"skills_commit": "unknown", "dirty": None, "kit": {"example": None, "modified": True}, "hrx_loom": "unknown"}
    BASIS = "hand-written package, no generator"

    def versions(self):
        return {"card_generator": "a" * 40, "package": {"skills_commit": {"commit": None, "basis": self.BASIS}}}

    def test_generator_commit(self):
        self.assertFalse(card_lib.stamped(self.GEN))
        self.assertEqual(card_lib.generator_commit(self.GEN, self.versions()), (None, "none"))
        self.assertEqual(card_lib.generator_commit(self.GEN, {"card_generator": "x"}), (None, None))
        rec = {"package": {"skills_commit": {"commit": "b" * 40, "recovered": True, "basis": "production card"}}}
        self.assertEqual(card_lib.generator_commit(self.GEN, rec), ("b" * 40, "recovered"))
        self.assertEqual(card_lib.generator_commit({"skills_commit": "c" * 40}, rec), ("c" * 40, "stamp"))

    def test_rendered_without_unknown(self):
        import render_card
        text = render_card.skills_text(self.GEN, self.versions())
        self.assertEqual(text, f"none ({self.BASIS})")
        self.assertNotIn("unknown", render_card.versions_sdk(self.GEN, self.versions(), {"hrx_loom": "0.3.0", "hrx_system_sha": "e" * 40}))

    def test_schema(self):
        schema = card_lib.load_json(card_lib.SCHEMA_PATH)
        sub = schema["properties"]["versions"]
        self.assertEqual(card_lib._check(self.versions(), sub, schema, "versions", []), [])
        bad = {"card_generator": "x", "package": {"skills_commit": {"commit": "nothex", "basis": "y"}}}
        self.assertNotEqual(card_lib._check(bad, sub, schema, "versions", []), [])


class KitKept(unittest.TestCase):
    """Migration-only programs whose kit build lost its A/B: the card says why in one generated line (sdk_delta.kit)."""
    REASON = "Strix Halo decode -4.2 % (confirmed in the same-job ABBA)"
    URL = "https://github.com/loom-fun/loom-skills/actions/runs/1/job/2"

    def test_lint_and_render(self):
        with tempfile.TemporaryDirectory() as t:
            d = Lint().card(t)
            info = json.loads((d / "info.json").read_text(encoding="utf-8"))
            info["sdk_delta"]["kit"] = {"reason": self.REASON, "url": self.URL}
            self.assertFalse([e for e in card_lib.validate(info, "") if "sdk_delta" in e])
            import render_card
            text = "\n".join(render_card.delta_lines(info, {"hrx_loom": "0.3.0"}, True, 3))
            self.assertIn(f"is not shipped: {self.REASON} ([evidence]({self.URL})).", text)
            bad = dict(info, sdk_delta=dict(info["sdk_delta"], kit={"why": "x"}))
            self.assertTrue([e for e in card_lib.validate(bad, "") if "sdk_delta" in e])

    def test_make_info_keeps_and_checks(self):
        ns = SimpleNamespace(previous_card=[], delta_explanations=None, kit_kept=None)
        old = {"kit": {"reason": self.REASON, "url": None}, "comparisons": []}
        self.assertEqual(make_info.sdk_delta(ns, old)["kit"], {"reason": self.REASON, "url": None})
        ns.kit_kept = {"url": self.URL}
        with self.assertRaises(SystemExit):
            make_info.sdk_delta(ns, None)
