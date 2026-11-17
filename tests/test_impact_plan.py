"""Impact selection must prove non-input identity, never trust a declared PR scope."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("impact_plan", ROOT / "fleet" / "impact_plan.py")
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)
MANIFEST = {"models": {
    "alpha": {"files": {"weights": {"dump": "alpha/gguf/weights.json.gz"}}},
    "beta": {"files": {"weights": {"dump": "beta/gguf/weights.json.gz"}}}
}}


class SelectionTests(unittest.TestCase):
    def test_non_inputs_do_not_build_even_with_architecture_scope(self):
        paths = ["README.md", "docs/index.html", "tests/test_card.py", ".github/workflows/impact.yml",
                 "skills/loom-benchmark/scripts/card/render_card.py", "skills/loom-fleet/reference/campaign.md"]
        self.assertEqual(PLAN.select(paths, MANIFEST)[0], [])

    def test_exact_model_inputs(self):
        for path in ("fleet/alpha/llm_config.json", "fleet/alpha/tokenizer_min.json", "fleet/alpha/gguf/weights.json.gz"):
            with self.subTest(path=path):
                self.assertEqual(PLAN.select([path], MANIFEST)[0], ["alpha"])

    def test_shared_header_dump_rebuilds_all_owners(self):
        manifest = json.loads(json.dumps(MANIFEST))
        manifest["models"]["beta"]["files"]["weights"]["dump"] = "alpha/gguf/weights.json.gz"
        self.assertEqual(PLAN.select(["fleet/alpha/gguf/weights.json.gz"], manifest)[0], ["alpha", "beta"])

    def test_shared_and_unknown_inputs_fall_back_to_full(self):
        for path in ("skills/loom-port-llm/starter/gen_llm.py", "skills/loom-port-llm/reference/gemma4-example/starter.patch",
                     "skills/loom-kb/model/alpha/scope.json", "skills/loom-kb/registry.py", "skills/loom-kb/device/rdna3/family.json",
                     "fleet/manifest.json", "fleet/gguf_stub.py", "fleet/build_fleet.py", "future-module.py",
                     "fleet/alpha/unknown-input.json", "fleet/new-model/llm_config.json"):
            with self.subTest(path=path):
                self.assertEqual(PLAN.select(["fleet/alpha/llm_config.json", path], MANIFEST)[0], ["alpha", "beta"])


class GitPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "test")
        self.git("config", "user.email", "test@example.com")
        self.write("fleet/manifest.json", json.dumps(MANIFEST))
        self.write("fleet/alpha/llm_config.json", "{}")
        self.write("fleet/beta/llm_config.json", "{}")
        self.write("skills/loom-port-llm/starter/gen_llm.py", "x=1")
        self.write("skills/loom-kb/choices.py", "x=1")
        self.base = self.commit()

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True).strip()

    def write(self, path, text):
        p = self.repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "test")
        return self.git("rev-parse", "HEAD")

    def test_docs_only_has_empty_build_matrix(self):
        self.write("README.md", "hello")
        self.commit()
        plan = PLAN.make_plan(self.repo, self.base)
        self.assertEqual(plan["builds"], 0)
        self.assertEqual(plan["unchanged_inputs"], ["alpha", "beta"])

    def test_shards_cover_both_revisions_exactly_once(self):
        self.write("skills/loom-port-llm/starter/gen_llm.py", "x=2")
        head = self.commit()
        plan = PLAN.make_plan(self.repo, self.base, shards=8)
        self.assertEqual(plan["builds"], 4)
        for side, ref in (("base", self.base), ("head", head)):
            jobs = [j for j in plan["matrix"]["include"] if j["side"] == side]
            self.assertEqual(sorted(s for j in jobs for s in j["slugs"].split(",")), ["alpha", "beta"])
            self.assertEqual({j["ref"] for j in jobs}, {ref})
        self.assertNotEqual(plan["matrix"]["include"][0]["key"], plan["matrix"]["include"][2]["key"])

    def test_cache_keys_include_head_fleet_inputs_for_base_build(self):
        self.write("fleet/alpha/llm_config.json", '{"value":1}')
        self.commit()
        first = PLAN.make_plan(self.repo, self.base)["matrix"]["include"][0]["key"]
        self.write("fleet/alpha/llm_config.json", '{"value":2}')
        self.commit()
        second = PLAN.make_plan(self.repo, self.base)["matrix"]["include"][0]["key"]
        self.assertNotEqual(first, second)

    def test_revision_is_part_of_cache_key_for_summary_identity(self):
        self.write("fleet/alpha/llm_config.json", '{"value":1}')
        self.commit()
        first = PLAN.make_plan(self.repo, self.base)["matrix"]["include"][1]["key"]
        self.write("README.md", "docs")
        self.commit()
        second = PLAN.make_plan(self.repo, self.base)["matrix"]["include"][1]["key"]
        self.assertNotEqual(first, second)

    def test_renamed_input_keeps_old_path_in_selection(self):
        self.git("mv", "fleet/alpha/llm_config.json", "README.md")
        self.commit()
        plan = PLAN.make_plan(self.repo, self.base)
        self.assertEqual(plan["selected"], ["alpha"])
        self.assertIn("fleet/alpha/llm_config.json", plan["changed"])


class AggregationTests(unittest.TestCase):
    def plan(self):
        return {"selected": ["alpha"], "unchanged_inputs": ["beta"], "reasons": ["model input"], "base": "a", "head": "b",
                "matrix": {"include": [{"side": "base", "ref": "a", "shard": 0, "slugs": "alpha"},
                                       {"side": "head", "ref": "b", "shard": 0, "slugs": "alpha"}]}}

    def summary(self, tmp, side, **extra):
        d = Path(tmp) / side
        d.mkdir(parents=True, exist_ok=True)
        data = {"ref": "a" if side == "base" else "b", "rdna4": "1", "models": {"alpha": 1}, "failures": {}, "skipped": {}}
        data.update(extra)
        (d / "summary-0.json").write_text(json.dumps(data))

    def test_missing_shard_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "incomplete impact shards"):
                PLAN.aggregate(self.plan(), tmp, Path(tmp) / "out")

    def test_failed_and_wrong_revision_shards_fail_closed(self):
        for extra in ({"failures": {"alpha": "build broke"}}, {"ref": "wrong"}, {"models": {}}, {"rdna4": "0"}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                self.summary(tmp, "base", **extra)
                self.summary(tmp, "head")
                with self.assertRaises(SystemExit):
                    PLAN.aggregate(self.plan(), tmp, Path(tmp) / "out")

    def test_empty_plan_is_explicit_about_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {"selected": [], "unchanged_inputs": ["alpha", "beta"], "reasons": ["no program build input changed"], "base": "a", "head": "b"}
            report = PLAN.aggregate(plan, tmp, Path(tmp) / "out")
            self.assertEqual(report["models"], {})
            text = (Path(tmp) / "out" / "impact.md").read_text()
            self.assertIn("0/2 models built", text)
            self.assertIn("unchanged build inputs", text)

    def test_complete_shards_find_real_program_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            for side, value in (("base", 1), ("head", 2)):
                model = Path(tmp) / side / "alpha"
                model.mkdir(parents=True)
                (model / "model.json").write_text("{}")
                (model / "llm_config.json").write_text('{"arch":"qwen3","heads":8,"kv_heads":8}')
                (model / "main.loom").write_text(f"kernel.def @g_test() {{\n  value {value}\n}}\n")
                self.summary(tmp, side)
            report = PLAN.aggregate(self.plan(), tmp, Path(tmp) / "out")
            self.assertIn("alpha", report["models"])
            self.assertEqual(report["summary"]["models"], 1)
            self.assertEqual(report["selection"]["unchanged_inputs"], ["beta"])

    def test_new_source_model_is_reported_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.summary(tmp, "base", skipped={"alpha": "new source format"})
            self.summary(tmp, "head")
            model = Path(tmp) / "head" / "alpha"
            model.mkdir()
            (model / "model.json").write_text("{}")
            (model / "main.loom").write_text("kernel.def @g_test() {}\n")
            report = PLAN.aggregate(self.plan(), tmp, Path(tmp) / "out")
            self.assertTrue(report["models"]["alpha"]["new_model"])
            self.summary(tmp, "head", skipped={"alpha": "unsupported at head"})
            with self.assertRaisesRegex(SystemExit, "failed impact shard"):
                PLAN.aggregate(self.plan(), tmp, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
