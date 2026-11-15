"""Model nodes (model/<slug>), shape match keys, device-scoped execution choices (exec.*), the fit rows built from them, the scope lint of
a per-device serving change, kb.py route --model, and ab_jobs.py --base-ref / --head-ref."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "skills" / "loom-kb"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(KB))
sys.path.insert(0, str(ROOT / "fleet"))
sys.path.insert(0, str(ROOT / "skills" / "loom-benchmark" / "scripts" / "card"))
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
import choices  # noqa: E402
import resolve  # noqa: E402
import card_lib  # noqa: E402
import scope_lint as SL  # noqa: E402
from test_scope_lint import Repo, impact, rules  # noqa: E402

SLUG = "unsloth-Qwen3-VL-8B-Instruct-loom"
CFG = json.loads((ROOT / "fleet" / SLUG / "llm_config.json").read_text())
R9 = "radeon-r9700"
STRIX = "strix-halo"


def kb_with(nodes):
    """A scratch copy of the knowledge base (registry code, device dirs, arch/facet nodes) plus NODES {scope path: node dict}."""
    d = tempfile.mkdtemp(prefix="kb_scopegaps_")
    shutil.copytree(KB, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("model", "__pycache__"))
    for scope, node in nodes.items():
        p = Path(d, scope)
        p.mkdir(parents=True, exist_ok=True)
        p.joinpath("scope.json").write_text(json.dumps({"scope": scope, "owner": "t", **node}))
        p.joinpath("README.md").write_text("# n\n")
    return d


def rule(devices, **s):
    return {"when": {"devices": devices}, "set": s, "evidence": "test"}


class ModelNodes(unittest.TestCase):
    def tearDown(self):
        for d in getattr(self, "dirs", []):
            shutil.rmtree(d, ignore_errors=True)

    def kb(self, nodes):
        d = kb_with(nodes)
        self.dirs = getattr(self, "dirs", []) + [d]
        return d

    def test_model_node_matches_by_slug_and_by_model_id(self):
        kb = self.kb({f"model/{SLUG}": {"match": {"slug": [SLUG]}, "choices": [rule([R9], **{"exec.prefill_capacity": 4096})]}})
        self.assertEqual(resolve.device_execution(CFG, R9, kb, slug=SLUG)["exec.prefill_capacity"]["value"], 4096)
        self.assertEqual(resolve.device_execution(CFG, R9, kb, slug=SLUG)["exec.prefill_capacity"]["from"], f"model/{SLUG}")
        self.assertEqual(resolve.device_execution(CFG, R9, kb), {})                 # no slug, no match
        self.assertEqual(resolve.device_execution(CFG, R9, kb, slug="other"), {})
        self.assertEqual(resolve.device_execution(dict(CFG, slug=SLUG), R9, kb)["exec.prefill_capacity"]["value"], 4096)   # cfg["slug"]
        kb2 = self.kb({"model/by-id": {"match": {"model_id": [CFG["model_id"]]}, "choices": [rule([R9], **{"exec.rows": 2})]}})
        self.assertEqual(resolve.device_execution(CFG, R9, kb2)["exec.rows"]["value"], 2)

    def test_execution_is_per_device_and_not_part_of_a_build(self):
        kb = self.kb({f"model/{SLUG}": {"match": {"slug": [SLUG]}, "choices": [rule([R9], **{"exec.prefill_capacity": 4096})]}})
        self.assertEqual(resolve.device_execution(CFG, STRIX, kb, slug=SLUG), {})
        self.assertEqual(resolve.device_execution(CFG, "rdna4", kb, slug=SLUG), {})   # the rule names the device, not the family
        r = resolve.resolve(CFG, ("rdna3", "rdna4"), kb, slug=SLUG)   # a dual-target build: no exec key, nothing exported
        self.assertEqual((r["choices"], r["env"]), ({}, {}))
        self.assertEqual(resolve.effective(CFG, ("rdna3", "rdna4"), kb, slug=SLUG)["all"]["exec.prefill_capacity"]["value"], None)

    def test_precedence_model_above_family_and_facet_below_the_package_override(self):
        fam = {"arch": ["qwen3vl"]}
        kb = self.kb({"arch/llm/dense/qwen3vl": {"match": fam, "choices": [rule([R9], **{"exec.prefill_capacity": 2048})]},
                      "facet/vision": {"match": {"facets": ["vision"]}, "choices": [rule([R9], **{"exec.prefill_capacity": 1024})]},
                      f"model/{SLUG}": {"match": {"slug": [SLUG]}, "choices": [rule([R9], **{"exec.prefill_capacity": 4096})]}})
        self.assertEqual(resolve.device_execution(CFG, R9, kb, slug=SLUG)["exec.prefill_capacity"]["value"], 4096)
        self.assertEqual(resolve.device_execution(CFG, R9, kb, slug="x")["exec.prefill_capacity"]["value"], 1024)   # the facet beats the family
        over = dict(CFG, choices={"exec.prefill_capacity": 8192})
        self.assertEqual(resolve.device_execution(over, R9, kb, slug=SLUG)["exec.prefill_capacity"], {"value": 8192, "from": "model", "evidence": "llm_config.json choices"})
        # two model nodes setting one key for one model: an error
        kb3 = self.kb({"model/a": {"match": {"slug": [SLUG]}, "choices": [rule([R9], **{"exec.rows": 2})]},
                       "model/b": {"match": {"slug": [SLUG]}, "choices": [rule([R9], **{"exec.rows": 3})]}})
        with self.assertRaises(resolve.ResolveError):
            resolve.device_execution(CFG, R9, kb3, slug=SLUG)

    def test_shape_match_keys_select_a_subset_of_a_family(self):
        kb = self.kb({"arch/llm/dense/qwen3vl": {"match": {"arch": ["qwen3vl"]}, "choices": [
            {"when": {"devices": [R9], "hidden": 4096}, "set": {"exec.prefill_capacity": 4096}, "evidence": "t"},
            {"when": {"devices": [R9], "hidden": [None, 2560], "layers": [30, 40]}, "set": {"exec.prefill_capacity": 2048}, "evidence": "t"},
            {"when": {"devices": [R9], "params_b": [7, 9]}, "set": {"exec.rows": 2}, "evidence": "t"}]}})
        got = {k: v["value"] for k, v in resolve.device_execution(CFG, R9, kb).items()}
        self.assertEqual(got, {"exec.prefill_capacity": 4096, "exec.rows": 2})   # hidden 4096, 8B parameters
        small = dict(CFG, hidden=2048, layers=36, ffn=6144)
        self.assertEqual({k: v["value"] for k, v in resolve.device_execution(small, R9, kb).items()}, {"exec.prefill_capacity": 2048})
        self.assertTrue(7 < resolve.params_b(CFG) < 9, resolve.params_b(CFG))

    def test_a_model_node_must_name_an_identity(self):
        kb = self.kb({"model/loose": {"match": {"arch": ["qwen3vl"]}, "choices": []}})
        self.assertTrue(any("must match an identity" in e for e in resolve.lint(kb)))
        kb = self.kb({"model/x": {"match": {"slug": ["x"]}, "choices": [rule([R9], **{"exec.nothing": 1})]}})
        self.assertTrue(any("unknown choice" in e for e in resolve.lint(kb)))


class Migration(unittest.TestCase):
    def test_fit_rows_carry_the_settings_the_table_used_to(self):
        before = json.loads((ROOT / "tests" / "data" / "r9700_exec_before_scope.json").read_text())
        rows = card_lib.fit_rows(str(ROOT / ".github" / "devlab" / "r9700_fit.json"), R9)
        got = {s: {k: r[k] for k in ("ctx_cap", "r9700_execution") if k in r} for s, r in rows.items() if "ctx_cap" in r or "r9700_execution" in r}
        self.assertEqual(got, before)
        table = json.loads((ROOT / ".github" / "devlab" / "r9700_fit.json").read_text())
        self.assertFalse([r["slug"] for r in table if "ctx_cap" in r or "r9700_execution" in r], "the table holds data only")

    def test_every_model_node_is_a_fleet_model_with_evidence(self):
        manifest = json.loads((ROOT / "fleet" / "manifest.json").read_text())["models"]
        for d in sorted((KB / "model").iterdir()):
            node = json.loads((d / "scope.json").read_text())
            self.assertIn(d.name, manifest)
            self.assertEqual(node["match"], {"slug": [d.name]})
            self.assertTrue((d / "README.md").is_file())
        self.assertEqual(resolve.lint(str(KB)), [])


class Route(unittest.TestCase):
    def test_route_model_prints_the_model_node_and_the_device_settings(self):
        out = subprocess.run([sys.executable, str(KB / "kb.py"), "route", "--model", "unsloth-Qwen3-0.6B-loom", "--device", R9, "--relative"],
                             capture_output=True, text=True, check=True).stdout
        lines = out.splitlines()
        self.assertIn("model/unsloth-Qwen3-0.6B-loom/README.md", lines)
        self.assertLess(lines.index("facet/vision/README.md") if "facet/vision/README.md" in lines else 0, lines.index("model/unsloth-Qwen3-0.6B-loom/README.md"))
        self.assertIn('"exec.prefill_capacity": 4096', out)
        none = subprocess.run([sys.executable, str(KB / "kb.py"), "route", "--model", SLUG, "--device", R9, "--relative"], capture_output=True, text=True, check=True).stdout
        self.assertNotIn("model/", "\n".join(l for l in none.splitlines() if not l.startswith("#")))


class ExecLint(unittest.TestCase):
    """A per-device serving change is targeted with scope = the model node + the device."""

    def setUp(self):
        self.r = Repo({"skills/loom-kb/choices.py": (KB / "choices.py").read_text(),   # the real registry (exec.* choices)
                       "skills/loom-kb/device/rdna4/radeon-r9700/README.md": "# r9700\n", "skills/loom-kb/device/rdna3/strix-halo/README.md": "# strix\n"})

    def tearDown(self):
        self.r.close()

    def node(self, slug, devices, **s):
        self.r.write(f"skills/loom-kb/model/{slug}/scope.json", json.dumps({"scope": f"model/{slug}", "match": {"slug": [slug]}, "owner": "t",
                     "choices": [rule(devices, **s)]}))
        self.r.write(f"skills/loom-kb/model/{slug}/README.md", "# m\n")
        self.r.commit()

    def test_targeted_model_plus_device_passes(self):
        self.node("m-dense", [R9], **{"exec.prefill_capacity": 4096})
        res = self.r.lint("kind: targeted\nscopes: model/m-dense, device/rdna4\n", impact())
        self.assertEqual(res.errors, [])
        self.assertEqual([(x["model"], x["device"], x["after"]) for x in res.info["exec_changed"]], [("m-dense", R9, {"exec.prefill_capacity": 4096})])

    def test_wrong_device_scope_fails(self):
        self.node("m-dense", [STRIX], **{"exec.prefill_capacity": 4096})
        res = self.r.lint("kind: targeted\nscopes: model/m-dense, device/rdna4\n", impact())
        self.assertEqual(rules(res), ["exec"])

    def test_a_family_rule_leaks_to_models_outside_the_declared_scope(self):
        self.r.write("skills/loom-kb/arch/llm/dense/qwen3/scope.json", json.dumps({"scope": "arch/llm/dense/qwen3", "match": {"arch": ["qwen3"]}, "owner": "t",
                     "choices": [rule([R9], **{"exec.prefill_capacity": 2048})]}))
        self.r.commit()
        ok = self.r.lint("kind: targeted\nscopes: arch/llm/dense/qwen3, device/rdna4\n", impact())
        self.assertEqual(ok.errors, [])
        bad = self.r.lint("kind: targeted\nscopes: arch/llm/dense/qwen3+drafted, device/rdna4\n", impact())   # m-dense is not drafted
        self.assertEqual(rules(bad), ["exec"])
        self.assertIn("m-dense", bad.errors[0]["message"])

    def test_tooling_and_docs_may_not_change_serving(self):
        self.node("m-dense", [R9], **{"exec.prefill_capacity": 4096})
        res = self.r.lint("kind: tooling\n", impact())
        self.assertIn("exec", rules(res))


class AbRefs(unittest.TestCase):
    def test_old_vs_new_from_two_refs(self):
        import ab_jobs
        # two exports of the checkout: base without the model node, head with it
        with tempfile.TemporaryDirectory(prefix="ab_refs_") as d:
            def export(name, extra):
                root = Path(d, name)
                for rel in ("skills/loom-kb", f"fleet/{SLUG}", ".github/devlab"):
                    shutil.copytree(ROOT / rel, root / rel, ignore=shutil.ignore_patterns("__pycache__", "model", "*.gz"))
                (root / "skills/loom-kb/model").mkdir(exist_ok=True)
                if extra:
                    n = root / "skills/loom-kb/model" / SLUG
                    n.mkdir()
                    n.joinpath("scope.json").write_text(json.dumps({"scope": f"model/{SLUG}", "match": {"slug": [SLUG]}, "owner": "t",
                                                                    "choices": [rule([R9], **{"exec.prefill_capacity": 4096})]}))
                    n.joinpath("README.md").write_text("# m\n")
                return str(root)
            base, head = export("base", False), export("head", True)
            old = ab_jobs.settings_from_root(R9, SLUG, base)
            new = ab_jobs.settings_from_root(R9, SLUG, head)
            self.assertNotIn("prefill_capacity", old[2]["execution"])
            self.assertEqual(new[2]["execution"]["prefill_capacity"], 4096)


if __name__ == "__main__":
    unittest.main()
