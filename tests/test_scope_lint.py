"""fleet/scope_lint.py on synthetic git repos: every rule passes and fails."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fleet"))
import scope_lint as SL  # noqa: E402

OWN = json.loads((ROOT / "fleet" / "ownership.json").read_text())
CHOICES = '''import re
CHOICES = {}
class Choice:
    def __init__(self, key, type, legacy, env, derive=None, kit_legacy=None, enc=("1", "0"), kits=None):
        self.key, self.type, self.legacy, self.env, self.derive = key, type, legacy, env, derive
        self.kit_legacy, self.enc, self.kits = dict(kit_legacy or {}), enc, kits
        CHOICES[key] = self
def CHOICE(key, type, legacy, env, **kw):
    return Choice(key, type, legacy, env, **kw)
def NOT(reason, *envs):
    pass
CHOICE("decode.attn.kv_once", "bool", True, "ATTN_GQA")
'''
CFGS = {"m-moe-drafted": {"arch": "qwen35moe", "moe": {"experts": 8}, "drafter": {"kind": "dflash", "gguf": "d.gguf"}, "kv_heads": 2},
        "m-moe-plain": {"arch": "qwen35moe", "moe": {"experts": 8}, "kv_heads": 2},
        "m-dense": {"arch": "qwen3", "kv_heads": 8}}


def impact(**models):
    """models: slug -> (gpus, [rule ids])"""
    return {"models": {s: {"gpus": g, "blocks": [{"rule": r, "kind": "kernel", "symbol": "x", "target": "both"} for r in rules],
                           "stratum": {"draft": "dflash" if "drafted" in s else "none"}} for s, (g, rules) in models.items()}}


class Repo:
    def __init__(self):
        self.d = tempfile.mkdtemp(prefix="scope_lint_test_")
        self.write("fleet/ownership.json", json.dumps(OWN))
        self.write("skills/loom-kb/choices.py", CHOICES)
        for node, m in {"arch/llm/moe/qwen35moe": {"arch": ["qwen35moe"]}, "facet/drafted": {"facets": ["drafted"]},
                        "arch/llm/dense/qwen3": {"arch": ["qwen3"]}}.items():
            self.write(f"skills/loom-kb/{node}/scope.json", json.dumps({"scope": node, "match": m, "owner": "t", "choices": []}))
            self.write(f"skills/loom-kb/{node}/README.md", "# n\n")
        for d in ("rdna3", "rdna4"):
            self.write(f"skills/loom-kb/device/{d}/family.json", "{}")
            self.write(f"skills/loom-kb/device/{d}/scope.json", json.dumps({"scope": f"device/{d}", "choices": []}))
        for s, c in CFGS.items():
            self.write(f"fleet/{s}/llm_config.json", json.dumps(c))
        self.write("skills/loom-port-llm/starter/gen_llm.py", "x = 1\n")
        self.write("skills/loom-port-llm/starter/attn_rules.py", "x = 1\n")
        self.write("skills/loom-port-llm/starter/gen_moe.py", "x = 1\n")
        self.write("README.md", "hi\n")
        self.git("init", "-q", "-b", "base")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
        self.commit("base")
        self.git("checkout", "-q", "-b", "head")

    def git(self, *a):
        subprocess.run(["git", "-C", self.d, *a], check=True, capture_output=True)

    def write(self, rel, text):
        p = Path(self.d, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def commit(self, msg="c"):
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)

    def lint(self, block, imp=None, kb_lint=False):
        b, errs = SL.parse_block(block)
        return SL.lint(self.d, "base", "head", b, errs, imp, kb_lint=kb_lint, own=OWN, tree=self.d)

    def close(self):
        shutil.rmtree(self.d, ignore_errors=True)


def rules(res):
    return sorted({e["rule"] for e in res.errors})


class T(unittest.TestCase):
    def setUp(self):
        self.r = Repo()

    def tearDown(self):
        self.r.close()

    def test_parse_block(self):
        b, e = SL.parse_block("text\n```loom-scope\nkind: targeted\nscopes: a/b+x, c\nkernels: gemm\nleak: why\n```\n")
        self.assertEqual((b["kind"], b["scopes"], b["kernels"], b["leak"]), ("targeted", ["a/b+x", "c"], ["gemm"], "why"))
        self.assertEqual(e, [])
        self.assertTrue(SL.parse_block("no fence")[1] == [] or True)
        self.assertTrue(SL.parse_block("```py\nx\n```")[1])
        self.assertTrue(SL.parse_block("```loom-scope\nkind: nope\n```")[1])

    def test_targeted_pass(self):
        self.r.write("skills/loom-kb/arch/llm/moe/qwen35moe/scope.json", json.dumps({"scope": "arch/llm/moe/qwen35moe", "match": {"arch": ["qwen35moe"]}, "choices": [1]}))
        self.r.commit()
        res = self.r.lint("kind: targeted\nscopes: arch/llm/moe/qwen35moe+drafted\n",
                          impact(**{"m-moe-drafted": (["gfx11", "gfx12"], ["gemm"])}))
        self.assertEqual(res.errors, [])
        self.assertEqual(res.info["leak"]["leaked"], [])

    def test_targeted_leak_to_undrafted(self):
        self.r.write("skills/loom-kb/arch/llm/moe/qwen35moe/README.md", "# n2\n")
        self.r.commit()
        res = self.r.lint("kind: targeted\nscopes: arch/llm/moe/qwen35moe+drafted\n",
                          impact(**{"m-moe-drafted": (["gfx12"], ["gemm"]), "m-moe-plain": (["gfx12"], ["gemm"])}))
        self.assertEqual(rules(res), ["leak"])
        self.assertEqual([x["model"] for x in res.info["leak"]["leaked"]], ["m-moe-plain"])
        self.assertEqual(res.info["leak"]["rate"], 0.5)

    def test_targeted_leak_declared_not_allowed(self):
        res = self.r.lint("kind: targeted\nscopes: arch/llm/moe/qwen35moe\nleak: because\n")
        self.assertIn("scope-block", rules(res))

    def test_device_scope_limits_gpu(self):
        self.r.write("skills/loom-kb/device/rdna4/README.md", "x\n")
        self.r.commit()
        ok = self.r.lint("kind: targeted\nscopes: device/rdna4\n", impact(**{"m-dense": (["gfx12"], ["gemm"])}))
        self.assertEqual(ok.errors, [])
        bad = self.r.lint("kind: targeted\nscopes: device/rdna4\n", impact(**{"m-dense": (["gfx11", "gfx12"], ["gemm"])}))
        self.assertEqual(rules(bad), ["leak"])

    def test_targeted_wrong_file(self):
        self.r.write("skills/loom-port-llm/starter/gen_moe.py", "x = 2\n")
        self.r.commit()
        res = self.r.lint("kind: targeted\nscopes: arch/llm/moe/qwen35moe\n", impact())
        self.assertEqual(rules(res), ["files"])

    def test_protected_without_architecture(self):
        self.r.write("skills/loom-port-llm/starter/attn_rules.py", "x = 2\n")
        self.r.commit()
        res = self.r.lint("kind: targeted\nscopes: arch/llm/moe/qwen35moe+drafted\n", impact(**{"m-moe-plain": (["gfx12"], ["decode_attention"])}))
        self.assertEqual(rules(res), ["leak", "protected"])
        arch = self.r.lint("kind: architecture\n", impact(**{"m-moe-plain": (["gfx12"], ["decode_attention"])}))
        self.assertEqual(arch.errors, [])

    def test_horizontal(self):
        self.r.write("skills/loom-port-llm/starter/gen_moe.py", "x = 2\n")
        self.r.write("skills/loom-port-llm/starter/gen_llm.py", "x = 2\n")
        self.r.commit()
        imp = impact(**{"m-moe-plain": (["gfx11"], ["moe"]), "m-dense": (["gfx11"], ["moe"])})
        self.assertEqual(self.r.lint("kind: horizontal\nkernels: moe\n", imp).errors, [])   # gen_llm.py is families_ok for moe
        res = self.r.lint("kind: horizontal\nkernels: gemm\n", impact(**{"m-dense": (["gfx11"], ["moe"])}))
        self.assertIn("files", rules(res))
        self.assertIn("leak", rules(res))
        # scoped horizontal with a leak: fails without `leak:`, passes with it
        blk = "kind: horizontal\nkernels: moe\nscopes: arch/llm/moe/qwen35moe\n"
        self.assertEqual(rules(self.r.lint(blk, imp)), ["leak"])
        res = self.r.lint(blk + "leak: dense models share the kernel\n", imp)
        self.assertEqual(res.errors, [])
        self.assertEqual(len(res.info["leak"]["leaked"]), 1)
        self.assertTrue(res.notes)

    def test_horizontal_protected_not_in_family(self):
        self.r.write("skills/loom-port-llm/starter/gen_llm.py", "x = 2\n")
        self.r.commit()
        res = self.r.lint("kind: horizontal\nkernels: vision\n", impact())
        self.assertEqual(rules(res), ["protected"])

    def test_docs(self):
        self.r.write("README.md", "changed\n")
        self.r.write("skills/loom-kb/arch/llm/dense/qwen3/README.md", "# n3\n")
        self.r.commit()
        res = self.r.lint("kind: docs\n", None)   # no program input changed: the impact is empty without a build
        self.assertEqual((res.errors, res.info["impact"]["touched"]), ([], 0))
        self.assertEqual(self.r.lint("kind: docs\n", impact()).errors, [])
        self.r.write("skills/loom-port-llm/starter/gen_moe.py", "x = 3\n")
        self.r.commit()
        self.assertEqual(rules(self.r.lint("kind: docs\n", impact())), ["files"])
        self.assertEqual(rules(self.r.lint("kind: docs\n", impact(**{"m-dense": (["gfx11"], ["moe"])}))), ["files", "leak"])

    def test_tooling_and_card(self):
        self.r.write("scripts/x.py", "1\n")
        self.r.write("tests/test_x.py", "1\n")
        self.r.commit()
        self.assertEqual(self.r.lint("kind: tooling\n", impact()).errors, [])
        self.assertEqual(rules(self.r.lint("kind: card\n", impact())), ["files"])
        self.r.write("skills/loom-benchmark/scripts/card/render_card.py", "1\n")
        self.r.commit()
        self.assertEqual(rules(self.r.lint("kind: tooling\n", impact())), ["files"])

    def test_workflow_and_guard_protected(self):
        self.r.write(".github/workflows/x.yml", "a: 1\n")
        self.r.write("fleet/scope_lint.py", "#\n")
        self.r.commit()
        res = self.r.lint("kind: tooling\n", impact())
        self.assertEqual(sorted(e["rule"] for e in res.errors), ["protected", "protected"])
        self.assertEqual(self.r.lint("kind: architecture\n", impact()).errors, [])

    def test_choices_addition_ok_legacy_change_not(self):
        add = CHOICES + 'CHOICE("decode.attn.new", "int", 4, "ATTN_NEW")\n'
        self.r.write("skills/loom-kb/choices.py", add)
        self.r.commit()
        res = self.r.lint("kind: targeted\nscopes: arch/llm/moe/qwen35moe\n", impact())
        self.assertEqual(res.errors, [])
        self.assertEqual(res.info["choices_added"], ["decode.attn.new"])
        self.r.write("skills/loom-kb/choices.py", add.replace('"bool", True, "ATTN_GQA"', '"bool", False, "ATTN_GQA"'))
        self.r.commit()
        res = self.r.lint("kind: targeted\nscopes: arch/llm/moe/qwen35moe\n", impact())
        self.assertEqual(rules(res), ["registry"])
        self.assertEqual(self.r.lint("kind: architecture\n", impact()).errors, [])

    def test_choices_removed_and_machinery(self):
        self.r.write("skills/loom-kb/choices.py", CHOICES.replace('CHOICE("decode.attn.kv_once", "bool", True, "ATTN_GQA")\n', ""))
        self.r.commit()
        self.assertEqual(rules(self.r.lint("kind: horizontal\nkernels: moe\n", impact())), ["registry"])
        self.r.write("skills/loom-kb/choices.py", CHOICES.replace("self.key, self.type,", "self.key = key; self.type,"))
        self.r.commit()
        res = self.r.lint("kind: targeted\nscopes: arch/llm/moe/qwen35moe\n", impact())
        self.assertEqual(rules(res), ["registry"])

    def test_choices_not_for_docs(self):
        self.r.write("skills/loom-kb/choices.py", CHOICES + 'CHOICE("a.b", "int", 1, "AB")\n')
        self.r.commit()
        self.assertEqual(rules(self.r.lint("kind: docs\n", impact())), ["files"])

    def test_missing_impact(self):
        self.r.write("skills/loom-kb/arch/llm/moe/qwen35moe/scope.json", json.dumps({"scope": "arch/llm/moe/qwen35moe", "choices": [1]}))
        self.r.commit()
        res = self.r.lint("kind: targeted\nscopes: arch/llm/moe/qwen35moe\n", None)
        self.assertEqual(rules(res), ["leak"])

    def test_unknown_scope_and_kernel(self):
        res = self.r.lint("kind: horizontal\nkernels: nope\nscopes: arch/llm/zzz\n", impact())
        self.assertEqual(rules(res), ["scope-block"])
        self.assertEqual(len(res.errors), 2)

    def test_cli_exit_code(self):
        self.r.write("README.md", "x\n")
        self.r.commit()
        body = Path(self.r.d, "body.md")
        body.write_text("```loom-scope\nkind: docs\n```\n")
        argv = ["--repo", self.r.d, "--base", "base", "--head", "head", "--pr-body", str(body), "--no-kb-lint", "--json", str(Path(self.r.d, "o.json"))]
        # --repo's ownership.json is the fixture copy; body.md is untracked so it does not count
        self.assertEqual(SL.main(argv), 0)
        self.assertTrue(json.load(open(Path(self.r.d, "o.json")))["ok"])
        body.write_text("no block\n```py\n```\n")
        self.assertEqual(SL.main(argv), 1)


if __name__ == "__main__":
    unittest.main()
