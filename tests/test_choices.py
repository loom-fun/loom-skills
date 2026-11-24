"""The choice registry (skills/loom-kb/choices.py), the resolver (resolve.py) and their build integration (kb_choices.py)."""
import ast
import glob
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "skills", "loom-kb")
STARTER = os.path.join(ROOT, "skills", "loom-port-llm", "starter")
FIX = os.path.join(ROOT, "tests", "data", "choices_kb")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KB)
sys.path.insert(0, STARTER)
import choices  # noqa: E402
import envscan  # noqa: E402
import resolve  # noqa: E402
import kb_choices  # noqa: E402

SCAN_ROOTS = [os.path.join(ROOT, "skills", "loom-port-llm", "starter"), os.path.join(ROOT, "skills", "loom-optimize", "reference"),
              os.path.join(ROOT, "skills", "loom-port-llm", "reference")]
PATCHES = sorted(glob.glob(os.path.join(ROOT, "skills", "loom-port-llm", "reference", "*", "starter.patch")))


def cfg(**kw):
    c = {"name": "t", "arch": "qwen3", "hidden": 1024, "layers": 4, "heads": 16, "kv_heads": 8, "head_dim": 128, "ffn": 3072,
         "tensors": {"blk.0.attn_q.weight": {"type": "q4_k", "shape": [1, 1]}}}
    c.update(kw)
    return c


def moe35(**kw):
    c = dict(arch="qwen35moe", kv_heads=2, head_dim=256, moe={"experts": 8, "used": 2, "ffn": 512}, recurrent_layers=[0, 1, 2],
             drafter={"kind": "dflash", "gguf": "d.gguf"})
    c.update(kw)
    return cfg(**c)


_TMP = []


def kb_copy(name):
    """A fixture knowledge base with the registry code beside it (kb_choices.py loads resolve.py from the KB root)."""
    import shutil
    d = tempfile.mkdtemp()
    _TMP.append(d)
    shutil.copytree(os.path.join(FIX, name), d, dirs_exist_ok=True)
    for f in ("resolve.py", "choices.py"):
        shutil.copy(os.path.join(KB, f), d)
    return d


def tearDownModule():
    import shutil
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


class Inventory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = envscan.scan(SCAN_ROOTS, PATCHES)

    def test_every_environment_variable_is_registered(self):
        missing = sorted(set(self.scan) - choices.registered_env())
        self.assertEqual(missing, [], "register these generator environment variables in skills/loom-kb/choices.py as a CHOICE "
                                      "(model-affecting) or in NOT_CHOICES (paths, tests, tuning, transport) with a reason: "
                                      + ", ".join(f"{m} ({os.path.relpath(self.scan[m][0][0], ROOT)}:{self.scan[m][0][1]})" for m in missing))

    def test_no_stale_registration(self):
        stale = sorted(choices.registered_env() - set(self.scan))
        self.assertEqual(stale, [], "registered but no generator reads them")

    def test_a_variable_is_in_one_place_only(self):
        self.assertEqual(set(choices.BY_ENV) & set(choices.NOT_CHOICES), set())
        self.assertTrue(all(v.strip() for v in choices.NOT_CHOICES.values()))

    def test_inventory_counts(self):
        self.assertEqual(len(choices.BY_ENV) + len(choices.NOT_CHOICES), len(self.scan))
        self.assertGreater(len(choices.CHOICES), 60)

    def test_choice_declarations(self):
        for c in choices.CHOICES.values():
            self.assertTrue(c.doc, c.key)
            self.assertIn(c.type, choices.TYPES)
            self.assertTrue(c.rows or c.kernels, f"{c.key}: name the rows or kernels it can move")
            if c.devices:
                self.assertTrue(set(c.devices) <= {"rdna3", "rdna4"}, c.key)
            if c.kits:
                self.assertTrue(set(c.kits) <= {"gemma4", "lfm2moe", "flint", "glimmer", "gpt-oss", "qwen4exp", "flint-moe", "deepseek2"}, c.key)

    def test_legacy_equals_the_generators_default(self):
        """The registry's legacy value is what the generators do when the variable is unset."""
        bad = []
        for c in choices.CHOICES.values():
            if c.execution:
                continue   # no generator reads it
            lits = [d for (p, ln, kind, d) in self.scan[c.env] if kind in ("get", "set") and not p.endswith(".patch")]
            strs = {d for d in lits if isinstance(d, str) and d not in ("<expr>", "<helper>")}
            nones = [d for d in lits if d is None]
            allowed = set(c.generator_defaults) if c.generator_defaults is not None else strs | ({None} if nones else set())
            if not allowed:
                self.assertTrue(c.dynamic or c.kits or c.derive, f"{c.key}: no literal default found; declare dynamic / kits")
                continue
            ok = c.legacy_env in allowed or (c.legacy in (None, False) and bool(allowed & {None, "", "0"}))
            if not ok:
                bad.append((c.key, c.env, c.legacy_env, sorted(x for x in allowed if x is not None)))
        self.assertEqual(bad, [])

    def test_kit_defaults_equal_the_patches(self):
        """A reference kit that defaults a knob differently says so in kit_legacy."""
        for patch in PATCHES:
            kit = os.path.basename(os.path.dirname(patch)).replace("-example", "")
            for name, ln, kind, d in envscan.scan_patch(patch):
                c = choices.BY_ENV.get(name)
                if c is None or kind not in ("get", "set") or not d:
                    continue
                try:
                    v = ast.literal_eval(d)
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(v, str):
                    continue
                accepted = {c.legacy_env, *(c.to_env(x) for x in c.kit_legacy.values())}
                self.assertIn(v, accepted, f"{name}: {os.path.basename(os.path.dirname(patch))} defaults to {v!r}; "
                                           f"add kit_legacy={{{kit!r}: ...}} to {c.key}")

    def test_defaults_do_not_change_legacy_serialization(self):
        for c in choices.CHOICES.values():
            if c.legacy is not None:
                self.assertEqual(c.check(c.legacy), c.legacy)
                if c.enc[1] is not None or c.legacy:
                    self.assertIsInstance(c.to_env(c.legacy), str, c.key)


class Resolution(unittest.TestCase):
    def test_the_real_knowledge_base_lints(self):
        self.assertEqual(resolve.lint(KB), [])

    def test_scope_rules_touch_declared_models(self):
        """D7 (the real knowledge base): the K/V-once decode body for head_dim 256 is off for drafted qwen35moe only, on both device
        families (the key is one kernel shared by both targets), exporting ATTN_GQA_D256=0 and nothing else."""
        for dev in (("rdna3",), ("rdna4",), ("rdna3", "rdna4"), ("strix-halo", "radeon-r9700")):
            for kind in ("dflash", "mtp"):
                r = resolve.resolve(moe35(drafter={"kind": kind, "gguf": "d.gguf"}), dev, KB, draft=kind)
                self.assertEqual(r["env"], {"ATTN_GQA_D256": "0"}, (dev, kind))
                self.assertEqual(r["choices"]["decode.attn.kv_once.d256"]["from"], "arch/llm/moe/qwen35moe")
        for other, draft in ((moe35(drafter=None), None), (moe35(arch="qwen35", moe=None), "dflash"), (cfg(), "dflash"),
                             (cfg(arch="qwen3moe", moe={"experts": 8}), "dflash"), (cfg(vision={"mmproj": "x"}), None)):
            self.assertEqual(resolve.resolve(other, ("rdna3", "rdna4"), KB, draft=draft)["env"], {}, other["arch"])

    def test_no_scope_sets_anything_by_default_means_no_exports(self):
        """Every program is as before: nothing is exported for any model shape, with the real knowledge base as of this change."""
        if any(s["choices"] for _, s in resolve.load_scopes(KB)):
            self.skipTest("the knowledge base has scope rules now; test_scope_rules_touch_declared_models covers them")
        for c in (cfg(), moe35(), cfg(vision={"mmproj": "x"}), cfg(mtp={"layers": [3]})):
            for dev in (("rdna3",), ("rdna3", "rdna4")):
                self.assertEqual(resolve.resolve(c, dev, KB)["env"], {})

    def test_facets(self):
        self.assertEqual(resolve.derive_facets(cfg(), None), [])
        f = resolve.derive_facets(moe35(vision={"mmproj": "m"}), "dflash")
        self.assertEqual(f, ["draft-dflash", "drafted", "hybrid-gdn", "kv-le2", "moe", "vision"])
        self.assertIn("sliding-window", resolve.derive_facets(cfg(sliding_window=512), None))
        self.assertEqual(resolve.auto_draft_kind(cfg(mtp={"layers": [1]})), "mtp")
        self.assertEqual(resolve.auto_draft_kind(cfg(assistant={"gguf": "a"})), "assist")
        self.assertIsNone(resolve.auto_draft_kind(cfg()))
        self.assertEqual(resolve.auto_draft_kind(cfg(drafter={"kind": "ngram"})), "ngram")
        # a built package's llm_config.json records the draft it was built with
        self.assertEqual(resolve.draft_kind(cfg(draft=None, mtp={"layers": [1]})), None)
        self.assertEqual(resolve.draft_kind(cfg(draft={"kind": "mtp", "depth": 3})), "mtp")

    def test_the_real_nodes_match_the_way_kb_py_reads_them(self):
        """The kb engineer's scope.json vocabulary: has/lacks/ranges with null, node `facets` implied by a family, facet nodes."""
        r = resolve.resolve(moe35(vision={"mmproj": "m"}), ("rdna3", "rdna4"), KB)
        self.assertIn("arch/llm/moe/qwen35moe", r["scopes"])
        self.assertIn("facet/drafted", r["scopes"])   # has: ["draft"] sees the draft the build uses
        self.assertIn("facet/vision", r["scopes"])
        self.assertTrue({"moe", "hybrid-gdn", "drafted", "vision", "kv-le2"} <= set(r["facets"]), r["facets"])
        r = resolve.resolve(cfg(arch="gemma3", sliding_window=512, kv_heads=1), ("rdna3",), KB)
        self.assertTrue({"sliding-window", "kv-le2"} <= set(r["facets"]))
        self.assertNotIn("facet/drafted", resolve.resolve(cfg(), ("rdna3",), KB, draft=None)["scopes"])
        self.assertEqual(resolve.resolve(cfg(), ())["devices"], ["rdna3", "rdna4"])   # kb.py route without --device

    def test_precedence_later_wins(self):
        kb = os.path.join(FIX, "precedence")
        k = "decode.attn.kv_once.from_keys"
        # arch root (1000) < topology (2000) < family leaf (3000) < facet (5000) < model
        r = resolve.resolve(moe35(), ("rdna3", "rdna4"), kb, draft=None)
        self.assertEqual((r["choices"][k]["value"], r["choices"][k]["from"]), (3000, "arch/llm/moe/qwen35moe"))
        r = resolve.resolve(moe35(), ("rdna3", "rdna4"), kb, draft="dflash")
        self.assertEqual((r["choices"][k]["value"], r["choices"][k]["from"]), (5000, "facet/drafted"))
        r = resolve.resolve(moe35(choices={k: 7}), ("rdna3", "rdna4"), kb, draft="dflash")
        self.assertEqual((r["choices"][k]["value"], r["choices"][k]["from"]), (7, "model"))
        # a dense qwen3 sees only the arch root
        r = resolve.resolve(cfg(), ("rdna3", "rdna4"), kb, draft=None)
        self.assertEqual((r["choices"][k]["value"], r["choices"][k]["from"]), (1000, "arch/llm"))
        self.assertEqual(r["scopes"][:2], ["device/rdna4", "quant/q4_k"])

    def test_device_and_quant_nodes(self):
        kb = os.path.join(FIX, "precedence")
        r = resolve.resolve(cfg(), ("rdna3", "rdna4"), kb)
        self.assertEqual(r["choices"]["prefill.attn.flash2_trim"]["from"], "device/rdna4")
        self.assertEqual(r["env"]["FLASH2_TRIM"], "1")
        self.assertEqual(r["env"]["MV_WG_LONG"], "512")
        r = resolve.resolve(cfg(), ("rdna3",), kb)
        self.assertNotIn("prefill.attn.flash2_trim", r["choices"])   # device/rdna4 does not match an rdna3-only build
        r = resolve.resolve(cfg(), ("radeon-r9700",), kb)   # a device slug maps to its family
        self.assertEqual(r["devices"], ["rdna4"])
        self.assertIn("prefill.attn.flash2_trim", r["choices"])
        r = resolve.resolve(cfg(tensors={"a": {"type": "q6_k"}}), ("rdna3",), kb)
        self.assertNotIn("decode.mv.wg_long", r["choices"])

    def test_the_d7_shape_of_rule_touches_only_drafted_qwen35moe(self):
        """The test-only rule: qwen35moe + drafted -> decode.attn.kv_once.d256 = false (ATTN_GQA_D256=0)."""
        kb = os.path.join(FIX, "precedence")
        hit = resolve.resolve_full(moe35(), ("rdna3", "rdna4"), kb, draft="dflash")
        self.assertEqual(hit["env"].get("ATTN_GQA_D256"), "0")
        self.assertFalse(hit["values"]["decode.attn.kv_once.d256"])
        for other, draft in ((moe35(), None), (cfg(), "dflash"), (cfg(arch="qwen3moe", moe={"experts": 8}), "dflash"),
                             (moe35(arch="qwen35"), "dflash")):
            r = resolve.resolve(other, ("rdna3", "rdna4"), kb, draft=draft)
            self.assertNotIn("ATTN_GQA_D256", r["env"], other["arch"])

    def test_same_level_conflict_is_an_error(self):
        kb = os.path.join(FIX, "conflict")
        with self.assertRaises(resolve.ResolveError) as e:
            resolve.resolve(moe35(), ("rdna3", "rdna4"), kb, draft="dflash")
        self.assertIn("decode.mv.wg1", str(e.exception))
        r = resolve.resolve(cfg(), ("rdna3", "rdna4"), kb)   # a model only one node matches resolves
        self.assertEqual(r["env"], {})
        self.assertTrue(resolve.lint(kb, {"drafted-moe": moe35(draft={"kind": "mtp"})}))
        # the model's own choice settles it
        r = resolve.resolve(moe35(choices={"decode.mv.wg1": 99}), ("rdna3", "rdna4"), kb, draft="dflash")
        self.assertEqual(r["env"]["MV_WG1"], "99")

    def test_a_shared_choice_cannot_differ_per_device(self):
        with self.assertRaises(resolve.ResolveError) as e:
            resolve.resolve(cfg(), ("rdna3", "rdna4"), os.path.join(FIX, "pertarget"))
        self.assertIn("cannot differ per device", str(e.exception))
        # a build for the one device it names is fine
        r = resolve.resolve(cfg(), ("rdna4",), os.path.join(FIX, "pertarget"))
        self.assertEqual(r["env"]["MV_WG1"], "1024")

    def test_per_target_rule_resolves_per_family(self):
        """qwen3moe (undrafted): the flash2 peel is off for the gfx11 kernel only; its gfx12 twin keeps it (PR #771 evidence)."""
        q = cfg(arch="qwen3moe", moe={"experts": 8})
        r = resolve.resolve_full(q, ("rdna3", "rdna4"), KB, draft=None)
        self.assertEqual(r["env"], {"FLASH2_PEEL": "0", "FLASH2_PEEL__RDNA4": "1"})
        self.assertEqual(r["choices"]["prefill.attn.flash2_peel"]["value"], {"rdna3": False, "rdna4": True})
        self.assertIs(r["values"]["prefill.attn.flash2_peel"], False)
        self.assertEqual(resolve.resolve(q, ("rdna3",), KB, draft=None)["env"], {"FLASH2_PEEL": "0"})
        self.assertEqual(resolve.resolve(q, ("rdna4",), KB, draft=None)["env"], {})
        self.assertEqual(resolve.resolve(q, ("rdna3", "rdna4"), KB, draft="dflash")["env"], {})
        self.assertEqual(resolve.resolve(cfg(), ("rdna3", "rdna4"), KB, draft=None)["env"], {})

    def test_per_target_rdna4_rule_and_default(self):
        """A rule for rdna4 only on a per_target key whose default is on: the base stays, the twin gets the value."""
        kb = kb_copy("precedence")
        os.makedirs(os.path.join(kb, "arch", "llm", "moe", "qwen3moe"), exist_ok=True)
        json.dump({"scope": "arch/llm/moe/qwen3moe", "match": {"arch": ["qwen3moe"]}, "owner": "t", "choices": [
            {"when": {"devices": ["rdna4"]}, "set": {"prefill.attn.flash2_peel": False}, "evidence": "t"},
            {"when": {"devices": ["rdna3"]}, "set": {"prefill.attn.flash2_peel": True}, "evidence": "t"}]},
            open(os.path.join(kb, "arch", "llm", "moe", "qwen3moe", "scope.json"), "w"))
        r = resolve.resolve_full(cfg(arch="qwen3moe", moe={"experts": 8}), ("rdna3", "rdna4"), kb, draft=None)
        self.assertEqual({k: v for k, v in r["env"].items() if "FLASH2_PEEL" in k}, {"FLASH2_PEEL__RDNA4": "0"})
        self.assertEqual(r["choices"]["prefill.attn.flash2_peel"]["value"], {"rdna3": True, "rdna4": False})

    def test_per_target_dirty_twin_variable(self):
        os.environ.pop("LOOM_KB_ROOT", None)
        q = cfg(arch="qwen3moe", moe={"experts": 8})
        env = {}
        self.assertEqual(kb_choices.apply(q, None, environ=env)["dirty"], [])
        self.assertEqual(env["FLASH2_PEEL__RDNA4"], "1")
        self.assertEqual(kb_choices.apply(q, None, environ={"FLASH2_PEEL__RDNA4": "0"})["dirty"], ["prefill.attn.flash2_peel"])

    def test_twin_generator_reads_the_twin_value(self):
        """gen_flash2.py: the gfx11 kernel takes FLASH2_PEEL, the K_rdna4 twin FLASH2_PEEL__RDNA4; unset = both as before."""
        import subprocess
        import shutil
        tmp = tempfile.mkdtemp()
        _TMP.append(tmp)
        c = os.path.join(tmp, "c.json")
        with open(c, "w") as f:
            json.dump({"kv_heads": 4, "heads": 32, "head_dim": 128, "layers": 4}, f)

        def gen(**kw):
            out = os.path.join(tmp, "o.loom")
            env = {k: v for k, v in os.environ.items() if not k.startswith("FLASH2_")}
            env.update(LLM_CONFIG=c, FLASH2_OUT=out, **kw)
            subprocess.run([sys.executable, os.path.join(STARTER, "gen_flash2.py")], check=True, env=env, capture_output=True)
            with open(out) as f:
                t = f.read()
            i = t.rindex("kernel.def", 0, t.index("target(@rdna4)") + 1)
            return t[:i], t[i:]
        legacy = gen()
        g11_off, tw_on = gen(FLASH2_PEEL="0", FLASH2_PEEL__RDNA4="1")
        g11_off2, tw_off = gen(FLASH2_PEEL="0")
        self.assertEqual(g11_off, g11_off2)
        self.assertEqual(tw_on, legacy[1])
        self.assertNotEqual(tw_on, tw_off)
        self.assertNotEqual(g11_off, legacy[0])
        self.assertEqual(gen(FLASH2_PEEL="1")[0], legacy[0])
        shutil.rmtree(tmp, ignore_errors=True)

    def test_per_target_shared_kernel_d7_twin(self):
        """decode.attn.kv_once.d256 is per_target: the D7 rule (drafted qwen35moe: off) plus a rule for one model on rdna4 only
        (on) gives the gfx11 kernel ATTN_GQA_D256=0 and the gfx12 twin ATTN_GQA_D256__RDNA4=1; a build for one device or a
        model without the rule exports what it always did."""
        kb = kb_copy("precedence")
        os.makedirs(os.path.join(kb, "model", "m"), exist_ok=True)
        json.dump({"scope": "model/m", "match": {"slug": ["m"]}, "owner": "t", "choices": [
            {"when": {"devices": ["rdna4"]}, "set": {"decode.attn.kv_once.d256": True}, "evidence": "t"}]},
            open(os.path.join(kb, "model", "m", "scope.json"), "w"))
        r = resolve.resolve_full(moe35(), ("rdna3", "rdna4"), kb, draft="dflash", slug="m")
        self.assertEqual({k: v for k, v in r["env"].items() if "GQA_D256" in k}, {"ATTN_GQA_D256": "0", "ATTN_GQA_D256__RDNA4": "1"})
        self.assertEqual(r["choices"]["decode.attn.kv_once.d256"]["value"], {"rdna3": False, "rdna4": True})
        def d256(*devs, slug):
            return {k: v for k, v in resolve.resolve(moe35(), devs, kb, draft="dflash", slug=slug)["env"].items() if "GQA_D256" in k}
        self.assertEqual(d256("rdna3", slug="m"), {"ATTN_GQA_D256": "0"})
        self.assertEqual(d256("rdna4", slug="m"), {})
        self.assertEqual(d256("rdna3", "rdna4", slug="other"), {"ATTN_GQA_D256": "0"})

    def test_real_kb_d7_model_rule_is_gfx12_only(self):
        """The real knowledge base: the Qwen3.6-35B-A3B package keeps the K/V-once decode body on gfx12 (D7 costs it R9700 decode_2k)."""
        slug = "ggml-org-Qwen3.6-35B-A3B-loom"
        r = resolve.resolve_full(moe35(), ("rdna3", "rdna4"), KB, draft="dflash", slug=slug)
        self.assertEqual({k: v for k, v in r["env"].items() if "GQA_D256" in k}, {"ATTN_GQA_D256": "0", "ATTN_GQA_D256__RDNA4": "1"})
        self.assertEqual(resolve.resolve(moe35(), ("rdna3",), KB, draft="dflash", slug=slug)["env"], {"ATTN_GQA_D256": "0"})
        self.assertEqual(resolve.resolve(moe35(), ("rdna4",), KB, draft="dflash", slug=slug)["env"], {})
        self.assertEqual(resolve.resolve(moe35(), ("rdna3", "rdna4"), KB, draft="dflash", slug="unsloth-Qwen3.5-35B-A3B-loom")["env"], {"ATTN_GQA_D256": "0"})

    def test_attn_rules_twin_reads_its_own_variable(self):
        """attn_rules.gqa(..., twin=True) follows ATTN_GQA_D256__RDNA4; unset it follows ATTN_GQA_D256 (no twin text then)."""
        import subprocess
        code = ("import attn_rules as a; print(a.gqa(True, False, 256), a.gqa(True, False, 256, twin=True))")
        def run(**kw):
            env = {k: v for k, v in os.environ.items() if not k.startswith("ATTN_")}
            env.update(kw)
            return subprocess.run([sys.executable, "-c", code], cwd=STARTER, env=env, capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(run(), ["True", "True"])
        self.assertEqual(run(ATTN_GQA_D256="0"), ["False", "False"])
        self.assertEqual(run(ATTN_GQA_D256="0", ATTN_GQA_D256__RDNA4="1"), ["False", "True"])

    def test_bad_scope_files(self):
        self.assertTrue(any("unknown choice" in e for e in resolve.lint(os.path.join(FIX, "badkey"))))
        self.assertTrue(any("evidence" in e for e in resolve.lint(os.path.join(FIX, "noevidence"))))
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/facet/x")
            json.dump({"scope": "facet/y", "choices": []}, open(f"{d}/facet/x/scope.json", "w"))
            self.assertTrue(resolve.lint(d))
            json.dump({"scope": "facet/x", "match": {"nope": 1}, "choices": []}, open(f"{d}/facet/x/scope.json", "w"))
            self.assertTrue(any("unknown match key" in e for e in resolve.lint(d)))
            json.dump({"scope": "facet/x", "choices": [{"set": {"decode.mv.wg1": "big"}, "evidence": "e"}]},
                      open(f"{d}/facet/x/scope.json", "w"))
            self.assertTrue(any("not a int" in e for e in resolve.lint(d)))

    def test_model_choices(self):
        r = resolve.resolve(cfg(choices={"decode.attn.unroll": 8, "prefill.gemm.m256": False}), ("rdna3", "rdna4"), KB)
        self.assertEqual(r["env"], {"ATTN_UNROLL": "8", "GEMM_M256": "0"})
        self.assertEqual(r["choices"]["decode.attn.unroll"]["from"], "model")
        with self.assertRaises(resolve.ResolveError):
            resolve.resolve(cfg(choices={"no.such": 1}), ("rdna3",), KB)
        with self.assertRaises(resolve.ResolveError):
            resolve.resolve(cfg(choices={"decode.attn.unroll": "8"}), ("rdna3",), KB)
        # a value equal to the default exports nothing
        self.assertEqual(resolve.resolve(cfg(choices={"decode.attn.unroll": 16}), ("rdna3",), KB)["env"], {})

    def test_cfg_driven_defaults(self):
        """rdna4_f16_ffn and gemm_n_fastest are the model's own values of choices; mtp drafting defaults the gdn form."""
        e = resolve.effective(cfg(rdna4_f16_ffn=True), ("rdna3", "rdna4"), KB)["all"]
        self.assertTrue(e["prefill.gemm.f16o_swiglu"]["value"])
        self.assertTrue(e["prefill.attn.flash2_trim"]["value"])
        self.assertEqual(e["prefill.attn.flash2_b2"]["value"], "2")
        e = resolve.effective(cfg(rdna4_f16_ffn=True), ("rdna3",), KB)["all"]
        self.assertFalse(e["prefill.gemm.f16o_swiglu"]["value"])   # a gfx11-only build has no twins to fuse
        e = resolve.effective(cfg(rdna4_f16_ffn=True), ("rdna3", "rdna4"), KB)
        self.assertEqual(e["env"], {})   # exporting nothing: the generators derive these from the cfg flag themselves
        self.assertFalse(resolve.effective(cfg(), ("rdna3", "rdna4"), KB)["all"]["prefill.gemm.f16o_swiglu"]["value"])
        # a scope that switches it off for a flagged model exports GEMM_F16O=0, and FLASH2_TRIM follows
        r = resolve.resolve_full(cfg(rdna4_f16_ffn=True, choices={"prefill.gemm.f16o_swiglu": False}), ("rdna3", "rdna4"), KB)
        self.assertEqual(r["env"], {"GEMM_F16O": "0"})
        self.assertFalse(r["values"]["prefill.attn.flash2_trim"])
        # on for a model without the flag: GEMM_F16O=swiglu, and the trims follow
        r = resolve.resolve_full(cfg(choices={"prefill.gemm.f16o_swiglu": True}), ("rdna3", "rdna4"), KB)
        self.assertEqual(r["env"], {"GEMM_F16O": "swiglu"})
        self.assertTrue(r["values"]["prefill.attn.flash2_trim"])
        self.assertEqual(resolve.effective(cfg(gemm_n_fastest=False), ("rdna3",), KB)["all"]["prefill.gemm.n_fastest"]["value"], False)
        e = resolve.effective(cfg(mtp={"layers": [3]}), ("rdna3",), KB, draft="mtp")["all"]
        self.assertEqual((e["hybrid.gdn.row_lanes"]["value"], e["hybrid.gdn.cols"]["value"]), (8, 4))
        e = resolve.effective(cfg(mtp={"layers": [3]}), ("rdna3",), KB, draft=None)["all"]
        self.assertEqual((e["hybrid.gdn.row_lanes"]["value"], e["hybrid.gdn.cols"]["value"]), (16, 2))

    def test_kit_defaults(self):
        e = resolve.effective(cfg(), ("rdna3",), KB, kit="gemma4")["all"]
        self.assertEqual(e["prefill.attn.flash2_kt"]["value"], 16)
        e = resolve.effective(cfg(), ("rdna3",), KB)["all"]
        self.assertEqual(e["prefill.attn.flash2_kt"]["value"], 32)
        # a kit-only knob set for another build is recorded but not exported
        r = resolve.resolve(cfg(choices={"prefill.ple.gemm": False}), ("rdna3",), KB)
        self.assertEqual(r["env"], {})
        r = resolve.resolve(cfg(choices={"prefill.ple.gemm": False}), ("rdna3",), KB, kit="gemma4")
        self.assertEqual(r["env"], {"PLE_GEMM": "0"})

    def test_value_spelling(self):
        c = choices.CHOICES
        self.assertEqual(c["prefill.gemm.f16o_swiglu"].to_env(True), "swiglu")
        self.assertEqual(c["prefill.gemm.off"].to_env(True), "1")
        self.assertIsNone(c["prefill.gemm.off"].to_env(False))
        self.assertEqual(c["decode.attn.unroll"].to_env(8), "8")
        with self.assertRaises(ValueError):
            c["decode.attn.unroll"].check(True)   # a bool is not an int
        with self.assertRaises(ValueError):
            c["prefill.gemm.m256"].check(1)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "llm_config.json")
            json.dump(moe35(draft={"kind": "dflash", "depth": 3}), open(p, "w"))
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(resolve.main(["--config", p, "--kb-root", os.path.join(FIX, "precedence"), "--all"]), 0)
                self.assertEqual(resolve.main(["--config", p, "--kb-root", os.path.join(FIX, "conflict")]), 2)
                self.assertEqual(resolve.main(["--config", p, "--lint"]), 0)
            self.assertIn("ATTN_GQA_D256", out.getvalue())


class Build(unittest.TestCase):
    def test_exports_and_caller_wins_and_dirty(self):
        # exports go into the environment the generators read
        env = {}
        if True:
            os.environ["LOOM_KB_ROOT"] = kb_copy("precedence")
            try:
                block = kb_choices.apply(moe35(), "dflash", environ=env)
                self.assertEqual(env, {"ATTN_GQA_D256": "0", "ATTN_GQA_FROM": "5000", "FLASH2_TRIM": "1", "MV_WG_LONG": "512", "MOE_R": "4"})
                self.assertEqual(block["dirty"], [])
                self.assertEqual(block["scopes"][0], "device/rdna4")
                self.assertEqual(block["choices"]["decode.attn.kv_once.d256"]["from"], "arch/llm/moe/qwen35moe")
                # a registered variable the caller set wins; it is dirty when it differs from the resolution
                env = {"ATTN_GQA_D256": "1", "MV_WG_LONG": "512", "MOE_Q8": "0"}
                block = kb_choices.apply(moe35(), "dflash", environ=env)
                self.assertEqual(env["ATTN_GQA_D256"], "1")
                self.assertEqual(sorted(block["dirty"]), ["decode.attn.kv_once.d256", "moe.decode.act_q8"])
                # NOT_CHOICES (paths, tests, tuning) never make a build dirty
                env = {"HF_HOME": "/x", "TUNE_LONG": "1", "KTEST_GEMM_ALL": "1"}
                self.assertEqual(kb_choices.apply(cfg(), None, environ=env)["dirty"], [])
            finally:
                del os.environ["LOOM_KB_ROOT"]

    def test_caller_value_equal_to_the_default_is_not_dirty(self):
        env = {"LOOM_RDNA4": "1", "ATTN_UNROLL": "16"}
        self.assertEqual(kb_choices.apply(cfg(), None, environ=env)["dirty"], [])
        env = {"LOOM_RDNA4": "0"}   # the gfx11-only build: the dual-target twins off
        b = kb_choices.apply(cfg(), None, environ=env)
        self.assertEqual(b["devices"], ["rdna3"])
        self.assertEqual(b["dirty"], [])   # LOOM_RDNA4=0 selects the single-target build; the resolution is not changed by it
        self.assertEqual(b["env"], {})

    def test_resolved_block_without_any_rule_is_empty(self):
        block = kb_choices.apply(cfg(), None, environ={})
        self.assertEqual((block["choices"], block["env"], block["dirty"]), ({}, {}, []))
        self.assertEqual(block["devices"], ["rdna3", "rdna4"])
        json.dumps(block)

    def test_tests_use_the_recorded_environment(self):
        env = {}
        kb_choices.apply_for_tests(cfg(resolved={"env": {"ATTN_GQA_D256": "0", "NO_GEMM": None}}), environ=env)
        self.assertEqual(env, {"ATTN_GQA_D256": "0"})
        env = {"ATTN_GQA_D256": "1"}
        kb_choices.apply_for_tests(cfg(resolved={"env": {"ATTN_GQA_D256": "0"}}), environ=env)
        self.assertEqual(env["ATTN_GQA_D256"], "1")   # the caller's value wins here too
        env = {"LLM_DRAFT": json.dumps({"kind": "dflash"})}
        os.environ["LOOM_KB_ROOT"] = kb_copy("precedence")
        try:
            kb_choices.apply_for_tests(moe35(), environ=env)   # a pre-build cfg.json: resolved with LLM_DRAFT's draft
        finally:
            del os.environ["LOOM_KB_ROOT"]
        self.assertEqual(env["ATTN_GQA_D256"], "0")

    def test_catalog_builds_are_exempt(self):
        b = kb_choices.exempt_block("catalog")
        self.assertEqual((b["exempt"], b["choices"], b["env"], b["dirty"]), ("catalog", {}, {}, []))

    def test_kit_example_and_kb_root_are_found(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"example": "gemma4", "kb_root": KB}, open(os.path.join(d, "KIT_SOURCE"), "w"))
            self.assertEqual(kb_choices.kit_example(d), "gemma4")
            old = os.environ.pop("LOOM_KB_ROOT", None)
            try:
                self.assertEqual(kb_choices.find_kb_root(d), KB)
            finally:
                if old:
                    os.environ["LOOM_KB_ROOT"] = old
        self.assertIsNone(kb_choices.kit_example("/nonexistent"))

    def test_kit_source_records_the_knowledge_base(self):
        import kit_source
        with tempfile.TemporaryDirectory() as d:
            kit = os.path.join(d, "kit")
            os.makedirs(kit)
            kit_source.main(["write", kit, "--skills-dir", os.path.join(ROOT, "skills", "loom-port-llm")])
            ks = json.load(open(os.path.join(kit, "KIT_SOURCE")))
            self.assertEqual(os.path.realpath(ks["kb_root"]), os.path.realpath(KB))

    def test_publish_refuses_a_dirty_package(self):
        spec = importlib.util.spec_from_file_location("publish_package", os.path.join(
            ROOT, "skills", "loom-benchmark", "scripts", "card", "publish_package.py"))
        pp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pp)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "llm_config.json")
            json.dump({"generator": {"skills_commit": "x", "dirty_choices": ["decode.attn.kv_once"]}}, open(p, "w"))
            with self.assertRaises(SystemExit) as e:
                pp.refuse_dirty(p)
            self.assertIn("decode.attn.kv_once", str(e.exception))
            pp.refuse_dirty(p, allow=True)
            json.dump({"generator": {"skills_commit": "x"}}, open(p, "w"))
            pp.refuse_dirty(p)
            pp.refuse_dirty(os.path.join(d, "missing.json"))


if __name__ == "__main__":
    unittest.main()
