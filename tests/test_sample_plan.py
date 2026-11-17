import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fleet"))
import loomtext  # noqa: E402
import replay_impact  # noqa: E402
import sample_lib as L  # noqa: E402
import sample_plan as SP  # noqa: E402


def cfg(arch="qwen3", hidden=2048, heads=16, kv=8, hd=128, draft=None, moe=False, ctx=40960):
    return {"arch": arch, "hidden": hidden, "heads": heads, "kv_heads": kv, "head_dim": hd, "context_length": ctx, "layers": 28,
            "draft": {"kind": draft} if draft else None, "moe": {"experts": 8} if moe else None}


class Fleet(SP.Fleet):
    """a Fleet with in-memory configs, weights and fit verdicts"""

    def __init__(self, cfgs, w=None, verdict=None, baselines=None):
        self.c, self.w, self.v = cfgs, w or {}, verdict or {}
        self.fit, self.baselines = {}, baselines or {}

    def config(self, slug):
        return self.c.get(slug)

    def weights(self, slug):
        return self.w.get(slug)

    def fits(self, slug, gpu):
        return not (gpu == "radeon-r9700" and self.v.get(slug) == "does not fit")


def impact(**models):
    return {"models": {s: {"kernels": ks} for s, ks in models.items()}}


FLASH = {"name": "g_attention_flash2", "target": "gfx11"}
SPLIT = {"name": "g_attention_split", "target": "gfx11"}
TWIN = {"name": "g_gemm_q4_k_store_rdna4", "target": "gfx12"}


class Rules(unittest.TestCase):
    def test_kernel_rows(self):
        self.assertEqual(L.kernel_family_rows("g_attention_flash2_partial"), ("attn_prefill", L.PREFILL_ROWS))
        self.assertEqual(L.kernel_family_rows("g_attention_split")[0], "attn_decode")
        self.assertEqual(L.kernel_family_rows("gd512_attention_split")[0], "attn_decode")
        self.assertEqual(L.kernel_family_rows("llm_verify")[1], L.DECODE_ROWS)
        self.assertEqual(L.kernel_family_rows("g_moe_gu_q4_k_k2048_r8")[1], L.ROWS)
        self.assertEqual(L.kernel_family_rows("ch_parse_calls"), ("text", []))
        self.assertEqual(L.kernel_family_rows("llm_advance")[0], "program")
        self.assertEqual(L.kernel_formats("g_moe_gu_q4_k_q6_k_k2304_r8"), ["q4_k", "q6_k"])

    def test_signature(self):
        s = L.signature_from_config(cfg(arch="qwen35moe", heads=16, kv=2, hd=256, draft="dflash", moe=True))
        self.assertEqual((s["G"], s["kv_bucket"], s["draft"], s["moe"]), (8, "<=2", "dflash", "moe"))
        self.assertEqual(L.kv_bucket(4), "4")
        self.assertEqual(L.kv_bucket(16), ">=8")


class Plan(unittest.TestCase):
    def one(self, imp, fleet, gpus=("strix-halo",), **kw):
        st, skipped = SP.plan(imp, list(gpus), fleet, **kw)
        return st, skipped

    def test_one_representative_per_stratum_heaviest_member(self):
        f = Fleet({"a": cfg(hidden=2048), "b": cfg(hidden=2560), "c": cfg(arch="llama", hd=64, kv=2, heads=14)},
                  w={"a": 2e9, "b": 3e9, "c": 2e9})
        st, _ = self.one(impact(a=[FLASH], b=[FLASH], c=[FLASH]), f)
        self.assertEqual(len(st), 2)    # qwen3 shape (a, b) and the llama shape (c)
        big = next(s for s in st if "a" in s["members"])
        self.assertEqual(sorted(big["members"]), ["a", "b"])
        self.assertEqual([r["slug"] for r in big["representatives"]], ["b"])      # largest hidden size, hidden ratio < 2: one rep
        self.assertEqual(big["rows"], L.PREFILL_ROWS)

    def test_wide_stratum_gets_two_representatives(self):
        f = Fleet({"s": cfg(hidden=1024), "m": cfg(hidden=2560), "l": cfg(hidden=5120)}, w={"s": 2e9, "m": 3e9, "l": 9e9})
        st, _ = self.one(impact(s=[FLASH], m=[FLASH], l=[FLASH]), f)
        reps = [r["slug"] for r in st[0]["representatives"]]
        self.assertEqual(reps, ["l", "s"])
        self.assertIn("opposite shape extreme", st[0]["representatives"][1]["why"][0])

    def test_sub_1b_avoided_for_decode_rows_only(self):
        f = Fleet({"tiny": cfg(hidden=1024), "mid": cfg(hidden=1024)}, w={"tiny": 0.4e9, "mid": 2e9})
        st, _ = self.one(impact(tiny=[SPLIT], mid=[SPLIT]), f)
        self.assertEqual([r["slug"] for r in st[0]["representatives"]], ["mid"])
        self.assertTrue(st[0]["notes"])
        # prefill-only change: the tiny model is a normal candidate (tie on hidden size: alphabetical, rotation decides later)
        st, _ = self.one(impact(tiny=[FLASH], mid=[FLASH]), f)
        self.assertEqual(st[0]["notes"], [])

    def test_rotation_takes_the_least_recently_measured(self):
        f = Fleet({"a": cfg(hidden=2048), "b": cfg(hidden=2560)}, w={"a": 2e9, "b": 3e9})
        hist = {"models": {"b|strix-halo": "2026-11-10"}}
        st, _ = self.one(impact(a=[FLASH], b=[FLASH]), f, history=hist)
        self.assertEqual(st[0]["representatives"][0]["slug"], "a")
        self.assertEqual(st[0]["representatives"][0]["why"], ["heaviest for the changed kernel"])
        hist["models"]["a|strix-halo"] = "2026-11-12"          # now b was measured earlier than a: b first
        st, _ = self.one(impact(a=[FLASH], b=[FLASH]), f, history=hist)
        self.assertEqual((st[0]["representatives"][0]["slug"], st[0]["representatives"][0]["why"]), ("b", ["least recently measured"]))

    def test_gpu_family_follows_the_kernel_target(self):
        f = Fleet({"a": cfg()}, w={"a": 2e9})
        st, _ = self.one(impact(a=[TWIN]), f, gpus=("strix-halo", "radeon-r9700"))
        self.assertEqual([s["gpu"] for s in st], ["radeon-r9700"])       # a gfx12 twin does not touch Strix
        st, _ = self.one(impact(a=[{"name": "llm_advance", "target": "both"}]), f, gpus=("strix-halo", "radeon-r9700"))
        self.assertEqual(sorted(s["gpu"] for s in st), ["radeon-r9700", "strix-halo"])

    def test_model_that_does_not_fit_is_skipped_on_that_gpu(self):
        f = Fleet({"big": cfg(), "ok": cfg()}, w={"big": 80e9, "ok": 2e9}, verdict={"big": "does not fit"})
        both = {"name": "g_attention_flash2", "target": "both"}
        st, skipped = self.one(impact(big=[both], ok=[both]), f, gpus=("strix-halo", "radeon-r9700"))
        r97 = next(s for s in st if s["gpu"] == "radeon-r9700")
        self.assertEqual(r97["members"], ["ok"])
        self.assertEqual([x["slug"] for x in skipped], ["big"])
        self.assertIn("big", next(s for s in st if s["gpu"] == "strix-halo")["members"])

    def test_rows_limited_to_published_rows(self):
        f = Fleet({"a": cfg()}, w={"a": 2e9}, baselines={"a": {"strix-halo": {r: {} for r in L.ROWS if r != "prefill_32k"}}})
        st, _ = self.one(impact(a=[FLASH]), f)
        self.assertNotIn("prefill_32k", st[0]["rows"])

    def test_text_only_change_is_functional_only(self):
        f = Fleet({"a": cfg()}, w={"a": 2e9})
        st, _ = self.one(impact(a=[{"name": "ch_parse_calls", "target": "both"}]), f)
        self.assertTrue(st[0]["functional_only"])
        self.assertEqual(st[0]["rows"], [])

    def test_strata_split_on_draft_kind_and_moe(self):
        f = Fleet({"d": cfg(draft="dflash"), "n": cfg(), "m": cfg(moe=True)}, w={"d": 2e9, "n": 2e9, "m": 2e9})
        st, _ = self.one(impact(d=[FLASH], n=[FLASH], m=[FLASH]), f)
        self.assertEqual(len(st), 3)
        st, _ = self.one(impact(d=[FLASH], n=[FLASH], m=[FLASH]), f, granularity="coarse")
        self.assertEqual(len(st), 3)

    def test_normalize_accepts_bare_names_and_target_lists(self):
        m = L.normalize_impact({"models": {"a": {"changed_kernels": ["g_attention_flash2", {"name": "x", "targets": ["gfx12"], "rows": ["decode"]}]}}})
        k = m["a"]["kernels"]
        self.assertEqual(k[0]["targets"], ["gfx11", "gfx12"])
        self.assertEqual((k[1]["targets"], k[1]["rows"]), (["gfx12"], ["decode"]))

    def test_diff_fleet_report_shape(self):
        rep = {"models": {"a": {"blocks": [{"symbol": "g_attention_split", "target": "gfx11", "rows": ["decode", "decode_32k"]},
                                           {"symbol": "ch_parse_calls", "target": "both", "rows": ["decode"]}],
                                "model_json": [], "gpus": ["gfx11"],
                                "stratum": {"arch": "qwen35moe", "head_dim": 256, "G": 8.0, "kv_heads": 2, "kv_bucket": "kv<=2", "draft": "dflash",
                                            "formats": [], "topology": "moe"}},
                         "b": {"blocks": [], "model_json": ["execution.draft_depth: 3 -> 2"], "stratum": {"arch": "qwen3", "kv_bucket": "kv>=8", "topology": "dense"}},
                         "c": {"error": "present in only one build"}}, "untouched": []}
        m = L.normalize_impact(rep)
        self.assertEqual(sorted(m), ["a", "b"])
        self.assertEqual(m["a"]["kernels"][0]["rows"], ["decode", "decode_32k"])    # the report's rows win over the name rules
        self.assertEqual(m["a"]["kernels"][1]["rows"], [])                           # text blocks never carry perf rows
        self.assertEqual((m["a"]["signature"]["kv_bucket"], m["a"]["signature"]["G"], m["a"]["signature"]["moe"]), ("<=2", 8, "moe"))
        self.assertEqual(m["b"]["kernels"][0]["family"], "config")                   # a model.json change can move every row
        self.assertEqual(m["b"]["kernels"][0]["targets"], ["gfx11", "gfx12"])
        st, _ = SP.plan(rep, ["strix-halo"], Fleet({"a": cfg(hidden=2048)}, w={"a": 3e9}))
        self.assertEqual([s["key"].split("|")[0] for s in st], ["arch=qwen35moe", "arch=qwen3"])

    def test_totals_and_table(self):
        f = Fleet({"a": cfg()}, w={"a": 6e9})
        st, sk = self.one(impact(a=[FLASH]), f)
        t = SP.totals(st)
        self.assertEqual((t["strata"], t["sampled_models"]), (1, 1))
        self.assertLess(t["canary_runner_hours"], t["full_runner_hours"])
        self.assertIn("strix-halo", SP.table(st, sk))


class Jobs(unittest.TestCase):
    JOB = {"name": "sab-x", "steps": ["perf:fun:1", "perf:staging:1", "perf:staging:1", "perf:fun:1", "perf:fun:1", "perf:staging:1",
                                      "perf:fun:1:long", "perf:staging:1:long", "c4:funnd", "c4:stagingnd", "textmatch:fun", "func:staging",
                                      "ktest:staging", "gates:staging:dev", "llama@rocm.np1:2"]}

    def test_short_rows_only(self):
        j = SP.filter_job(self.JOB, ["decode", "prefill_1k"], 2)
        self.assertEqual([s for s in j["steps"] if s.endswith("long")], [])
        self.assertFalse(any(s.startswith(("c4", "gates", "llama")) for s in j["steps"]))
        self.assertEqual(sum(s.startswith("perf:") for s in j["steps"]), 4)      # 2 sessions per side, ABBA kept
        self.assertIn("textmatch:fun", j["steps"])
        self.assertTrue(j["name"].startswith("t1-"))

    def test_long_and_c4_rows(self):
        j = SP.filter_job(self.JOB, ["decode_32k", "streams4"], 2)
        self.assertTrue(any(s.endswith("long") for s in j["steps"]))
        self.assertFalse(any(s.startswith("perf:") and not s.endswith("long") for s in j["steps"]))
        self.assertIn("c4:funnd", j["steps"])


class Texts(unittest.TestCase):
    def tree(self, root, slug, body, twin=False, rdna3_attr=False):
        d = Path(root, slug, "program")
        d.mkdir(parents=True)
        head = "target.decl @device\namdgpu.target<gfx11-generic> @rdna3\namdgpu.target<gfx12-generic> @rdna4\n" if twin else ""
        k = "kernel.def target(@rdna3) export(\"g_k\") @g_k(%a: index) {" if rdna3_attr else "kernel.def export(\"g_k\") @g_k(%a: index) {"
        t = ("kernel.def target(@rdna4) export(\"g_k_rdna4\") @g_k_rdna4(%a: index) {\n  %x = index.constant 7 : index\n}\n"
             "template.decl @sel1_g_k(%g0: index)\ntemplate.def<@sel1_g_k> target(@rdna3) @sel1_g_k_rdna3(%g0: index) {\n"
             "  kernel.launch @g_k[%g0](%g0) : [index](index)\n  template.return\n}\n"
             "template.def<@sel1_g_k> target(@rdna4) @sel1_g_k_rdna4(%g0: index) {\n  kernel.launch @g_k_rdna4[%g0](%g0) : [index](index)\n  template.return\n}\n") if twin else ""
        prog = ("command.def public target(@device) @llm_advance() launch(%w: buffer) {\n  template.apply<@sel1_g_k>(%t, %t) : (index, index)\n}\n" if twin
                else "command.def public @llm_advance() launch(%w: buffer) {\n  kernel.launch @g_k[%t](%t) : [index](index)\n}\n")
        (d / "llm.loom").write_text(f"{head}// Generated by gen_llm.py from /x/{slug}.json.\n{k}\n{body}\n}}\n{t}{prog}")

    def test_view_normalizes_twins_selection_and_headers(self):
        with tempfile.TemporaryDirectory() as t:
            self.tree(t, "old", "  %x = index.constant 1 : index")
            self.tree(t, "new", "  %x = index.constant 1 : index", twin=True, rdna3_attr=True)
            a, b = loomtext.view(os.path.join(t, "old", "program")), loomtext.view(os.path.join(t, "new", "program"))
            self.assertEqual(loomtext.diff_views(a, b), [])
            self.assertNotIn("kernel.def:g_k_rdna4", b)
            self.assertIn("kernel.def:g_k_rdna4", loomtext.view(os.path.join(t, "new", "program"), "gfx12"))

    def test_replay_impact_names_the_changed_kernel_only(self):
        with tempfile.TemporaryDirectory() as t:
            self.tree(t + "/b", "m", "  %x = index.constant 1 : index")
            self.tree(t + "/h", "m", "  %x = index.constant 2 : index", twin=True, rdna3_attr=True)
            self.tree(t + "/b", "same", "  %x = index.constant 1 : index")
            self.tree(t + "/h", "same", "  %x = index.constant 1 : index", twin=True, rdna3_attr=True)
            res = replay_impact.impact(t + "/b", t + "/h")
            self.assertEqual([k["name"] for k in res["models"]["m"]["kernels"]], ["g_k"])
            self.assertEqual(res["models"]["same"]["kernels"], [])


if __name__ == "__main__":
    unittest.main()
