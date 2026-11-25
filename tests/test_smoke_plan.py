"""Release smoke: the plan (fleet/smoke_plan.py, sample_plan granularity "arch") and the runner's pure helpers (.github/scripts/devlab_smoke.py)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fleet"))
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
import sample_plan as SP  # noqa: E402
import smoke_plan as P  # noqa: E402
import devlab_smoke as D  # noqa: E402

STRIX = P.render_card.gpu_label("strix-halo")
R9700 = P.render_card.gpu_label("radeon-r9700")
G16 = P.render_card.gpu_label("gfx1201-16gb")


def block(org_repo, preset, out, serve_extra, pin="0.3.0", listen=""):
    return ("```bash\n"
            f'pipx install "hrx-loom[hf]=={pin}"\n'
            f"loom-fetch huggingface:{org_repo}:{preset} --output={out}\n"
            f"loom-serve --model={out} '--device=amdgpu://?completion_spin_ns=4294967295' {serve_extra}{listen}\n"
            "```\n")


def readme(repo, blocks, request_model="Org/Model"):
    if len(blocks) == 1 and blocks[0][0] is None:
        qs = "\n" + blocks[0][1]
    else:
        qs = "Open your GPU and copy its commands.\n\n" + "".join(
            f"<details>\n<summary>{lab.replace(chr(34), '&quot;') if False else lab}</summary>\n\n{b}\n- note\n\n</details>\n\n" for lab, b in blocks)
    return (f"# {repo}\n\n## Quick start\n\n{qs}\n## Send a request\n\n```bash\ncurl http://127.0.0.1:8080/v1/chat/completions "
            f"-d '{{\"model\": \"{request_model}\", \"messages\": []}}'\n```\n\n## Performance\n")


def cfg(arch, hidden=2048, kv=8, draft=None, moe=False, vision=False):
    return {"arch": arch, "hidden": hidden, "heads": 16, "kv_heads": kv, "head_dim": 128, "context_length": 40960, "layers": 28,
            "draft": {"kind": draft} if draft else None, "moe": {"experts": 8} if moe else None, "vision": {"x": 1} if vision else None,
            "sdk": {"hrx_loom": "0.3.0"}}


def info(devices=("strix-halo", "radeon-r9700"), tools=True, states=None):
    devs = {d: {"performance": {}} for d in devices}
    devs.update({d: {"state": s} for d, s in (states or {}).items()})
    return {"devices": devs, "card": {"features": {"tool_calling": {"status": "yes" if tools else "n/a"},
                                                   "thinking": {"status": "yes"}}}}


def card(slug, config, blocks, inf=None, rev="a" * 40):
    repo = f"org/{slug}"
    return P.describe({"repo": repo, "revision": rev, "readme": readme(repo, blocks), "llm_config.json": config, "info.json": inf or info()})


class Fleet(P.CardFleet):
    def __init__(self, cards, w):
        self.cards, self.w, self.fit, self.baselines = cards, w, {}, {}

    def weights(self, slug):
        return self.w.get(slug)


class PlanTest(unittest.TestCase):
    def fleet(self, slugs_cfg, w, two=True):
        cards = {}
        for s, c, extra in slugs_cfg:
            bl = [(STRIX, block(f"org/{s}", "chat", s, extra.get("strix", "--pool_backing=fixed")))]
            if two:
                bl.append((R9700, block(f"org/{s}", "chat-r9700", s, "--pool_backing=fixed")))
                bl.append((G16, block(f"org/{s}", "chat-16gb", s, "--memory_bytes=15000000000")))
            cards[s] = card(s, c, bl)
        return cards

    def run_plan(self, cards, w, **kw):
        orig = P.CardFleet
        P.CardFleet = lambda cards_, fit, base: Fleet(cards_, w)
        try:
            return P.plan(cards, ["strix-halo", "radeon-r9700"], **kw)
        finally:
            P.CardFleet = orig

    def test_one_lightest_per_arch_stratum_and_memory_variants(self):
        cards = self.fleet([("qwen3-8b", cfg("qwen3", 4096), {}), ("qwen3-1.7b", cfg("qwen3", 2048), {}),
                            ("qwen3-30b", cfg("qwen3", 2048, moe=True), {}),
                            ("odd", cfg("llama"), {"strix": "--workers=8"})], {})
        w = {"qwen3-8b": 5e9, "qwen3-1.7b": 1.2e9, "qwen3-30b": 18e9, "odd": 3e9}
        rows, skipped = self.run_plan(cards, w, windows=["strix-halo"])
        strix = sorted(r["slug"] for r in rows if r["os"] == "linux" and r["device"] == "strix-halo")
        # qwen3 dense shapes (4096 vs 2048 hidden) are one stratum: the smaller one; MoE and llama are their own strata
        self.assertEqual(strix, ["odd", "qwen3-1.7b", "qwen3-30b"])
        odd = next(r for r in rows if r["slug"] == "odd" and r["device"] == "strix-halo" and r["os"] == "linux")
        self.assertIn("stratum", odd["why"])   # in the sample already; its different flags add no second row
        g16 = [r for r in rows if r["block_device"] == "gfx1201-16gb"]
        self.assertEqual([(r["device"], r["block"]) for r in g16], [("radeon-r9700", G16)])
        self.assertEqual(sorted(r["slug"] for r in rows if r["os"] == "windows"), ["odd", "qwen3-1.7b", "qwen3-30b"])

    def test_memory_flag_card_added_and_windows_skip(self):
        cards = self.fleet([("a", cfg("qwen3", 2048), {}), ("b", cfg("qwen3", 4096), {"strix": ""}), ("c", cfg("qwen3", 5120), {})], {})
        rows, skipped = self.run_plan(cards, {"a": 1.2e9, "b": 5e9, "c": 9e9}, windows=["strix-halo"], windows_skip={"b": "known limit"})
        strix = {r["slug"]: r["why"] for r in rows if r["os"] == "linux" and r["device"] == "strix-halo"}
        self.assertEqual(sorted(strix), ["a", "b"])
        self.assertIn("memory flags", strix["b"])
        self.assertEqual([(s["os"], s["slug"], s["reason"]) for s in skipped], [("windows", "b", "known limit")])

    def test_single_block_goes_to_the_claimed_runner_gpu(self):
        c = card("quark", cfg("qwen35"), [(None, block("org/quark", "chat-r9700", "q", "--pool_backing=fixed"))],
                 info(devices=["radeon-r9700"], states={"strix-halo": "not_a_target"}))
        self.assertEqual(list(c["blocks"]), ["radeon-r9700"])
        rows, _ = self.run_plan({"quark": c}, {"quark": 5e9})
        self.assertEqual([(r["os"], r["device"], r["block"]) for r in rows], [("linux", "radeon-r9700", None)])

    def test_jobs_pack_by_os_and_gpu(self):
        rows = [{"os": "linux", "device": "strix-halo", "slug": f"s{i}", "weights": (i + 1) * 1e9} for i in range(7)]
        js = P.jobs(rows, 3)
        self.assertEqual((len(js), sum(len(j["cards"]) for j in js), max(len(j["cards"]) for j in js)), (3, 7, 3))
        self.assertEqual({j["runs_on"][0] for j in js}, {"strix-halo"})
        self.assertTrue(all(j["stock_memlock"] for j in js))
        win = P.jobs([{"os": "windows", "device": "strix-halo", "slug": "w", "weights": 1e9}], 3)
        self.assertEqual(win[0]["runs_on"], ["windows", "strix-halo", "devlab-dispatch"])
        self.assertFalse(win[0]["stock_memlock"])

    def test_cli_with_eligible_revisions(self):
        cards = self.fleet([("a", cfg("qwen3"), {})], {})
        files = {"a": cards["a"]}
        with tempfile.TemporaryDirectory() as d:
            el = os.path.join(d, "eligible.json")
            json.dump({"a": {"staging_revision": "b" * 40}}, open(el, "w"))
            seen = []
            orig = P.load_card
            P.load_card = lambda repo, rev, cache=None: seen.append((repo, rev)) or dict(files["a"], revision=rev)
            try:
                P.main(["--org", "org", "--eligible", el, "--windows", "", "--out", os.path.join(d, "s.json"),
                        "--hook-job", "lem:strix-halo:a:.github/devlab/smoke/lemonade.py:LEMONADE_REVISION=x"])
            finally:
                P.load_card = orig
            doc = json.load(open(os.path.join(d, "s.json")))
        self.assertEqual(seen, [("org/a", "b" * 40)])
        hook = doc["jobs"][-1]
        self.assertEqual((hook["name"], hook["hook"], hook["env"], hook["cards"][0]["revision"]),
                         ("lem", ".github/devlab/smoke/lemonade.py", {"LEMONADE_REVISION": "x"}, "b" * 40))


class SamplePlanArchTest(unittest.TestCase):
    def test_arch_granularity_merges_shapes_keeps_draft_and_vision(self):
        class F(SP.Fleet):
            def __init__(self, c, w):
                self.c, self.w, self.fit, self.baselines = c, w, {}, {}

            def config(self, s):
                return self.c[s]

            def weights(self, s):
                return self.w[s]

            def fits(self, s, g):
                return True
        c = {"a": cfg("qwen3", 2048, kv=8), "b": cfg("qwen3", 5120, kv=4), "c": cfg("qwen3", 2048, draft="dflash"),
             "d": cfg("qwen3", 2048, vision=True)}
        f = F(c, {"a": 2e9, "b": 9e9, "c": 2e9, "d": 2e9})
        st, _ = SP.plan({"models": {s: {"model_json": True} for s in c}}, ["strix-halo"], f, granularity="arch", prefer="lightest")
        self.assertEqual(sorted(len(s["members"]) for s in st), [1, 1, 2])
        merged = next(s for s in st if len(s["members"]) == 2)
        self.assertEqual(merged["representatives"][0]["slug"], "a")
        st_h, _ = SP.plan({"models": {s: {"model_json": True} for s in c}}, ["strix-halo"], f, granularity="arch")
        self.assertEqual(next(s for s in st_h if len(s["members"]) == 2)["representatives"][0]["slug"], "b")


class RunnerHelpersTest(unittest.TestCase):
    def test_block_selection_and_lines(self):
        rd = readme("org/x", [(STRIX, block("org/x", "chat", "X", "--pool_backing=fixed")), (R9700, block("org/x", "chat-r9700", "X", ""))])
        lines = D.quick_start_block(rd, R9700)
        kinds, other = D.classify(lines)
        self.assertEqual(sorted(kinds), ["fetch", "install", "serve"])
        self.assertEqual(other, [])
        self.assertIn("chat-r9700", kinds["fetch"])
        with self.assertRaises(ValueError):
            D.quick_start_block(rd, None)
        with self.assertRaises(ValueError):
            D.quick_start_block(rd, "AMD Nothing")
        single = readme("org/y", [(None, block("org/y", "chat", "Y", ""))])
        self.assertEqual(len(D.quick_start_block(single, None)), 3)

    def test_html_escaped_summary_matches(self):
        rd = readme("org/x", [(STRIX.replace('"', "&quot;"), block("org/x", "chat", "X", ""))])
        self.assertEqual(len(D.quick_start_block(rd, STRIX)), 3)

    def test_fetch_pin_and_selector(self):
        ln = "loom-fetch huggingface:org/x:chat --output=X"
        self.assertEqual(D.pin_fetch(ln, "org/x", "c" * 40), f"loom-fetch huggingface:org/x:chat@{'c' * 40} --output=X")
        self.assertEqual(D.fetch_selector(ln, "org/x"), ("chat", None))
        pinned = "loom-fetch huggingface:org/x:chat@main --output=X"
        self.assertEqual(D.pin_fetch(pinned, "org/x", "c" * 40), pinned)
        self.assertEqual(D.fetch_selector("loom-fetch huggingface:org/x --output=X", "org/x"), (None, None))
        self.assertIsNone(D.fetch_selector(ln, "org/other"))
        self.assertEqual(D.install_pin('pipx install "hrx-loom[hf]==0.3.0"'), "0.3.0")

    def test_listen_and_memlock(self):
        self.assertEqual(D.listen_of("loom-serve --model=X"), ("127.0.0.1", 8080))
        self.assertEqual(D.with_listen("loom-serve --model=X --listen=127.0.0.1:8080", 9000), "loom-serve --model=X --listen=127.0.0.1:9000")
        self.assertEqual(D.with_listen("loom-serve --model=X", 9000), "loom-serve --model=X --listen=127.0.0.1:9000")
        lim = "Limit Soft Limit Hard Limit Units\nMax locked memory         8388608              8388608              bytes\n"
        self.assertEqual(D.memlock_of(lim), (8388608, 8388608))
        self.assertEqual(D.memlock_of("Max locked memory unlimited unlimited bytes"), (-1, -1))

    def test_stock_shell_and_replies(self):
        if os.name != "nt":
            self.assertEqual(D.shell_argv("loom-serve --model=X", True), ["bash", "-c", "ulimit -l 8192 && exec loom-serve --model=X"])
        b = D.chat_body("m", {"thinking": True, "reasoning_effort": ["low", "high"]}, "hi")
        self.assertEqual((b["chat_template_kwargs"], b["reasoning_effort"]), ({"enable_thinking": False}, "low"))
        self.assertTrue(D.chat_ok({"choices": [{"message": {"content": "Paris."}}]}))
        self.assertFalse(D.chat_ok({"choices": [{"message": {"content": "", "reasoning_content": "Paris"}}]}))
        call = {"choices": [{"message": {"tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}}]}
        self.assertTrue(D.tool_ok(call))
        self.assertFalse(D.tool_ok({"choices": [{"message": {"content": "sunny"}}]}))
        self.assertEqual(D.request_model_id(readme("org/x", [(None, block("org/x", "c", "X", ""))], "Qwen/Qwen3-0.6B")), "Qwen/Qwen3-0.6B")
        img = "## Send a request\n\n```bash\ncurl x/v1/images/generations \\\n  -d '{\"model\": \"m\", \"prompt\": \"a cat\", \"seed\": 1}' | jq\n```\n"
        self.assertEqual(D.image_request(img)["prompt"], "a cat")
        self.assertEqual(P.image_request(img)["seed"], 1)


class HookCredentials(unittest.TestCase):
    def test_hooks_get_no_credentials(self):
        """a smoke hook builds and runs another repository's code: devlab_smoke.py strips HF and GitHub tokens from its environment"""
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "scripts", "devlab_smoke.py")).read()
        self.assertIn('k.startswith("HF_TOKEN") or k in ("GITHUB_TOKEN", "GH_TOKEN")', src)
        self.assertIn("henv.pop(k)", src)


if __name__ == "__main__":
    unittest.main()
