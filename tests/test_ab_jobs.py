"""ab_jobs.py (device-generic A/B jobs): the Strix wrapper's perf.json is byte-identical to the former strix_ab_jobs.py, the R9700 jobs
serve the new card exactly as r9700_jobs.py does, card references pin revisions, devlab_perf.sh parses."""
import contextlib
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, ".github", "scripts")
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
sys.path.insert(0, SCRIPTS)
import card_lib  # noqa: E402
import ab_jobs  # noqa: E402

FAST, SLOW = "unsloth-Qwen3-0.6B-loom", "ggml-org-Qwen3-4B-loom"


def run_script(name, args):
    """stdout of .github/scripts/NAME ARGS with the published decode rates stubbed (no network): the 0.6B is fast, the rest are not."""
    old = (card_lib.published_decode, sys.argv)
    card_lib.published_decode = lambda slug, org="loom-fun", gpu="strix-halo": 300.0 if "0.6B" in slug else 60.0
    sys.argv = [name] + args
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(SCRIPTS, name), run_name="__main__")
    finally:
        card_lib.published_decode, sys.argv = old
    return buf.getvalue()


GOLD = [(["unsloth-Qwen3-0.6B-loom", "ggml-org-Qwen3-4B-loom"]), (["--baselines", "--suffix=-r2", "--reps", "4", "--llama-reps", "3", "--gates", "fast", SLOW]),
        (["--old-org", "loom-fun-x", "--new-org", "y", FAST])]


class StrixWrapper(unittest.TestCase):
    def test_byte_identical_to_the_former_script(self):
        for i, args in enumerate(GOLD, 1):
            gold = open(os.path.join(ROOT, "tests", "data", "ab_jobs", f"strix_{i}.json")).read()
            self.assertEqual(run_script("strix_ab_jobs.py", args), gold, args)
            self.assertEqual(run_script("ab_jobs.py", ["--device", "strix-halo"] + args), gold, args)


class R9700(unittest.TestCase):
    def refs(self, obj):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        p = os.path.join(d, "refs.json")
        with open(p, "w") as f:
            json.dump(obj, f)
        return p

    def jobs(self, *args):
        return json.loads(run_script("ab_jobs.py", ["--device", "radeon-r9700", "--fit", os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"), *args]))

    def test_new_variants_are_r9700_jobs_card_and_nodraft(self):
        fit = os.path.join(ROOT, ".github", "devlab", "r9700_fit.json")
        for slug in (FAST, SLOW, "InternScience-Sage-A1.5-loom", "bartowski-Ornith-1.5-35B-A3B-loom"):
            ref = json.loads(run_script("r9700_jobs.py", ["--fit", fit, slug]))["jobs"][0]
            ab = self.jobs(slug)["jobs"][0]
            v = {x["name"]: x for x in ab["variants"]}
            self.assertEqual(v["staging"], {**ref["variants"][0], "name": "staging"}, slug)
            self.assertEqual(v["stagingnd"], {**ref["variants"][1], "name": "stagingnd"}, slug)
            for k in ("require_preset_r9700", "memory_bytes_r9700", "served_context", "llama_kv_rocm", "card", "kit", "llamacpp", "llamacpp_vulkan"):
                self.assertEqual(ab.get(k), ref.get(k), (slug, k))

    def test_old_variants_get_the_same_settings(self):
        j = self.jobs("--old-rev", "abc", "--new-rev", "def", "--baselines", FAST)["jobs"][0]
        v = {x["name"]: x for x in j["variants"]}
        self.assertEqual(j["card"], f"loom-fun-staging/{FAST}@def")
        self.assertEqual(v["fun"]["card"], f"loom-fun/{FAST}@abc")
        self.assertEqual(v["fun"]["execution"], {"prefill_capacity": 4096})
        self.assertEqual(v["funnd"]["execution"], {"prefill_capacity": 4096, "draft_depth": 0})
        self.assertGreater(v["fun"]["cap_pool"], 0)
        self.assertNotIn("require_preset_r9700", v["fun"])
        self.assertIn("llama@vulkan.np1:6", j["steps"])   # fast model: 6 sessions, llama.cpp as many runs
        self.assertEqual(j["steps"].count("perf:fun:1"), 6)

    def test_sampling_plan_follows_old_org_not_the_pinned_ref(self):
        # an A/A against a staging revision still takes the fast-model plan from the published loom-fun decode rate
        j = self.jobs("--refs", self.refs({FAST: {"old": f"loom-fun-staging/{FAST}@abc"}}), FAST)["jobs"][0]
        self.assertEqual(j["repeat_rows"], 3)
        self.assertEqual(j["steps"].count("perf:fun:1"), 6)

    def test_no_fit_no_job(self):
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "ab_jobs.py"), "--device", "radeon-r9700", "bartowski-Ornith-1.5-122B-loom"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)

    def test_refs_map_wins(self):
        j = self.jobs("--refs", self.refs({SLOW: {"old": f"loom-fun-staging/{SLOW}@aaa", "new": f"loom-fun-staging/{SLOW}"}}), SLOW)["jobs"][0]
        self.assertEqual(j["card"], f"loom-fun-staging/{SLOW}")
        self.assertEqual(j["variants"][0]["card"], f"loom-fun-staging/{SLOW}@aaa")


class SdkAb(unittest.TestCase):
    """--sdk-ab: old = the published card on the old SDK, new = built in the job on the new SDK (variant "sdk", job "sdk"), same ABBA steps."""

    def jobs(self, device, *args):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        ref = f"loom-fun-staging/{FAST}@abc"
        p = os.path.join(d, "refs.json")
        json.dump({FAST: {"old": ref, "new": ref}}, open(p, "w"))
        return json.loads(run_script("ab_jobs.py", ["--device", device, "--sdk-ab", "--baselines", "--refs", p, *args, FAST]))["jobs"][0]

    def test_strix_variants_name_their_sdk_and_the_new_one_is_built(self):
        j = self.jobs("strix-halo")
        v = {x["name"]: x for x in j["variants"]}
        self.assertEqual(j["sdk"], "0.2.0")
        self.assertEqual((v["fun"]["sdk"], v["funnd"]["sdk"]), ("0.1.0", "0.1.0"))
        self.assertEqual((v["staging"]["sdk"], v["stagingnd"]["sdk"]), ("0.2.0", "0.2.0"))
        self.assertEqual(v["fun"]["card"], f"loom-fun-staging/{FAST}@abc")
        for n in ("staging", "stagingnd"):
            self.assertEqual(v[n]["build"], {})
            self.assertEqual(v[n]["build_args"], ["--memory-gib", "4096"])
            self.assertNotIn("card", v[n])
        self.assertEqual(v["stagingnd"]["execution"], {"draft_depth": 0})

    def test_abba_order_of_old_and_new_sessions(self):
        steps = self.jobs("strix-halo")["steps"]
        self.assertEqual(steps[:4], ["perf:fun:1", "perf:staging:1", "perf:staging:1", "perf:fun:1"])
        c4 = [s for s in steps if s.startswith("c4:")]
        self.assertEqual(c4[:4], ["c4:funnd", "c4:stagingnd", "c4:stagingnd", "c4:funnd"])
        self.assertIn("gates:staging:dev", steps)
        self.assertIn("llama@rocm.np1:6", steps)   # the card's own baselines in the same job
        self.assertFalse([s for s in steps if s.startswith("ab:")])   # the one-server step cannot host two SDKs

    def test_r9700_new_variants_take_the_old_ones_serving_and_no_preset(self):
        j = self.jobs("radeon-r9700", "--fit", os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"))
        v = {x["name"]: x for x in j["variants"]}
        self.assertTrue(j["require_preset_r9700"])
        for old, new in (("fun", "staging"), ("funnd", "stagingnd")):
            self.assertEqual(v[new]["execution"], v[old]["execution"])
            self.assertEqual(v[new]["cap_pool"], v[old]["cap_pool"])
            self.assertIs(v[new]["require_preset_r9700"], False)   # a rebuilt package carries no <first>-r9700 preset
            self.assertEqual(v[new]["build_args"][-2:], ["--memory-gib", "4096"])   # built as build_fleet.py builds the published package
            self.assertNotIn("context", v[new])   # the served context is an execution override, never a build flag

    def loom_only(self, device, *args):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        ref = f"loom-fun-staging/{FAST}@abc"
        p = os.path.join(d, "refs.json")
        json.dump({FAST: {"old": ref, "new": ref}}, open(p, "w"))
        return json.loads(run_script("ab_jobs.py", ["--device", device, "--sdk-ab", "--loom-only", "--refs", p, *args, FAST]))["jobs"][0]

    def test_loom_only_serves_the_new_package_alone(self):
        for device in ("strix-halo", "radeon-r9700"):
            j = self.loom_only(device)
            self.assertEqual({v["name"] for v in j["variants"]}, {"staging", "stagingnd"})
            self.assertEqual(j["sdk"], "0.2.0")
            steps = j["steps"]
            self.assertEqual([s for s in steps if s.startswith("perf:") and ":long" not in s], ["perf:staging:1"] * 6)   # the fast model's 6 sessions, as the ABBA job
            self.assertEqual([s for s in steps if s.startswith("c4:")], ["c4:stagingnd"] * 4)
            fields = [f for x in steps for f in x.split(":")]
            for gone in ("fun", "funnd", "textmatch", "extend"):
                self.assertNotIn(gone, fields, (device, steps))
            self.assertFalse([x for x in steps if x.startswith("llama")], steps)
            for need in ("func:staging", "ktest:staging", "gates:staging:dev"):
                self.assertIn(need, steps)

    def test_loom_only_is_a_form_of_sdk_ab(self):
        with self.assertRaises(SystemExit):
            run_script("ab_jobs.py", ["--device", "strix-halo", "--loom-only", FAST])

    def test_without_the_flag_nothing_changes(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        j = json.loads(run_script("ab_jobs.py", ["--device", "strix-halo", FAST]))["jobs"][0]
        self.assertNotIn("sdk", json.dumps(j))


@unittest.skipIf(os.name == "nt", "devlab_perf.sh runs on the Linux DevLab runners only")
class DevlabPerfSdk(unittest.TestCase):
    """devlab_perf.sh's per-variant SDK roots, run against stub ensure_sdk.sh / loom roots (no network, no GPU)."""
    SCRIPT = os.path.join(SCRIPTS, "devlab_perf.sh")

    def functions(self):
        lines, on = [], False
        for ln in open(self.SCRIPT):
            if ln.startswith("sdk_root()"):
                on = True
            if on:
                lines.append(ln)
                if ln.startswith("}"):
                    break
        return "".join(lines)

    def run_sdk(self, variants, call):
        w = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, w, True)
        sv = os.path.join(w, "sv")
        os.makedirs(sv)
        with open(os.path.join(sv, "ensure_sdk.sh"), "w") as f:   # echoes the root it was asked for and records which SDK it was told
            f.write('#!/bin/bash\necho "ensure ${LOOM_VERSION:--} $1 $2" >> "$SVLOG"\nmkdir -p "$1"; echo \'{"hrx_loom": "\'"${LOOM_VERSION:-card}"\'"}\' > "$1/loom-install.json"; echo "$1"\n')
        os.chmod(os.path.join(sv, "ensure_sdk.sh"), 0o755)
        exp = os.path.join(w, "e.json")
        json.dump({"jobs": [{"name": "j", "variants": variants, "steps": []}]}, open(exp, "w"))
        script = (f'exp={exp}; job=j; SV={sv}; D={w}/d; out={w}/out; mkdir -p $out; WORK={w}; ROOT_BASE={w}/loom; troot={w}/tools; SVLOG={w}/log; export SVLOG\n'
                  "q() { python3 -c \"import json,sys; e=json.load(open('$exp')); j=next(j for j in e['jobs'] if j['name']=='$job'); $1\"; }\n"
                  "qv() { python3 -c \"import json,sys; e=json.load(open('$exp')); j=next(j for j in e['jobs'] if j['name']=='$job'); v=next(v for v in j['variants'] if v['name']=='$1'); $2\"; }\n"
                  f"declare -A SDKROOT\n{self.functions()}\n{call}")
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        log = open(os.path.join(w, "log")).read() if os.path.exists(os.path.join(w, "log")) else ""
        return r.stdout.strip().splitlines(), log, w

    def test_explicit_sdks_get_their_own_roots_and_the_built_card_variant_the_tools(self):
        out, log, w = self.run_sdk([{"name": "fun", "sdk": "0.1.0"}, {"name": "staging", "sdk": "0.2.0", "build": {}}, {"name": "tools", "build": {}}],
                                   'for v in fun staging tools; do sdk_root $v >/dev/null; echo "$v $SDKR"; done')
        roots = dict(x.split(" ", 1) for x in out)
        self.assertEqual(roots["fun"], f"{w}/loom-sdk-0.1.0")
        self.assertEqual(roots["staging"], f"{w}/loom-sdk-0.2.0")
        self.assertEqual(roots["tools"], f"{w}/tools")   # sdk "card" of a built variant = the job's tools SDK
        self.assertIn("ensure 0.1.0 ", log)
        self.assertIn("ensure 0.2.0 ", log)
        self.assertTrue(os.path.exists(os.path.join(w, "out", "sdk_fun.json")))

    def test_card_variant_follows_the_pin_of_its_own_package(self):
        out, log, w = self.run_sdk([{"name": "old"}], 'sdk_root old >/dev/null; echo "$SDKR"')
        self.assertEqual(out, [f"{w}/loom"])
        self.assertIn(f"ensure - {w}/loom {w}/d/old/pkg/llm_config.json", log)   # no LOOM_VERSION: ensure_sdk.sh reads the package's pin

    def test_one_install_per_variant_even_when_asked_twice(self):
        _, log, _ = self.run_sdk([{"name": "fun", "sdk": "0.1.0"}], 'sdk_root fun >/dev/null; sdk_root fun >/dev/null')
        self.assertEqual(log.count("ensure"), 1)

    def test_ab_step_refuses_variants_with_different_sdks(self):
        s = open(self.SCRIPT).read()
        self.assertIn("different SDKs", s)
        self.assertIn("ONE loom-serve", s)


class DevlabPerf(unittest.TestCase):
    def test_syntax_and_card_pin(self):
        p = os.path.join(SCRIPTS, "devlab_perf.sh")
        if os.name != "nt":   # a Windows checkout may hold CRLF line endings, which bash rejects (the script only runs on Linux runners)
            self.assertEqual(subprocess.run(["bash", "-n", p]).returncode, 0)
        s = open(p).read()
        for needle in ("card_revision.txt", "card_revision_$v.txt", "want=${ref#*@}", "cap_pool"):
            self.assertIn(needle, s)


if __name__ == "__main__":
    unittest.main()
