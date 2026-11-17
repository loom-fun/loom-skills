"""skills/loom-migrate scripts: the v0.1.0 -> v0.2.0 source rewrite and the version-agnostic parse/link sweep (fake SDK tools, no GPU)."""
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills/loom-migrate/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rw = load("rewrite_v010_v020")
sweep = load("sweep")

OLD = """\
func.def public @publish_epoch(%plan: buffer, %count: i32) {
  %zero = vector.constant 0.0 : vector<8xf32>
  %mask = vector.constant 252645135 : vector<4xi32>
  %flag = vector.constant 1 : vector<4xi1>
  %h = vector.constant 1.0 : vector<2xbf16>
  %tab = vector.constanti [-127, -104] : vector<2xi8>
  return
}
func.decl @publish_epoch(%plan: buffer, %count: i32)
func.def public @text_publish_epoch_v1(%plan: buffer, %count: i32) {
  func.call @publish_epoch(%plan, %count) : (buffer, i32) -> ()
  return
}
command.program.def public target(@device) @go() launch(%w: buffer) {
  command.program.launch @go(%w) : (buffer) -> ()
}
command.program.decl @x()
"""


class Rewrite(unittest.TestCase):
    def test_text(self):
        new, c, man = rw.rewrite_text(OLD)
        self.assertEqual({k: v for k, v in c.items() if v}, {"splati": 2, "splatf": 2, "command": 3, "publish_epoch": 4})
        self.assertIn("%zero = vector.splatf 0.0 : vector<8xf32>", new)
        self.assertIn("%h = vector.splatf 1.0 : vector<2xbf16>", new)
        self.assertIn("%mask = vector.splati 252645135 : vector<4xi32>", new)
        self.assertIn("%flag = vector.splati 1 : vector<4xi1>", new)
        self.assertIn("vector.constanti [-127, -104]", new)            # the dense-list op is untouched
        self.assertNotIn("command.program", new)
        self.assertIn("command.def public target(@device) @go()", new)
        self.assertIn("@publish_epoch(%control_state: buffer, %plan: buffer, %count: i32) {", new)
        self.assertIn("func.decl @publish_epoch(%control_state: buffer, %plan", new)
        self.assertIn("@text_publish_epoch_v1(%control_state: buffer, %plan", new)
        self.assertIn("func.call @publish_epoch(%control_state, %plan, %count) : (buffer, buffer, i32) -> ()", new)
        self.assertEqual(man, [])

    def test_weights_key_range(self):
        old = ('func.decl import("weights") @parameter(%source: i32, %key: buffer) -> (buffer, i64)\n'
               'func.decl import("weights") @input(%source: i32, %key: buffer, %offset: i64, %length: i64, %alignment: i64) -> (i64)\n'
               "func.def public @w(%key: buffer) -> (i64) {\n"
               "    %md, %size = func.call @parameter(%source, %key) : (i32, buffer) -> (buffer, i64)\n"
               "    %first = func.call @input(%source, %key, %zero, %length, %align) : (i32, buffer, i64, i64, i64) -> (i64)\n}\n")
        new, c, _ = rw.rewrite_text(old)
        self.assertEqual(c["weights_key_range"], 4)
        self.assertIn("@parameter(%source: i32, %key: buffer, %key_offset: i64, %key_length: i64) -> (buffer, i64)", new)
        self.assertIn("@input(%source: i32, %key: buffer, %key_offset: i64, %key_length: i64, %offset: i64, %length: i64, %alignment: i64)", new)
        self.assertIn("    %wk1_len = index.cast %wk1_ext : offset to i64\n    %first = func.call @input(%source, %key, %wk1_off, %wk1_len, %zero, %length, %align) : (i32, buffer, i64, i64, i64, i64, i64) -> (i64)", new)
        self.assertIn("func.call @parameter(%source, %key, %wk2_off, %wk2_len) : (i32, buffer, i64, i64) -> (buffer, i64)", new)
        self.assertEqual(rw.rewrite_text(new)[0], new)
        self.assertNotIn("%wk", rw.rewrite_text(old, calls=False)[0])      # generator text: declarations only
        prep = 'func.call @parameter(%stage, %c0_i32, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)\n'   # the prepare import, not the weights one
        self.assertEqual(rw.rewrite_text(prep)[0], prep)

    def test_idempotent(self):
        once, _, _ = rw.rewrite_text(OLD)
        twice, c, _ = rw.rewrite_text(once)
        self.assertEqual(once, twice)
        self.assertEqual(sum(c.values()), 0)

    def test_publish_epoch_optional_and_already_new(self):
        _, c, _ = rw.rewrite_text(OLD, publish_epoch=False)
        self.assertEqual(c["publish_epoch"], 0)
        new = "func.def public @text_publish_epoch_v1(%control_state: buffer, %plan: buffer) {\n}\n"
        self.assertEqual(rw.rewrite_text(new)[0], new)

    def test_manual_cases(self):
        txt = "  %a = vector.constant 1 : vector<4x{ty}>\n  %b = vector.constant 1 : vector<4xweird>\n"
        new, c, man = rw.rewrite_text(txt)
        self.assertEqual(new, txt)
        self.assertEqual([m[1] for m in man], [1, 2])
        _, _, man = rw.rewrite_text("func.def public @text_image_prepare_batch_v1(%state: buffer, %tags: buffer, %ranges: buffer, %packed: buffer) -> (i32) {\n")
        self.assertTrue(any("vision" in m[0] and "section 2.4" in m[0] and m[1] == 1 for m in man))
        self.assertEqual(man[0][1], 1)

    def test_generator_source_literal_element(self):
        py = 'f"  %{o} = vector.constant {v!r} : vector<{n}xf32>\\n"'
        new, c, man = rw.rewrite_text(py)
        self.assertIn("vector.splatf {v!r} : vector<{n}xf32>", new)

    def test_pool_bounds_by_context(self):
        src = (
            "  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index\n"
            "config.decl @llm.pool : %value: index where [range(%value, 16, 1048576)]\n"
            "  %bounded_physical = scalar.assume %physical32 [range(%physical32, 0, 1048575)] : i32\n"
            "    %kh_phb = scalar.assume %kh_ph32 [range(%kh_ph32, 0, 1048575)] : i32\n"
            "    %rw0 = index.assume %rw0_v [range(%rw0_v, 0, 1048560), lt(%rw0_v, %pl_rows_m)] : index\n"
            "  %fm_lr = index.assume %fm_lr_v [range(%fm_lr_v, 0, 1048575), lt(%fm_lr_v, %pool)] : index\n")
        keep = (
            "  %bounded_row = scalar.assume %row [range(%row, 0, 1048575)] : i32\n"
            "  %bounded_rows = scalar.assume %rows [range(%rows, 1, 1048576)] : i64\n"
            "  %pacc_dec = index.constant 1048576 : offset\n"
            "  %x = index.assume %pool_arg [range(%pool_arg, 8, 1048576)] : index\n")
        new, c, man = rw.rewrite_text(src + keep)
        self.assertEqual({k: c[k] for k in ("pool_arg", "pool_decl", "physical_page", "flash2_rows")},
                         {"pool_arg": 1, "pool_decl": 1, "physical_page": 2, "flash2_rows": 2})
        self.assertIn("[range(%pool_arg, 16, 2147483584)]", new)
        self.assertIn("[range(%value, 16, 2147483584)]", new)
        self.assertEqual(new.count("0, 134217727)"), 2)
        self.assertIn("0, 2147483568), lt(%rw0_v, %pl_rows_m)", new)
        self.assertIn("0, 2147483583), lt(%fm_lr_v, %pool)", new)
        self.assertTrue(new.endswith(keep))                       # the unrelated bounds are untouched
        self.assertEqual(rw.rewrite_text(src + keep, pool=False)[0], src + keep)
        self.assertEqual(rw.rewrite_text(new)[0], new)
        self.assertEqual(man, [])

    CONV = ("        %back0 = index.constant 3 : index\n"
            "        %has0_u0 = index.cmp uge, %rel, %back0 : index\n"
            "        %x0 = scf.if %has0_u0 -> (f32) {\n"
            "          %src0 = index.sub %token, %back0 : index\n")

    def test_conv_guard(self):
        new, c, man = rw.rewrite_text(self.CONV)
        self.assertEqual(c["conv_guard"], 1)
        self.assertIn("        %has_rel0_u0 = index.cmp uge, %rel, %back0 : index\n"
                      "        %has_token0_u0 = index.cmp uge, %token, %back0 : index\n"
                      "        %has0_u0 = scalar.andi %has_rel0_u0, %has_token0_u0 : i1\n", new)
        self.assertEqual(man, [])
        self.assertEqual(rw.rewrite_text(new)[0], new)
        other = "\n  %h = index.cmp uge, %rel, %back0 : index\n  %y = scf.if %h -> (f32) {\n"     # the guard of another shape: manual, file:line
        new, c, man = rw.rewrite_text(other)
        self.assertEqual((new, c["conv_guard"]), (other, 0))
        self.assertEqual([(m[1], "section 2.9" in m[0]) for m in man], [(2, True)])

    V1 = ("// Image component control (generated by gen_component.py).\n"
          'func.decl import("input") @require(%condition: i32, %diagnostic: buffer)\n'
          "func.def public @text_image_prepare_batch_v1(%state: buffer, %tags: buffer, %ranges: buffer, %packed: buffer) -> (i32, buffer, buffer) {\n"
          "  %origin = index.constant 0 : offset\n  func.return %a, %b, %c : i32, buffer, buffer\n}\n\n"
          "func.def public @text_image_run_v1(%stage: i32, %input: hal.buffer, %output: hal.buffer) {\n"
          "  %work = func.call @execute_2(%stage, %input, %output) : (i32, hal.buffer, hal.buffer) -> (i64)\n  func.return\n}\n")

    def test_vision_abi(self):
        new, c, man = rw.rewrite_text(self.V1)
        self.assertEqual((c["vision_abi"], man), (1, []))
        lines = new.split("\n")
        self.assertEqual(lines[0], self.V1.split("\n")[0])
        self.assertEqual(lines[1], 'type.decl import("finite", "batch") @FiniteBatch')
        self.assertIn("@text_image_prepare_batch_v1(%state: buffer, %tags: buffer, %batch: ptr<@FiniteBatch>) -> (i32, buffer, buffer) {\n"
                      "  %ranges, %packed = func.call @finite_batch_pack(%batch) : (ptr<@FiniteBatch>) -> (buffer, buffer)\n  %origin", new)
        self.assertIn("@text_image_run_v1(%state: buffer, %stage: i32, %input: hal.buffer, %output: hal.buffer, %batch: ptr<@FiniteBatch>) {", new)
        self.assertTrue(new.endswith("  func.call @finite_batch_finish(%batch) : (ptr<@FiniteBatch>)\n  func.return\n}\n"))
        self.assertIn("@text_image_collect_batch_v1(%state: buffer, %batch: ptr<@FiniteBatch>) {", new)
        self.assertEqual(rw.rewrite_text(new)[0], new)
        odd = self.V1.replace("%output: hal.buffer) {", "%output: hal.buffer, %extra: i32) {")      # another shape: untouched, manual with the section
        new, c, man = rw.rewrite_text(odd)
        self.assertEqual((new, c["vision_abi"]), (odd, 0))
        self.assertEqual([(m[1], "section 2.4" in m[0]) for m in man], [(3, True)])

    def test_main_prints_manual_file_line(self):
        import contextlib, io
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.loom").write_text("\n  %b = vector.constant 1 : vector<4xweird>\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rw.main([d, "--check"])
            self.assertIn("manual: " + str(Path(d, "a.loom")) + ":2:", out.getvalue())

    def test_card(self):
        mid = '{\n "execution": {\n  "rows": 4,\n  "pool_capacity": 106496,\n  "max_tokens": 4096\n }\n}'
        last = '{\n "execution": {\n  "rows": 4,\n  "pool_capacity": 106496\n }\n}'
        for t in (mid, last):
            new, n = rw.rewrite_card(t)
            self.assertEqual(n, 1)
            json.loads(new)
            self.assertNotIn("pool_capacity", new)
        self.assertEqual(rw.rewrite_card(new)[1], 0)

    def test_tree_and_check(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d, "pkg")
            (pkg / "program").mkdir(parents=True)
            (pkg / "program/llm.loom").write_text(OLD)
            (pkg / "model.json").write_text('{\n "execution": {\n  "pool_capacity": 1,\n  "rows": 4\n }\n}')
            (pkg / "notes.py").write_text("vector.constant 1 : vector<4xi32>")
            self.assertEqual(rw.main(["--check", str(pkg)]), 1)
            self.assertIn("vector.constant ", (pkg / "program/llm.loom").read_text())   # --check wrote nothing
            self.assertEqual(rw.main([str(pkg), "--json", str(Path(d, "o.json"))]), 0)
            self.assertEqual(rw.main(["--check", str(pkg)]), 0)
            self.assertNotIn("pool_capacity", (pkg / "model.json").read_text())
            self.assertIn("vector.constant", (pkg / "notes.py").read_text())             # .py only with --py
            rw.main([str(pkg), "--py"])
            self.assertIn("vector.splati", (pkg / "notes.py").read_text())
            self.assertEqual(json.loads(Path(d, "o.json").read_text())["totals"]["pool_capacity"], 1)


FAKE_OPT = """#!{py}
import sys
t = open([a for a in sys.argv[1:] if not a.startswith("--")][0]).read()
if "vector.constant " in t:
    sys.stderr.write("program/x.loom:3:5: error [PARSE/006]: unknown op 'vector.constant'\\n")
    sys.exit(1)
"""
FAKE_LINK = """#!{py}
import sys
if "--config=llm.pool=7" in sys.argv:
    sys.stderr.write("error [SYMBOL/013]: symbol '@llm.pool' has incompatible contract field 'range'\\n")
    sys.exit(1)
"""


@unittest.skipIf(sys.platform == "win32", "the fake SDK tools are shell-less python scripts with a shebang")
class Sweep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        bin_ = t / "venv/bin"
        bin_.mkdir(parents=True)
        for n, body in (("loom-opt", FAKE_OPT), ("loom-link", FAKE_LINK)):
            p = bin_ / n
            p.write_text(body.format(py=sys.executable))
            p.chmod(p.stat().st_mode | stat.S_IEXEC)
        self.fleet = t / "fleet"
        for slug, src, pool in (("ok", "func.def public @a.b() {\n}\n", 1), ("old", "%x = vector.constant 1 : vector<4xi32>\n", 1), ("badlink", "func.def public @a() {\n}\n", 7)):
            p = self.fleet / slug
            (p / "program").mkdir(parents=True)
            (p / "program/x.loom").write_text(src)
            card = {"programs": {"source": {"inputs": ["program/x.loom"]}},
                    "variants": {"v": {"services": {"s": {"programs": ["source"], "config": {"llm.pool": pool}}}}}}
            (p / "model.json").write_text(json.dumps(card))

    def tearDown(self):
        self.tmp.cleanup()

    def test_sweep(self):
        out = Path(self.tmp.name, "o.json")
        rc = sweep.main(["--sdk", str(Path(self.tmp.name, "venv")), str(self.fleet), "--json", str(out), "--jobs", "2"])
        self.assertEqual(rc, 1)
        d = json.loads(out.read_text())
        self.assertEqual(d["totals"], {"packages": 3, "parse_failing": 1, "link_failing": 1, "link_skipped": 1, "parse_error_files": 1})
        self.assertEqual(d["first_error_class"], {"PARSE/006 unknown op 'X'": 1, "SYMBOL/013 symbol 'X' has incompatible contract field 'X'": 1})
        self.assertEqual(d["packages"]["ok"]["parse"], [])
        self.assertTrue(d["packages"]["old"]["link_skipped"])
        self.assertIn("badlink", [s for s, r in d["packages"].items() if r["link"]])

    def test_single_package_and_no_link(self):
        rc = sweep.main(["--sdk", str(Path(self.tmp.name, "venv", "bin")), str(self.fleet / "badlink"), "--no-link"])
        self.assertEqual(rc, 0)

    def test_dotted_roots(self):
        self.assertEqual(sweep.ROOTS.findall("func.def public @qwen38.packet_capacity(%a: i32) {\ncommand.def public @go()"), ["qwen38.packet_capacity", "go"])


class Wiring(unittest.TestCase):
    """loom-migrate is wired into the suite like the other skills; every version entry keeps the template's section list."""
    SK = ROOT / "skills/loom-migrate"

    def test_skill_frontmatter(self):
        text = (self.SK / "SKILL.md").read_text()
        head = text.split("---")[1]
        self.assertIn("name: loom-migrate", head)
        self.assertIn("description:", head)
        self.assertLess(len(text.splitlines()), 500)

    def test_listed_in_readme_router_checklist(self):
        self.assertIn("| `loom-migrate` |", (ROOT / "README.md").read_text())
        self.assertIn("`loom-migrate` (", (ROOT / "skills/loom/SKILL.md").read_text())
        self.assertIn("skills/loom-migrate/SKILL.md", (ROOT / "skills/loom-fleet/reference/practices-checklist.md").read_text())

    def test_entries_follow_template(self):
        import re
        heads = lambda p: [re.sub(r"^\d+\. ", "", h) for h in re.findall(r"^## (.+)$", p.read_text(), re.M)]
        want = heads(self.SK / "versions/TEMPLATE.md")
        self.assertEqual(len(want), 12)
        entries = sorted((self.SK / "versions").glob("v*-to-*.md"))
        self.assertTrue(entries)
        for e in entries:
            self.assertEqual(heads(e), want, e.name)

    def test_script_and_link_targets_exist(self):
        import re
        for md in [self.SK / "SKILL.md", *(self.SK / "versions").glob("*.md")]:
            for rel in re.findall(r"\]\((?!http)([^)#]+)\)", md.read_text()):
                self.assertTrue((md.parent / rel).exists(), f"{md.name}: {rel}")
        for s in ("sweep.py", "rewrite_v010_v020.py"):
            self.assertIn(s, (self.SK / "SKILL.md").read_text())
            self.assertTrue((self.SK / "scripts" / s).exists())

    def test_single_package_recipe_commands_exist(self):
        """The entry's section 3a is the self-sufficient recipe: every script it calls is in the repo."""
        import re
        text = (self.SK / "versions/v0.1.0-to-v0.2.0.md").read_text()
        s3 = text[text.index("### 3a."):text.index("### 3b.")]
        for name in set(re.findall(r'\$LOOM_SETUP_DIR/scripts/([\w.]+)', s3)):
            self.assertTrue((ROOT / "skills/loom-setup/scripts" / name).exists(), name)
        for name in set(re.findall(r'\$LOOM_SKILL_DIR/scripts/([\w.]+)', s3)):
            self.assertTrue((self.SK / "scripts" / name).exists(), name)
        for pr in ("#979", "#980", "#981", "#986", "#987", "#990"):
            self.assertIn(pr, text)



class TagOrg(unittest.TestCase):
    """scripts/tag_org.py with a fake Hub: create, skip, refuse a migrated head, report a tag on another commit, survive one repo's error."""

    def run_tag(self, repos, args=()):
        import importlib.util, io, contextlib, json as _json, tempfile
        from unittest import mock
        spec = importlib.util.spec_from_file_location("tag_org", ROOT / "skills/loom-migrate/scripts/tag_org.py")
        t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
        api = mock.Mock(); created = []
        def info(repo, revision=None):
            r = repos[repo]
            if r.get("error"):
                raise OSError("429")
            return mock.Mock(sha=r["tags"].get(revision, r["head"]) if revision else r["head"])
        api.model_info.side_effect = info
        def refs(repo):
            m = mock.Mock(); m.tags = []
            for n in repos[repo]["tags"]:
                x = mock.Mock(); x.name = n; m.tags.append(x)
            return m
        api.list_repo_refs.side_effect = refs
        api.create_tag.side_effect = lambda repo, tag, revision, tag_message: created.append(repo)
        d = tempfile.mkdtemp()
        def dl(repo, name, revision):
            p = os.path.join(d, repo.replace("/", "_") + ".json"); _json.dump({"sdk": {"hrx_loom": repos[repo]["pin"]}} if repos[repo]["pin"] else {}, open(p, "w")); return p
        out = io.StringIO()
        with mock.patch("huggingface_hub.hf_hub_download", dl), contextlib.redirect_stdout(out):
            rc = t.main(["--org", "o", "--tag", "v0.1.0", "--only", *[k.split("/")[1] for k in repos], *args], api=api)
        return rc, out.getvalue(), created

    @unittest.skipUnless(__import__("importlib.util").util.find_spec("huggingface_hub"), "needs huggingface_hub")
    def test_statuses_and_exit_code(self):
        repos = {"o/a": {"head": "h1", "tags": {}, "pin": None}, "o/b": {"head": "h2", "tags": {"v0.1.0": "h2"}, "pin": None},
                 "o/c": {"head": "h3", "tags": {}, "pin": "0.2.0"}, "o/d": {"head": "h4", "tags": {}, "pin": None, "error": True}}
        rc, out, created = self.run_tag(repos)
        self.assertEqual(created, ["o/a"])
        self.assertIn("o/b: exists", out); self.assertIn("o/c: refused", out); self.assertIn("o/d: FAILED", out)
        self.assertIn("summary:", out)
        self.assertEqual(rc, 1)
        rc, out, created = self.run_tag({"o/a": {"head": "h1", "tags": {}, "pin": "0.1.0"}}, ["--dry-run"])
        self.assertEqual((rc, created), (0, []))
        self.assertIn("would-create", out)

if __name__ == "__main__":
    unittest.main()
