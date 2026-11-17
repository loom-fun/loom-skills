"""The kit's migration to hrx-loom 0.2.0: IR spellings, publish_epoch / vision ABI, the pool range, the SDK pin, and the fleet tools reading both
SDK spellings (base = v0.1 programs, head = v0.2)."""
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills" / "loom-port-llm" / "starter"
sys.path[:0] = [str(ROOT / "fleet"), str(STARTER), str(STARTER / "vision")]
import diff_fleet  # noqa: E402
import link_check  # noqa: E402
import loomtext  # noqa: E402
import kit_source  # noqa: E402

V01 = """target.decl @device

kernel.def export("g_k") @g_k(%n: index) {
  %zero = vector.constant 0.0 : vector<8xf32>
  %mask = vector.constant 15 : vector<16xi8>
  %half = vector.constant 1.5 : vector<8xf16>
  %wide = vector.constant 3 : vector<4xi32>
  kernel.return
}

command.program.decl @llm_prefetch(%a: buffer)
command.program.def public @llm_advance() launch(%a: buffer) {
  kernel.launch @g_k[%a](%a) : [index](buffer)
  command.return
}
"""


class SpellingTests(unittest.TestCase):
    def test_upgrade_v01_maps_element_types(self):
        t = loomtext.upgrade_v01(V01)
        self.assertIn("vector.splatf 0.0 : vector<8xf32>", t)
        self.assertIn("vector.splatf 1.5 : vector<8xf16>", t)
        self.assertIn("vector.splati 15 : vector<16xi8>", t)
        self.assertIn("vector.splati 3 : vector<4xi32>", t)
        self.assertIn("command.def public @llm_advance()", t)
        self.assertIn("command.decl @llm_prefetch", t)
        self.assertNotIn("vector.constant", t)
        self.assertNotIn("command.program", t)
        self.assertEqual(loomtext.upgrade_v01(t), t)   # idempotent on v0.2 text

    def test_blocks_read_both_spellings_as_one_key(self):
        a, b = diff_fleet.blocks(V01), diff_fleet.blocks(loomtext.upgrade_v01(V01))
        self.assertIn(("program", "llm_advance", "", 0), a)
        self.assertEqual(a, b)   # a v0.1 base against its v0.2 rewrite is no change
        k1 = [(k, s) for k, s, *_ in loomtext.split_blocks(V01)]
        k2 = [(k, s) for k, s, *_ in loomtext.split_blocks(loomtext.upgrade_v01(V01))]
        self.assertEqual(k1, k2)

    def test_raw_spelling_shows_the_renames(self):
        diff_fleet.RAW_SPELLING = True
        try:
            self.assertNotEqual(diff_fleet.blocks(V01), diff_fleet.blocks(loomtext.upgrade_v01(V01)))
        finally:
            diff_fleet.RAW_SPELLING = False

    def test_link_roots_keep_dotted_names(self):
        self.assertEqual(link_check.ROOTS.findall("func.def public @qwen38.packet_capacity(%a: i64)\ncommand.def public @llm_advance()\n"),
                         ["qwen38.packet_capacity", "llm_advance"])

    def test_no_emitter_writes_a_v01_spelling(self):
        bad = []
        for p in list((ROOT / "skills").rglob("*.py")) + list((ROOT / "skills").rglob("*.loom")) + list((ROOT / "skills").rglob("*.patch")):
            if "qwen4exp-example" in p.as_posix() or "/test/" in p.as_posix() or "skills/loom-migrate/" in p.as_posix():   # the migration skill names the old spellings on purpose
                continue
            t = p.read_text(errors="replace")
            if re.search(r"vector\.constant\b(?!i)|command\.program\.(def|decl|launch)|command\\\.program", t):
                bad.append(p.relative_to(ROOT).as_posix())
        self.assertEqual(bad, [])


class AbiTests(unittest.TestCase):
    def test_publish_epoch_takes_the_control_state_first(self):
        for f in ("templates/control.loom", "templates/service.loom", "draft/build_draft.py"):
            for ln in (STARTER / f).read_text().splitlines():
                if re.search(r"(func\.(def|decl)\b.*@\w*publish_epoch\w*\(%|func\.call @\w*publish_epoch\()", ln):
                    self.assertRegex(ln, r"publish_epoch\w*\(%control_state[,:]", f"{f}: {ln[:100]}")
                    if "func.call" in ln:
                        self.assertIn(": (buffer, buffer, buffer, i32,", ln)   # control_state + plan + media_plan + count

    @unittest.skipUnless(importlib.util.find_spec("gguf") and importlib.util.find_spec("numpy"), "vision generator needs gguf + numpy")
    def test_vision_component_uses_the_finite_batch_abi(self):
        import gen_component
        t = gen_component.control_program({"patch": 14, "merge": 2}, 1024, 256, 0)
        self.assertIn('type.decl import("finite", "batch") @FiniteBatch', t)
        self.assertRegex(t, r"@text_image_prepare_batch_v1\(%state: buffer, %tags: buffer, %batch: ptr<@FiniteBatch>\)")
        self.assertIn("%ranges, %packed = func.call @finite_batch_pack(%batch)", t)
        self.assertRegex(t, r"@text_image_run_v1\(%state: buffer, %stage: i32, %input: hal.buffer, %output: hal.buffer, %batch: ptr<@FiniteBatch>\)")
        self.assertRegex(t, r"@text_image_collect_batch_v1\(%state: buffer, %batch: ptr<@FiniteBatch>\) \{\n\s+func.call @finite_batch_finish\(%batch\)")

    def test_weights_import_carries_a_key_range(self):
        t = (STARTER / "draft" / "prep_weights.py").read_text()
        self.assertIn("@input(%source: i32, %key: buffer, %key_offset: i64, %key_length: i64, %offset: i64, %length: i64, %alignment: i64)", t)
        self.assertIn("@input(%source, %key, %zero, %key_length, %zero, %length, %align)", t)

    def test_gdn_and_short_conv_guard_the_token_too(self):
        for p in (STARTER / "gen_gdn.py", ROOT / "skills/loom-port-llm/reference/lfm2moe-example/gen_sconv.py"):
            t = p.read_text()
            self.assertIn("%has_token{j} = index.cmp uge, %token, %back{j} : index", t, p.name)
            self.assertIn("%has{j} = scalar.andi %has_rel{j}, %has_token{j} : i1", t, p.name)


class PoolTests(unittest.TestCase):
    """The KV pool is the SDK's (--memory_bytes): the kernels' pool range covers every pool it can hand out, and the physical page range follows."""
    def test_pool_range_and_page_assumptions_are_widened_together(self):
        pool_max, page_max = 2147483584, 134217727   # upstream's range(0, 2147483584); pages = pool / 16
        for p in STARTER.rglob("*.py"):
            t = p.read_text()
            self.assertNotRegex(t, r"range\(%pool_arg, 16, 1048576\)", p.name)
            self.assertNotRegex(t, r"@llm\.pool : %value: index where \[range\(%value, 16, 1048576\)\]", p.name)
            self.assertNotRegex(t, r"%bounded_physical = scalar\.assume %physical32 \[range\(%physical32, 0, 1048575\)\]", p.name)
        self.assertIn(f"config.decl @llm.pool : %value: index where [range(%value, 16, {pool_max})]", (STARTER / "gen_llm.py").read_text())
        self.assertIn(f"range(%physical32, 0, {page_max})", (STARTER / "gen_llm.py").read_text())

    def test_card_has_no_pool_capacity(self):
        t = (STARTER / "build.py").read_text()
        self.assertNotRegex(t, r'"pool_capacity"\s*:')


class PinTests(unittest.TestCase):
    def test_kit_pin_schema_matches_sdk_pin(self):
        sys.path.insert(0, str(ROOT / "skills" / "loom-setup" / "scripts"))
        import sdk_pin
        pin = kit_source.SDK_PIN
        self.assertEqual(pin["hrx_loom"], "0.2.0")
        self.assertEqual(pin["source"], "pypi")   # the stable release is on PyPI; sdk_pin.py still reads source "release" (pre-releases)
        self.assertRegex(pin["wheel_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(pin["hrx_system_sha"], r"^[0-9a-f]{40}$")
        self.assertLessEqual(set(pin), {"hrx_loom", "source", "artifact_id", "hrx_system_sha", "wheel_sha256"})
        self.assertTrue(json.dumps(pin))



class GeneratedCommentTests(unittest.TestCase):
    def test_llm_comment_names_the_config_file_not_its_path(self):
        # the comment is line 5 of a dual-target llm.loom (after the target decls): card_lib.programs_sha256 strips it only on line 1, so a
        # path in it made every in-job build's programs hash depend on the runner's work directory
        src = (STARTER / "gen_llm.py").read_text()
        self.assertIn("// Generated by gen_llm.py from {os.path.basename(", src)

if __name__ == "__main__":
    unittest.main()
