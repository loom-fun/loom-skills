"""Cheap proposals for a BF16 NextN block and head (safetensors checkpoints, draft/prep_weights.py): the bf16 row-subset head copy
(PROPOSE_HEAD_ROWS over a BF16 head, source "bf16r"), the MTP block's Q8_0 / Q4_0 copies (MTP_BLOCK_PREP), and what the generated
programs of the Quark MXFP4 card do with them: the verified path (llm_advance / llm_verify, their parameters) is byte-identical with
the choices on and off; only the MTP warm / propose programs and the weight policies change."""
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTER = os.path.join(ROOT, "skills", "loom-port-llm", "starter")
sys.path.insert(0, os.path.join(STARTER, "draft"))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-kb"))
import prep_weights as pw  # noqa: E402
import choices  # noqa: E402

V, B, H = 248320, 248044, 5120
SLUG = "amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom"


class Bf16Rows(unittest.TestCase):
    def test_entry_stages_two_ranges(self):
        """Only the bytes the copy needs are staged: the head rows [0, N) and the added-token rows (0.85 GB of the 2.54 GB head)."""
        src, dst, blocks, (len1, split, zero, off2, len2) = pw.subset_entry("bf16r", "q4_0", 81920, B, V, H)
        self.assertEqual((src, dst, zero), ("bf16r", "q4_0", 0))
        self.assertEqual((len1, off2, len2), (81920 * H * 2, B * H * 2, (V - B) * H * 2))
        self.assertEqual(blocks, (81920 + V - B) * (H // 32))
        self.assertEqual(split, 81920 * H // 32)
        self.assertEqual(blocks - split, len2 // 64)      # one 64-byte bf16 block per output block of the tail
        # N inside the added tokens: everything from N on is the tail range
        _, _, _, (l1, sp, _, o2, l2) = pw.subset_entry("bf16r", "q4_0", B + 10, B, V, H)
        self.assertEqual((l1, o2, l2), ((B + 10) * H * 2, (B + 10) * H * 2, (V - B - 10) * H * 2))

    def test_converter(self):
        lib = pw.library({("bf16r", "q4_0")})
        self.assertIn("g_prep_bf16r_q4_0", lib)
        self.assertIn("llm.prep_split", lib)
        self.assertIn("view<32xbf16>", lib)              # the source is read as 64-byte bf16 blocks
        self.assertNotIn("g_q6_k_values", lib)
        self.assertNotIn("%sb_bytes", lib)               # no superblock arithmetic
        self.assertEqual(lib.count("{"), lib.count("}"))
        self.assertIn("view<32xbf16>", pw.library({("bf16", "q8_0")}))

    def test_policy(self):
        e = {"lm_head.weight": pw.subset_entry("bf16r", "q4_0", 81920, B, V, H),
             "mtp.fc.weight": ("bf16", "q8_0", 10240 * 5120 // 32)}
        t = pw.policy(e, "x")
        self.assertIn(f"scalar.constant {81920 * H * 2} : i64", t)       # the head rows are staged ...
        self.assertIn(f"scalar.constant {(V - B) * H * 2} : i64", t)     # ... and the added-token rows as a second range
        self.assertEqual(t.count("func.call @input("), 2)
        self.assertIn("scalar.constant 34 : i64", t)                      # the block's matrices are Q8_0
        self.assertEqual(t.count("{"), t.count("}"))

    def test_registry(self):
        c = choices.CHOICES["draft.propose.block"]
        self.assertEqual((c.legacy, c.env, c.type), (None, "MTP_BLOCK_PREP", "str"))


class Programs(unittest.TestCase):
    """The Quark MXFP4 card built from its fleet entry (CPU only), choices off and on."""

    @classmethod
    def setUpClass(cls):
        try:
            import gguf, jinja2  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("fleet builds need gguf and jinja2")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dirs = {}
        # off / q8 / q4: the card's own trunk (int8-activation MXFP4 matvecs); t_off / t_q8: a trunk without them (MXFP4S_Q8=0), where the
        # int8 matvecs exist in the MTP programs alone
        for tag, env in (("off", {"PROPOSE_HEAD_PIECES": "0", "MTP_BLOCK_PREP": "", "PROPOSE_HEAD_ROWS": "0"}),
                         ("q8", {"PROPOSE_HEAD_PIECES": "0", "MTP_BLOCK_PREP": "q8_0", "PROPOSE_HEAD_ROWS": "81920"}),
                         ("q4", {"PROPOSE_HEAD_PIECES": "0", "MTP_BLOCK_PREP": "q4_0", "PROPOSE_HEAD_ROWS": "49152"}),
                         ("t_off", {"PROPOSE_HEAD_PIECES": "0", "MTP_BLOCK_PREP": "", "PROPOSE_HEAD_ROWS": "0", "MXFP4S_Q8": "0"}),
                         ("t_q8", {"PROPOSE_HEAD_PIECES": "0", "MTP_BLOCK_PREP": "q8_0", "PROPOSE_HEAD_ROWS": "81920", "MXFP4S_Q8": "0"})):
            out = os.path.join(cls.tmp.name, tag)
            p = subprocess.run([sys.executable, os.path.join(ROOT, "fleet", "build_fleet.py"), "--tree", ROOT, "--out", out, "--slugs", SLUG],
                               capture_output=True, text=True, env=dict(os.environ, **env))
            if p.returncode:
                raise AssertionError(p.stdout + p.stderr)
            cls.dirs[tag] = os.path.join(out, SLUG)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def llm(self, tag):
        return open(os.path.join(self.dirs[tag], "program", "llm.loom")).read()

    @staticmethod
    def program(text, name):
        i = text.index("@" + name + "()")
        return text[i:text.index("\n}\n", i)]

    def test_verified_path_is_unchanged(self):
        for base, tags in (("off", ("q8", "q4")), ("t_off", ("t_q8",))):
            off = self.llm(base)
            for tag in tags:
                on = self.llm(tag)
                for name in ("llm_advance", "llm_verify"):
                    self.assertEqual(self.program(off, name), self.program(on, name), f"{name} ({tag})")
        # the verified head is read as stored
        self.assertIn('"lm_head.weight" : view<2542796800xi8>', self.program(self.llm("q8"), "llm_verify"))

    def test_proposals_read_the_copies(self):
        p = self.program(self.llm("q8"), "llm_mtp_propose")
        self.assertIn('%w_phead, "lm_head.weight" : view<236724480xi8>', p)          # (81920 + 276) rows x 5120 / 32 x 18 bytes
        self.assertIn('"mtp.layers.0.mlp.down_proj.weight" : view<94699520xi8>', p)    # 17408 x 5120 / 32 x 34 bytes
        self.assertIn("g_head_select_p", p)
        self.assertNotIn("g_mv_bf16", p)
        q = self.program(self.llm("q4"), "llm_mtp_propose")
        self.assertIn('"mtp.layers.0.mlp.down_proj.weight" : view<50135040xi8>', q)    # 18 bytes per 32 values
        off = self.program(self.llm("off"), "llm_mtp_propose")
        self.assertIn("g_mv_bf16", off)
        self.assertIn('"lm_head.weight" : view<2542796800xi8>', off)

    def test_policies(self):
        w = open(os.path.join(self.dirs["q8"], "weights.loom")).read()
        for hf in ("mtp.fc.weight", "mtp.layers.0.mlp.down_proj.weight", "mtp.layers.0.self_attn.q_proj.weight"):
            self.assertIn(f"// \"{hf}\"", w)
        self.assertNotIn("model.language_model.layers.0.mlp", w)                      # the trunk's MXFP4 weights are not prepared
        self.assertTrue(os.path.exists(os.path.join(self.dirs["q8"], "head_weights.loom")))
        self.assertFalse(os.path.exists(os.path.join(self.dirs["off"], "head_weights.loom")))
        off_w = open(os.path.join(self.dirs["off"], "weights.loom")).read()
        self.assertNotIn("mtp.layers.0.mlp.down_proj.weight\"\nglobal", off_w.replace("// \"mtp.layers.0.mlp.down_proj.weight\"\nglobal", "x"))
        full = self.llm("q8")
        self.assertEqual(full.count("config.decl @llm.prep_blocks"), 1)               # one declaration for the head's and the block's converters
        self.assertEqual(full.count('kernel.def export("g_prep_bf16_q8_0")'), 1)

    def test_no_int8_in_a_trunk_without_it(self):
        trunk = re.sub(r"command\.def public (?:target\(@\w+\) )?@llm_mtp_(?:warm|propose)\(\).*?\n\}\n", "", self.llm("t_q8"), flags=re.S)
        launches = re.findall(r"kernel\.launch @(g_(?:mv|quant)\w*q8\w*)\[", trunk)
        self.assertEqual(launches, [], "int8 activations in the trunk")


if __name__ == "__main__":
    unittest.main()
