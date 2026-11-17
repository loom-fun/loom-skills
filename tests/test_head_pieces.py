"""Heads read as several load-time prepared tensors (gen_llm.HEAD_PIECES, draft/prep_weights.piece_entries): the proposal head's Q4_0 copy
(draft.propose.head_pieces) and the verified head's Q8_0 copy (decode.head.verify) of a BF16 safetensors head. hrx-loom stages a prepared
tensor's input whole through four lanes, so pieces bound the load-time staging by the piece, not the copy. The piece entries cover exactly
the copied rows, every head launch becomes one launch per piece over its own parameter, and nothing else of the programs changes."""
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


class Entries(unittest.TestCase):
    def check_cover(self, N, P, dst="q4_0"):
        """The pieces' source ranges are exactly the rows subset_entry copies, in order, and each stages at most P rows."""
        e, sizes = pw.piece_entries(dst, "lm_head.weight", N, B, V, H, P, "k#p")
        n, tail, rows = pw.subset_rows(N, B, V)
        self.assertEqual(sum(sizes), rows)
        self.assertTrue(all(s == P for s in sizes[:-1]) and 0 < sizes[-1] <= P)
        src_rows = []
        for k, s in enumerate(sizes):
            src, d, blocks, (l1, split, z, o2, l2, o1, key) = e[f"k#p{k}"]
            self.assertEqual((src, d, z, key), ("bf16r", dst, 0, "lm_head.weight"))
            self.assertEqual(blocks, s * H // 32)
            self.assertEqual(split, l1 // 64)                 # one 64-byte bf16 block per output block of the first range
            self.assertEqual(l1 + l2, s * H * 2)              # staged bytes: the piece's rows only
            src_rows += list(range(o1 // (2 * H), (o1 + l1) // (2 * H))) + (list(range(o2 // (2 * H), (o2 + l2) // (2 * H))) if l2 else [])
        self.assertEqual(src_rows, list(range(n)) + list(range(tail, V)))
        return e, sizes

    def test_cover(self):
        self.check_cover(65536, 16384)
        self.check_cover(60000, 16384)                        # a piece straddles the head rows and the added-token rows
        self.check_cover(V, 16384, "q8_0")                    # the whole head (the verified copy)
        _, s = self.check_cover(10 ** 6, 8192)                # more rows than the vocabulary: every row
        self.assertEqual(sum(s), V)

    def test_policy(self):
        e, sizes = pw.piece_entries("q4_0", "lm_head.weight", 65536, B, V, H, 16384, "lm_head.weight#p")
        t = pw.policy(e, "x")
        self.assertIn('// source "lm_head.weight"', t)                         # the pieces read the head under its own key
        self.assertIn("%off1_", t)
        self.assertIn(f"scalar.constant {16384 * H * 2} : i64", t)          # a piece stages 16384 rows
        self.assertNotIn(f"scalar.constant {65536 * H * 2} : i64", t)
        self.assertEqual(t.count("{"), t.count("}"))
        names = re.findall(r"^\s*(%\w+) = ", t, re.M)
        self.assertEqual(len(names), len(set(names)), "SSA names are unique (many entries)")
        # a policy without pieces is unchanged by the piece support
        one = {"lm_head.weight": pw.subset_entry("bf16r", "q4_0", 49152, B, V, H)}
        self.assertNotIn("%off1_", pw.policy(one, "x"))

    def test_library_dedup(self):
        a = pw.library({("bf16r", "q4_0")})
        b = pw.library({("bf16r", "q4_0"), ("bf16r", "q8_0")}, have=a)
        self.assertNotIn('export("g_prep_bf16r_q4_0")', b)
        self.assertIn('export("g_prep_bf16r_q8_0")', b)
        self.assertNotIn("config.decl", b)

    def test_registry(self):
        self.assertEqual(choices.CHOICES["decode.head.verify"].env, "VERIFY_HEAD_PREP")
        self.assertEqual(choices.CHOICES["draft.propose.head_pieces"].env, "PROPOSE_HEAD_PIECES")
        self.assertEqual(choices.CHOICES["decode.head.piece_rows"].legacy, 16384)


class Programs(unittest.TestCase):
    """The Quark MXFP4 card built from its fleet entry (CPU only): the node's config, proposal pieces, the verified Q8_0 head."""

    @classmethod
    def setUpClass(cls):
        try:
            import gguf, jinja2  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("fleet builds need gguf and jinja2")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dirs = {}
        for tag, env in (("off", {"PROPOSE_HEAD_PIECES": "0"}),
                         ("pieces", {"PROPOSE_HEAD_PIECES": "1", "PROPOSE_HEAD_ROWS": "65536"}),
                         ("verify", {"VERIFY_HEAD_PREP": "q8_0"})):
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

    def test_proposal_pieces(self):
        off, on = self.llm("off"), self.llm("pieces")
        for name in ("llm_advance", "llm_verify", "llm_mtp_warm"):
            self.assertEqual(self.program(off, name), self.program(on, name), name)
        p = self.program(on, "llm_mtp_propose")
        rows = [16384] * 4 + [V - B]
        for k, n in enumerate(rows):
            self.assertIn(f'%head_p{k} = command.parameter %w_phead, "lm_head.weight#p{k}" : view<{n * H // 32 * 18}xi8>', p)
        self.assertNotIn('"lm_head.weight" :', p)
        self.assertEqual(len(re.findall(r"@g_head_mv_t4_q8_p40_pc128\[", p)), 4)
        self.assertEqual(len(re.findall(r"@g_head_mv_t4_q8_p40_pc3\[", p)), 1)
        self.assertIn("%head_c4 = index.constant 512 : index", p)
        self.assertIn("g_head_select_p", p)
        # one load-time preparation plan: the main weight policy prepares the pieces, the proposals' head root is bound with it
        self.assertFalse(os.path.exists(os.path.join(self.dirs["pieces"], "head_weights.loom")))
        w = open(os.path.join(self.dirs["pieces"], "weights.loom")).read()
        self.assertEqual(len(re.findall(r'^// "lm_head.weight#p\d+"$', w, re.M)), 5)
        prep = open(os.path.join(self.dirs["pieces"], "program", "prepare.loom")).read()
        self.assertNotIn("%head_policy", prep)
        self.assertIn("func.call @parameter(%pstage, %pstage_b2, %c0_source, %checkpoint, %policy)", prep)
        for k in ("g_head_mv_q8_p40_pc128", "g_head_mv_t16_q8_p40_pc3"):
            self.assertIn(f'kernel.def export("{k}")', on)

    def test_verified_q8(self):
        on = self.llm("verify")
        for name in ("llm_advance", "llm_verify"):
            p = self.program(on, name)
            self.assertNotIn('"lm_head.weight"', p)                    # the BF16 head is no longer a parameter (not resident)
            self.assertIn('"lm_head.weight#v15" : view<13926400xi8>', p)  # 2560 rows x 5120 / 32 x 34 bytes
            self.assertNotIn("@g_head_mv[", p)
            self.assertIn("@g_head_mv_t4_q8_pc128[", p)
        w = open(os.path.join(self.dirs["verify"], "weights.loom")).read()
        self.assertEqual(len(re.findall(r'^// "lm_head.weight#v\d+"$', w, re.M)), 16)
        self.assertIn('// source "lm_head.weight"', w)
        self.assertEqual(on.count("config.decl @llm.prep_split"), 1)
        self.assertEqual(on.count('kernel.def export("g_prep_bf16r_q8_0")'), 1)
        # the proposals read their own pieced Q4_0 copy (the model node: draft.propose.head_pieces)
        self.assertIn('%w_phead, "lm_head.weight#p0" :', self.program(on, "llm_mtp_propose"))


if __name__ == "__main__":
    unittest.main()
