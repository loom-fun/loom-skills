"""Image input for a safetensors checkpoint: the vision tower is read from the checkpoint's own bf16 tensors (model.visual.*).

The tower of amd/Qwen3.8-27B-Quark-AWQ-MXFP4 is the model's own tower, the same one llama.cpp's converter wrote into the bartowski
mmproj; these tests hold the safetensors path to it: the name map (hf_names/qwen35.json "vision") covers exactly the mmproj's
tensors with the same shapes (from the fleet header dumps, stdlib only), the encoder config derived from config.json equals the one
derived from the mmproj metadata, the generated image programs read HF keys (bf16 matrices as stored, f32 prepared copies of the small
tensors through the weight policy), and, when both files are in the HF cache, the checkpoint's tensors are the mmproj's values (up to
f16 rounding of the matrices) and the numpy reference encoder (ref_vit.py) gives the same rows from either."""
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:   # the CI python has no numpy: the tests below that need it skip
    np = None
try:
    import gguf
except ImportError:
    gguf = None

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vision"))
import inspect_safetensors as I  # noqa: E402
import st_source  # noqa: E402

SLUG = "amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom"
DUMP = ROOT / "fleet" / SLUG / "safetensors/model.safetensors.json.gz"
MMPROJ_DUMP = ROOT / "fleet/bartowski-Qwen3.8-27B-loom/gguf/mmproj-Qwen3.8-27B-f16.gguf.json.gz"
HF_CACHE = Path(os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")))


def cached(repo, pattern):
    d = HF_CACHE / f"models--{repo.replace('/', '--')}/snapshots"
    return next(iter(sorted(d.glob(pattern))), None) if d.exists() else None


REAL = cached("amd/Qwen3.8-27B-Quark-AWQ-MXFP4", "*/model.safetensors")
MMPROJ = cached("bartowski/Qwen3.8-27B-GGUF", "*/mmproj-Qwen3.8-27B-f16.gguf")


def load_cfg():
    src = I.load_source(str(DUMP))
    return src, I.build_config(src, drop_chat=True)


class VisionBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src, cls.cfg = load_cfg()
        cls.v = cls.cfg["vision"]

    def test_names_and_shapes_are_the_mmproj_converters(self):
        d = json.load(gzip.open(MMPROJ_DUMP, "rt"))
        ref = {n: shape for n, shape, _, _ in d["tensors"]}
        got = self.v["tensors"]
        expect = set(ref) - {"v.patch_embd.weight.1"}   # both temporal taps are the one conv3d tensor here
        self.assertEqual(set(got), expect)
        for n, t in got.items():
            self.assertEqual(t["shape"], ref[n], n)   # GGUF order: the HF shape reversed (the patch weight keeps the converter's per-tap shape)

    def test_matrices_as_stored_small_tensors_prepared(self):
        for n, t in self.v["tensors"].items():
            hf = self.src.header[t["hf"]]
            self.assertEqual(hf["dtype"], "BF16", n)
            if t.get("prep"):
                self.assertEqual((t["type"], t["bytes"], t["prep"]), ("F32", 4 * hf["bytes"] // 2, "copy"), n)
                self.assertTrue(n.endswith(("bias", "ln1.weight", "ln2.weight", "post_ln.weight")) or n == "v.position_embd.weight", n)
            else:
                self.assertEqual((t["type"], t["bytes"]), ("BF16", hf["bytes"]), n)
                self.assertTrue(n.endswith("weight") and (n.startswith(("mm.", "v.blk.")) or n == "v.patch_embd.weight"), n)
        self.assertEqual(self.v["tensors"]["v.patch_embd.weight"]["bytes"], 2 * 1152 * 3 * 2 * 16 * 16)   # both taps
        self.assertEqual(sum(self.v["tensors"][n]["bytes"] for n in self.v["tensors"]) > 0.9e9, True)

    def test_config_and_markers(self):
        c = self.v["config"]
        self.assertEqual((c["patch"], c["E"], c["NH"], c["D"], c["ffn"], c["layers"], c["merge"], c["out"], c["pos_n"], c["act"], c["fmt"]),
                         (16, 1152, 16, 72, 4304, 27, 2, 5120, 48, "gelu_tanh", "bf16"))
        self.assertEqual((self.v["family"], self.v["img_begin"], self.v["img_end"], c["mean"], c["std"]),
                         ("qwen3vl_merger", "<|vision_start|>", "<|vision_end|>", [0.5] * 3, [0.5] * 3))
        self.assertEqual(self.cfg["mrope"], {"sections": [11, 11, 10, 0], "interleaved": True})   # the GGUF's rope.dimension_sections
        self.assertEqual(c["deepstack"], [])

    def test_prepared_keys_are_the_policy_entries(self):
        prep = st_source.prepared(self.v)
        self.assertEqual(len(prep), 27 * 8 + 4 + 2)   # per block: 4 biases + 4 norm tensors; merger: 2 biases + norm w/b; patch bias, positions
        self.assertTrue(all(k.startswith("model.visual.") for k in prep))
        self.assertEqual(prep["model.visual.pos_embed.weight"], 2304 * 1152)

    def test_text_tensors_untouched(self):
        self.assertFalse(any(t.get("hf", "").startswith("model.visual.") for t in self.cfg["tensors"].values()))

    def test_mismatches_stop_with_the_reason(self):
        vmap = I.family_for(self.src.config)["vision"]
        bad = json.loads(json.dumps(self.src.config))
        bad["vision_config"]["temporal_patch_size"] = 1
        with self.assertRaises(SystemExit) as e:
            st_source.block(bad, self.src.header, vmap)
        self.assertIn("temporal_patch_size", str(e.exception))
        h = dict(self.src.header)
        h.pop("model.visual.merger.linear_fc2.bias")
        with self.assertRaises(SystemExit) as e:
            st_source.block(self.src.config, h, vmap)
        self.assertIn("merger.linear_fc2.bias", str(e.exception))
        bad = json.loads(json.dumps(self.src.config))
        bad["vision_config"]["hidden_act"] = "relu"
        with self.assertRaises(SystemExit):
            st_source.block(bad, self.src.header, vmap)

    def test_preprocessor_config_sets_mean_std(self):
        vmap = I.family_for(self.src.config)["vision"]
        v = st_source.block(self.src.config, self.src.header, vmap, {"image_mean": [0.48, 0.46, 0.41], "image_std": [0.27, 0.26, 0.28]})
        self.assertEqual(v["config"]["mean"], [0.48, 0.46, 0.41])
        with self.assertRaises(SystemExit):
            st_source.block(self.src.config, self.src.header, vmap, {"patch_size": 14})


@unittest.skipUnless(gguf, "gguf")
class ProgramsAndConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import encoder as EN
        import gen_component as GC
        cls.EN, cls.GC = EN, GC
        cls.src, cls.cfg = load_cfg()
        cls.c = EN.config_st(cls.cfg["vision"])

    def test_config_equals_the_mmproj_metadata_config(self):
        sys.path.insert(0, str(ROOT / "fleet"))
        import gguf_stub
        d = gguf_stub.load(str(MMPROJ_DUMP))
        with tempfile.TemporaryDirectory() as t:
            p = gguf_stub.write_stub(d, os.path.join(t, "mmproj.gguf"))
            g = self.EN.config(p)
        for k in ("family", "patch", "E", "NH", "D", "ffn", "layers", "out", "merge", "pos_n", "act", "deepstack", "out_row", "mean", "std"):
            self.assertEqual(self.c[k], g[k], k)
        self.assertAlmostEqual(self.c["eps"], g["eps"], places=9)
        self.assertEqual(self.c["fmt"], "bf16")   # llama.cpp stores f16: the checkpoint's own values are read as bf16
        self.assertEqual(g["fmt"], "f16")

    def test_plan_reads_the_hf3d_patch_embedding_once(self):
        p = self.EN.build(self.c)
        self.assertEqual([n for n in p.kernels if n.startswith("v_patch")], ["v_patch_16_1152_bf16_block2_dual_hf3d_bias"])
        pe = [o for o in p.ops if o[0].startswith("v_patch")]
        self.assertEqual(len(pe), 1)
        self.assertEqual([a for a in pe[0][2] if a[0] == "W"], [("W", "v.patch_embd.weight"), ("W", "v.patch_embd.bias")])

    def test_encoder_program_reads_hf_keys(self):
        text = self.GC.encoder_program(self.c)
        keys = [l.split('"')[1] for l in text.splitlines() if "command.parameter %weights" in l]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys), len(self.c["tensors"]))
        self.assertTrue(all(k.startswith("model.visual.") for k in keys), [k for k in keys if not k.startswith("model.visual.")][:3])
        for line in text.splitlines():
            if "command.parameter %weights" in line:
                key = line.split('"')[1]
                t = next(t for t in self.c["tensors"].values() if t["hf"] == key)
                self.assertIn(f"view<{t['bytes']}xi8>", line)

    def test_prepare_program_binds_the_vision_policy(self):
        text = self.GC.prepare_program(self.c, (512, 512), self.c["out_row"], 1024)
        self.assertIn("vision_weights.loom".encode().hex(), text)

    def test_gguf_programs_unchanged(self):
        # the GGUF path (no "hf" map, no hf3d layout) keeps its policy name and parameter names
        d = json.load(gzip.open(MMPROJ_DUMP, "rt"))
        self.assertTrue(all(not n.startswith("model.visual.") for n, *_ in d["tensors"]))
        import gen_vision as G
        name, text = G.patch_embed(16, 1152, "f16", "block2", True, True)
        self.assertEqual(name, "v_patch_16_1152_f16_block2_dual_bias")
        self.assertIn("%w2: buffer", text)
        self.assertNotIn("hf3d", text)


@unittest.skipUnless(np and gguf and REAL and MMPROJ, "numpy, gguf, the checkpoint and the bartowski mmproj in the HF cache")
class RealTower(unittest.TestCase):
    """The checkpoint's tower is the model's tower: the same values as the converter's mmproj, the same encoder output."""

    @classmethod
    def setUpClass(cls):
        import ref_vit as R
        import st_load
        cls.R = R
        _, cfg = load_cfg()
        cls.W, cls.KV = st_load.load([str(REAL)], cfg["vision"])
        cls.Wg, cls.KVg = R.load(str(MMPROJ))

    def test_tensors_are_the_mmproj_values(self):
        self.assertEqual(sorted(self.W), sorted(self.Wg))
        for n, g in self.Wg.items():
            a = self.W[n]
            self.assertEqual(a.shape, g.shape, n)
            # matrices: bf16 values (checkpoint) vs the same values rounded to f16 by llama.cpp's converter (differences ~1e-8)
            np.testing.assert_allclose(a, g, atol=1e-6, rtol=0, err_msg=n)

    def test_reference_encoder_rows_agree(self):
        H = Wd = 96
        yy, xx = np.mgrid[0:H, 0:Wd]
        img = np.stack([(127 + 120 * np.sin(xx / 7.0)), (127 + 120 * np.cos(yy / 5.0)), ((xx + yy) % 256)], -1).astype(np.uint8).astype(np.float32) / 255.0
        a = self.R.qwen_vl(self.W, self.KV, img)
        b = self.R.qwen_vl(self.Wg, self.KVg, img)
        self.assertEqual(a.shape, b.shape)
        cos = [float(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)) for x, y in zip(a, b)]
        self.assertGreater(min(cos), 0.99999)


if __name__ == "__main__":
    unittest.main()
