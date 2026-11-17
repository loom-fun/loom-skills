"""The safetensors source path: header parse, the family name map (hf_names/qwen35.json), the Quark producer convention
(hf_names/quark.json), the mxfp4s weight format (nibble plane + E8M0 scale plane) against an independent reimplementation of
vLLM's decode (and, when the checkpoint is in the HF cache, on real rows), and the generated programs of a tiny Qwen3.5-style
checkpoint (HF keys in command.parameter, scale operands, the weight policy that prepares bf16 norms / A_log / conv1d)."""
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import math

try:
    import numpy as np
except ImportError:   # the CI python has no numpy: the tests below that need it skip
    np = None

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))
import inspect_safetensors as I  # noqa: E402
st_tensors = __import__("st_tensors") if np else None   # numpy-based reader

HF_CACHE = Path(os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")))
REAL = next(iter(sorted((HF_CACHE / "models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots").glob("*/model.safetensors"))), None) \
    if (HF_CACHE / "models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots").exists() else None

HID, HEADS, KVH, HD, FFN, VOCAB = 128, 2, 1, 256, 256, 512
GDN = dict(linear_conv_kernel_dim=4, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32)
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "quantization_config": {"quant_method": "quark", "exclude": ["lm_head", "mtp.fc"], "global_quant_config": {}},
    "text_config": dict(hidden_size=HID, num_hidden_layers=4, num_attention_heads=HEADS, num_key_value_heads=KVH, head_dim=HD,
                        intermediate_size=FFN, max_position_embeddings=4096, rms_norm_eps=1e-6, layer_types=LAYER_TYPES,
                        mtp_num_hidden_layers=1, rope_parameters={"rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                                                                  "mrope_interleaved": True}, **GDN),
}
C = 2 * 2 * 32 + 4 * 32   # gated delta-net channels (q, k, v)


def tiny_header():
    """The tensors of a 4-layer Qwen3.5-style checkpoint with its MTP block, shapes only."""
    h = {}

    def bf(name, *shape):
        h[name] = {"dtype": "BF16", "shape": list(shape), "bytes": 2 * math.prod(shape)}

    def q(stem, rows, k):   # a Quark MXFP4 linear
        h[stem + ".weight"] = {"dtype": "U8", "shape": [rows, k // 2], "bytes": rows * k // 2}
        h[stem + ".weight_scale"] = {"dtype": "U8", "shape": [rows, k // 32], "bytes": rows * k // 32}

    bf("model.language_model.embed_tokens.weight", VOCAB, HID)
    bf("lm_head.weight", VOCAB, HID)
    bf("model.language_model.norm.weight", HID)
    for i, kind in enumerate(LAYER_TYPES):
        p = f"model.language_model.layers.{i}."
        bf(p + "input_layernorm.weight", HID)
        bf(p + "post_attention_layernorm.weight", HID)
        q(p + "mlp.gate_proj", FFN, HID); q(p + "mlp.up_proj", FFN, HID); q(p + "mlp.down_proj", HID, FFN)
        if kind == "full_attention":
            q(p + "self_attn.q_proj", HEADS * 2 * HD, HID); q(p + "self_attn.k_proj", KVH * HD, HID); q(p + "self_attn.v_proj", KVH * HD, HID)
            q(p + "self_attn.o_proj", HID, HEADS * HD)
            bf(p + "self_attn.q_norm.weight", HD); bf(p + "self_attn.k_norm.weight", HD)
        else:
            q(p + "linear_attn.in_proj_qkv", C, HID); q(p + "linear_attn.in_proj_z", 4 * 32, HID)
            q(p + "linear_attn.in_proj_a", 4, HID); q(p + "linear_attn.in_proj_b", 4, HID); q(p + "linear_attn.out_proj", HID, 4 * 32)
            bf(p + "linear_attn.conv1d.weight", C, 1, 4); bf(p + "linear_attn.A_log", 4); bf(p + "linear_attn.dt_bias", 4)
            bf(p + "linear_attn.norm.weight", 32)
    bf("mtp.fc.weight", HID, 2 * HID); bf("mtp.norm.weight", HID); bf("mtp.pre_fc_norm_embedding.weight", HID); bf("mtp.pre_fc_norm_hidden.weight", HID)
    p = "mtp.layers.0."
    bf(p + "input_layernorm.weight", HID); bf(p + "post_attention_layernorm.weight", HID)
    bf(p + "mlp.gate_proj.weight", FFN, HID); bf(p + "mlp.up_proj.weight", FFN, HID); bf(p + "mlp.down_proj.weight", HID, FFN)
    bf(p + "self_attn.q_proj.weight", HEADS * 2 * HD, HID); bf(p + "self_attn.k_proj.weight", KVH * HD, HID); bf(p + "self_attn.v_proj.weight", KVH * HD, HID)
    bf(p + "self_attn.o_proj.weight", HID, HEADS * HD); bf(p + "self_attn.q_norm.weight", HD); bf(p + "self_attn.k_norm.weight", HD)
    bf("model.visual.patch_embed.proj.weight", 8, 3)
    return h


def tiny_source():
    tok = {"eos_token": "<|im_end|>", "add_bos_token": False}
    added = [{"content": "<|endoftext|>", "id": 500, "special": True}, {"content": "<|im_end|>", "id": 501, "special": True}]
    return I.Source(tiny_header(), CONFIG, {"eos_token_id": [501, 500]}, tok, "", added,
                    {"kind": "safetensors", "repo": "x/Tiny-Quark-MXFP4", "revision": "0" * 40}, {"model.safetensors": 12345})


def scalar_dequant(nibbles, scales):
    """vLLM mxfp4_w4a8_gemv_rdna3.cu, element by element: kE2M1x2 table (2 * e2m1), low nibble = even k, scale 2^(e - 127)."""
    tab = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
    rows, half = nibbles.shape
    out = np.zeros((rows, half * 2), np.float32)
    for r in range(rows):
        for k in range(half * 2):
            byte = int(nibbles[r, k // 2])
            code = (byte >> 4) & 15 if k & 1 else byte & 15
            out[r, k] = tab[code] * 0.5 * 2.0 ** (int(scales[r, k // 32]) - 127)
    return out


class Header(unittest.TestCase):
    def test_parse_header(self):
        meta = {"a": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6]}, "__metadata__": {"format": "pt"}}
        raw = json.dumps(meta).encode()
        blob = struct.pack("<Q", len(raw)) + raw + bytes(6)
        h = I.parse_header(lambda o, n: blob[o:o + n])
        self.assertEqual(h, {"a": {"dtype": "U8", "shape": [2, 3], "bytes": 6}})

    def test_dump_roundtrip(self):
        s = tiny_source()
        s2 = I.Source.from_dump(json.loads(json.dumps(s.dump())))
        self.assertEqual(I.build_config(s, drop_chat=True), I.build_config(s2, drop_chat=True))


class Config(unittest.TestCase):
    def setUp(self):
        self.cfg = I.build_config(tiny_source(), ["/ck/model.safetensors"], drop_chat=True)

    def test_geometry_matches_the_gguf_inspector(self):
        c = self.cfg
        self.assertEqual((c["arch"], c["hidden"], c["layers"], c["heads"], c["kv_heads"], c["head_dim"], c["ffn"], c["vocab"]),
                         ("qwen35", HID, 4, HEADS, KVH, HD, FFN, VOCAB))
        self.assertEqual(c["rope_dims"], 64)
        self.assertEqual(c["recurrent_layers"], [0, 1, 2])
        self.assertEqual(c["kv_layers"], 1)
        self.assertEqual(c["attn_gate"], "fused_q")
        self.assertEqual(c["ffn_norm_tensor"], "post_attention_norm")
        self.assertEqual(c["mtp"], {"layers": [4]})
        self.assertEqual((c["eos_id"], c["extra_eos_ids"]), (501, [500]))
        self.assertEqual(c["gdn"], {"conv_kernel": 4, "k_heads": 2, "v_heads": 4, "head_k": 32, "head_v": 32, "inner": 128, "v_order": "grouped"})
        self.assertEqual(c["weights"], {"format": "safetensors", "files": [os.path.abspath("/ck/model.safetensors")]})
        self.assertNotIn("mrope", c)   # image positions only: wired with the image component
        self.assertEqual(c["source"]["kind"], "safetensors")   # the keys loom-kb routes on: source, quark (the checkpoint's quantization_config), formats
        self.assertEqual(c["quark"], CONFIG["quantization_config"])
        self.assertEqual(c["formats"], ["bf16", "f32", "mxfp4s"])

    def test_tensor_table(self):
        t = self.cfg["tensors"]
        q = t["blk.3.attn_q.weight"]
        self.assertEqual((q["type"], q["shape"], q["hf"], q["hf_scale"]),
                         ("mxfp4s", [HID, HEADS * 2 * HD], "model.language_model.layers.3.self_attn.q_proj.weight",
                          "model.language_model.layers.3.self_attn.q_proj.weight_scale"))
        self.assertEqual(t["blk.0.ssm_alpha.weight"]["type"], "mxfp4s")   # the quantized in_proj_a / in_proj_b
        self.assertEqual(t["token_embd.weight"]["type"], "bf16")
        self.assertEqual(t["output.weight"]["hf"], "lm_head.weight")
        # excluded layers stay bf16 matrices: the MTP block and its projection
        self.assertEqual({t[f"blk.4.{n}.weight"]["type"] for n in ("attn_q", "attn_k", "ffn_gate", "ffn_down", "nextn.eh_proj")}, {"bf16"})
        self.assertEqual(t["blk.4.nextn.eh_proj.weight"]["shape"], [2 * HID, HID])
        self.assertEqual(t["blk.4.nextn.shared_head_norm.weight"]["hf"], "mtp.norm.weight")

    def test_prepared_tensors(self):
        t = self.cfg["tensors"]
        prep = lambda n: (t[n]["type"], t[n]["prep"]["op"], t[n]["shape"])
        self.assertEqual(prep("blk.0.attn_norm.weight"), ("f32", "add1", [HID]))
        self.assertEqual(prep("output_norm.weight"), ("f32", "add1", [HID]))
        self.assertEqual(prep("blk.3.attn_q_norm.weight"), ("f32", "add1", [HD]))
        self.assertEqual(prep("blk.0.ssm_a"), ("f32", "negexp", [4]))
        self.assertEqual(prep("blk.0.ssm_dt.bias"), ("f32", "copy", [4]))
        self.assertEqual(prep("blk.0.ssm_norm.weight"), ("f32", "copy", [32]))    # the gated norm is not zero-centred
        self.assertEqual(prep("blk.0.ssm_conv1d.weight"), ("f32", "copy", [4, C]))
        self.assertEqual(prep("blk.4.nextn.enorm.weight"), ("f32", "add1", [HID]))

    def test_every_text_tensor_is_mapped(self):
        used = {e[k] for e in self.cfg["tensors"].values() for k in ("hf", "hf_scale") if k in e}
        left = [k for k in tiny_header() if k not in used and not k.startswith("model.visual.")]
        self.assertEqual(left, [])

    def test_quantized_but_excluded_is_refused(self):
        src = tiny_source()
        src.config = json.loads(json.dumps(CONFIG))
        src.config["quantization_config"]["exclude"].append("model.language_model.layers.0.mlp.gate_proj")
        with self.assertRaises(SystemExit):
            I.build_config(src, drop_chat=True)

    def test_scale_shape_checked(self):
        src = tiny_source()
        src.header["model.language_model.layers.0.mlp.up_proj.weight_scale"]["shape"] = [FFN, 3]
        with self.assertRaises(SystemExit):
            I.build_config(src, drop_chat=True)

    def test_unknown_family_and_producer(self):
        with self.assertRaises(SystemExit):
            I.family_for({"architectures": ["NoSuchModel"]})
        with self.assertRaises(SystemExit):
            I.producer_for({"quantization_config": {"quant_method": "nosuch"}})


@unittest.skipUnless(np, "numpy")
class Mxfp4s(unittest.TestCase):
    def test_dequant_matches_vllm_reference_on_random_planes(self):
        rng = np.random.default_rng(1)
        nib = rng.integers(0, 256, (7, 96), dtype=np.uint8)
        sc = rng.integers(100, 130, (7, 6), dtype=np.uint8)
        np.testing.assert_array_equal(st_tensors.dequant_mxfp4(nib, sc), scalar_dequant(nib, sc))

    def test_element_order_and_scale(self):
        nib = np.zeros((1, 16), np.uint8)
        nib[0, 0] = 0x21          # low nibble 1 (0.5) = element 0, high nibble 2 (1.0) = element 1
        nib[0, 1] = 0x7F          # element 2: code 15 = -6, element 3: code 7 = 6
        out = st_tensors.dequant_mxfp4(nib, np.array([[128]], np.uint8))   # 2^(128-127) = 2
        self.assertEqual(out[0, :4].tolist(), [1.0, 2.0, -12.0, 12.0])

    @unittest.skipUnless(REAL, "amd/Qwen3.8-27B-Quark-AWQ-MXFP4 is not in the HF cache")
    def test_real_rows(self):
        ck = st_tensors.Checkpoint([str(REAL)])
        for stem in ("model.language_model.layers.0.mlp.gate_proj", "model.language_model.layers.3.self_attn.q_proj",
                     "model.language_model.layers.63.mlp.down_proj"):
            rows, half = ck.t[stem + ".weight"][4]
            nib = np.asarray(ck.array(stem + ".weight")[:64])
            sc = np.asarray(ck.array(stem + ".weight_scale")[:64])
            ours = st_tensors.dequant_mxfp4(nib, sc)
            np.testing.assert_array_equal(ours, scalar_dequant(nib, sc))
            self.assertTrue(np.isfinite(ours).all())
            self.assertTrue(0 < np.abs(ours).mean() < 1.0, np.abs(ours).mean())
            self.assertTrue((sc >= 90).all() and (sc <= 135).all(), (sc.min(), sc.max()))   # plausible E8M0 exponents


class Programs(unittest.TestCase):
    """gen_llm.py on the tiny checkpoint's config."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="stgen-", dir=os.environ.get("LOOM_TEST_TMP"))
        cfg = I.build_config(tiny_source(), ["/ck/model.safetensors"], drop_chat=True)
        cls.cfgp = os.path.join(cls.tmp.name, "cfg.json")
        json.dump(cfg, open(cls.cfgp, "w"))
        env = dict(os.environ, LLM_CONFIG=cls.cfgp, LLM_OUT=cls.tmp.name, LLM_TOKENIZER=os.path.join(cls.tmp.name, "tok.json"), LLM_DRAFT="null")
        json.dump({"added_tokens": [{"content": "<|endoftext|>", "id": 500}]}, open(env["LLM_TOKENIZER"], "w"))
        p = subprocess.run([sys.executable, str(STARTER / "gen_llm.py"), cls.cfgp], capture_output=True, text=True, env=env, cwd=cls.tmp.name)
        assert p.returncode == 0, p.stderr[-2000:]
        cls.text = open(os.path.join(cls.tmp.name, "model.loom")).read()
        cls.policy = open(os.path.join(cls.tmp.name, "weights.loom")).read()
        cls.cfg = cfg

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def params(self):
        return re.findall(r'(%\w+) = command\.parameter %weights, "([^"]+)" : (view<[^>]+>)', self.text)

    def test_parameters_are_hf_keys(self):
        names = {n for _, n, _ in self.params()}
        self.assertNotIn("token_embd.weight", names)
        self.assertIn("model.language_model.embed_tokens.weight", names)
        self.assertIn("lm_head.weight", names)
        self.assertIn("model.language_model.layers.3.self_attn.q_proj.weight_scale", names)
        self.assertFalse([n for n in names if n.startswith("blk.")], "an internal name leaked into a command.parameter")
        self.assertFalse([n for n in names if "#scale" in n])

    def test_every_split_weight_has_its_scale_plane(self):
        by = {v: (n, t) for v, n, t in self.params()}
        for var, (n, t) in by.items():
            if var.endswith("_sc") or n.endswith("weight_scale"):
                continue
            if n.endswith(".weight") and f"{var}_sc" in by:
                nb = int(re.search(r"view<(\d+)x", t).group(1)); sb = int(re.search(r"view<(\d+)x", by[f"{var}_sc"][1]).group(1))
                self.assertEqual(nb, sb * 16, var)

    def test_kernels_and_launch_operands(self):
        self.assertIn("@g_mxfp4s_values(%w: buffer, %ws: buffer, %origin: offset, %scale_origin: offset, %group: index)", self.text)
        self.assertIn('export("g_gemm_mxfp4s_store")', self.text)
        self.assertRegex(self.text, r"kernel\.launch @g_mv_mxfp4s_k128\w*\[[^\]]*\]\([^)]*%wq_3, %wq_3_sc, [^)]*\) : \[index, index\]\(index, index, i32, buffer, view<\d+xi8>, view<\d+xi8>, buffer\)|kernel\.launch @g_mv_mxfp4s_k\d+\w*\[")
        self.assertNotIn("g_gemm_mxfp4s_swiglu", self.text)   # gate and up are two projections and g_silu_mul

    def test_value_heads_are_grouped(self):
        self.assertIn("%kh = index.div %vh, %vratio : index", self.text)
        self.assertNotIn("%kh = index.rem %vh, %chk : index", self.text)

    def test_weight_policy_lists_the_prepared_tensors(self):
        prepared = {e["hf"] for e in self.cfg["tensors"].values() if e.get("prep") and not e["hf"].startswith("mtp.")}   # no drafter in this build
        keys = set(re.findall(r"// \"([^\"]+)\"\nglobal\.rodata\.def @k\d+", self.policy))
        self.assertEqual(keys, prepared)
        for op in ("copy", "add1", "negexp"):
            self.assertIn(f"g_prep_bf16e_f32_{op}", self.text)
        for e in self.cfg["tensors"].values():   # a prepared tensor is read as f32 by the programs
            if e.get("prep") and not e["hf"].startswith("mtp."):
                n = math.prod(e["shape"])
                self.assertIn(f'"{e["hf"]}" : view<{n}xf32>', self.text)


@unittest.skipUnless(np, "numpy")
class StTensors(unittest.TestCase):
    def test_prepared_values(self):
        v = np.array([0.5, -1.0, 2.0], np.float32)
        np.testing.assert_allclose(st_tensors.prepared(v, "add1"), [1.5, 0.0, 3.0])
        np.testing.assert_allclose(st_tensors.prepared(v, "negexp"), -np.exp(v), rtol=1e-6)
        np.testing.assert_array_equal(st_tensors.prepared(v, "copy"), v)

    def test_bf16_widening(self):
        f = np.array([1.0, -2.5, 0.15625], np.float32)
        b = (f.view(np.uint32) >> 16).astype(np.uint16)
        np.testing.assert_array_equal(st_tensors.bf16_to_f32(b), f)


if __name__ == "__main__":
    unittest.main()
