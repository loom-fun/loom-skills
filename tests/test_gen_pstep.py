"""Fused / persistent verify matvecs (starter gen_pstep.py; choices decode.verify.fuse, decode.verify.persist) on the AMD MXFP4 card's programs."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills/loom-port-llm/starter"
sys.path.insert(0, str(STARTER))
import gen_pstep  # noqa: E402

SLUG = "amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom"
FLEET = ROOT / "fleet" / SLUG


def loom_tool(name):
    for c in (os.environ.get("LOOM_" + name.upper().replace("-", "_")), shutil.which(name), str(Path.home() / f"loom/.venv/bin/{name}")):
        if c and os.path.exists(c):
            return c


def build(work, env):
    """programs-only build of the card from the fleet header dump (as build_fleet.py does for one safetensors model)"""
    shutil.copy(FLEET / "tokenizer_min.json", work)
    subprocess.run([sys.executable, str(STARTER / "inspect_safetensors.py"), str(FLEET / "safetensors/model.safetensors.json.gz"), "fresh.json",
                    "--checkpoint", os.path.join(work, "model.safetensors")], cwd=work, check=True, capture_output=True)
    cfg = json.load(open(os.path.join(work, "fresh.json")))
    cfg.update(json.load(open(FLEET / "llm_config.json")))
    json.dump(cfg, open(os.path.join(work, "cfg.json"), "w"))
    e = dict(os.environ, LOOM_KB_ROOT=str(ROOT / "skills/loom-kb"), **env)
    e.pop("LLM_CONFIG", None)
    subprocess.run([sys.executable, str(STARTER / "build.py"), "cfg.json", "pkg", "--slug", SLUG, "--tokenizer", "tokenizer_min.json", "--model-id", "x",
                    "--memory-gib", "4096", "--context", "65536", "--pool", "65536", "--prefill", "1024", "--rows", "4", "--draft-depth", "3"],
                   cwd=work, check=True, env=e, capture_output=True)
    return open(os.path.join(work, "pkg/program/llm.loom")).read()


def verify(prog):
    s = prog.index("@llm_verify()")
    return prog[s:prog.index("\n}\n", s)]


@unittest.skipUnless((FLEET / "safetensors/model.safetensors.json.gz").exists(), "fleet dump of the AMD card")
class FuseVerify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        try:
            import numpy  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("numpy")
        os.makedirs(os.path.join(cls.tmp, "off")); os.makedirs(os.path.join(cls.tmp, "on"))
        cls.off = build(os.path.join(cls.tmp, "off"), {})
        cls.on = build(os.path.join(cls.tmp, "on"), {"VERIFY_FUSE": "full", "MV_PERSIST": "256"})
        os.makedirs(os.path.join(cls.tmp, "swq"))
        cls.swq = build(os.path.join(cls.tmp, "swq"), {"VERIFY_FUSE": "swqn"})
        os.makedirs(os.path.join(cls.tmp, "swq1"))
        cls.swq1 = build(os.path.join(cls.tmp, "swq1"), {"VERIFY_FUSE": "swq"})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_other_programs_unchanged(self):
        """only llm_verify and the kernel library change"""
        def progs(p):
            return {m.group(1): m.group(0) for m in re.finditer(r"^command\.def[^\n]*@(\w+)\(\)(?:.|\n)*?^\}\n", p, re.M)}
        a, b = progs(self.off), progs(self.on)
        self.assertEqual(set(a), set(b))
        for k in a:
            if k != "llm_verify":
                self.assertEqual(a[k], b[k], k)

    def test_layers_fused(self):
        v = verify(self.on)
        layers = re.findall(r"^  // ---- layer (\d+)", v, re.M)
        self.assertEqual(len(layers), 64)
        self.assertEqual(v.count("  scf.if %pz {\n"), 64)
        fused = re.findall(r"^  scf\.if %pz \{\n(.*?)^  \} else \{\n", v, re.M | re.S)
        self.assertEqual(len(fused), 64)
        for i, f in enumerate(fused):
            # the 4-token %pz copy keeps the GDN alpha / beta blocks' GEMM-epoch branch (prefill.gdn.ab_q8: g_quant_q8 of h there), never taken at 4 rows
            f = re.sub(r"(scf\.if %gemm \{\n)\s*kernel\.launch @g_quant_q8\[%t, %nH\]\(%t, %nH, %h, %h16q\)[^\n]*\n", r"\1", f)
            for k in ("@g_post_add_norm", "@g_quant_q8", "@g_silu_mul", "@g_mv_mxfp4s_k5120_t4_q8", "@g_mv_mxfp4s_k17408_t4_q8"):
                self.assertNotIn(k, f, f"layer {i}")
            self.assertEqual(len(re.findall(r"@g_pmv_mxfp4s_k5120_t4_norm_\w+\[", f)), 2, f"layer {i}: pre-mixer + gate/up")
            self.assertEqual(len(re.findall(r"@g_pmv_mxfp4s_k17408_t4_silu_\w+\[", f)), 1)
            down = re.search(r"@g_pmv_mxfp4s_k17408_t4_silu_\w+\[[^\n]*", f).group(0)
            self.assertIn(", %y) :" if i == 63 else ", %x) :", down, f"layer {i}: down accumulates into x (the last layer stores y)")
        # the else branches are the layers as generated
        self.assertEqual(verify(self.off).count("@g_post_add_norm["), verify(self.on).count("@g_post_add_norm["))

    def test_swq(self):
        """swq: only the FFN's gate + up + silu + quant change, into one swiglu_q8 launch writing the down projection's input"""
        v = verify(self.swq)
        fused = re.findall(r"^  scf\.if %pz \{\n(.*?)^  \} else \{\n(.*?)^  \}\n", v, re.M | re.S)
        self.assertEqual(len(fused), 64)
        for i, (f, orig) in enumerate(fused):
            self.assertNotIn("@g_silu_mul", f)
            self.assertEqual(f.count("@g_pmv_mxfp4s_k5120_t4_swq_r32["), 1, i)
            # gate: 4 branch launches -> 1, up: 4, g_silu_mul, the f quant block (row scale | quant), and each
            # post_add_norm + quant(h) pair (2 per layer, layer 0's first is the embedding's quant: kept)
            pairs = len(re.findall(r"@g_post_add_norm_q8\[", f))
            self.assertEqual(pairs, 1 if i == 0 else 2, i)
            self.assertEqual(len(re.findall(r"kernel\.launch", orig)) - len(re.findall(r"kernel\.launch", f)), 3 + 4 + 1 + 2 + pairs, i)
        self.assertIn('export("g_pmv_mxfp4s_k5120_t4_swq_r32")', self.swq)
        self.assertNotIn("g_post_add_norm_q8", self.swq1)   # swq alone keeps the norm + quant launches
        self.assertEqual(verify(self.swq1).count("@g_pmv_mxfp4s_k5120_t4_swq_r32["), 64)

    def test_hout_where_f32_projections_read_h(self):
        for f in re.findall(r"^  scf\.if %pz \{\n(.*?)^  \} else \{\n", verify(self.on), re.M | re.S):
            reads_h = "%zero_i32, %h, %walpha" in f
            self.assertEqual(bool(re.search(r"@g_pmv_\w+_h_p256\[", f)), reads_h)

    def test_kernels_compile(self):
        lc = loom_tool("loom-compile")
        if not lc:
            self.skipTest("loom-compile")
        prog = self.on
        def block(kind, name):
            return re.search(rf"^{kind}[^\n]*@{re.escape(name)}\((?:.|\n)*?^\}}\n", prog, re.M).group(0)
        ks = sorted(set(re.findall(r'^kernel\.def export\("(g_pmv_[^"]+)"\)', prog, re.M)))
        self.assertTrue(ks)
        for arch, tgt in (("gfx11-generic", "amdgpu:gfx1151"), ("gfx12-generic", "amdgpu:gfx1201")):
            src = (f"target.decl @device\namdgpu.target<{arch}> @t\n\n" + "".join(block("func.def", f) for f in ("g_mxfp4s_icodes", "g_store_row"))
                   + "".join(block("kernel.def", k).replace("kernel.def export(", "kernel.def target(@t) export(", 1) for k in ks))
            p = os.path.join(self.tmp, f"k_{arch}.loom")
            open(p, "w").write(src)
            r = subprocess.run([lc, p, "--format=amdgpu-hsaco", f"--target={tgt}", "--output=" + p + ".hsaco"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])

    def test_program_type_checks(self):
        lo = loom_tool("loom-opt")
        if not lo:
            self.skipTest("loom-opt")
        p = os.path.join(self.tmp, "on/pkg/program/llm.loom")
        r = subprocess.run([lo, "--pass=canonicalize", "--output=/dev/null", p], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])


class Pmv(unittest.TestCase):
    def test_quant_halves_match_g_quant_q8_ops(self):
        """the prologue quant is g_quant_q8's op sequence per element (absf, maxnum reduce, divf by 127, divf, roundf, select, fptosi)"""
        t = gen_pstep.quant_lines(0, "v")
        for op in ("vector.absf", "vector.reduce<maxnumf>", "scalar.divf %qamax0, %pq_c127", "vector.divf", "vector.roundf", "vector.select", "vector.fptosi",
                   "vector.dot4i<s8s8>", "scalar.fptrunc", "scalar.extf"):
            self.assertIn(op, t)


if __name__ == "__main__":
    unittest.main()
