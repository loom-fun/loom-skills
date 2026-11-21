"""Text-level checks of the UD-quant kit switches (q38xl): prefill.gemm.prune, the IQ4 f16 staging and its G32 requirement."""
import ast
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills/loom-port-llm/starter"


def _prune_gemm():
    tree = ast.parse((STARTER / "gen_llm.py").read_text())
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prune_gemm"]
    ns = {"re": re}
    exec(compile(ast.Module(body=picked, type_ignores=[]), "prune_gemm", "exec"), ns)
    return ns["prune_gemm"]


def _gen_gemm(code, **env):
    """run `code` with gen_gemm imported under env (module-level switches read at import); returns (rc, stdout, stderr)"""
    e = {k: v for k, v in os.environ.items() if not k.startswith(("GEMM_", "LOOM_RDNA"))}
    e.update(env)
    p = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(STARTER)!r}); import gen_gemm\n{code}"],
                       env=e, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


class PruneGemm(unittest.TestCase):
    LIB = "\n".join((
        "kernel.def @g_gemm_a_m256(%x: buffer) {\n}",
        "kernel.def target(@rdna4) @g_gemm_a_m256_rdna4(%x: buffer) {\n}",
        "kernel.def @g_gemm_b(%x: buffer) {\n  // calls itself only: @g_gemm_b[%t](%x)\n}",
        "kernel.def target(@rdna4) @g_gemm_b_rdna4(%x: buffer) {\n}",
        "kernel.def @g_other(%x: buffer) {\n}",
        "command.def public @llm_prefill() {\n  kernel.launch @g_gemm_a_m256[%t](%x)\n}",
        ""))

    def test_keeps_launched_kernels_and_their_twins(self):
        out, pruned = _prune_gemm()(self.LIB)
        self.assertEqual(pruned, ["g_gemm_b", "g_gemm_b_rdna4"])
        self.assertIn("@g_gemm_a_m256(", out)
        self.assertIn("@g_gemm_a_m256_rdna4(", out)
        self.assertIn("@g_other(", out)   # only g_gemm_* kernels are pruned
        self.assertNotIn("@g_gemm_b", out)


class Iq4Staging(unittest.TestCase):
    def test_g32_requires_f16lut(self):
        rc, _, err = _gen_gemm("gen_gemm._iq4_f16lut_on()", GEMM_IQ4_NATIVE="1", GEMM_IQ4_G32="1")
        self.assertNotEqual(rc, 0)
        self.assertIn("GEMM_IQ4_F16LUT", err)
        rc, out, err = _gen_gemm("print(gen_gemm._iq4_f16lut_on())", GEMM_IQ4_NATIVE="1", GEMM_IQ4_G32="1", GEMM_IQ4_F16LUT="1")
        self.assertEqual((rc, out.strip()), (0, "True"), err)

    def test_f16_offset_uses_the_rounded_scale(self):
        """(u - 128) * f16(dl) exactly: the -128 offset comes from the f16 scale (extf), never from the f32 dl"""
        rc, out, err = _gen_gemm("print(gen_gemm.iq4_dequant16('iq4_xs', 'w', 'v'))", GEMM_IQ4_NATIVE="1", GEMM_IQ4_F16LUT="1")
        self.assertEqual(rc, 0, err)
        self.assertIn("%v_dlh32 = scalar.extf %v_dlh : f16 to f32", out)
        self.assertIn("%v_off32 = scalar.mulf %v_dlh32, %v_m128 : f32", out)
        self.assertNotRegex(out, r"scalar\.mulf [^\n]*: f16")   # hrx-loom 0.2.0 miscompiles a scalar f16 mulf feeding a splat


if __name__ == "__main__":
    unittest.main()
