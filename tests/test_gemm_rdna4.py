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
import gen_gemm_rdna4 as R  # noqa: E402

FIXTURE = (ROOT / "tests/data/dual_target_gemm.loom").read_text()   # a q4_k GEMM from a generated llm.loom, with helpers
SPEC = "q4_k:store+acc+swiglu,q6_k:store,q8_0:store,q4_0:store+swiglu,bf16:store,f32:store,mxfp4:store"
HEADER = "target.decl @device\namdgpu.target<gfx11-generic> @rdna3\namdgpu.target<gfx12-generic> @rdna4\n\n"


def gen(rdna4):
    env = dict(os.environ, GEMM_FORMATS=SPEC, LOOM_RDNA4="1" if rdna4 else "0")
    return subprocess.run([sys.executable, str(STARTER / "gen_gemm.py")], capture_output=True, text=True, check=True, env=env).stdout


def kernels(text):
    return [k.strip() for k in re.split(r"(?m)^(?=kernel\.def )", text) if k.strip()]


def loom_compile():
    for c in (os.environ.get("LOOM_COMPILE"), shutil.which("loom-compile"), "/home/jfowers/lt/loom010/.venv/bin/loom-compile"):
        if c and os.path.exists(c):
            return c


class GemmTwinTests(unittest.TestCase):
    def test_off_has_no_twins_and_on_keeps_every_kernel(self):
        off, on = gen(False), gen(True)
        self.assertNotIn("_rdna4", off)
        base = [k for k in kernels(on) if not k.startswith("kernel.def target(@rdna4)")]
        self.assertEqual(base, kernels(off))   # every gfx11 kernel's text is unchanged
        twins = [k for k in kernels(on) if k.startswith("kernel.def target(@rdna4)")]
        self.assertEqual(len(twins), len([k for k in kernels(off) if k.startswith("kernel.def")]))   # one twin per WMMA kernel

    def test_twin_shape(self):
        for k in kernels(gen(True)):
            if not k.startswith("kernel.def target(@rdna4)"):
                continue
            name = re.search(r'export\("(\w+_rdna4)"\)', k).group(1)
            self.assertNotIn("vector<16xf16>, vector<16xf16>", k, name)
            self.assertNotIn("vector.fragment<rhs>", k, name)
            self.assertIn("vector.fragment.load<rhs>", k, name)
            self.assertRegex(k, r"vector\.mma [^\n]*: vector<8xf16>, vector<8xf16>, vector<8xf32>", name)

    @unittest.skipUnless(loom_compile(), "loom-compile not available")
    def test_twin_compiles_for_gfx1201(self):
        kern = re.search(r'(?ms)^kernel\.def export\("g_gemm_q4_k_store"\).*?^\}\n', FIXTURE).group(0)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "t.loom"
            lib = re.sub(r'(?ms)^kernel\.def target\(@rdna4\) export\("g_gemm_q4_k_store_rdna4"\).*?^\}\n', "", FIXTURE)   # the fixture's own twin
            f.write_text(HEADER + lib.split("command.program.def")[0] + "\n" + R.twin(kern))
            r = subprocess.run([loom_compile(), str(f), "--target=amdgpu:gfx1201", "--root=g_gemm_q4_k_store_rdna4"], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-800:])


if __name__ == "__main__":
    unittest.main()
