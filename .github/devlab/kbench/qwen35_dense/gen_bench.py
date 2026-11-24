"""Benchmark alternative activation and weight staging against the archived Qwen 35B Strix kernel."""
import os
from pathlib import Path
import re
import sys

import numpy as np

root, out = map(Path, sys.argv[1:3])
sys.path.insert(0, str(root / ".github/devlab/kbench/gemm_q38"))
import gen_q38 as gen

dense = Path(__file__).parent
base = (dense / "staged.loom").read_text()
shapes = [("qkv", 2048, 8192), ("gate", 2048, 4096), ("out", 4096, 2048)]
arms = {"base": {}, "db": {"GEMM_M256_DB": "1"},
        "xg": {"GEMM_M256_XG": "1"}, "xg512": {"GEMM_M256_XG": "2"},
        "xg512db": {"GEMM_M256_XG": "2", "GEMM_M256_XG_DB": "1"},
        "edq": {"GEMM_M256_EDQ": "1"}}
target = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n"
rng = np.random.default_rng(3256)
for name, k, n in shapes:
    blocks = rng.integers(-127, 128, (k*n//32, 34), dtype=np.int8)
    scales = rng.uniform(.001, .02, len(blocks)).astype("<f2")
    blocks[:, :2] = scales.view(np.int8).reshape(-1, 2)
    np.save(out / f"{name}.npy", blocks.ravel())

def case(shape, k, n, rows, verify):
    ty = f"tensor<{rows}x{n}xf32>"
    x = f"tensor<{rows}x{k}xf16>"
    s = f"tensor<{rows}x2xf32>"
    w = f"tensor<{k*n//32*34}xi8>"
    body = f"""check.case public @case_{shape} {{
  %seed = check.literal value(711) : i64
  %rows = check.literal value({rows}) : index
  %k = check.literal value({k}) : index
  %n = check.literal value({n}) : index
  %x = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : {x}
  %s = check.generate.random.uniform seed(%seed) range(0.5 to 2.0) : {s}
  %w = check.file.read.npy path("{shape}.npy") : {w}
"""
    for kernel in (("reference","candidate") if verify else ("candidate",)):
        body += f"  %{kernel} = check.generate.fill value(0.0) : {ty}\n"
        body += f"  kernel.launch @{kernel}[%rows, %n](%rows, %k, %n, %x, %s, %w, %{kernel}) : [index, index](index, index, index, {x}, {s}, {w}, {ty})\n"
    if verify:
        body += f"  check.expect.approximate actual(%candidate) expected(%reference) atol(1.0e-4) rtol(1.0e-4) nan(different) : {ty}, {ty}\n"
    body += "  check.return\n}\n"
    if not verify:
        body += f"check.benchmark<@case_{shape}> @{shape}_bm\n"
    return body

for name, env in arms.items():
    body = gen.kernel("q8_0", "store", env) if env else base
    leaf = "g_gemm_q8_0_store_m256"
    candidate = body.replace(leaf, "candidate") + "\n"
    benchmark = target + candidate + "".join(case(label,k,n,4096,False) for label,k,n in shapes)
    (out / ("bench.loom" if name == "base" else f"bench_{name}.loom")).write_text(benchmark)
    if name != "base":
        checks = target + base.replace(leaf,"reference") + "\n" + candidate
        (out/f"test_{name}.loom").write_text(checks + "".join(case(label,k,n,768,True) for label,k,n in shapes))
