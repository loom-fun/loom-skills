#!/usr/bin/env python3
"""DevLab kbench suite gemm_iq4_udxl (campaign q38xl, lane iq4): the IQ4 m256 GEMMs of unsloth Qwen3.8-27B UD-Q4_K_XL at its
real shapes (gen_q38.py Q38_SET=udxl), one bench_<variant>.loom per variant holding every shape. The production env of the
port package (GEMM_IQ4_NATIVE, G32, MIXED_BAL) is on for every variant; a variant adds its own switches.
usage: gen_udxl.py CHECKOUT RUNDIR VARIANT...   VARIANT: base | g32 (GEMM_IQ4_G32) | lut (GEMM_IQ4_F16LUT) | name=ENV=VAL[+...]
Weights: weights/<name>.npy absent on a runner -> random int8 (devlab_kbench.sh): timing only (the dequant has no data-dependent path)."""
import os, re, sys
ROOT, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
for k, v in (("Q38_SET", "udxl"), ("GEMM_IQ4_NATIVE", "1"), ("GEMM_M256_G32", "1"), ("GEMM_M256_MIXED_BAL", "1")):
    os.environ.setdefault(k, v)
NAMED = {"base": {}, "g32": {"GEMM_IQ4_G32": "1"}, "lut": {"GEMM_IQ4_F16LUT": "1"},
         "g32nf0": {"GEMM_IQ4_G32": "1", "Q38_NF": "0"}, "g32swz10": {"GEMM_IQ4_G32": "1", "GEMM_M256_SWZ": "10", "Q38_NF": "0"}}
args = [a for a in sys.argv[3:] if not a.startswith("shapes:")]
_sh = [a[7:] for a in sys.argv[3:] if a.startswith("shapes:")]
ONLY = [x for x in (_sh[0] if _sh else os.environ.get("Q38_SHAPES", "")).split(",") if x]   # or a "shapes:a,b" argument (kbench.json gen_variants)
sys.argv = [sys.argv[0], ROOT, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_q38 as Q

os.makedirs(OUT, exist_ok=True)
for a in args:
    n, _, spec = a.partition("=")
    env = NAMED[n] if not spec else dict(kv.split("=", 1) for kv in spec.split("+") if kv)
    parts = [Q.helpers()]
    for sh, fmt, mode, rows, K, N, ws in Q.SHAPES:
        if ONLY and sh not in ONLY:
            continue
        wbs = [N * K // Q.WB[f][1] * Q.WB[f][0] for f in fmt.split(":")]
        wbs = wbs * len(ws) if len(wbs) == 1 else wbs
        t = Q.kernel(fmt, mode, env)
        old = re.search(r'export\("(\w+)"\)', t).group(1)
        kn = f"k_{n}_{sh}"
        parts.append(t.replace(f'export("{old}") @{old}(', f'export("{kn}") @{kn}('))
        decl = "".join(f'  %w{i} = check.file.read.npy path("weights/{w}.npy") : tensor<{wbs[i]}xi8>\n' for i, w in enumerate(ws))
        wargs = ", ".join(f"%w{i}" for i in range(len(ws))); wtys = ", ".join(f"tensor<{wbs[i]}xi8>" for i in range(len(ws)))
        parts.append(f"""check.case public @{kn}_case {{
  %seed = check.literal value(12345) : i64
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({N}) : index
  %x = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{rows}x{K}xf16>
  %s = check.generate.fill value(1.0) : tensor<{rows}x2xf32>
{decl}  %y = check.generate.fill value(0.0) : tensor<{rows}x{N}xf32>
  kernel.launch @{kn}[%rows, %n](%rows, %k, %n, %x, %s, {wargs}, %y) : [index, index](index, index, index, tensor<{rows}x{K}xf16>, tensor<{rows}x2xf32>, {wtys}, tensor<{rows}x{N}xf32>)
  check.return
}}
check.benchmark<@{kn}_case> @{kn}_bm
""")
    open(os.path.join(OUT, "bench.loom" if n == "base" else f"bench_{n}.loom"), "w").write("\n".join(parts))
