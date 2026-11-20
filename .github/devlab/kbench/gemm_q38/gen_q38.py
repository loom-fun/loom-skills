#!/usr/bin/env python3
"""Qwen3.8-27B prefill GEMM microbenchmarks (campaign q38pf, lane gemm): the m256 kernels of gen_gemm.gemm_big256 at the
model's real shapes (1024 rows), one file per shape holding every variant, for an interleaved
`loom-benchmark --compare=@base_<shape>,@<v>_<shape>,... --interleave=round_robin` run.

usage: gen_q38.py ROOT OUTDIR [VARIANT ...]   VARIANT: name=ENV=VAL[+ENV=VAL...] (gen_gemm env switches; base = none).
Weights: check.file.read.npy of weights/<name>.npy (raw GGUF bytes of one real tensor; absent files get random int8 from
devlab_kbench.sh). Activations: random f16 in [-1, 1].
Output: OUTDIR/q38_<shape>.loom and OUTDIR/q38_<shape>.cmp (the --compare list)."""
import os, re, sys
ROOT = os.path.abspath(sys.argv[1]); OUT = sys.argv[2]
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-port-llm", "starter"))
import gen_gemm as G

SHAPES = [  # (shape name, fmt, mode, rows, K, N, weight npy names)
    ("down", "q4_k", "store", 1024, 17408, 5120, ["down_q4k"]),
    ("swiglu", "q4_k", "swiglu", 1024, 5120, 17408, ["gate_q4k", "up_q4k"]),
    ("qkv", "q4_k", "store", 1024, 5120, 10240, ["qkv_q4k"]),
    ("gate", "q4_k", "store", 1024, 5120, 17408, ["gate_q4k"]),
    ("up6", "q6_k", "store", 1024, 5120, 17408, ["up_q6k"]),
    ("ssmout6", "q6_k", "store", 1024, 6144, 5120, ["ssmout_q6k"]),
    ("attnout8", "q8_0", "store", 1024, 6144, 5120, ["attnout_q80"]),
    ("mixed", "q4_k:q6_k", "swiglu", 1024, 5120, 17408, ["gate_q4k", "up_q6k"]),   # gate q4_k + up q6_k in one kernel
]
WB = {"q4_k": (144, 256), "q6_k": (210, 256), "q8_0": (34, 32)}


def helpers():
    src = open(os.path.join(ROOT, "skills", "loom-port-llm", "starter", "gen_llm.py")).read()
    out = []
    for fn in ("g_k_scale_minimum", "g_k_metadata"):
        s = src.index(f"func.def inline @{fn}(")
        out.append(src[s: src.index("\n}\n", s) + 3])
    return "\n".join(out)


def ablate(t, abl):
    """Timing-only ablations (wrong results), ABL=a,b: nofrag (MMA fragments sliced from the staged registers: no LDS
    fragment reads), nobar (no barriers), nostore (no LDS stage stores)."""
    for a in abl.split(","):
        if a == "nofrag":
            t = re.sub(r"(%l\d+_\d+) = vector\.fragment\.load<lhs> [^\n]*", r"\1_v = vector.slice %cr_x[0] : vector<32xf16> -> vector<16xf16>\n    \1 = vector.fragment<lhs> \1_v shape [%c16, %c16] : vector<16xf16>", t)
            t = re.sub(r"(%rv\d+_\d+) = vector\.load %w_wave_view[^\n]*", r"\1 = vector.slice %cr_x1[16] : vector<32xf16> -> vector<16xf16>", t)
        elif a == "nobar":
            t = re.sub(r"^\s*kernel\.barrier<workgroup>[^\n]*\n(?=(?:.|\n)*scf\.yield)", "", t, count=2, flags=re.M)
        elif a.startswith("lds"):   # ldsN: pad the activation stage allocation to make the kernel's LDS N bytes
            t = t.replace("%x_bytes = index.constant 36864 : offset", f"%x_bytes = index.constant {int(a[3:]) - 18432} : offset")
        elif a == "nometa":   # repack bound: the K-quant group scale/min as ready f16 values (the header's d, dmin), no 6-bit decode
            t = re.sub(r"(%\S+), (%\S+) = func\.call @g_k_metadata\((%\S+), (%\S+)\) : \(vector<4xi32>, index\) -> \(f32, f32\)",
                       lambda m: (f"{m.group(1)}_hv = vector.bitcast {m.group(3)} : vector<4xi32> to vector<8xf16>\n"
                                  f"      {m.group(1)}_h0 = vector.extract {m.group(1)}_hv[0] : vector<8xf16> -> f16\n"
                                  f"      {m.group(1)}_h1 = vector.extract {m.group(1)}_hv[1] : vector<8xf16> -> f16\n"
                                  f"      {m.group(1)} = scalar.extf {m.group(1)}_h0 : f16 to f32\n"
                                  f"      {m.group(2)} = scalar.extf {m.group(1)}_h1 : f16 to f32"), t)
        elif a == "nostore":
            t = re.sub(r"^\s*vector\.store [^\n]*%(?:x|w)_stage_view\[[^\n]*\n", "", t, flags=re.M)
    return t


def kernel(fmt, mode, env):
    env = dict(env)
    abl = env.pop("ABL", "")
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        f1, _, f2 = fmt.partition(":")
        t = G.gemm_m256_variant(f1, mode, f2 or None) or G.gemm_big256(f1, mode, f2 or None)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    if os.environ.get("Q38_NF", "1") != "0":   # production order (gen_gemm GEMM_N_FASTEST, the program default): output tiles fastest
        t = t.replace("workgroups(%row_tiles, %out_tiles, %c1)", "workgroups(%out_tiles, %row_tiles, %c1)")
        t = t.replace("  %row_group_id = kernel.workgroup.id<x> : index\n  %out_group_id = kernel.workgroup.id<y> : index",
                      "  %row_group_id = kernel.workgroup.id<y> : index\n  %out_group_id = kernel.workgroup.id<x> : index")
    return ablate(t, abl) if abl else t


def main():
    variants = [("base", {})]
    for a in sys.argv[3:]:
        n, _, spec = a.partition("=")
        variants.append((n, dict(kv.split("=", 1) for kv in spec.split("+") if kv)))
    only = os.environ.get("Q38_SHAPES")
    os.makedirs(OUT, exist_ok=True)
    for sh, fmt, mode, rows, K, N, ws in SHAPES:
        if only and sh not in only.split(","): continue
        rows = int(os.environ.get("Q38_ROWS", rows))
        wbs = [N * K // WB[f][1] * WB[f][0] for f in fmt.split(":")]
        wbs = wbs * len(ws) if len(wbs) == 1 else wbs
        parts, names = [helpers()], []
        for vn, env in variants:
            t = kernel(fmt, mode, env)
            old = re.search(r'export\("(\w+)"\)', t).group(1)
            kn = f"k_{vn}_{sh}"
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
            names.append("@" + kn + "_bm")
        open(os.path.join(OUT, f"q38_{sh}.loom"), "w").write("\n".join(parts))
        # A/B correctness (loom-test ab_<shape>.loom-test --xfail=@ab_negative=...): every variant against base on the
        # same inputs (real weights, random activations, random positive row scales); the negative control must fail.
        ab = [helpers()] + [p for p in parts[1:] if p.lstrip().startswith("kernel.def")]
        ty = f"tensor<{rows}x{N}xf32>"
        launch = lambda kn, y: f"  kernel.launch @{kn}[%rows, %n](%rows, %k, %n, %x, %s, {wargs}, {y}) : [index, index](index, index, index, tensor<{rows}x{K}xf16>, tensor<{rows}x2xf32>, {wtys}, {ty})\n"
        head = f"""  %seed = check.literal value(777) : i64
  %seed2 = check.literal value(778) : i64
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({N}) : index
  %x = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{rows}x{K}xf16>
  %s = check.generate.random.uniform seed(%seed2) range(0.5 to 2.0) : tensor<{rows}x2xf32>
{decl}  %yb = check.generate.fill value(0.0) : {ty}
"""
        for vn, _ in variants[1:]:
            ab.append(f"check.case public @ab_{vn} {{\n{head}  %yv = check.generate.fill value(0.0) : {ty}\n" + launch(f"k_base_{sh}", "%yb") + launch(f"k_{vn}_{sh}", "%yv") +
                      f"  check.expect.approximate actual(%yv) expected(%yb) atol(1.0e-4) rtol(1.0e-4) nan(different) : {ty}, {ty}\n  check.return\n}}\n")
        ab.append(f"check.case public @ab_negative {{\n{head}  %z = check.generate.fill value(0.0) : {ty}\n" + launch(f"k_base_{sh}", "%yb") +
                  f"  check.expect.approximate actual(%yb) expected(%z) atol(0.0) rtol(0.0) nan(different) : {ty}, {ty}\n  check.return\n}}\n")
        open(os.path.join(OUT, f"ab_{sh}.loom-test"), "w").write("\n".join(ab))
        open(os.path.join(OUT, f"q38_{sh}.cmp"), "w").write(",".join(names) + f"\n{2 * rows * K * N * (2 if mode == 'swiglu' else 1)}\n")


if __name__ == "__main__":
    main()
