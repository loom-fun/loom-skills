#!/usr/bin/env python3
"""Generate the R9700 GEMM microbenchmark files of the gemm_r9700 kbench suite from the starter's generators.

usage: gen_gemm_bench.py ROOT OUTDIR [VARIANT ...]   ROOT: the loom-skills checkout (its skills/loom-port-llm/starter generates the kernels);
VARIANT: base (the twin as generated; file bench.loom), nfast (output-column tiles fastest), nomma / nodq / nobar (ablations: no MMA
and LDS fragment reads / no dequantization ALU / no barriers: timing only, wrong results); file bench_<variant>.loom.
Each bench file holds one benchmark per (kernel, shape): k_<fmt>_<mode>_<m128|m256>_<rows>x<K>x<N> (the gfx12 twin, real
random operands), plus k_peak: a register-only WMMA loop (the device's real f16 WMMA ceiling on this runner)."""
import os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(sys.argv[1], "skills", "loom-port-llm", "starter"))
import gen_gemm as G
import gen_gemm_rdna4 as R

SHAPES = [  # (see ALL_SHAPES for the full list)
    ("q4_k", "swiglu", 1024, 1536, 6144), ("q4_k", "store", 1024, 6144, 1536),
    ("q4_k", "swiglu", 1024, 4096, 12288), ("q4_k", "store", 1024, 12288, 4096),
    ("q4_k", "swiglu", 1024, 1024, 3072), ("q4_k", "store", 1024, 3072, 1024),
]
EXTRA = [("bf16", "store", 1024, 6144, 1536), ("bf16", "swiglu", 1024, 1536, 6144), ("bf16", "store", 1024, 12288, 4096),
         ("q6_k", "store", 1024, 6144, 1536), ("q6_k", "store", 1024, 12288, 4096)]
ALL_SHAPES = [  # (fmt, mode, rows, K, N): the prefill GEMMs of Gemma-4-E2B (k1536), Qwen3-8B (k4096) and Qwen3-0.6B (k1024)
    ("q4_k", "swiglu", 1024, 1536, 6144), ("q4_k", "store", 1024, 6144, 1536), ("q4_k", "store", 1024, 1536, 2048),
    ("q4_k", "swiglu", 1024, 4096, 12288), ("q4_k", "store", 1024, 12288, 4096), ("q4_k", "store", 1024, 4096, 4096),
    ("q4_k", "swiglu", 1024, 1024, 3072), ("q4_k", "store", 1024, 3072, 1024),
    ("q4_k", "swiglu", 4096, 1536, 6144), ("q4_k", "store", 4096, 6144, 1536),
]

NATIVE = {"dq2": R.q4k_m256_native, "sb": R.q4k_m256_sb, "sbnb": lambda t: R.q4k_m256_sb(t, nb=True),
          "sbsc": lambda t: R.q4k_m256_sb(t, sc=True), "sbnbsc": lambda t: R.q4k_m256_sb(t, nb=True, sc=True)}   # variants that swap the m256 q4_k twin for a native gfx12 kernel (no ablation transform)

def kernel_text(text, name):
    m = re.search(r'kernel\.def (?:target\(@\w+\) )?export\("%s"\)' % re.escape(name), text)
    s = text.index("kernel.def", m.start()) if m else None
    e = text.index("  kernel.return\n}\n", s) + len("  kernel.return\n}\n")
    return text[s:e]

import re as _re
def transform(tw, variant):
    """Ablations of one twin kernel text (timing only, wrong results): nomma, nodq, nobar, ablw (no per-step weight loads: the
    carried raw values are reused), ablx (no per-step activation loads)."""
    if variant == "nox":      # no activation staging at all (loads and LDS stores)
        tw = _re.sub(r"^\s*vector\.store (?:%cr_x1?|%x_rawn1?_u\d), %x_stage_view\[[^\n]*\n", "", tw, flags=_re.M)
    elif variant == "xhalf":  # the second activation row of each thread is not staged
        tw = _re.sub(r"^\s*vector\.store (?:%cr_x1|%x_rawn1_u\d), %x_stage_view\[[^\n]*\n", "", tw, flags=_re.M)
    elif variant == "nowst":  # no weight staging stores (dequantization and loads vanish with them)
        tw = _re.sub(r"^\s*vector\.store %dq_(?:lo|hi)_u\d, %w_stage_view\[[^\n]*\n", "", tw, flags=_re.M)
    elif variant == "ablw":
        tw = _re.sub(r"^(\s*)(%nx_(?:pk|hdr)\w*) = vector\.load [^\n]*-> vector<4xi32>$", lambda m: "%s%s = vector.andi %s, %s : vector<4xi32>" % (m.group(1), m.group(2), "%cr_pk" if "pk" in m.group(2) else "%cr_hdr", "%cr_pk" if "pk" in m.group(2) else "%cr_hdr"), tw, flags=_re.M)
    elif variant == "ablx":
        tw = _re.sub(r"^(\s*)(%x_rawn1?_u\d) = vector\.load [^\n]*$", lambda m: "%s%s = vector.bitcast %%cr_x%s : vector<32xf16> to vector<32xf16>" % (m.group(1), m.group(2), "1" if "rawn1" in m.group(2) else ""), tw, flags=_re.M)
    if variant == "nomma":
        tw = _re.sub(r"^(\s*)(%\w+) = vector\.mma (%\w+), (%\w+), (%\w+) : vector<8xf16>, vector<8xf16>, vector<8xf32>$", r"\1\2 = vector.addf \5, \5 : vector<8xf32>", tw, flags=_re.M)
    elif variant == "nodq":
        tw = tw.replace("  %origin = index.constant 0 : offset\n", "  %origin = index.constant 0 : offset\n  %zero16 = vector.constant 0.0 : vector<16xf16>\n", 1)
        tw = _re.sub(r"vector\.store %dq\w+, ", "vector.store %zero16, ", tw)
    elif variant == "nfast":   # output-column tiles fastest (GEMM_N_FASTEST): consecutive workgroups share an activation row tile
        tw = tw.replace("workgroups(%row_tiles, %out_tiles, %c1)", "workgroups(%out_tiles, %row_tiles, %c1)")
        tw = tw.replace("  %row_group_id = kernel.workgroup.id<x> : index\n  %out_group_id = kernel.workgroup.id<y> : index",
                        "  %row_group_id = kernel.workgroup.id<y> : index\n  %out_group_id = kernel.workgroup.id<x> : index")
    elif variant == "nobar":
        tw = _re.sub(r"^\s*kernel\.barrier<workgroup>[^\n]*\n", "", tw, flags=_re.M)
    return tw

def build(variant=""):
    out = []
    for fmt, mode, rows, K, N in SHAPES + (EXTRA if not variant else []):
        for big in ("m128", "m256"):
            if big == "m256" and rows % 256: continue
            g = G.gemm_big(fmt, mode) if big == "m128" else G.gemm_big256(fmt, mode)
            tw = R.twin(g)
            nat, _, abl = variant.partition("_")
            if nat in NATIVE:
                if big != "m256": continue
                tw = NATIVE[nat](tw)
            elif abl == "" and variant not in ("", "nfast", "nomma", "nodq", "nobar", "ablw", "ablx", "nox", "xhalf", "nowst"):
                raise SystemExit("unknown variant " + variant)
            tw = transform(tw, abl if nat in NATIVE else variant)
            m = re.search(r'export\("(\w+)"\)', tw); old = m.group(1)
            kn = f"k_{fmt}_{mode}_{big}_{rows}x{K}x{N}"
            tw = tw.replace(f'export("{old}") @{old}(', f'export("{kn}") @{kn}(')
            wb = N * K * 2 if fmt == "bf16" else N * K // 256 * (210 if fmt == "q6_k" else 144)
            two = mode == "swiglu"
            wdecl = f"  %w = check.generate.random.uniform seed(%seed) range(-128 to 127) : tensor<{wb}xi8>\n"
            wargs, wtys = "%w", f"tensor<{wb}xi8>"
            if two:
                wdecl += f"  %u = check.generate.random.uniform seed(%seed) range(-128 to 127) : tensor<{wb}xi8>\n"; wargs += ", %u"; wtys += f", tensor<{wb}xi8>"
            out.append(tw)
            out.append(f"""check.case public @{kn}_case {{
  %seed = check.literal value(12345) : i64
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({N}) : index
  %x = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{rows}x{K}xf16>
  %s = check.generate.fill value(1.0) : tensor<{rows}x2xf32>
{wdecl}  %y = check.generate.fill value(0.0) : tensor<{rows}x{N}xf32>
  kernel.launch @{kn}[%rows, %n](%rows, %k, %n, %x, %s, {wargs}, %y) : [index, index](index, index, index, tensor<{rows}x{K}xf16>, tensor<{rows}x2xf32>, {wtys}, tensor<{rows}x{N}xf32>)
  check.return
}}
check.benchmark<@{kn}_case> @{kn}_bm
""")
    return out


PEAK = """kernel.def target(@rdna4) export("k_peak") @k_peak() {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %groups = index.constant 512 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%x_arg: buffer, %y_arg: buffer) {
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c8 = index.constant 8 : index
  %c4096 = index.constant 4096 : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %xv = buffer.view %x_arg[%origin] : buffer -> view<256x8xf16>
  %yv = buffer.view %y_arg[%origin] : buffer -> view<256x8xf32>
  %a = vector.load %xv[%lane, %c0] : view<256x8xf16> -> vector<8xf16>
  %b = vector.load %xv[%lane, %c0] : view<256x8xf16> -> vector<8xf16>
  %lhs = vector.fragment<lhs> %a shape [%c16, %c16] : vector<8xf16>
  %rhs = vector.fragment<rhs> %b shape [%c16, %c16] : vector<8xf16>
  %zero = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero shape [%c16, %c16] : vector<8xf32>
  %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7 = scf.for %i = [%c0 to %c4096 step %c1](%a0 = %init : vector<8xf32>, %a1 = %init : vector<8xf32>, %a2 = %init : vector<8xf32>, %a3 = %init : vector<8xf32>, %a4 = %init : vector<8xf32>, %a5 = %init : vector<8xf32>, %a6 = %init : vector<8xf32>, %a7 = %init : vector<8xf32>) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {
    %n0 = vector.mma %lhs, %rhs, %a0 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %n1 = vector.mma %lhs, %rhs, %a1 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %n2 = vector.mma %lhs, %rhs, %a2 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %n3 = vector.mma %lhs, %rhs, %a3 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %n4 = vector.mma %lhs, %rhs, %a4 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %n5 = vector.mma %lhs, %rhs, %a5 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %n6 = vector.mma %lhs, %rhs, %a6 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    %n7 = vector.mma %lhs, %rhs, %a7 : vector<8xf16>, vector<8xf16>, vector<8xf32>
    scf.yield %n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
  }
  %s01 = vector.addf %r0, %r1 : vector<8xf32>
  %s23 = vector.addf %r2, %r3 : vector<8xf32>
  %s45 = vector.addf %r4, %r5 : vector<8xf32>
  %s67 = vector.addf %r6, %r7 : vector<8xf32>
  %s03 = vector.addf %s01, %s23 : vector<8xf32>
  %s47 = vector.addf %s45, %s67 : vector<8xf32>
  %sum = vector.addf %s03, %s47 : vector<8xf32>
  vector.store %sum, %yv[%lane, %c0] : vector<8xf32>, view<256x8xf32>
  kernel.return
}
check.case public @k_peak_case {
  %x = check.generate.fill value(0.01) : tensor<256x8xf16>
  %y = check.generate.fill value(0.0) : tensor<256x8xf32>
  %groups = check.literal value(512) : index
  kernel.launch @k_peak[](%x, %y) : [](tensor<256x8xf16>, tensor<256x8xf32>)
  check.return
}
check.benchmark<@k_peak_case> @k_peak_bm
"""

def mix(nv, nw=8):
    """k_mix<nv>: k_peak's WMMA loop with nv extra independent f16 vector FMAs (4 v_pk_fma_f16 each) per iteration: does VALU work overlap the matrix unit?"""
    p = PEAK.replace("k_peak", "k_mix%d" % nv)
    fm = "".join("    %%f{0}_n = vector.fmaf %%f{0}, %%a_c, %%b_c : vector<8xf16>\n".format(i % 4) if False else "" for i in range(0))
    chains = "".join("%%f%d = %%a : vector<8xf16>, " % i for i in range(4))
    # carried f16 chains appended to the loop carries
    p = p.replace("%a0 = %init : vector<8xf32>,", "%a0 = %init : vector<8xf32>,", 1)
    carry_decl = ", ".join("%%g%d = %%a : vector<8xf16>" % i for i in range(4))
    ty = ", ".join(["vector<8xf16>"] * 4)
    p = p.replace(") -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {",
                  ", " + carry_decl + ") -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, " + ty + ") {", 1)
    p = p.replace("%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7 = scf.for", "%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %q0, %q1, %q2, %q3 = scf.for", 1)
    body = ""
    cur = ["%%g%d" % i for i in range(4)]
    for n in range(nv):
        c = n % 4
        body += "    %%h%d_%d = vector.fmaf %s, %%b, %%a : vector<8xf16>\n" % (c, n, cur[c])
        cur[c] = "%%h%d_%d" % (c, n)
    p = p.replace("    scf.yield %n0,", body + "    scf.yield %n0,", 1)
    p = p.replace("%n7 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>\n  }",
                  "%n7, " + ", ".join(cur) + " : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, " + ty + "\n  }", 1)
    p = p.replace("  %sum = vector.addf %s03, %s47 : vector<8xf32>", "  %sum0 = vector.addf %s03, %s47 : vector<8xf32>\n  %qq = vector.addf %q0, %q1 : vector<8xf16>\n  %qr = vector.addf %q2, %q3 : vector<8xf16>\n  %qs = vector.addf %qq, %qr : vector<8xf16>\n  %qx = vector.extf %qs : vector<8xf16> to vector<8xf32>\n  %sum = vector.addf %sum0, %qx : vector<8xf32>", 1)
    return p

HEAD = "amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n\n" + open(os.path.join(HERE, "gemm_helpers.loom")).read() + "\n"
if __name__ == "__main__" and len(sys.argv) > 2:
    d = sys.argv[2]
    os.makedirs(d, exist_ok=True)
    for v in (sys.argv[3:] or ["base"]):
        v = "" if v == "base" else v
        open(os.path.join(d, "bench%s.loom" % ("_" + v if v else "")), "w").write(HEAD + "\n".join(build(v)) + ("\n" + PEAK + "".join("\n" + mix(n) for n in (0, 8, 16, 32, 64)) if not v else ""))
