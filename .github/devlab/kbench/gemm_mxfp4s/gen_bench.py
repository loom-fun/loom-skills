#!/usr/bin/env python3
"""Generate the gemm_mxfp4s kbench suite: the gfx12 MXFP4 split-plane fp8-WMMA prefill GEMMs (gen_mxfp4s.py) at the Qwen3.8-27B shapes.

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   ROOT: the loom-skills checkout; VARIANT: base (file bench.loom), or a name of VARIANTS (bench_<name>.loom).
Writes bench*.loom (one benchmark k_<mode>_<K>x<N>_<rows> per shape: random planar weights, timing only) and test.loom (check.cases against a numpy
float64 reference on the exactly dequantized weights and the e4m3 activations given; run by devlab_kbench.sh's "tests" step with loom-test)."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-port-llm", "starter"))
import numpy as np
import gen_mxfp4s as G

HEAD = "amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n\n"
ROWS = 2048
SHAPES = [("swiglu", 5120, 17408), ("store", 17408, 5120), ("store", 5120, 12288), ("store", 5120, 10240), ("store", 5120, 6144), ("store", 6144, 5120)]
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float64)
VARIANTS = {}   # name -> function(kernel text, mode) -> kernel text (ablations and levers, filled in as they are tried)


def _ablation(name):
    def deco(f):
        VARIANTS[name] = f
        return f
    return deco


@_ablation("nomma")      # timing only (wrong results): no WMMA
def _nomma(t, mode):
    return re.sub(r"^(\s*)(%\w+) = vector\.mma (%\w+), (%\w+), (%\w+) : vector<2xi32>, vector<2xi32>, vector<8xf32>$", r"\1\2 = vector.addf \5, \5 : vector<8xf32>", t, flags=re.M)


@_ablation("nodq")       # no weight decode ALU: the raw nibble words are stored as they are
def _nodq(t, mode):
    t = re.sub(r"vector\.store %(?:lo16|hi16|blk)_(\d), (%w_stage_st\[[^\n]*)vector<32xi8>, ", r"vector.store %zero32, \2vector<32xi8>, ", t)
    t = re.sub(r"vector\.store %(?:lo16|hi16)_(\d), (%w_stage_st\[[^\n]*)vector<16xi8>, ", r"vector.store %zero16, \2vector<16xi8>, ", t)
    return t.replace("  %origin = index.constant 0 : offset\n", "  %origin = index.constant 0 : offset\n  %zero16 = vector.splati 0 : vector<16xi8>\n  %zero32 = vector.splati 0 : vector<32xi8>\n", 1)


@_ablation("nofrag")     # LDS fragment reads of the first K substep only (every substep reuses them): ~1/8 of the LDS read traffic
def _nofrag(t, mode):
    return re.sub(r"(vector\.fragment\.load<(?:lhs|rhs)> %\w+\[%?\w+, )%fk\d+\]", r"\1%fk0]", t)


@_ablation("noload")     # no next-step global loads of activations and weights: the carried registers are reused
def _noload(t, mode):
    t = re.sub(r"^(\s*)%xrn(\d) = vector\.load %x_view[^\n]*$", r"\1%xrn\2 = vector.andi %xrp\2, %xrp\2 : vector<16xi8>", t, flags=re.M)
    t = re.sub(r"^(\s*)%wq(\d)n = vector\.load [^\n]*-> vector<4xi32>$", r"\1%wq\2n = vector.andi %wq\2p, %wq\2p : vector<4xi32>", t, flags=re.M)
    return t


@_ablation("noa")        # activation fragments of K substeps 1..7 reuse substep 0's (the global loads of A vanish)
def _noa(t, mode):
    return re.sub(r"(vector\.fragment\.load<lhs> %\w+\[%\w+, )%(?:fk|kg)\d+\]", lambda m: m.group(1) + ("%kg0]" if "%x_g8" in m.group(0) else "%fk0]"), t)


@_ablation("nob")        # weight fragments of K substeps 1..7 reuse substep 0's (the LDS reads of B vanish)
def _nob(t, mode):
    return re.sub(r"(vector\.fragment\.load<rhs> %\w+\[)%fk\d+,", r"\1%fk0,", t)


@_ablation("nobar")      # no workgroup barriers (wrong results)
def _nobar(t, mode):
    return re.sub(r"^\s*kernel\.barrier<workgroup>[^\n]*\n", "", t, flags=re.M)


@_ablation("nox")        # no activation staging stores
def _nox(t, mode):
    return re.sub(r"^\s*vector\.store %cr_x\d, %x_stage_st[^\n]*\n", "", t, flags=re.M)


GEN = {"staged": {"a_direct": False, "rf": 4}, "adirect": {"a_direct": True, "rf": 4}, "adbuf": {"a_direct": True, "dbuf": True}, "adk2": {"a_direct": True, "kmult": 2}, "adr8": {"a_direct": True, "rf": 8}, "adr2": {"a_direct": True, "rf": 2}, "adr2pf": {"a_direct": True, "rf": 2, "a_pf": True}, "adn128": {"a_direct": True, "nt": 128}, "adn128r2": {"a_direct": True, "nt": 128, "rf": 2}, "adr2k2": {"a_direct": True, "rf": 2, "kmult": 2},
       # pfgemm round 1 (a swiglu entry in "sw" overrides the lever for the swiglu kernels: tn there is per weight)
       "t128x256": {"bm": 128, "tn": 256, "sw": {"bm": 128, "tn": 128}}, "t256x256": {"bm": 256, "tn": 256, "sw": {"bm": 256, "tn": 128}},
       "swp": {"swp": True}, "swz": {"swz": True}, "pfa2": {"pfa": 2}, "pfa1": {"pfa": 1},
       "a2swp": {"rf": 2, "swp": True}, "yf16": {"yf16": True},
       "a2pfa": {"rf": 2, "swp": True, "pfa": 2}, "p0": {"pad": 0}, "p32": {"pad": 32}, "p64": {"pad": 64}, "p80": {"pad": 80}, "p112": {"pad": 112},
       # ogemm round 1: lane-vector operand loads (one 16-byte load per lane = two K substeps, A from global with immediate offsets, B ds_load_b128), swiglu at rf 4
       "lv": {"lv": True, "sw": {"rf": 4}}, "lvpf1": {"lv": True, "lvpf": 1, "sw": {"rf": 4}}, "lvpf2": {"lv": True, "lvpf": 2, "sw": {"rf": 4}},
       "lvk2": {"lv": True, "kmult": 2, "sw": {"rf": 4}}, "lvsw2": {"lv": True, "lvpf": 1, "sw": {"rf": 4, "lvpf": 2}},
       "lvs": {"lv": True, "lvs": True, "sw": {"rf": 4}}, "lvk2s": {"lv": True, "lvs": True, "kmult": 2, "sw": {"rf": 4}},
       # ogemm round 3: wave-specialized (4 producer waves decode the next step into a second LDS buffer while 8 WMMA waves multiply; one barrier per step)
       "ws": {"ws": 4}, "wsr4": {"ws": 4, "sw": {"rf": 4}}, "wsl": {"ws": 4, "lvs": True, "sw": {"rf": 4}},
       # ogemm round 4/5: pipelining modes lvs 1 (A + B of the next pair) / 2 (A only), with the next step's first activation pair before the barrier (lvpf)
       "lvs2": {"lv": True, "lvs": 2, "sw": {"rf": 4}}, "lvs2pf1": {"lv": True, "lvs": 2, "lvpf": 1, "sw": {"rf": 4}},
       # ogemm round 6: 16-wave workgroups (8 accumulators per wave, ~150 VGPR: 2 workgroups = 32 waves per WGP instead of 24)
       "w16a": {"rf": 2, "cf": 4, "lv": True, "lvs": 1, "sw": {"rf": 2, "cf": 2}}, "w16b": {"rf": 4, "cf": 2, "lv": True, "lvs": 1, "sw": {"rf": 4, "cf": 1}},
       "w16v": {"rf": 2, "cf": 4, "sw": {"rf": 2, "cf": 2}}}   # variants that are generator parameters (the levers), not text ablations


# the variants whose results are checked against float64 (the others are timing-only levers that lost or tied; hrx-loom 0.1.0 fails to verify a module of ~65k lines
# ("view type references encoding value ... not defined at this point", notes/sdk_blockers.md), so the checked set stays at ~60 cases)
TESTED = ["staged", "adirect", "adr2", "adn128", "t128x256", "swp", "pfa2", "yf16", "lvs", "lvs2pf1", "w16a", "w16b", "w16v"]


def ytype(variant):
    """the output element type of a variant's kernels"""
    return "f16" if GEN.get(variant.split("+")[0] if variant else "", {}).get("yf16") else "f32"


def kernel(mode, kn, variant="", order="pairs"):
    """variant: `lever[+ablation...]`: a GEN lever (generator parameters), then text ablations applied on top of it"""
    parts = variant.split("+") if variant else []
    lever = parts[0] if parts and parts[0] in GEN else ""
    kw = dict(GEN.get(lever, {}))
    sw = kw.pop("sw", None)
    if mode == "swiglu" and sw:
        kw.update(sw)
    t = G.gemm_mxfp4s_f8_m256(mode, order=order, **kw)
    variant = "+".join(x for x in parts if x != lever)
    old = f"g_gemm_mxfp4s_{mode}_f8_m256_rdna4"
    t = t.replace(f'export("{old}") @{old}(', f'export("{kn}") @{kn}(')
    for ab in variant.split("+") if variant else []:
        t = VARIANTS[ab](t, mode)
    return t


def build(variant=""):
    """a variant token `r1024` / `r4096` sets the row count of the benchmarks (the prefill chunk; default 2048)"""
    global ROWS
    toks = variant.split("+") if variant else []
    ROWS = 2048
    zd = "zd" in toks   # zero operands (activations and weight nibbles 0): the power / toggle sensitivity of the kernel (timing only)
    for t in toks:
        if re.fullmatch(r"r\d+", t):
            ROWS = int(t[1:])
    variant = "+".join(t for t in toks if not re.fullmatch(r"r\d+", t) and t != "zd")
    out = []
    for mode, K, N in SHAPES:
        kn = f"k_{mode}_{K}x{N}_{ROWS}"
        out.append(kernel(mode, kn, variant))
        YT = ytype(variant)
        planes = ["w"] + (["u"] if mode == "swiglu" else [])
        decl = "".join((f"  %{p} = check.generate.fill value(0) : tensor<{N * K // 2}xi8>\n" if zd else f"  %{p} = check.generate.random.uniform seed(%seed) range(-128 to 127) : tensor<{N * K // 2}xi8>\n") +
                       f"  %{p}s = check.generate.random.uniform seed(%seed) range(116 to 126) : tensor<{N * K // 32}xi8>\n" for p in planes)
        args = ", ".join(f"%{p}, %{p}s" for p in planes)
        tys = ", ".join(f"tensor<{N * K // 2}xi8>, tensor<{N * K // 32}xi8>" for _ in planes)
        out.append(f"""check.case public @{kn}_case {{
  %seed = check.literal value(12345) : i64
  %rows = check.literal value({ROWS}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({N}) : index
  %x = {f"check.generate.fill value(0) : tensor<{ROWS}x{K}xi8>" if zd else f"check.generate.random.uniform seed(%seed) range(-100 to 100) : tensor<{ROWS}x{K}xi8>"}
  %s = check.generate.fill value(1.0) : tensor<{ROWS}x2xf32>
{decl}  %y = check.generate.fill value(0.0) : tensor<{ROWS}x{N}x{YT}>
  kernel.launch @{kn}[%rows, %n](%rows, %k, %n, %x, %s, {args}, %y) : [index, index](index, index, index, tensor<{ROWS}x{K}xi8>, tensor<{ROWS}x2xf32>, {tys}, tensor<{ROWS}x{N}x{YT}>)
  check.return
}}
check.benchmark<@{kn}_case> @{kn}_bm
""")
    return out


PEAK8 = """kernel.def target(@rdna4) export("k_peak8") @k_peak8() {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %groups = index.constant 512 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%y_arg: buffer) {
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c256 = index.constant 256 : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %yv = buffer.view %y_arg[%origin] : buffer -> view<256x8xf32>
  %stage_bytes = index.constant 8192 : offset
  %stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %a_schema = encoding.define #encoding.operand<element_format=f8e4m3, payload_elements=8, payload_registers=2> : encoding<schema>
  %b_schema = encoding.define #encoding.operand<element_format=f8e5m2, payload_elements=8, payload_registers=2> : encoding<schema>
  %a_layout = encoding.layout.dense : encoding<layout>
  %b_layout = encoding.layout.strided [1, 64] : encoding<layout>
  %a_storage = encoding.define #encoding.storage {layout = %a_layout : encoding<layout>, schema = %a_schema : encoding<schema>} : encoding<storage>
  %b_storage = encoding.define #encoding.storage {layout = %b_layout : encoding<layout>, schema = %b_schema : encoding<schema>} : encoding<storage>
  %a_view = buffer.view %stage[%origin] : buffer -> view<64x64xf8E4M3, %a_storage>
  %b_view = buffer.view %stage[%origin] : buffer -> view<64x64xf8E5M2, %b_storage>
  %lhs = vector.fragment.load<lhs> %a_view[%c0, %c0] shape [%c16, %c16] : view<64x64xf8E4M3, %a_storage> -> vector<2xi32>
  %rhs = vector.fragment.load<rhs> %b_view[%c0, %c0] shape [%c16, %c16] : view<64x64xf8E5M2, %b_storage> -> vector<2xi32>
  %zero = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero shape [%c16, %c16] : vector<8xf32>
  %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7 = scf.for %i = [%c0 to %c256 step %c1](%a0 = %init : vector<8xf32>, %a1 = %init : vector<8xf32>, %a2 = %init : vector<8xf32>, %a3 = %init : vector<8xf32>, %a4 = %init : vector<8xf32>, %a5 = %init : vector<8xf32>, %a6 = %init : vector<8xf32>, %a7 = %init : vector<8xf32>) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {
""" + "".join(f"    %n{i} = vector.mma %lhs, %rhs, %a{i} : vector<2xi32>, vector<2xi32>, vector<8xf32>\n" for i in range(8)) + """    scf.yield %n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
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
check.case public @k_peak8_case {
  %y = check.generate.fill value(0.0) : tensor<256x8xf32>
  kernel.launch @k_peak8[](%y) : [](tensor<256x8xf32>)
  check.return
}
check.benchmark<@k_peak8_case> @k_peak8_bm
"""   # 4096 waves x 256 iterations x 8 WMMA of 8192 flop = 68.7 TFLOP: the fp8 WMMA ceiling of this runner (register-only chains)


PEAK8R = """kernel.def target(@rdna4) export("k_peak8r") @k_peak8r() {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %groups = index.constant 512 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%src: buffer, %y_arg: buffer) {
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c256 = index.constant 256 : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %yv = buffer.view %y_arg[%origin] : buffer -> view<256x8xf32>
  %stage_bytes = index.constant 8192 : offset
  %stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %a_schema = encoding.define #encoding.operand<element_format=f8e4m3, payload_elements=8, payload_registers=2> : encoding<schema>
  %b_schema = encoding.define #encoding.operand<element_format=f8e5m2, payload_elements=8, payload_registers=2> : encoding<schema>
  %a_layout = encoding.layout.dense : encoding<layout>
  %b_layout = encoding.layout.strided [1, 64] : encoding<layout>
  %a_storage = encoding.define #encoding.storage {layout = %a_layout : encoding<layout>, schema = %a_schema : encoding<schema>} : encoding<storage>
  %b_storage = encoding.define #encoding.storage {layout = %b_layout : encoding<layout>, schema = %b_schema : encoding<schema>} : encoding<storage>
  %src_view = buffer.view %src[%origin] : buffer -> view<256x32xi8>
  %stage_st = buffer.view %stage[%origin] : buffer -> view<256x32xi8>
  %c0v = index.constant 0 : index
  %c16v = index.constant 16 : index
  %q0 = vector.load %src_view[%lane, %c0v] : view<256x32xi8> -> vector<16xi8>
  %q1 = vector.load %src_view[%lane, %c16v] : view<256x32xi8> -> vector<16xi8>
  vector.store %q0, %stage_st[%lane, %c0v] : vector<16xi8>, view<256x32xi8>
  vector.store %q1, %stage_st[%lane, %c16v] : vector<16xi8>, view<256x32xi8>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %a_view = buffer.view %stage[%origin] : buffer -> view<64x64xf8E4M3, %a_storage>
  %b_view = buffer.view %stage[%origin] : buffer -> view<64x64xf8E5M2, %b_storage>
  %lhs = vector.fragment.load<lhs> %a_view[%c0, %c0] shape [%c16, %c16] : view<64x64xf8E4M3, %a_storage> -> vector<2xi32>
  %rhs = vector.fragment.load<rhs> %b_view[%c0, %c0] shape [%c16, %c16] : view<64x64xf8E5M2, %b_storage> -> vector<2xi32>
  %zero = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero shape [%c16, %c16] : vector<8xf32>
  %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7 = scf.for %i = [%c0 to %c256 step %c1](%a0 = %init : vector<8xf32>, %a1 = %init : vector<8xf32>, %a2 = %init : vector<8xf32>, %a3 = %init : vector<8xf32>, %a4 = %init : vector<8xf32>, %a5 = %init : vector<8xf32>, %a6 = %init : vector<8xf32>, %a7 = %init : vector<8xf32>) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {
""" + "".join(f"    %n{i} = vector.mma %lhs, %rhs, %a{i} : vector<2xi32>, vector<2xi32>, vector<8xf32>\n" for i in range(8)) + """    scf.yield %n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
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
check.case public @k_peak8r_case {
  %seed = check.literal value(5) : i64
  %src = check.generate.random.uniform seed(%seed) range(-100 to 100) : tensor<8192xi8>
  %y = check.generate.fill value(0.0) : tensor<256x8xf32>
  kernel.launch @k_peak8r[](%src, %y) : [](tensor<8192xi8>, tensor<256x8xf32>)
  check.return
}
check.benchmark<@k_peak8r_case> @k_peak8r_bm
"""   # 4096 waves x 256 iterations x 8 WMMA of 8192 flop = 68.7 TFLOP: the fp8 WMMA ceiling of this runner (register-only chains)




# -- correctness cases --------------------------------------------------------------------------------------------------
def e4m3_round(x):
    x = np.asarray(x, np.float64); a = np.minimum(np.abs(x), 448.0)
    e = np.maximum(np.floor(np.log2(np.maximum(a, 2.0 ** -20))), -6)
    q = 2.0 ** (e - 3)
    return np.sign(x) * np.minimum(np.rint(a / q) * q, 448.0)


def e4m3_bytes(v):
    v = np.asarray(v, np.float64); a = np.abs(v)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -20))).astype(np.int64)
    sub = a < 2.0 ** -6
    ef = np.where(sub, 0, e + 7); mant = np.where(sub, np.rint(a / 2.0 ** -9), np.rint((a / 2.0 ** e - 1) * 8)).astype(np.int64)
    b = np.where(a == 0, 0, (ef << 3) | mant) | np.where(v < 0, 0x80, 0)
    return b.astype(np.uint8).view(np.int8)


def dequant(nib, sc, order):
    """nib [n, K/2] U8, sc [n, K/32] U8 -> float64 [n, K]: element values fp4 * 2^(e - 127), the block's elements in `order`."""
    n, kh = nib.shape
    lo, hi = FP4[nib & 7] * np.where(nib & 8, -1.0, 1.0), FP4[(nib >> 4) & 7] * np.where(nib & 128, -1.0, 1.0)
    lo, hi = lo.reshape(n, -1, 16), hi.reshape(n, -1, 16)
    blk = np.stack([lo, hi], 3).reshape(n, -1, 32) if order == "pairs" else np.concatenate([lo, hi], 2)
    return (blk * 2.0 ** (sc.astype(np.float64)[:, :, None] - 127)).reshape(n, -1)


def save(d, name, a):
    np.save(os.path.join(d, name + ".npy"), a)


def tests(outdir):
    rng = np.random.default_rng(7)
    out = []
    for order, variant in [("pairs", ""), ("split16", "")] + [("pairs", v) for v in TESTED]:
        for mode, K, n in (("store", 5120, 1024), ("swiglu", 5120, 512), ("store", 17408, 256)):
            rows = 512
            kn = f"t_{mode}_{order}{'_' + variant if variant else ''}_{K}"
            YT = ytype(variant)
            nib = [rng.integers(0, 256, (n, K // 2)).astype(np.uint8) for _ in range(2 if mode == "swiglu" else 1)]
            sc = [rng.integers(108, 127, (n, K // 32)).astype(np.uint8) for _ in nib]
            W = [dequant(a, b, order) for a, b in zip(nib, sc)]
            xf = e4m3_round(np.clip(rng.standard_normal((rows, K)) * (3.0 if YT == "f16" else 30.0), -440, 440))
            inv = rng.uniform(0.25, 2.0, rows).astype(np.float32)
            s = np.stack([np.ones(rows, np.float32), inv * np.float32(2.0 ** -8)], 1)
            g = (xf @ W[0].T) * inv[:, None].astype(np.float64)
            if mode == "swiglu":
                u = (xf @ W[1].T) * inv[:, None].astype(np.float64)
                y = g / (1 + np.exp(np.clip(-g, -80, 80))) * u
            else:
                y = g
            save(outdir, kn + "_x", e4m3_bytes(xf)); save(outdir, kn + "_s", s); save(outdir, kn + "_y", y.astype(np.float16 if YT == "f16" else np.float32))
            decl, args, tys = "", [], []
            for i, (a, b) in enumerate(zip(nib, sc)):
                save(outdir, f"{kn}_w{i}", a.reshape(-1).view(np.int8)); save(outdir, f"{kn}_ws{i}", b.reshape(-1).view(np.int8))
                decl += (f'  %w{i} = check.file.read.npy path("{kn}_w{i}.npy") : tensor<{a.size}xi8>\n'
                         f'  %ws{i} = check.file.read.npy path("{kn}_ws{i}.npy") : tensor<{b.size}xi8>\n')
                args += [f"%w{i}", f"%ws{i}"]; tys += [f"tensor<{a.size}xi8>", f"tensor<{b.size}xi8>"]
            tol = 1.0e-3 * float(np.abs(y).std() + 1e-6)
            out.append(kernel(mode, kn, variant, order=order))
            out.append(f"""check.case public @{kn}_case {{
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({n}) : index
  %x = check.file.read.npy path("{kn}_x.npy") : tensor<{rows}x{K}xi8>
  %s = check.file.read.npy path("{kn}_s.npy") : tensor<{rows}x2xf32>
{decl}  %e = check.file.read.npy path("{kn}_y.npy") : tensor<{rows}x{n}x{YT}>
  %y = check.generate.fill value(0.0) : tensor<{rows}x{n}x{YT}>
  kernel.launch @{kn}[%rows, %n](%rows, %k, %n, %x, %s, {", ".join(args)}, %y) : [index, index](index, index, index, tensor<{rows}x{K}xi8>, tensor<{rows}x2xf32>, {", ".join(tys)}, tensor<{rows}x{n}x{YT}>)
  check.expect.close actual(%y) expected(%e) atol({tol:.3e}) rtol(2.0e-3) nan(different) : tensor<{rows}x{n}x{YT}>
  check.return
}}
""")
    return out


if __name__ == "__main__" and len(sys.argv) > 2:
    d = sys.argv[2]
    os.makedirs(d, exist_ok=True)
    for v in (sys.argv[3:] or ["base"]):
        if v != "base" and not all(x in VARIANTS or x in GEN or x == "zd" or re.fullmatch(r"r\d+", x) for x in v.split("+")): raise SystemExit("unknown variant " + v)
        v = "" if v == "base" else v
        open(os.path.join(d, "bench%s.loom" % ("_" + v if v else "")), "w").write(HEAD + "\n".join(build(v)) + ("\n" + PEAK8 + "\n" + PEAK8R if not v else ""))
    # hrx-loom 0.1.0: float comparison is check.expect.approximate with the actual and expected types (as make_kernel_tests.py does)
    text = re.sub(r"check\.expect\.close (actual\(.*\)) : (.*)$", r"check.expect.approximate \1 : \2, \2", HEAD + "\n".join(tests(d)), flags=re.M)
    open(os.path.join(d, "test.loom"), "w").write(text)
