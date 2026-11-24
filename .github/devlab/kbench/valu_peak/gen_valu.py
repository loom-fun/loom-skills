#!/usr/bin/env python3
"""Strix Halo VALU peak: register-only loops of independent vector ALU ops, by op family (the roofline's vector-ALU rate).

usage: gen_valu.py CHECKOUT RUNDIR VARIANT...   (devlab_kbench.sh "gen"; also: gen_valu.py - OUTDIR VARIANT... locally)

Each thread loads x and y once (random data), then runs ITER steps over NACC independent accumulators of W elements
(acc = op(acc, x, y)), then stores their sum. No memory traffic in the loop: the rate is the vector ALU's (and the
power limit's). The roofline (skills/loom-roofline/roofline.py, format-aware ceiling) counts dequant/softmax work in
VALU lane-instructions (one 32-bit VALU instruction for one lane: an f32 op, a packed-f16 op on 2 values, an int32
bit op); the variants measure the sustained rate of each family:
  base (f32_fma):  acc = fma(acc, x, y), vector<8xf32>       1 v_fma_f32 (or VOPD dual fmac) per element
  f16_pk_fma:      acc = fma(acc, x, y), vector<16xf16>      1 v_pk_fma_f16 per 2 elements
  i32_and_or:      acc = (acc & x) | y, vector<8xi32>        1 v_and_or_b32 per element (or 2 ops)
  f32_exp:         acc = exp(acc * x), vector<8xf32>         1 v_exp_f32 + 1-2 multiplies per element (softmax's exp)
Elements per run = G * NT * ITER * NACC * W; instructions per element come from the compiled listing
(--compile-report / target listing), recorded in the device preset's note.
"""
import os, sys

G, NT, ITER, NACC = 2560, 256, 1024, 8   # workgroups, threads, loop steps, independent accumulators per thread

VARIANTS = {  # name: (element type, width, op)
    "base": ("f32", 8, "fma"), "f16_pk_fma": ("f16", 16, "fma"),
    "i32_and_or": ("i32", 8, "andor"), "f32_exp": ("f32", 8, "exp"),
}


def kernel(et, W, op):
    lt = "f16" if et == "f16" else "f32"   # load type (i32 variants bitcast random f32 bits)
    vt, lv = f"vector<{W}x{et}>", f"vector<{W}x{lt}>"
    L = [f"""kernel.def target(@w32) @k_valu() {{
  %c1 = index.constant 1 : index
  %groups = index.constant {G} : index
  %threads = index.constant {NT} : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%threads, %c1, %c1) : index
}} launch(%x_arg: buffer, %y_arg: buffer, %out_arg: buffer) {{
  %x, %y, %out = buffer.assume.noalias %x_arg, %y_arg, %out_arg : buffer, buffer, buffer
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cnt = index.constant {NT} : index
  %citer = index.constant {ITER} : index
  %origin = index.constant 0 : offset
  %xv = buffer.view %x[%origin] : buffer -> view<{G * NT}x{W}x{lt}>
  %yv = buffer.view %y[%origin] : buffer -> view<{G * NT}x{W}x{lt}>
  %ov = buffer.view %out[%origin] : buffer -> view<{G * NT}x{W}x{lt}>
  %group = kernel.workgroup.id<x> : index
  %tid = kernel.workitem.id<x> : index
  %t = index.assume %tid [range(%tid, 0, {NT - 1})] : index
  %gt = index.madd %group, %cnt, %t : index
  %g = index.assume %gt [range(%gt, 0, {G * NT - 1})] : index
  %xl = vector.load %xv[%g, %c0] : view<{G * NT}x{W}x{lt}> -> {lv}
  %yl = vector.load %yv[%g, %c0] : view<{G * NT}x{W}x{lt}> -> {lv}"""]
    if et == "i32":
        L.append(f"  %xa = vector.bitcast %xl : {lv} to {vt}\n  %ya = vector.bitcast %yl : {lv} to {vt}")
    else:
        L.append("  %xa = vector.addf %xl, %xl : " + vt + "\n  %ya = vector.addf %yl, %yl : " + vt)
    # distinct initial accumulators: y + j*x (keeps the chains independent and data-dependent)
    for j in range(NACC):
        if et == "i32":
            L.append(f"  %sh{j} = vector.splati {j} : {vt}\n  %a0_{j} = vector.xori %ya, %sh{j} : {vt}")
        else:
            L.append(f"  %sh{j} = vector.splatf {0.125 * j} : {vt}\n  %a0_{j} = vector.addf %ya, %sh{j} : {vt}")
    L.append("  " + ", ".join(f"%r{j}" for j in range(NACC)) + " = scf.for %it = [%c0 to %citer step %c1](" +
             ", ".join(f"%i{j} = %a0_{j} : {vt}" for j in range(NACC)) + ") -> (" + ", ".join(vt for _ in range(NACC)) + ") {")
    for j in range(NACC):
        if op == "fma":
            L.append(f"    %n{j} = vector.fmaf %i{j}, %xa, %ya : {vt}")
        elif op == "andor":
            L.append(f"    %m{j} = vector.andi %i{j}, %xa : {vt}\n    %n{j} = vector.ori %m{j}, %ya : {vt}")
        elif op == "exp":
            L.append(f"    %m{j} = vector.mulf %i{j}, %xa : {vt}\n    %n{j} = vector.expf<afn> %m{j} : {vt}")
    L.append("    scf.yield " + ", ".join(f"%n{j}" for j in range(NACC)) + " : " + ", ".join(vt for _ in range(NACC)) + "\n  }")
    add = "vector.xori" if et == "i32" else "vector.addf"
    prev = "%r0"
    for j in range(1, NACC):
        L.append(f"  %s{j} = {add} {prev}, %r{j} : {vt}")
        prev = f"%s{j}"
    if et == "i32":
        L.append(f"  %sf = vector.bitcast {prev} : {vt} to {lv}")
        prev = "%sf"
    L.append(f"""  vector.store {prev}, %ov[%g, %c0] : {lv}, view<{G * NT}x{W}x{lt}>
  kernel.return
}}
""")
    return "\n".join(L)


def bench(name):
    et, W, op = VARIANTS[name]
    lt = "f16" if et == "f16" else "f32"
    n = G * NT * W
    lo = "-1.0" if op != "exp" else "-0.5"   # exp: |acc * x| < 1, no overflow
    return f"""// VALU peak, {name}: {et} x{W}, {op} ({G} workgroups x {NT} threads x {ITER} steps x {NACC} accumulators x {W} elements)
amdgpu.target<gfx11-generic> @w32 {{subgroup_size = 32}}
{kernel(et, W, op)}
check.case public @peak {{
  %seed = check.literal value(12345) : i64
  %x = check.generate.random.uniform seed(%seed) range({lo} to 0.5) : tensor<{n}x{lt}>
  %y = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x{lt}>
  %out = check.generate.fill value(0.0) : tensor<{n}x{lt}>
  kernel.launch @k_valu(%x, %y, %out) : (tensor<{n}x{lt}>, tensor<{n}x{lt}>, tensor<{n}x{lt}>)
  check.return
}}

check.benchmark<@peak> @peak_bm
"""


def main():
    checkout, rundir, variants = sys.argv[1], sys.argv[2], sys.argv[3:] or list(VARIANTS)
    os.makedirs(rundir, exist_ok=True)
    for v in variants:
        open(os.path.join(rundir, "bench.loom" if v == "base" else f"bench_{v}.loom"), "w").write(bench(v))
    if checkout != "-":   # the GPU's power/clock state during the suite (wmma_peak's probe: read-only sysfs)
        sys.path.insert(0, os.path.join(checkout, ".github", "devlab", "kbench", "wmma_peak"))
        from gen_peak import power_probe
        power_probe(rundir)


if __name__ == "__main__":
    main()
