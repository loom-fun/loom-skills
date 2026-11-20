#!/usr/bin/env python3
"""Strix Halo WMMA peak: register-only 16x16x16 MMA loops, by operand type and data pattern.

usage: gen_peak.py CHECKOUT RUNDIR VARIANT...   (devlab_kbench.sh "gen"; also: gen_peak.py - OUTDIR VARIANT... locally)

Each wave holds 4 A and 4 B fragments in registers (loaded once from the inputs) and runs ITER steps of the
16 (4 x 4) MMAs, then stores the sum of its accumulators. No memory traffic in the loop: the rate is the
matrix pipe's (and the power limit's). FLOPs per MMA = 2 * 16^3 = 8192 for f16, bf16 and iu8 alike.
Data patterns are .npy inputs written here (numpy on the runner), so the operands' bit activity is controlled:
  f16: rand (uniform -1..1), const (all 1.0), zero, q4int (B = integers 0..15, A rand), q4dq (B = d*q - m per 32
       values, the Q4_K dequant's values), act_small (A = rand rounded to 4 mantissa bits, B rand)
  i8:  rand (-128..127), const (all 1), w4 (B 0..15, A rand), w4s (B -8..7, A rand), a7 (A -64..63, B 0..15)
  bf16: rand
On the runner it also logs the GPU's power/clock state (sysfs, read-only) and samples it while the suite runs.
"""
import os, subprocess, sys
import numpy as np

G, NT, ITER = 2560, 256, 256          # workgroups, threads (8 waves), loop steps
NW = NT // 32
NFRAG = G * NW * 8                     # fragments per input (4 A + 4 B per wave; A and B from separate inputs)

def kernel(kind, name):
    if kind == "i8":
        ty, acc, dt = "vector<4xi32>", "vector<8xi32>", "i32"
        sch = "  %schema = encoding.define #encoding.operand<element_format=i8, payload_elements=16, payload_registers=4> : encoding<schema>\n"
        use = " using {schema = %schema : encoding<schema>}"
        zero = "  %zero = vector.splati 0 : vector<8xi32>\n"
        per = 4
    else:
        e = "bf16" if kind == "bf16" else "f16"
        ty, acc, dt = f"vector<16x{e}>".replace("x", "x", 1), "vector<8xf32>", e
        ty = f"vector<16x{e}>"
        sch, use = "", ""
        zero = "  %zero = vector.splatf 0.0 : vector<8xf32>\n"
        per = 16
    L = [f"""kernel.def target(@w32) @{name}() {{
  %c1 = index.constant 1 : index
  %groups = index.constant {G} : index
  %threads = index.constant {NT} : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%threads, %c1, %c1) : index
}} launch(%a_arg: buffer, %b_arg: buffer, %out_arg: buffer) {{
  %a, %b, %out = buffer.assume.noalias %a_arg, %b_arg, %out_arg : buffer, buffer, buffer
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c4 = index.constant 4 : index
  %cnw = index.constant {NW} : index
  %citer = index.constant {ITER} : index
  %origin = index.constant 0 : offset
{sch}{zero}  %init = vector.fragment<init> %zero shape [%c16, %c16] : {acc}
  %av = buffer.view %a[%origin] : buffer -> view<{NFRAG // 2}x32x{per}x{dt}>
  %bv = buffer.view %b[%origin] : buffer -> view<{NFRAG // 2}x32x{per}x{dt}>
  %ov = buffer.view %out[%origin] : buffer -> view<{G * NW * 16}x16x{'i32' if kind == 'i8' else 'f32'}>
  %group = kernel.workgroup.id<x> : index
  %sg = kernel.subgroup.id : index
  %wave = index.assume %sg [range(%sg, 0, {NW - 1})] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_b = index.assume %lane [range(%lane, 0, 31)] : index
  %gw = index.madd %group, %cnw, %wave : index
  %gwb = index.assume %gw [range(%gw, 0, {G * NW - 1})] : index"""]
    for i in range(4):
        L.append(f"""  %fi{i}_c = index.constant {i} : index
  %fi{i} = index.madd %gwb, %c4, %fi{i}_c : index
  %fb{i} = index.assume %fi{i} [range(%fi{i}, 0, {NFRAG // 2 - 1})] : index
  %ad{i} = vector.load %av[%fb{i}, %lane_b, %c0] : view<{NFRAG // 2}x32x{per}x{dt}> -> {ty}
  %bd{i} = vector.load %bv[%fb{i}, %lane_b, %c0] : view<{NFRAG // 2}x32x{per}x{dt}> -> {ty}
  %A{i} = vector.fragment<lhs> %ad{i} shape [%c16, %c16]{use} : {ty}
  %B{i} = vector.fragment<rhs> %bd{i} shape [%c16, %c16]{use} : {ty}""")
    accs = [f"{r}{c}" for r in range(4) for c in range(4)]
    L.append("  " + ", ".join(f"%o{a}" for a in accs) + " = scf.for %s = [%c0 to %citer step %c1](" +
             ", ".join(f"%i{a} = %init : {acc}" for a in accs) + ") -> (" + ", ".join(acc for _ in accs) + ") {")
    for a in accs:
        L.append(f"    %m{a} = vector.mma %A{a[0]}, %B{a[1]}, %i{a} : {ty}, {ty}, {acc}")
    L.append("    scf.yield " + ", ".join(f"%m{a}" for a in accs) + " : " + ", ".join(acc for _ in accs) + "\n  }")
    add = "vector.addi" if kind == "i8" else "vector.addf"
    prev = "%o00"
    for j, a in enumerate(accs[1:]):
        L.append(f"  %s{j} = {add} {prev}, %o{a} : {acc}")
        prev = f"%s{j}"
    L.append(f"""  %row = index.mul %gwb, %c16 : index
  vector.fragment.store<result> {prev}, %ov[%row, %c0] shape [%c16, %c16] : {acc}, view<{G * NW * 16}x16x{'i32' if kind == 'i8' else 'f32'}>
  kernel.return
}}
""")
    return "\n".join(L)

VARIANTS = {  # name: (kind, A pattern, B pattern)
    "base": ("f16", "rand", "rand"), "f16_const": ("f16", "one", "one"), "f16_zero": ("f16", "zero", "zero"),
    "f16_q4int": ("f16", "rand", "q4int"), "f16_q4dq": ("f16", "rand", "q4dq"), "f16_asmall": ("f16", "small", "rand"),
    "f16_both_small": ("f16", "small", "q4int"),

    "i8_rand": ("i8", "i8rand", "i8rand"), "i8_const": ("i8", "i8one", "i8one"), "i8_w4": ("i8", "i8rand", "u4"),
    "i8_w4s": ("i8", "i8rand", "s4"), "i8_a7w4": ("i8", "a7", "u4"),
}

def data(pat, n, kind, rng):
    if kind in ("f16", "bf16"):
        if pat == "rand": v = rng.uniform(-1, 1, n)
        elif pat == "one": v = np.ones(n)
        elif pat == "zero": v = np.zeros(n)
        elif pat == "q4int": v = rng.integers(0, 16, n).astype(np.float64)
        elif pat == "q4dq":
            q = rng.integers(0, 16, n); d = np.repeat(rng.uniform(0.001, 0.02, n // 32), 32); m = np.repeat(rng.uniform(0, 0.1, n // 32), 32)
            v = d * q - m
        elif pat == "small":   # rand with 4 mantissa bits
            v = rng.uniform(-1, 1, n); e = np.floor(np.log2(np.abs(v) + 1e-30)); v = np.round(v / 2 ** (e - 4)) * 2 ** (e - 4)
        if kind == "bf16":
            u = (v.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            return u.view(np.uint16)
        return v.astype(np.float16)
    if pat == "i8rand": return rng.integers(-128, 128, n).astype(np.int8)
    if pat == "i8one": return np.ones(n, np.int8)
    if pat == "u4": return rng.integers(0, 16, n).astype(np.int8)
    if pat == "s4": return rng.integers(-8, 8, n).astype(np.int8)
    if pat == "a7": return rng.integers(-64, 64, n).astype(np.int8)

def bench(name):
    kind, pa, pb = VARIANTS[name]
    per = 16 if kind != "i8" else 16      # elements per lane per fragment (i8: 16 bytes = 4 x i32)
    n = (NFRAG // 2) * 32 * per
    et = {"f16": "f16", "bf16": "i16", "i8": "i8"}[kind]
    k = kernel(kind, "k_peak")
    return (f"""// WMMA peak, {kind}: A {pa}, B {pb} ({G} workgroups x {NW} waves x {ITER} steps x 16 MMAs)
amdgpu.target<gfx11-generic> @w32 {{subgroup_size = 32}}
{k}
check.case public @peak {{
  %a = check.file.read.npy path("data/{kind}_{pa}.npy") : tensor<{n}x{et}>
  %b = check.file.read.npy path("data/{kind}_{pb}_b.npy") : tensor<{n}x{et}>
  %out = check.generate.fill value({"0" if kind == "i8" else "0.0"}) : tensor<{G * NW * 256}x{'i32' if kind == 'i8' else 'f32'}>
  kernel.launch @k_peak(%a, %b, %out) : (tensor<{n}x{et}>, tensor<{n}x{et}>, tensor<{G * NW * 256}x{'i32' if kind == 'i8' else 'f32'}>)
  check.return
}}

check.benchmark<@peak> @peak_bm
""", [(f"data/{kind}_{pa}.npy", pa, kind, n), (f"data/{kind}_{pb}_b.npy", pb, kind, n)])

def power_probe(rundir):
    out = os.path.join(rundir, "..", "out"); os.makedirs(out, exist_ok=True)
    sh = r'''for c in /sys/class/drm/card*/device; do echo "== $c"; for f in power_dpm_force_performance_level pp_power_profile_mode pp_dpm_sclk pp_dpm_mclk pp_dpm_fclk pp_od_clk_voltage; do [ -r $c/$f ] && { echo "-- $f"; cat $c/$f; }; done;
  for h in $c/hwmon/hwmon*; do echo "-- $h"; for f in $h/name $h/power1_cap $h/power1_cap_max $h/power1_cap_default $h/power1_average $h/power1_input $h/freq1_input $h/temp1_input; do [ -r $f ] && echo "$f $(cat $f)"; done; done; done
  cat /sys/firmware/acpi/platform_profile 2>/dev/null | sed 's/^/platform_profile /'; cat /sys/firmware/acpi/platform_profile_choices 2>/dev/null | sed 's/^/platform_profile_choices /'
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null | sed 's/^/cpu_governor /'; uname -a; which amd-smi rocm-smi 2>/dev/null; amd-smi metric 2>/dev/null | head -80'''
    with open(os.path.join(out, "power_state.txt"), "w") as f:
        subprocess.run(["bash", "-c", sh], stdout=f, stderr=subprocess.STDOUT)
    samp = r'''for i in $(seq 1 1800); do t=$(date +%s.%N); for c in /sys/class/drm/card*/device; do s=$(grep '\*' $c/pp_dpm_sclk 2>/dev/null | tr -d '\n'); p=""; for h in $c/hwmon/hwmon*; do p="$p $(cat $h/power1_average 2>/dev/null || cat $h/power1_input 2>/dev/null) $(cat $h/freq1_input 2>/dev/null)"; done; echo "$t $c sclk[$s] pw/freq[$p]"; done; sleep 0.5; done'''
    subprocess.Popen(["bash", "-c", samp], stdout=open(os.path.join(out, "power_samples.txt"), "w"), stderr=subprocess.STDOUT, start_new_session=True)

def main():
    checkout, rundir, variants = sys.argv[1], sys.argv[2], sys.argv[3:] or list(VARIANTS)
    os.makedirs(os.path.join(rundir, "data"), exist_ok=True)
    rng = np.random.default_rng(1)
    for v in variants:
        text, files = bench(v)
        open(os.path.join(rundir, "bench.loom" if v == "base" else f"bench_{v}.loom"), "w").write(text)
        for path, pat, kind, n in files:
            p = os.path.join(rundir, path)
            if not os.path.exists(p): np.save(p, data(pat, n, kind, rng))
    if checkout != "-": power_probe(rundir)

if __name__ == "__main__":
    main()
