#!/usr/bin/env python3
"""Generate the pstep_r9700 kbench suite (campaign mxfp4st, lane ostep): the drafted verify step's FFN block at 4 tokens on the
AMD MXFP4 card's shapes, as the program launches it today vs persistent / prologue-fused matvecs (starter gen_pstep.py), plus
the cost of a device-scope grid barrier vs a kernel launch.

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   ROOT: the loom-skills checkout; VARIANT: base (bench.loom, test.loom).
The kernels are the card's own: a programs-only build of fleet/amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom (header dump, model node
choices via --slug) gives llm.loom; the chain kernels are taken from it verbatim and gen_pstep transforms its matvecs.

Benchmarks (one case = one FFN block of one layer, t = 4 tokens; weights random, 142 MB > the 64 MB MALL):
  k_case_chain        post_add_norm, quant_q8, gate, up, silu_mul, quant_q8, down           7 launches (today's program)
  k_case_chain_p<G>   the same with persistent row-loop matvecs (G workgroups)              7 launches
  k_case_fuse         gate+up in one launch with the rmsnorm+quant prologue, down with silu+quant   2 launches
  k_case_fuse_p<G>    k_case_fuse with persistent row loops                                  2 launches
  k_case_gb_chain_<G> 64 dependent phases of a tiny kernel as 64 launches (G workgroups)
  k_case_gb_bar_<G>   the same 64 phases in one launch with a device-scope grid barrier between phases
test.loom: the fused kernels' outputs equal the launch chain's (check.expect.equal), and the grid barrier delivers every phase."""
import json, os, re, subprocess, sys, shutil
ROOT = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
STARTER = os.path.join(ROOT, "skills", "loom-port-llm", "starter")
sys.path.insert(0, STARTER)
import gen_pstep  # noqa: E402

SLUG = "amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom"
T = 4
GS = (128, 256, 512, 1024)
GB = (64, 256)
PHASES = 64


def build():
    """programs-only build of the card from the fleet header dump (build_fleet.py's steps for one safetensors model)"""
    fl = os.path.join(ROOT, "fleet", SLUG)
    work = os.path.join(OUT, "_pkg")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    shutil.copy(os.path.join(fl, "tokenizer_min.json"), work)
    subprocess.run([sys.executable, os.path.join(STARTER, "inspect_safetensors.py"), os.path.join(fl, "safetensors", "model.safetensors.json.gz"),
                    "fresh.json", "--checkpoint", os.path.join(work, "model.safetensors")], cwd=work, check=True, stdout=subprocess.DEVNULL)
    cfg = json.load(open(os.path.join(work, "fresh.json")))
    cfg.update(json.load(open(os.path.join(fl, "llm_config.json"))))
    json.dump(cfg, open(os.path.join(work, "cfg.json"), "w"))
    env = dict(os.environ, LOOM_KB_ROOT=os.path.join(ROOT, "skills", "loom-kb"))
    env.pop("LLM_CONFIG", None)
    subprocess.run([sys.executable, os.path.join(STARTER, "build.py"), "cfg.json", "pkg", "--slug", SLUG, "--tokenizer", "tokenizer_min.json",
                    "--model-id", "x", "--memory-gib", "4096", "--context", "65536", "--pool", "65536", "--prefill", "1024", "--rows", "4",
                    "--draft-depth", "3"], cwd=work, check=True, env=env, stdout=subprocess.DEVNULL)
    return cfg, open(os.path.join(work, "pkg", "program", "llm.loom")).read()


def gen_mv(cfg, K, R):
    """gen_llm.mv of the card's int8 matvec at R rows (the build's choices: kb_choices for the slug)"""
    work = os.path.join(OUT, "_pkg")
    os.environ["LOOM_KB_ROOT"] = os.path.join(ROOT, "skills", "loom-kb")
    import kb_choices
    kb_choices.apply(cfg, "mtp", slug=SLUG)
    os.environ["LLM_CONFIG"] = os.path.join(work, "cfg.json")
    os.environ["LLM_DRAFT"] = json.dumps({"kind": "mtp", "depth": 3})
    sys.argv = [os.path.join(STARTER, "gen_llm.py")]
    import gen_llm
    return gen_llm.mv("mxfp4s", K, R=R, TG=T, suffix=f"_r{R}", iq=True)


def block(prog, kind, name):
    m = re.search(rf'^{kind}[^\n]*@{re.escape(name)}\((?:.|\n)*?^\}}\n', prog, re.M)
    assert m, name
    return m.group(0)


def funcs_for(prog, text, seen=None):
    seen = seen if seen is not None else {}
    for f in re.findall(r"func\.call @(\w+)\(", text):
        if f not in seen:
            seen[f] = block(prog, "func.def", f)
            funcs_for(prog, seen[f], seen)
    return seen


def kdef(t):
    return t.replace("kernel.def export(", "kernel.def target(@rdna4) export(", 1)


HEAD = "target.decl @device\namdgpu.target<gfx12-generic> @rdna4\n\n"


def gb_kernels(G):
    """grid-barrier probe: phase p stores out[p % 2][me] = out[(p - 1) % 2][neighbor] + 1 (phase 0: 1)"""
    n = G * 64
    common = f"""  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c64 = index.constant 64 : index
  %cg = index.constant {G} : index
  %out_v = buffer.view %out[%origin] : buffer -> view<2x{n}xi32>
  %wg0 = kernel.workgroup.id<x> : index
  %wg = index.assume %wg0 [range(%wg0, 0, {G - 1})] : index
  %lane0 = kernel.workitem.id<x> : index
  %lane = index.assume %lane0 [range(%lane0, 0, 63)] : index
  %me0 = index.mul %wg, %c64 : index
  %me1 = index.add %me0, %lane : index
  %me = index.assume %me1 [range(%me1, 0, {n - 1})] : index
  %nb0 = index.add %wg, %c1 : index
  %nb1 = index.rem %nb0, %cg : index
  %nb2 = index.mul %nb1, %c64 : index
  %nb3 = index.add %nb2, %lane : index
  %nb = index.assume %nb3 [range(%nb3, 0, {n - 1})] : index
  %one = scalar.constant 1 : i32
"""
    def phase(p):
        return f"""  %pp_{p}0 = index.rem %{p}, %c2 : index
  %pp_{p} = index.assume %pp_{p}0 [range(%pp_{p}0, 0, 1)] : index
  %pq_{p}0 = index.sub %c1, %pp_{p} : index
  %pq_{p} = index.assume %pq_{p}0 [range(%pq_{p}0, 0, 1)] : index
  %first_{p} = index.cmp eq, %{p}, %c0 : index
  %prev_{p} = scf.if %first_{p} -> (i32) {{
    %z = scalar.constant 0 : i32
    scf.yield %z : i32
  }} else {{
    %v = view.load %out_v[%pq_{p}, %nb] : view<2x{n}xi32> -> i32
    scf.yield %v : i32
  }}
  %val_{p} = scalar.addi %prev_{p}, %one : i32
  view.store %val_{p}, %out_v[%pp_{p}, %me] : i32, view<2x{n}xi32>
"""
    chain = f"""kernel.def target(@rdna4) export("k_gb_chain_{G}") @k_gb_chain_{G}() {{
  %c1 = index.constant 1 : index
  %cg = index.constant {G} : index
  %c64 = index.constant 64 : index
  kernel.launch.config workgroups(%cg, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
}} launch(%p_arg: index, %out: buffer) {{
  %pa = index.assume %p_arg [range(%p_arg, 0, {PHASES - 1})] : index
{common}{phase("pa")}  kernel.return
}}
"""
    bar = f"""kernel.def target(@rdna4) export("k_gb_bar_{G}") @k_gb_bar_{G}() {{
  %c1 = index.constant 1 : index
  %cg = index.constant {G} : index
  %c64 = index.constant 64 : index
  kernel.launch.config workgroups(%cg, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
}} launch(%out: buffer, %cnt: buffer) {{
{common}  %cnt_v = buffer.view %cnt[%origin] : buffer -> view<1xi32>
  %is0 = index.cmp eq, %lane, %c0 : index
  %g32 = scalar.constant {G} : i32
  %cph = index.constant {PHASES} : index
  scf.for %pa0 = [%c0 to %cph step %c1] {{
    %pa = index.assume %pa0 [range(%pa0, 0, {PHASES - 1})] : index
{phase("pa")}    buffer.fence scope(device) ordering(release)
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.if %is0 {{
      %old = view.atomic.rmw<addi> %one, %cnt_v[0] {{ordering = acq_rel, scope = device}} : i32, view<1xi32> -> i32
      %gen = scalar.divui %old, %g32 : i32
      %gen1 = scalar.addi %gen, %one : i32
      %target = scalar.muli %gen1, %g32 : i32
      %t, %v2 = scf.while(%b = %old : i32) -> (i32, i32) {{
        %cur = view.atomic.load %cnt_v[0] {{ordering = acquire, scope = device}} : view<1xi32> -> i32
        %lt = scalar.cmpi slt, %cur, %target : i32
        scf.condition %lt, %b, %cur : i1, i32, i32
      }} do(%pw: i32, %qw: i32) {{
        scf.yield %pw : i32
      }}
      scf.yield
    }}
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    buffer.fence scope(device) ordering(acquire)
  }}
  kernel.return
}}
"""
    return chain, bar


def gb_case(G, mode, check=False):
    n = G * 64
    if mode == "chain":
        body = "".join(f"  %p{i} = check.literal value({i}) : index\n  kernel.launch @k_gb_chain_{G}[](%p{i}, %out) : [](index, tensor<2x{n}xi32>)\n" for i in range(PHASES))
        cnt = ""
    else:
        cnt = "  %cnt = check.generate.fill value(0) : tensor<1xi32>\n"
        body = f"  kernel.launch @k_gb_bar_{G}[](%out, %cnt) : [](tensor<2x{n}xi32>, tensor<1xi32>)\n"
    name = f"gb_{mode}_{G}"
    s = f"check.case public @{name}_case {{\n  %out = check.generate.fill value(0) : tensor<2x{n}xi32>\n{cnt}{body}"
    if check:
        s += f"  %e = check.file.read.npy path(\"gb_expected_{G}.npy\") : tensor<2x{n}xi32>\n  check.expect.equal actual(%out) expected(%e) : tensor<2x{n}xi32>\n"
    s += "  check.return\n}\n"
    if not check:
        s += f"check.benchmark<@{name}_case> @k_case_{name}_bm\n"
    return s


def main():
    cfg, prog = build()
    H, FF, EPS = cfg["hidden"], cfg["ffn"], cfg["rms_eps"]
    names = {"gate": re.search(r"kernel\.launch @(g_mv_mxfp4s_k%d_t4_q8_r\d+)\[%%nFF" % H, prog).group(1),
             "down": re.search(r"kernel\.launch @(g_mv_mxfp4s_k%d_t4_q8_r\d+)\[%%nH" % FF, prog).group(1)}
    kt = {k: block(prog, "kernel.def", v) for k, v in names.items()}
    chain_k = {n: block(prog, "kernel.def", n) for n in ("g_post_add_norm", "g_quant_q8", "g_silu_mul")}
    # standalone (no launch-site facts) the silu kernel's 4-wide tail is unproven: pin n to the FFN width (the program's only launch)
    chain_k["g_silu_mul"] = re.sub(r"\[range\(%n_arg, 4, 65536\), multiple_of\(%n_arg, 4\)\]", f"[range(%n_arg, {cfg['ffn']}, {cfg['ffn']}), multiple_of(%n_arg, 1024)]", chain_k["g_silu_mul"])
    assert f"range(%n_arg, {cfg['ffn']}" in chain_k["g_silu_mul"]
    kernels = dict(chain_k)
    kernels.update({names["gate"]: kt["gate"], names["down"]: kt["down"]})
    fused = {}
    for G in (0,) + GS:
        a = gen_pstep.pmv(kt["gate"], "norm", mats=2, G=G, eps=EPS, count=float(H))
        b = gen_pstep.pmv(kt["down"], "silu", G=G)
        fused[G] = (re.search(r'export\("([^"]+)"', a).group(1), re.search(r'export\("([^"]+)"', b).group(1))
        kernels[fused[G][0]], kernels[fused[G][1]] = a, b
    persist = {}
    for G in GS:
        a, b = gen_pstep.pmv(kt["gate"], G=G), gen_pstep.pmv(kt["down"], G=G)
        persist[G] = (re.search(r'export\("([^"]+)"', a).group(1), re.search(r'export\("([^"]+)"', b).group(1))
        kernels[persist[G][0]], kernels[persist[G][1]] = a, b
    swq = gen_pstep.swiglu_q8(gen_mv(cfg, H, 32))
    swq_name = re.search(r'export\("([^"]+)"', swq).group(1)
    kernels[swq_name] = swq
    funcs = {}
    for t in kernels.values():
        funcs_for(prog, t, funcs)
    lib = HEAD + "".join(funcs.values()) + "\n" + "".join(kdef(t) + "\n" for t in kernels.values())
    for G in GB:
        c, b = gb_kernels(G)
        lib += c + b

    GH, GF = H // 32, FF // 32
    hq, fq = T * H + T * GH * 8, T * FF + T * GF * 8
    def ffn_case(name, variant, check=False):
        d = [f"  %seed = check.literal value(777) : i64",
             f"  %t = check.literal value({T}) : index", f"  %nH = check.literal value({H}) : index", f"  %nFF = check.literal value({FF}) : index",
             f"  %zero_i32 = check.literal value(0) : i32",
             f"  %x = check.generate.random.uniform seed(%seed) range(-2.0 to 2.0) : tensor<{T}x{H}xf32>",
             f"  %y = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{T}x{H}xf32>",
             f"  %nw = check.generate.random.uniform seed(%seed) range(0.5 to 1.5) : tensor<{H}xf32>",
             f"  %wg = check.generate.random.uniform seed(%seed) range(-128 to 127) : tensor<{FF * H // 2}xi8>",
             f"  %wgs = check.generate.random.uniform seed(%seed) range(116 to 124) : tensor<{FF * GH}xi8>",
             f"  %wu = check.generate.random.uniform seed(%seed) range(-128 to 127) : tensor<{FF * H // 2}xi8>",
             f"  %wus = check.generate.random.uniform seed(%seed) range(116 to 124) : tensor<{FF * GH}xi8>",
             f"  %wd = check.generate.random.uniform seed(%seed) range(-128 to 127) : tensor<{H * FF // 2}xi8>",
             f"  %wds = check.generate.random.uniform seed(%seed) range(116 to 124) : tensor<{H * GF}xi8>",
             f"  %yo = check.generate.fill value(0.0) : tensor<{T}x{H}xf32>"]
        TX, TH, TF = f"tensor<{T}x{H}xf32>", f"tensor<{H}xf32>", f"tensor<{T}x{FF}xf32>"
        WG, WGS, WD, WDS = f"tensor<{FF * H // 2}xi8>", f"tensor<{FF * GH}xi8>", f"tensor<{H * FF // 2}xi8>", f"tensor<{H * GF}xi8>"
        def chain(gate, down, sfx=""):
            return [f"  %h{sfx} = check.generate.fill value(0.0) : {TX}", f"  %hq{sfx} = check.generate.fill value(0) : tensor<{hq}xi8>",
                    f"  %f{sfx} = check.generate.fill value(0.0) : {TF}", f"  %fu{sfx} = check.generate.fill value(0.0) : {TF}",
                    f"  %fq{sfx} = check.generate.fill value(0) : tensor<{fq}xi8>",
                    f"  kernel.launch @g_post_add_norm[%t](%t, %y, %nw, %x, %nw, %h{sfx}) : [index](index, {TX}, {TH}, {TX}, {TH}, {TX})",
                    f"  kernel.launch @g_quant_q8[%t, %nH](%t, %nH, %h{sfx}, %hq{sfx}) : [index, index](index, index, {TX}, tensor<{hq}xi8>)",
                    f"  kernel.launch @{gate}[%nFF, %t](%nFF, %t, %zero_i32, %hq{sfx}, %wg, %wgs, %f{sfx}) : [index, index](index, index, i32, tensor<{hq}xi8>, {WG}, {WGS}, {TF})",
                    f"  kernel.launch @{gate}[%nFF, %t](%nFF, %t, %zero_i32, %hq{sfx}, %wu, %wus, %fu{sfx}) : [index, index](index, index, i32, tensor<{hq}xi8>, {WG}, {WGS}, {TF})",
                    f"  kernel.launch @g_silu_mul[%t](%t, %nFF, %f{sfx}, %fu{sfx}) : [index](index, index, {TF}, {TF})",
                    f"  kernel.launch @g_quant_q8[%t, %nFF](%t, %nFF, %f{sfx}, %fq{sfx}) : [index, index](index, index, {TF}, tensor<{fq}xi8>)",
                    f"  kernel.launch @{down}[%nH, %t](%nH, %t, %zero_i32, %fq{sfx}, %wd, %wds, %yo) : [index, index](index, index, i32, tensor<{fq}xi8>, {WD}, {WDS}, {TX})"]
        def fuse(k1, k2, out):
            return [f"  %f2 = check.generate.fill value(0.0) : {TF}", f"  %fu2 = check.generate.fill value(0.0) : {TF}",
                    f"  kernel.launch @{k1}[%nFF, %nFF, %t](%nFF, %nFF, %t, %zero_i32, %x, %nw, %wg, %wgs, %wu, %wus, %f2, %fu2) : [index, index, index](index, index, index, i32, {TX}, {TH}, {WG}, {WGS}, {WG}, {WGS}, {TF}, {TF})",
                    f"  kernel.launch @{k2}[%nH, %t](%nH, %t, %zero_i32, %f2, %fu2, %wd, %wds, {out}) : [index, index](index, index, i32, {TF}, {TF}, {WD}, {WDS}, {TX})"]
        def swq_chain(out, xn="%x"):
            return [f"  %h = check.generate.fill value(0.0) : {TX}", f"  %hq = check.generate.fill value(0) : tensor<{hq}xi8>",
                    f"  %fq3 = check.generate.fill value(0) : tensor<{fq}xi8>",
                    f"  kernel.launch @g_post_add_norm[%t](%t, %y, %nw, {xn}, %nw, %h) : [index](index, {TX}, {TH}, {TX}, {TH}, {TX})",
                    f"  kernel.launch @g_quant_q8[%t, %nH](%t, %nH, %h, %hq) : [index, index](index, index, {TX}, tensor<{hq}xi8>)",
                    f"  kernel.launch @{swq_name}[%nFF, %t](%nFF, %t, %zero_i32, %hq, %wg, %wgs, %wu, %wus, %fq3) : [index, index](index, index, i32, tensor<{hq}xi8>, {WG}, {WGS}, {WG}, {WGS}, tensor<{fq}xi8>)",
                    f"  kernel.launch @{names['down']}[%nH, %t](%nH, %t, %zero_i32, %fq3, %wd, %wds, {out}) : [index, index](index, index, i32, tensor<{fq}xi8>, {WD}, {WDS}, {TX})"]
        if variant == "swq":
            if check:   # the launch chain on (x, y), then the swq chain on a fresh copy of x (same seed): equal outputs
                d += chain(names["gate"], names["down"], "c")
                d += [f"  %x2 = check.generate.random.uniform seed(%seed) range(-2.0 to 2.0) : {TX}", f"  %y3 = check.generate.fill value(0.0) : {TX}"]
                d += swq_chain("%y3", "%x2")
                d.append(f"  check.expect.equal actual(%y3) expected(%yo) : {TX}")
            else:
                d += swq_chain("%yo")
        elif variant == "chain":
            d += chain(names["gate"], names["down"])
        elif variant.startswith("chain_p"):
            G = int(variant[7:])
            d += chain(*persist[G])
        else:
            G = int(variant[6:]) if variant.startswith("fuse_p") else 0
            if check:   # the chain on (x, y) leaves x = x + y, the fused kernels read that residual (the producer's accumulate)
                d += chain(names["gate"], names["down"])
                d.append(f"  %y2 = check.generate.fill value(0.0) : {TX}")
                d += fuse(*fused[G], "%y2")
                d.append(f"  check.expect.equal actual(%y2) expected(%yo) : {TX}")
            else:
                d += fuse(*fused[G], "%yo")
        s = f"check.case public @{name}_case {{\n" + "\n".join(d) + "\n  check.return\n}\n"
        if not check:
            s += f"check.benchmark<@{name}_case> @k_case_{name}_bm\n"
        return s

    bench = lib
    # round 1 (runs 43809470161, 43810902803) measured chain_p<G>, fuse, fuse_p<G> and the grid barrier: VARIANTS=all brings them back
    allv = os.environ.get("VARIANTS") == "all"
    for v in ["swq", "chain"] + ([f"chain_p{G}" for G in GS] + ["fuse"] + [f"fuse_p{G}" for G in GS] if allv else []):
        bench += ffn_case(v, v)
    for G in (GB if allv else ()):
        bench += gb_case(G, "chain") + gb_case(G, "bar")
    open(os.path.join(OUT, "bench.loom"), "w").write(bench)
    test = lib
    for v in ["swq", "fuse"] + [f"fuse_p{G}" for G in GS]:
        test += ffn_case("eq_" + v, v, check=True)
    import numpy as np
    for G in GB:
        e = np.zeros((2, G * 64), np.int32)
        e[0], e[1] = PHASES - 1, PHASES
        np.save(os.path.join(OUT, f"gb_expected_{G}.npy"), e)
        test += gb_case(G, "bar", check=True)
    open(os.path.join(OUT, "test.loom"), "w").write(test)
    print("wrote bench.loom test.loom")


main()
