#!/usr/bin/env python3
"""Generate the gdn_fused kbench suite: the gfx12 fused gated delta-net prefill (gen_gdn_fused.py: g_gdn_fa_rdna4 + g_gdn_fb_rdna4) against the
production chain it replaces (g_gdn_conv, g_gdn_conv_commit, g_gdn_qk_norm, g_gdn_chunk_rdna4, g_gdn_norm_gate), Qwen3.8-27B geometry (16 key heads,
48 value heads grouped, 128 x 128, conv 4).

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   VARIANT: base (bench.loom) or rNNN (bench_rNNN.loom: NNN prefill rows, default 1024)
Writes bench*.loom (timing: the chain, the two fused launches together and each alone, one continuing span) and test.loom (check.cases against a
numpy float64 reference of the whole token mixer on four spans: fresh / continuing, a 2-token span that shifts the conv history; outputs, state
and conv history; run by devlab_kbench.sh's "tests" step with loom-test)."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-port-llm", "starter"))
import numpy as np
# the AMD card's build: MTP drafting (one recurrence form, 8 x 4; verify spans do not publish), chunked GDN in a dual group, the fused pair
for _k, _v in (("GDN_CHUNK", "1"), ("GDN_FUSED2", "1"), ("GDN_ROW_LANES", "8"), ("GDN_COLS", "4"), ("LLM_DRAFT", '{"kind": "mtp"}')):
    os.environ.setdefault(_k, _v)
import gen_gdn as GG
import gen_gdn_fused as GF

HEAD = "amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n\n"
CFG = {"gdn": {"conv_kernel": 4, "k_heads": 16, "v_heads": 48, "head_k": 128, "head_v": 128, "inner": 6144, "v_order": "grouped"},
       "recurrent_layers": [i for i in range(64) if i % 4 != 3]}
EPS = 1e-6
G = GG.gdn_geometry(CFG)
HK, HV, DK, DV, K, C, VD, LB, SB = (G[k] for k in ("HK", "HV", "DK", "DV", "K", "C", "VD", "layer_bytes", "state_bytes"))
RATIO = HV // HK


ABL_A = [("nosolve",), ("nostore",), ("noconv",)]   # timing-only ablations of g_gdn_fa_rdna4 (gen_gdn_fused.phase_a abl=)
ABL_B = [("nonorm",)]                                # of g_gdn_fb_rdna4


def kernels():
    t = GG.gdn_kernels(CFG, EPS)
    NG, LB_ = len(CFG["recurrent_layers"]), G["layer_bytes"]
    for a in ABL_A:
        t += GF.phase_a(G, NG, LB_, EPS, True, RATIO, abl=a)
    for a in ABL_B:
        t += GF.phase_b(G, NG, LB_, EPS, True, RATIO, abl=a)
    return t


def save(d, name, a):
    np.save(os.path.join(d, name + ".npy"), a)


def reference(spans, rng, t):
    """spans: [(length, start position, continues a stored state)]; returns the arrays of one test."""
    x = (rng.standard_normal((t, C)) * 1.5).astype(np.float32)
    conv_w = (rng.standard_normal((C, K)) * 0.5).astype(np.float32)
    a = (-np.exp(rng.uniform(-2, 1, HV))).astype(np.float32); dt = rng.uniform(-1, 1, HV).astype(np.float32)
    nw = rng.uniform(0.5, 1.5, DV).astype(np.float32)
    alpha = rng.standard_normal((t, HV)).astype(np.float32); beta = rng.standard_normal((t, HV)).astype(np.float32)
    z = rng.standard_normal((t, VD)).astype(np.float32)
    ns = len(spans)
    arena = np.zeros(2 * ns * LB // 4, np.float32); meta = np.zeros(4 + 9 * ns, np.int32); meta[0] = ns
    origins = np.zeros((ns, 5), np.int64); o = np.zeros((t, HV, DV)); first = 0
    expect_arena = None; finals = []
    for si, (L, p0, cont) in enumerate(spans):
        rd, wr = 2 * si * LB, (2 * si + 1) * LB
        origins[si] = [rd, wr, 0, p0, 0]; meta[4 + 9 * si:4 + 9 * si + 5] = [L, p0, 0, si, first]
        S = np.zeros((HV, DV, DK)); hist = np.zeros((K - 1, C))
        if cont:
            S0 = (0.3 * rng.standard_normal((HV, DV, DK))).astype(np.float32); h0 = rng.standard_normal((K - 1, C)).astype(np.float32)
            arena[rd // 4:rd // 4 + SB // 4] = S0.reshape(-1); arena[rd // 4 + SB // 4:rd // 4 + SB // 4 + (K - 1) * C] = h0.reshape(-1)
            S = S0.astype(np.float64); hist = h0.astype(np.float64)
        full = np.concatenate([hist, x[first:first + L].astype(np.float64)], 0)
        cv = np.stack([(full[i:i + K] * conv_w.T).sum(0) for i in range(L)])
        sv = cv / (1 + np.exp(-cv))
        qs, ks, vs = sv[:, :HK * DK].reshape(L, HK, DK), sv[:, HK * DK:2 * HK * DK].reshape(L, HK, DK), sv[:, 2 * HK * DK:].reshape(L, HV, DV)
        qs = qs / np.sqrt((qs ** 2).sum(-1, keepdims=True) + EPS) / np.sqrt(DK); ks = ks / np.sqrt((ks ** 2).sum(-1, keepdims=True) + EPS)
        for i in range(L):
            for h in range(HV):
                kh = h // RATIO
                g = np.log1p(np.exp(float(alpha[first + i, h]) + dt[h])) * a[h]
                St = S[h] * np.exp(g)
                d = (vs[i, h] - St @ ks[i, kh]) / (1 + np.exp(-float(beta[first + i, h])))
                S[h] = St + np.outer(d, ks[i, kh]); o[first + i, h] = S[h] @ qs[i, kh]
        finals.append((wr, S, full[-(K - 1):]))
        first += L
    zz = z.reshape(t, HV, DV).astype(np.float64)
    og = o / np.sqrt((o ** 2).mean(-1, keepdims=True) + EPS) * nw * (zz / (1 + np.exp(-zz)))
    expect = arena.copy()
    for wr, S, h1 in finals:
        expect[wr // 4:wr // 4 + SB // 4] = S.astype(np.float32).reshape(-1)
        expect[wr // 4 + SB // 4:wr // 4 + SB // 4 + (K - 1) * C] = h1.astype(np.float32).reshape(-1)
    sscale = float(max(np.abs(S).max() for _, S, _ in finals))
    return dict(x=x, w=conv_w.reshape(-1), a=a, dt=dt, nw=nw, alpha=alpha, beta=beta, z=z, arena=arena, meta=meta,
                orig=origins.reshape(-1).view(np.int32), o=og.reshape(t, VD).astype(np.float32), expect=expect), sscale


def launches(t, ns, AT, MT, OT, WT, which):
    T = {"t": f"tensor<{t}x{C}xf32>", "w": f"tensor<{C * K}xf32>", "hv": f"tensor<{t}x{HV}xf32>", "a": f"tensor<{HV}xf32>",
         "nw": f"tensor<{DV}xf32>", "vd": f"tensor<{t}x{VD}xf32>", "inv": f"tensor<{t}x{2 * HK}xf32>"}
    fa = f"  kernel.launch @g_gdn_fa_rdna4[%t, %spans](%t, %spans, %layer, %meta, %orig, %arena, %x, %w, %alpha, %beta, %a, %dt, %work) : [index, index](index, index, index, {MT}, {OT}, {AT}, {T['t']}, {T['w']}, {T['hv']}, {T['hv']}, {T['a']}, {T['a']}, {WT})\n"
    fb = f"  kernel.launch @g_gdn_fb_rdna4[%spans](%t, %spans, %layer, %meta, %orig, %arena, %x, %work, %nw, %z, %o) : [index](index, index, index, {MT}, {OT}, {AT}, {T['t']}, {WT}, {T['nw']}, {T['vd']}, {T['vd']})\n"
    chain = (f"  kernel.launch @g_gdn_conv[%t](%t, %spans, %layer, %meta, %info, %orig, %arena, %x, %w, %s) : [index](index, index, index, {MT}, tensor<{t}x4xi32>, {OT}, {AT}, {T['t']}, {T['w']}, {T['t']})\n"
             f"  kernel.launch @g_gdn_conv_commit[%spans](%t, %spans, %layer, %meta, %orig, %arena, %x) : [index](index, index, index, {MT}, {OT}, {AT}, {T['t']})\n"
             f"  kernel.launch @g_gdn_qk_norm[%t](%t, %s, %inv) : [index](index, {T['t']}, {T['inv']})\n"
             f"  kernel.launch @g_gdn_chunk_rdna4[%spans](%t, %spans, %layer, %meta, %orig, %arena, %s, %inv, %alpha, %beta, %a, %dt, %raw) : [index](index, index, index, {MT}, {OT}, {AT}, {T['t']}, {T['inv']}, {T['hv']}, {T['hv']}, {T['a']}, {T['a']}, {T['vd']})\n"
             f"  kernel.launch @g_gdn_norm_gate[%t](%t, %info, %raw, %nw, %z, %o) : [index](index, tensor<{t}x4xi32>, {T['vd']}, {T['nw']}, {T['vd']}, {T['vd']})\n")
    if which.startswith("fa_"):
        return fa.replace("@g_gdn_fa_rdna4[", f"@g_gdn_fa_rdna4_{which[3:]}[")
    if which.startswith("fb_"):
        return fb.replace("@g_gdn_fb_rdna4[", f"@g_gdn_fb_rdna4_{which[3:]}[")
    return {"fa": fa, "fb": fb, "fused": fa + fb, "chain": chain}[which]


def operands(pfx, t, ns, AT, MT, OT, WT, chain=False):
    s = f"""  %t = check.literal value({t}) : index
  %spans = check.literal value({ns}) : index
  %layer = check.literal value(0) : index
  %meta = check.file.read.npy path("{pfx}_meta.npy") : {MT}
  %orig = check.file.read.npy path("{pfx}_orig.npy") : {OT}
  %arena = check.file.read.npy path("{pfx}_arena.npy") : {AT}
  %x = check.file.read.npy path("{pfx}_x.npy") : tensor<{t}x{C}xf32>
  %w = check.file.read.npy path("{pfx}_w.npy") : tensor<{C * K}xf32>
  %alpha = check.file.read.npy path("{pfx}_alpha.npy") : tensor<{t}x{HV}xf32>
  %beta = check.file.read.npy path("{pfx}_beta.npy") : tensor<{t}x{HV}xf32>
  %a = check.file.read.npy path("{pfx}_a.npy") : tensor<{HV}xf32>
  %dt = check.file.read.npy path("{pfx}_dt.npy") : tensor<{HV}xf32>
  %nw = check.file.read.npy path("{pfx}_nw.npy") : tensor<{DV}xf32>
  %z = check.file.read.npy path("{pfx}_z.npy") : tensor<{t}x{VD}xf32>
  %work = check.generate.fill value(0.0) : {WT}
  %o = check.generate.fill value(0.0) : tensor<{t}x{VD}xf32>
"""
    if chain:
        s += f"""  %info = check.file.read.npy path("{pfx}_info.npy") : tensor<{t}x4xi32>
  %s = check.generate.fill value(0.0) : tensor<{t}x{C}xf32>
  %inv = check.generate.fill value(0.0) : tensor<{t}x{2 * HK}xf32>
  %raw = check.generate.fill value(0.0) : tensor<{t}x{VD}xf32>
"""
    return s


def types(t, ns):
    nslots = (t + 31) // 32 + ns + 1
    return (f"tensor<{2 * ns * LB // 4}xf32>", f"tensor<{4 + 9 * ns}xi32>", f"tensor<{10 * ns}xi32>", f"tensor<{nslots * GF.nslots_bytes(G) // 4}xf32>")


def write_inputs(d, pfx, arrs, t):
    for k, v in arrs.items():
        if k not in ("o", "expect"):
            save(d, f"{pfx}_{k}", v)
    info = np.zeros((t, 4), np.int32); info[:, 2] = 1
    save(d, f"{pfx}_info", info)


def tests(d):
    rng = np.random.default_rng(11)
    spans = [(130, 0, False), (64, 10, True), (97, 0, False), (2, 20, True)]
    t = sum(L for L, _, _ in spans); ns = len(spans)
    arrs, sscale = reference(spans, rng, t)
    write_inputs(d, "tf", arrs, t); save(d, "tf_eo", arrs["o"]); save(d, "tf_expect", arrs["expect"])
    AT, MT, OT, WT = types(t, ns)
    oscale = float(np.abs(arrs["o"]).max())
    return [f"""check.case public @gdn_fused_spans {{
{operands("tf", t, ns, AT, MT, OT, WT)}  %eo = check.file.read.npy path("tf_eo.npy") : tensor<{t}x{VD}xf32>
  %expect = check.file.read.npy path("tf_expect.npy") : {AT}
{launches(t, ns, AT, MT, OT, WT, "fused")}  check.expect.close actual(%o) expected(%eo) atol({1e-2 * oscale:.3e}) rtol(2.0e-2) nan(different) : tensor<{t}x{VD}xf32>
  check.expect.close actual(%arena) expected(%expect) atol({3e-3 * sscale:.3e}) rtol(1.0e-2) nan(different) : {AT}
  check.return
}}
"""]


def bench(d, rows):
    rng = np.random.default_rng(5)
    spans = [(rows, 4096, True)]
    t, ns = rows, 1
    x = (rng.standard_normal((t, C)) * 1.5).astype(np.float32)
    arena = np.zeros(2 * LB // 4, np.float32); arena[:SB // 4] = (0.3 * rng.standard_normal(SB // 4)).astype(np.float32)
    meta = np.zeros(13, np.int32); meta[0] = 1; meta[4:9] = [t, 4096, 0, 0, 0]
    orig = np.array([0, LB, 0, 4096, 0], np.int64).view(np.int32)
    arrs = dict(x=x, w=(rng.standard_normal(C * K) * 0.5).astype(np.float32), a=(-np.exp(rng.uniform(-2, 1, HV))).astype(np.float32),
                dt=rng.uniform(-1, 1, HV).astype(np.float32), nw=np.ones(DV, np.float32), alpha=rng.standard_normal((t, HV)).astype(np.float32),
                beta=rng.standard_normal((t, HV)).astype(np.float32), z=rng.standard_normal((t, VD)).astype(np.float32), arena=arena, meta=meta, orig=orig)
    pfx = f"b{rows}"
    write_inputs(d, pfx, arrs, t)
    AT, MT, OT, WT = types(t, ns)
    out = []
    for which in ["chain", "fused", "fa", "fb"] + ["fa_" + "_".join(a) for a in ABL_A] + ["fb_" + "_".join(a) for a in ABL_B]:
        out.append(f"""check.case public @{which}_{rows}_case {{
{operands(pfx, t, ns, AT, MT, OT, WT, chain=which == "chain")}{launches(t, ns, AT, MT, OT, WT, which)}  check.return
}}
check.benchmark<@{which}_{rows}_case> @{which}_{rows}_bm
""")
    return out


if __name__ == "__main__" and len(sys.argv) > 2:
    d = sys.argv[2]
    os.makedirs(d, exist_ok=True)
    ktext = kernels()
    for v in (sys.argv[3:] or ["base"]):
        rows = int(v[1:]) if re.fullmatch(r"r\d+", v) else 1024
        open(os.path.join(d, "bench%s.loom" % ("" if v == "base" else "_" + v)), "w").write(HEAD + ktext + "\n" + "\n".join(bench(d, rows)))
    text = re.sub(r"check\.expect\.close (actual\(.*\)) : (.*)$", r"check.expect.approximate \1 : \2, \2", HEAD + ktext + "\n" + "\n".join(tests(d)), flags=re.M)
    open(os.path.join(d, "test.loom"), "w").write(text)
