#!/usr/bin/env python3
"""Generate the gdn_fold kbench suite: the gated delta-net part of a drafted verify step (MTP depth 3: one 4-token speculative span,
3 accepted) as the AMD MXFP4 card runs it today (capture + conv + commit + q/k norms + recurrence per layer in the verification, then
restore + conv + commit + q/k norms + recurrence per layer in the replay) against decode.spec.gdn_fold (gen_gdn.fold_kernels:
g_gdn_conv_bank + commit + q/k norms + g_gdn_recurrence_rec per layer, then one g_gdn_fold and one g_gdn_fold_hist for all layers),
Qwen3.8-27B geometry (48 gated delta-net layers, 16 key heads, 48 value heads grouped, 128 x 128, conv 4).

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   VARIANT: base (bench.loom: 48 layers) or lNN (bench_lNN.loom: NN layers)
Writes bench*.loom (timing: whole GDN part old / new, verification alone, replay alone, fold alone) and test.loom (the fold path's
verification outputs and published state + conv history EQUAL to the replay path's on three speculative spans, and close to numpy)."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "../../../.."))
ST = os.path.join(ROOT, "skills", "loom-port-llm", "starter")
sys.path.insert(0, ST)
sys.path.insert(0, os.path.join(ST, "draft"))
import numpy as np
# the AMD card's build: MTP drafting (one recurrence form, 8 x 4; verify spans do not publish), chunked GDN (dual group), the fold
for _k, _v in (("GDN_CHUNK", "1"), ("GDN_FOLD", "1"), ("GDN_ROW_LANES", "8"), ("GDN_COLS", "4"), ("LLM_DRAFT", '{"kind": "mtp"}')):
    os.environ.setdefault(_k, _v)
import gen_gdn as GG
import gen_draft as GD

HEAD = "amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n\n"
CFG = {"gdn": {"conv_kernel": 4, "k_heads": 16, "v_heads": 48, "head_k": 128, "head_v": 128, "inner": 6144, "v_order": "grouped"},
       "recurrent_layers": [i for i in range(64) if i % 4 != 3]}
EPS = 1e-6
G = GG.gdn_geometry(CFG)
HK, HV, DK, DV, K, C, VD, LB, SB = (G[k] for k in ("HK", "HV", "DK", "DV", "K", "C", "VD", "layer_bytes", "state_bytes"))
NG = len(CFG["recurrent_layers"])
RATIO = HV // HK
FW = GG.FOLD_W


def kernels():
    return GG.gdn_kernels(CFG, EPS) + "\n" + GD.gdn_copy_kernels(G, NG)


def save(d, name, a):
    np.save(os.path.join(d, name + ".npy"), a)


def types(t, ns, nl):
    T = dict(AT=f"tensor<{2 * ns * LB // 4 if nl is None else nl * LB // 4}xf32>", MT=f"tensor<{4 + 9 * ns}xi32>", OT=f"tensor<{10 * ns}xi32>",
             IT=f"tensor<{t}x4xi32>", XT=f"tensor<{t}x{C}xf32>", ST=f"tensor<{t}x{2 * HK}xf32>", VT=f"tensor<{t}x{VD}xf32>", HT=f"tensor<{t}x{HV}xf32>",
             WT=f"tensor<{C * K}xf32>", PT=f"tensor<{HV}xf32>")
    L = nl or 1
    T.update(BX=f"tensor<{L * ns * FW}x{C}xf32>", BK=f"tensor<{L * ns * FW}x{HK * DK}xf32>", BD=f"tensor<{L * ns * FW}x{VD + HV}xf32>",
             CT=f"tensor<{NG * t * (C + 2 * HV)}xf32>")
    return T


def L_verify_old(T, lay, arena="%arena", s="%s", inv="%inv", raw="%raw", cap=True):
    out = f"  kernel.launch @g_gdn_capture[%t](%t, {lay}, %qkv, %alpha, %beta, %cap) : [index](index, index, {T['XT']}, {T['HT']}, {T['HT']}, {T['CT']})\n" if cap else ""
    return out + (f"  kernel.launch @g_gdn_conv[%t](%t, %spans, {lay}, %meta, %info, %orig, {arena}, %qkv, %w, {s}) : [index](index, index, index, {T['MT']}, {T['IT']}, {T['OT']}, {T['AT']}, {T['XT']}, {T['WT']}, {T['XT']})\n"
                  f"  kernel.launch @g_gdn_conv_commit[%spans](%t, %spans, {lay}, %meta, %orig, {arena}, %qkv) : [index](index, index, index, {T['MT']}, {T['OT']}, {T['AT']}, {T['XT']})\n"
                  f"  kernel.launch @g_gdn_qk_norm[%t](%t, {s}, {inv}) : [index](index, {T['XT']}, {T['ST']})\n"
                  f"  kernel.launch @g_gdn_recurrence_rdna4[%spans](%t, %spans, {lay}, %meta, %orig, {arena}, {s}, {inv}, %alpha, %beta, %a, %dt, {raw}) : [index](index, index, index, {T['MT']}, {T['OT']}, {T['AT']}, {T['XT']}, {T['ST']}, {T['HT']}, {T['HT']}, {T['PT']}, {T['PT']}, {T['VT']})\n")


def L_replay_old(T, lay, arena="%arena", s="%s", inv="%inv", raw="%raw", cap=True):
    out = f"  kernel.launch @g_gdn_restore[%t](%t, {lay}, %qkv, %alpha, %beta, %cap) : [index](index, index, {T['XT']}, {T['HT']}, {T['HT']}, {T['CT']})\n" if cap else ""
    return out + (f"  kernel.launch @g_gdn_conv[%t](%t, %spans, {lay}, %comm, %info2, %orig, {arena}, %qkv, %w, {s}) : [index](index, index, index, {T['MT']}, {T['IT']}, {T['OT']}, {T['AT']}, {T['XT']}, {T['WT']}, {T['XT']})\n"
                  f"  kernel.launch @g_gdn_conv_commit[%spans](%t, %spans, {lay}, %comm, %orig, {arena}, %qkv) : [index](index, index, index, {T['MT']}, {T['OT']}, {T['AT']}, {T['XT']})\n"
                  f"  kernel.launch @g_gdn_qk_norm[%t](%t, {s}, {inv}) : [index](index, {T['XT']}, {T['ST']})\n"
                  f"  kernel.launch @g_gdn_recurrence_rdna4[%spans](%t, %spans, {lay}, %comm, %orig, {arena}, {s}, {inv}, %alpha, %beta, %a, %dt, {raw}) : [index](index, index, index, {T['MT']}, {T['OT']}, {T['AT']}, {T['XT']}, {T['ST']}, {T['HT']}, {T['HT']}, {T['PT']}, {T['PT']}, {T['VT']})\n")


def L_verify_new(T, lay, arena="%arena", s="%s", inv="%inv", raw="%raw"):
    return (f"  kernel.launch @g_gdn_conv_bank[%t](%t, %spans, {lay}, %meta, %info, %orig, {arena}, %qkv, %w, {s}, %bx, %bk) : [index](index, index, index, {T['MT']}, {T['IT']}, {T['OT']}, {T['AT']}, {T['XT']}, {T['WT']}, {T['XT']}, {T['BX']}, {T['BK']})\n"
            f"  kernel.launch @g_gdn_conv_commit[%spans](%t, %spans, {lay}, %meta, %orig, {arena}, %qkv) : [index](index, index, index, {T['MT']}, {T['OT']}, {T['AT']}, {T['XT']})\n"
            f"  kernel.launch @g_gdn_qk_norm[%t](%t, {s}, {inv}) : [index](index, {T['XT']}, {T['ST']})\n"
            f"  kernel.launch @g_gdn_recurrence_rec_rdna4[%spans](%t, %spans, {lay}, %meta, %orig, {arena}, {s}, {inv}, %alpha, %beta, %a, %dt, {raw}, %bd) : [index](index, index, index, {T['MT']}, {T['OT']}, {T['AT']}, {T['XT']}, {T['ST']}, {T['HT']}, {T['HT']}, {T['PT']}, {T['PT']}, {T['VT']}, {T['BD']})\n")


def L_fold(T, arena="%arena"):
    return (f"  kernel.launch @g_gdn_fold[%spans, %layers](%t, %spans, %meta, %comm, %orig, {arena}, %bk, %bd) : [index, index](index, index, {T['MT']}, {T['MT']}, {T['OT']}, {T['AT']}, {T['BK']}, {T['BD']})\n"
            f"  kernel.launch @g_gdn_fold_hist[%spans, %layers](%t, %spans, %meta, %comm, %orig, {arena}, %bx) : [index, index](index, index, {T['MT']}, {T['MT']}, {T['OT']}, {T['AT']}, {T['BX']})\n")


def tests(d):
    """Three speculative spans: 4 tokens at a fork (reader != writer) accepting 2, 4 tokens accepting 4, 3 tokens accepting 1."""
    rng = np.random.default_rng(7)
    spans = [(4, 10, 2, True), (4, 20, 4, False), (3, 30, 1, False)]
    t = sum(L for L, _, _, _ in spans); ns = len(spans)
    x = (rng.standard_normal((t, C)) * 1.5).astype(np.float32)
    w = (rng.standard_normal((C, K)) * 0.5).astype(np.float32)
    a = (-np.exp(rng.uniform(-2, 1, HV))).astype(np.float32); dt = rng.uniform(-1, 1, HV).astype(np.float32)
    alpha = rng.standard_normal((t, HV)).astype(np.float32); beta = rng.standard_normal((t, HV)).astype(np.float32)
    arena = np.zeros(2 * ns * LB // 4, np.float32); meta = np.zeros(4 + 9 * ns, np.int32); meta[0] = ns
    comm = np.zeros(4 + 9 * ns, np.int32); comm[0] = ns
    orig = np.zeros((ns, 5), np.int64); info = np.zeros((t, 4), np.int32); info2 = np.zeros((t, 4), np.int32)
    first = 0; finals = []
    for si, (L, p0, acc, fork) in enumerate(spans):
        rd, wr = 2 * si * LB, (2 * si + 1) * LB
        orig[si] = [rd, wr, 0, p0, 0] if fork else [rd, rd, 0, 0, 0]
        S0 = (0.3 * rng.standard_normal((HV, DV, DK))).astype(np.float32); h0 = rng.standard_normal((K - 1, C)).astype(np.float32)
        arena[rd // 4:rd // 4 + SB // 4] = S0.reshape(-1); arena[rd // 4 + SB // 4:rd // 4 + SB // 4 + (K - 1) * C] = h0.reshape(-1)
        meta[4 + 9 * si:13 + 9 * si] = [L, p0, 0, si, first, 0, 4, 0, 0]
        comm[4 + 9 * si:13 + 9 * si] = [acc, p0, 0, si, first, 0, 0, 0, 0]
        info[first:first + L] = [[0, p0 + i, 1, si] for i in range(L)]
        info2[first:first + L] = [[0, p0 + i, int(i < acc), si] for i in range(L)]
        full = np.concatenate([h0, x[first:first + acc]], 0).astype(np.float64)
        cv = np.stack([(full[i:i + K] * w.T).sum(0) for i in range(acc)]); sv = cv / (1 + np.exp(-cv))
        ks, vs = sv[:, HK * DK:2 * HK * DK].reshape(acc, HK, DK), sv[:, 2 * HK * DK:].reshape(acc, HV, DV)
        ks = ks / np.sqrt((ks ** 2).sum(-1, keepdims=True) + EPS)
        S = S0.astype(np.float64)
        for i in range(acc):
            for h in range(HV):
                St = S[h] * np.exp(np.log1p(np.exp(float(alpha[first + i, h]) + dt[h])) * a[h])
                dd = (vs[i, h] - St @ ks[i, h // RATIO]) / (1 + np.exp(-float(beta[first + i, h])))
                S[h] = St + np.outer(dd, ks[i, h // RATIO])
        finals.append((wr if fork else rd, S, full[-(K - 1):]))
        first += L
    expect = arena.copy()
    for w_, S, h1 in finals:
        expect[w_ // 4:w_ // 4 + SB // 4] = S.astype(np.float32).reshape(-1)
        expect[w_ // 4 + SB // 4:w_ // 4 + SB // 4 + (K - 1) * C] = h1.astype(np.float32).reshape(-1)
    for k_, v_ in dict(meta=meta, comm=comm, orig=orig.reshape(-1).view(np.int32), info=info, info2=info2, arena=arena, expect=expect,
                       x=x, w=w.reshape(-1), a=a, dt=dt, alpha=alpha, beta=beta).items():
        save(d, "tf_" + k_, v_)
    T = types(t, ns, None)
    sscale = float(max(np.abs(S).max() for _, S, _ in finals))
    ops = "".join(f'  %{n} = check.file.read.npy path("tf_{f}.npy") : {T[ty]}\n' for n, f, ty in (
        ("meta", "meta", "MT"), ("comm", "comm", "MT"), ("orig", "orig", "OT"), ("info", "info", "IT"), ("info2", "info2", "IT"),
        ("arena1", "arena", "AT"), ("arena2", "arena", "AT"), ("expect", "expect", "AT"), ("qkv", "x", "XT"), ("w", "w", "WT"),
        ("a", "a", "PT"), ("dt", "dt", "PT"), ("alpha", "alpha", "HT"), ("beta", "beta", "HT")))
    fills = "".join(f"  %{n} = check.generate.fill value(0.0) : {T[ty]}\n" for n, ty in (
        ("s1", "XT"), ("i1", "ST"), ("r1", "VT"), ("bx", "BX"), ("bk", "BK"), ("bd", "BD"), ("s2", "XT"), ("i2", "ST"), ("r2", "VT"),
        ("s3", "XT"), ("i3", "ST"), ("r3", "VT")))
    return [f"""check.case public @gdn_fold_vs_replay {{
  %t = check.literal value({t}) : index
  %spans = check.literal value({ns}) : index
  %layer = check.literal value(0) : index
  %layers = check.literal value(1) : index
{ops}{fills}{L_verify_new(T, "%layer", "%arena1", "%s1", "%i1", "%r1")}{L_fold(T, "%arena1")}{L_verify_old(T, "%layer", "%arena2", "%s2", "%i2", "%r2", cap=False)}{L_replay_old(T, "%layer", "%arena2", "%s3", "%i3", "%r3", cap=False)}  check.expect.equal actual(%r1) expected(%r2) : {T['VT']}
  check.expect.equal actual(%arena1) expected(%arena2) : {T['AT']}
  check.expect.close actual(%arena2) expected(%expect) atol({3e-3 * sscale:.3e}) rtol(1.0e-2) nan(different) : {T['AT']}
  check.return
}}
"""]


def bench(d, nl):
    """One 4-token speculative span at position 4096 continuing its slot (reader == writer), 3 accepted, `nl` layers."""
    rng = np.random.default_rng(5)
    t, ns, acc = 4, 1, 3
    meta = np.zeros(13, np.int32); meta[0] = 1; meta[4:13] = [t, 4096, 0, 0, 0, 0, 4, 0, 0]
    comm = meta.copy(); comm[4] = acc; comm[10] = 0
    orig = np.array([0, 0, 0, 0, 0], np.int64).view(np.int32)
    info = np.array([[0, 4096 + i, 1, 0] for i in range(t)], np.int32); info2 = np.array([[0, 4096 + i, int(i < acc), 0] for i in range(t)], np.int32)
    pfx = f"b{nl}"
    for k_, v_ in dict(meta=meta, comm=comm, orig=orig, info=info, info2=info2, x=(rng.standard_normal((t, C))).astype(np.float32),
                       w=(rng.standard_normal(C * K) * 0.5).astype(np.float32), a=(-np.exp(rng.uniform(-2, 1, HV))).astype(np.float32),
                       dt=rng.uniform(-1, 1, HV).astype(np.float32), alpha=rng.standard_normal((t, HV)).astype(np.float32),
                       beta=rng.standard_normal((t, HV)).astype(np.float32)).items():
        save(d, f"{pfx}_{k_}", v_)
    T = types(t, ns, nl)
    ops = "".join(f'  %{n} = check.file.read.npy path("{pfx}_{f}.npy") : {T[ty]}\n' for n, f, ty in (
        ("meta", "meta", "MT"), ("comm", "comm", "MT"), ("orig", "orig", "OT"), ("info", "info", "IT"), ("info2", "info2", "IT"),
        ("qkv", "x", "XT"), ("w", "w", "WT"), ("a", "a", "PT"), ("dt", "dt", "PT"), ("alpha", "alpha", "HT"), ("beta", "beta", "HT")))
    ops += "".join(f"  %{n} = check.generate.fill value({v}) : {T[ty]}\n" for n, ty, v in (
        ("arena", "AT", "0.01"), ("s", "XT", "0.0"), ("inv", "ST", "0.0"), ("raw", "VT", "0.0"), ("cap", "CT", "0.0"),
        ("bx", "BX", "0.0"), ("bk", "BK", "0.0"), ("bd", "BD", "0.0")))
    lays = "".join(f"  %l{i} = check.literal value({i}) : index\n" for i in range(nl))
    head = f"  %t = check.literal value({t}) : index\n  %spans = check.literal value({ns}) : index\n  %layers = check.literal value({nl}) : index\n" + lays + ops
    body = {
        "old": "".join(L_verify_old(T, f"%l{i}") for i in range(nl)) + "".join(L_replay_old(T, f"%l{i}") for i in range(nl)),
        "new": "".join(L_verify_new(T, f"%l{i}") for i in range(nl)) + L_fold(T),
        "vold": "".join(L_verify_old(T, f"%l{i}") for i in range(nl)),
        "vnew": "".join(L_verify_new(T, f"%l{i}") for i in range(nl)),
        "replay": "".join(L_replay_old(T, f"%l{i}") for i in range(nl)),
        "fold": L_fold(T),
    }
    return [f"""check.case public @{k}_{nl}_case {{
{head}{v}  check.return
}}
check.benchmark<@{k}_{nl}_case> @{k}_{nl}_bm
""" for k, v in body.items()]


if __name__ == "__main__" and len(sys.argv) > 2:
    d = sys.argv[2]
    os.makedirs(d, exist_ok=True)
    ktext = kernels()
    for v in (sys.argv[3:] or ["base"]):
        nl = int(v[1:]) if re.fullmatch(r"l\d+", v) else NG
        open(os.path.join(d, "bench%s.loom" % ("" if v == "base" else "_" + v)), "w").write(HEAD + ktext + "\n" + "\n".join(bench(d, nl)))
    text = re.sub(r"check\.expect\.close (actual\(.*\)) : (.*)$", r"check.expect.approximate \1 : \2, \2", HEAD + ktext + "\n" + "\n".join(tests(d)), flags=re.M)
    open(os.path.join(d, "test.loom"), "w").write(text)
