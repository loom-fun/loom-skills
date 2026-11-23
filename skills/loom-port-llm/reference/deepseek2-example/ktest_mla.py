"""Kernel tests of the MLA attention (gen_mla.py + gen_llm's split-K / listed attention at head_dim kv_lora + rope), on the
real tensors of one layer, against numpy (gguf dequantize). make_kernel_tests.py calls cases(ns) with its globals.

Cases: g_mla_rms_q; g_mla_rope_store (rotary Q halves and the F16 latent cache row at plane = layer); g_mla_kb_<fmt> and
g_mla_vb_<fmt> (per-head matvecs, 9 tokens: two token groups); and the whole chain on a span that continues stored KV
(history written by g_mla_rope_store, then the span: rope store -> k_b -> decode split-K + combine AND the prefill listed
split-K + combine (g16_attention_split / combine: the body of the _listed kernels the program dispatches indirectly; loom-test cannot
launch from a device count) -> v_b), both paths vs numpy, with a dirty KV pool and padding tokens."""
import os
import numpy as np
import gen_mla_dec   # decode attention: the WMMA kernels (MLA_DEC=wmma, default) or the per-head split-K (MLA_DEC=split)

DSPLIT, DCOMB = ("g_mla_dsplit", "g_mla_dcombine") if gen_mla_dec.on() else ("g_attention_split", "g_attention_combine")
SDEC = max(64, gen_mla_dec.SMAX)   # partial slots per (token, head): enough for either layout


def rope_norm(a, pos, theta):
    """llama.cpp NORM RoPE over the last axis (pairs 2p, 2p+1) of a (tokens, ..., RD) at positions pos."""
    rd = a.shape[-1]
    inv = theta ** (-np.arange(rd // 2) * 2.0 / rd)
    ang = pos.astype(np.float64)[:, None] * inv[None]
    shp = (len(pos),) + (1,) * (a.ndim - 2) + (rd // 2,)
    c, s = np.cos(ang).reshape(shp), np.sin(ang).reshape(shp)
    o = a.astype(np.float64).copy()
    a1, a2 = a[..., 0::2].astype(np.float64), a[..., 1::2].astype(np.float64)
    o[..., 0::2] = a1 * c - a2 * s
    o[..., 1::2] = a2 * c + a1 * s
    return o


def cases(ns):
    cfg, T, deq, raw, save, ty, rng, out_cases = ns["cfg"], ns["T"], ns["deq"], ns["raw"], ns["save"], ns["ty"], ns["rng"], ns["cases"]
    list_launch, flash_covered = ns["list_launch"], ns["flash_covered"]
    m = cfg["mla"]
    QH, KL, RD, NOPE, VD, QL = cfg["heads"], m["kv_lora"], m["rope_dims"], m["nope_dims"], m["v_dims"], m["q_lora"]
    D, QB, eps, theta, scale = KL + RD, NOPE + RD, cfg["rms_eps"], cfg["rope_theta"], m["scale"]
    li = 1 if cfg["layers"] > 1 else 0
    b = f"blk.{li}."
    fmt = lambda n: T[b + n].tensor_type.name.lower()
    def deq3(n):   # GGUF [K, N, heads] -> (heads, N, K)
        t = T[b + n]
        from gguf.quants import dequantize
        return dequantize(np.asarray(t.data), t.tensor_type).reshape([int(x) for x in reversed(t.shape)]).astype(np.float64)
    def rms(x, w):
        x = x.astype(np.float64)
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
    qan, kvan = deq(b + "attn_q_a_norm.weight").astype(np.float32), deq(b + "attn_kv_a_norm.weight").astype(np.float32)
    kb, vb = deq3("attn_k_b.weight"), deq3("attn_v_b.weight")            # (QH, KL, NOPE), (QH, VD, KL)
    kbr, vbr = raw(b + "attn_k_b.weight"), raw(b + "attn_v_b.weight")
    save("mla_qan", qan); save("mla_kvan", kvan); save("mla_kb_w", kbr); save("mla_vb_w", vbr)
    KBT, VBT = ty(kbr), ty(vbr)

    # 1. q_a norm (in place)
    t = 4
    x = (rng.standard_normal((t, QL)) * 0.8).astype(np.float32)
    save("mla_rq_x", x); save("mla_rq_e", rms(x, qan).astype(np.float32))
    out_cases.append(f"""check.case public @mla_rms_q {{
  %t = check.literal value({t}) : index
  %x = check.file.read.npy path("mla_rq_x.npy") : tensor<{t}x{QL}xf32>
  %w = check.file.read.npy path("mla_qan.npy") : tensor<{QL}xf32>
  %e = check.file.read.npy path("mla_rq_e.npy") : tensor<{t}x{QL}xf32>
  kernel.launch @g_mla_rms_q[%t](%t, %x, %w) : [index](index, tensor<{t}x{QL}xf32>, tensor<{QL}xf32>)
  check.expect.close actual(%x) expected(%e) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{QL}xf32>
  check.return
}}
""")

    # 2. rope + latent cache store at plane = layer 1 (pool = ctx positions per plane), positions incl. a large one
    t, ctx, layer = 4, 1024, 1
    pos = np.array([0, 1, 7, 1000])
    qb = rng.standard_normal((t, QH, QB)).astype(np.float32)
    kv = (rng.standard_normal((t, D)) * 2).astype(np.float32)
    qe = np.zeros((t, QH, D), np.float32)
    qe[..., KL:] = rope_norm(qb[..., NOPE:], pos, theta)
    rowk = np.concatenate([rms(kv[:, :KL], kvan), rope_norm(kv[:, KL:], pos, theta)], -1).astype(np.float16)
    blocks = ctx // 16
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)
    info = np.zeros((t, 4), np.int32); info[:, 1] = pos; info[:, 2] = 1
    arena = np.zeros((2, ctx, D), np.float16); arena[layer, pos] = rowk
    for n, a in [("mla_rs_qb", qb.reshape(t, -1)), ("mla_rs_kv", kv), ("mla_rs_qe", qe.reshape(t, -1)), ("mla_rs_info", info),
                 ("mla_rs_orig", orig), ("mla_rs_arena", arena.reshape(-1))]:
        save(n, a)
    AR = f"tensor<{2 * ctx * D}xf16>"
    out_cases.append(f"""check.case public @mla_rope_store {{
  %t = check.literal value({t}) : index
  %layer = check.literal value({layer}) : index
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({ctx}) : index
  %qb = check.file.read.npy path("mla_rs_qb.npy") : tensor<{t}x{QH * QB}xf32>
  %kv = check.file.read.npy path("mla_rs_kv.npy") : tensor<{t}x{D}xf32>
  %w = check.file.read.npy path("mla_kvan.npy") : tensor<{KL}xf32>
  %info = check.file.read.npy path("mla_rs_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("mla_rs_orig.npy") : tensor<{10 + blocks}xi32>
  %qe = check.file.read.npy path("mla_rs_qe.npy") : tensor<{t}x{QH * D}xf32>
  %ae = check.file.read.npy path("mla_rs_arena.npy") : {AR}
  %q = check.generate.fill value(0.0) : tensor<{t}x{QH * D}xf32>
  %arena = check.generate.fill value(0.0) : {AR}
  kernel.launch @g_mla_rope_store[%t](%t, %layer, %ctx, %pool, %qb, %kv, %w, %q, %info, %orig, %arena) : [index](index, index, index, index, tensor<{t}x{QH * QB}xf32>, tensor<{t}x{D}xf32>, tensor<{KL}xf32>, tensor<{t}x{QH * D}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, {AR})
  check.expect.close actual(%q) expected(%qe) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{QH * D}xf32>
  check.expect.close actual(%arena) expected(%ae) atol(2.0e-3) rtol(4.0e-3) nan(different) : {AR}
  check.return
}}
""")

    # 3. per-head matvecs, 9 tokens (two 8-token groups: the second one partial)
    t = 9
    xq = rng.standard_normal((t, QH, QB)).astype(np.float32)
    ykb = np.zeros((t, QH, D), np.float64); ykb[..., :KL] = np.einsum("hrk,thk->thr", kb, xq[..., :NOPE].astype(np.float64))
    xo = (rng.standard_normal((t, QH, D)) * 0.05).astype(np.float32)
    yvb = np.einsum("hvk,thk->thv", vb, xo[..., :KL].astype(np.float64))
    for n, a in [("mla_kb_x", xq.reshape(t, -1)), ("mla_kb_e", ykb.reshape(t, -1).astype(np.float32)),
                 ("mla_vb_x", xo.reshape(t, -1)), ("mla_vb_e", yvb.reshape(t, -1).astype(np.float32))]:
        save(n, a)
    out_cases.append(f"""check.case public @mla_kb {{
  %t = check.literal value({t}) : index
  %x = check.file.read.npy path("mla_kb_x.npy") : tensor<{t}x{QH * QB}xf32>
  %w = check.file.read.npy path("mla_kb_w.npy") : {KBT}
  %e = check.file.read.npy path("mla_kb_e.npy") : tensor<{t}x{QH * D}xf32>
  %y = check.generate.fill value(0.0) : tensor<{t}x{QH * D}xf32>
  kernel.launch @g_mla_kb_{fmt("attn_k_b.weight")}[%t](%t, %x, %w, %y) : [index](index, tensor<{t}x{QH * QB}xf32>, {KBT}, tensor<{t}x{QH * D}xf32>)
  check.expect.close actual(%y) expected(%e) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{QH * D}xf32>
  check.return
}}
check.case public @mla_vb {{
  %t = check.literal value({t}) : index
  %x = check.file.read.npy path("mla_vb_x.npy") : tensor<{t}x{QH * D}xf32>
  %w = check.file.read.npy path("mla_vb_w.npy") : {VBT}
  %e = check.file.read.npy path("mla_vb_e.npy") : tensor<{t}x{QH * VD}xf32>
  %y = check.generate.fill value(0.0) : tensor<{t}x{QH * VD}xf32>
  kernel.launch @g_mla_vb_{fmt("attn_v_b.weight")}[%t](%t, %x, %w, %y) : [index](index, tensor<{t}x{QH * D}xf32>, {VBT}, tensor<{t}x{QH * VD}xf32>)
  check.expect.close actual(%y) expected(%e) atol(1.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{QH * VD}xf32>
  check.return
}}
""")

    # 3b. their prefill WMMA twins (epochs of 64+ tokens, a multiple of 64): 128 tokens, f16 operands
    t = 128
    xq = rng.standard_normal((t, QH, QB)).astype(np.float32)
    ykb = np.zeros((t, QH, D), np.float64); ykb[..., :KL] = np.einsum("hrk,thk->thr", kb, xq[..., :NOPE].astype(np.float16).astype(np.float64))
    xo = (rng.standard_normal((t, QH, D)) * 0.05).astype(np.float32)
    yvb = np.einsum("hvk,thk->thv", vb, xo[..., :KL].astype(np.float16).astype(np.float64))
    for n, a in [("mla_kbm_x", xq.reshape(t, -1)), ("mla_kbm_e", ykb.reshape(t, -1).astype(np.float32)),
                 ("mla_vbm_x", xo.reshape(t, -1)), ("mla_vbm_e", yvb.reshape(t, -1).astype(np.float32))]:
        save(n, a)
    lv = os.environ.get("MLA_BMM_LIVE", "0") not in ("", "0")   # the products take the epoch's %info (all tokens live here)
    LVI = f"  %info = check.generate.fill value(1) : tensor<{t}x4xi32>\n" if lv else ""
    LVA, LVT = (", %info", f", tensor<{t}x4xi32>") if lv else ("", "")
    for nm, w, X, Y in (("kb", "attn_k_b", QH * QB, QH * D), ("vb", "attn_v_b", QH * D, QH * VD)):
        WT = KBT if nm == "kb" else VBT
        out_cases.append(f"""check.case public @mla_{nm}m {{
  %t = check.literal value({t}) : index
  %x = check.file.read.npy path("mla_{nm}m_x.npy") : tensor<{t}x{X}xf32>
  %w = check.file.read.npy path("mla_{nm}_w.npy") : {WT}
  %e = check.file.read.npy path("mla_{nm}m_e.npy") : tensor<{t}x{Y}xf32>
  %y = check.generate.fill value(0.0) : tensor<{t}x{Y}xf32>
{LVI}  kernel.launch @g_mla_{nm}m_{fmt(w + ".weight")}[%t](%t, %x, %w, %y{LVA}) : [index](index, tensor<{t}x{X}xf32>, {WT}, tensor<{t}x{Y}xf32>{LVT})
  check.expect.close actual(%y) expected(%e) atol(2.0e-3) rtol(5.0e-3) nan(different) : tensor<{t}x{Y}xf32>
  check.return
}}
""")

    # 4. the whole chain on a span continuing stored KV: p0 history tokens, then t span tokens (the last `pad` dead)
    def chain(p0, t, pad, tag, flash=False):
        n = p0 + t
        ctx = ((n + 15) // 16) * 16 + 64
        blocks = ctx // 16
        allpos = np.arange(n)
        qb = rng.standard_normal((n, QH, QB)).astype(np.float32)
        kv = (rng.standard_normal((n, D)) * 2).astype(np.float32)
        live = np.ones(t, bool); live[t - pad:] = False
        Kc = np.concatenate([rms(kv[:, :KL], kvan), rope_norm(kv[:, KL:], allpos, theta)], -1).astype(np.float16).astype(np.float64)
        qabs = np.einsum("hrk,thk->thr", kb, qb[p0:, :, :NOPE].astype(np.float64))
        Q = np.concatenate([qabs, rope_norm(qb[p0:, :, NOPE:], allpos[p0:], theta)], -1)    # (t, QH, D)
        vo = np.zeros((t, QH, VD))
        for i in range(t):
            if not live[i]:
                continue
            p = p0 + i
            sc = np.einsum("hd,sd->hs", Q[i], Kc[:p + 1]) * scale
            pr = np.exp(sc - sc.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
            o = pr @ Kc[:p + 1, :KL]                     # (QH, KL)
            vo[i] = np.einsum("hvk,hk->hv", vb, o)
        orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)
        hinfo = np.zeros((p0, 4), np.int32); hinfo[:, 1] = allpos[:p0]; hinfo[:, 2] = 1
        sinfo = np.zeros((t, 4), np.int32); sinfo[:, 1] = allpos[p0:]; sinfo[:, 2] = live
        planes = li + 1
        if not flash and not ns.get("MLA_FLASH_ON"):   # no flash kernels: the work list holds every token of the span (runs of 9+ included)
            fc = np.zeros(64, np.int32); fc[0] = t; save(f"{tag}_fcount", fc)
        vo_pre = np.where(flash_covered(sinfo)[:, None, None], vo, 0.0) if flash else vo
        for nn, a in [("vop", vo_pre.reshape(t, -1).astype(np.float32)), ("hqb", qb[:p0].reshape(p0, -1)), ("hkv", kv[:p0]), ("hinfo", hinfo), ("sqb", qb[p0:].reshape(t, -1)), ("skv", kv[p0:]),
                      ("sinfo", sinfo), ("orig", orig), ("vo", vo.reshape(t, -1).astype(np.float32)),
                      ("arena", (rng.standard_normal(planes * ctx * D) * 4).astype(np.float16))]:   # a used pool: stale rows everywhere
            save(f"{tag}_{nn}", a)
        print(f"{tag}: layer {li} p0={p0} t={t} pad={pad} |vo| max {np.abs(vo).max():.4f} mean {np.abs(vo).mean():.5f}")
        AR = f"tensor<{planes * ctx * D}xf16>"
        S, S16 = SDEC, 16
        QT, OT = f"tensor<{t}x{QH * D}xf32>", f"tensor<{t}x{QH * VD}xf32>"
        def st(tt, pre):
            return (f"  kernel.launch @g_mla_rope_store[%{pre}t](%{pre}t, %layer, %ctx, %pool, %{pre}qb, %{pre}kv, %w, %{pre}q, %{pre}info, %orig, %arena) : "
                    f"[index](index, index, index, index, tensor<{tt}x{QH * QB}xf32>, tensor<{tt}x{D}xf32>, tensor<{KL}xf32>, tensor<{tt}x{QH * D}xf32>, tensor<{tt}x4xi32>, tensor<{10 + blocks}xi32>, {AR})\n")
        PA, PM = f"tensor<{t * QH * S * D}xf32>", f"tensor<{t * QH * S * 2}xf32>"
        PA16, PM16 = f"tensor<{t * QH * S16 * D}xf32>", f"tensor<{t * QH * S16 * 2}xf32>"
        fl = "".join(f"  kernel.launch @g_attention_flash2{k}[%st](%st, %layer, %sliding, %ctx, %pool, %sq, %sq, %sinfo, %tiles, %orig, %arena, %o_pre) : "
                     f"[index](index, index, i32, index, index, {QT}, {QT}, tensor<{t}x4xi32>, tensor<{t}xi32>, tensor<{10 + blocks}xi32>, {AR}, {QT})\n"
                     for k in ("", "_partial")) if flash else ""
        out_cases.append(f"""check.case public @{tag} {{
  %ht = check.literal value({p0}) : index
  %st = check.literal value({t}) : index
  %layer = check.literal value({li}) : index
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({ctx}) : index
  %hqb = check.file.read.npy path("{tag}_hqb.npy") : tensor<{p0}x{QH * QB}xf32>
  %hkv = check.file.read.npy path("{tag}_hkv.npy") : tensor<{p0}x{D}xf32>
  %hinfo = check.file.read.npy path("{tag}_hinfo.npy") : tensor<{p0}x4xi32>
  %sqb = check.file.read.npy path("{tag}_sqb.npy") : tensor<{t}x{QH * QB}xf32>
  %skv = check.file.read.npy path("{tag}_skv.npy") : tensor<{t}x{D}xf32>
  %sinfo = check.file.read.npy path("{tag}_sinfo.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("{tag}_orig.npy") : tensor<{10 + blocks}xi32>
  %w = check.file.read.npy path("mla_kvan.npy") : tensor<{KL}xf32>
  %kbw = check.file.read.npy path("mla_kb_w.npy") : {KBT}
  %vbw = check.file.read.npy path("mla_vb_w.npy") : {VBT}
  %evo = check.file.read.npy path("{tag}_vo.npy") : {OT}
  %arena = check.file.read.npy path("{tag}_arena.npy") : {AR}
  %hq = check.generate.fill value(0.0) : tensor<{p0}x{QH * D}xf32>
  %sq = check.generate.fill value(0.0) : {QT}
  %pacc = check.generate.fill value(0.0) : {PA}
  %pml = check.generate.fill value(0.0) : {PM}
  %pacc16 = check.generate.fill value(0.0) : {PA16}
  %pml16 = check.generate.fill value(0.0) : {PM16}
  %o_dec = check.generate.fill value(0.0) : {QT}
  %o_pre = check.generate.fill value(0.0) : {QT}
  %vo_dec = check.generate.fill value(0.0) : {OT}
  %vo_pre = check.generate.fill value(0.0) : {OT}
{st(p0, "h")}{st(t, "s")}  kernel.launch @g_mla_kb_{fmt("attn_k_b.weight")}[%st](%st, %sqb, %kbw, %sq) : [index](index, tensor<{t}x{QH * QB}xf32>, {KBT}, {QT})
{list_launch("%st", t, "%sinfo")}{fl}{"" if flash else "  "}{"// " if flash else ""}kernel.launch @g16_attention_split[%st](%st, %layer, %sliding, %ctx, %pool, %sq, %sinfo, %orig, %arena, %pacc16, %pml16) : [index](index, index, i32, index, index, {QT}, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, {AR}, {PA16}, {PM16})
  {"// " if flash else ""}kernel.launch @g16_attention_combine[%st](%st, %sliding, %sinfo, %pacc16, %pml16, %sq, %o_pre) : [index](index, i32, tensor<{t}x4xi32>, {PA16}, {PM16}, {QT}, {QT})
  kernel.launch @{DSPLIT}[%st](%st, %layer, %sliding, %ctx, %pool, %sq, %sinfo, %orig, %arena, %pacc, %pml) : [index](index, index, i32, index, index, {QT}, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, {AR}, {PA}, {PM})
  kernel.launch @{DCOMB}[%st](%st, %sliding, %sinfo, %pacc, %pml, %sq, %o_dec) : [index](index, i32, tensor<{t}x4xi32>, {PA}, {PM}, {QT}, {QT})
  kernel.launch @g_mla_vb_{fmt("attn_v_b.weight")}[%st](%st, %o_dec, %vbw, %vo_dec) : [index](index, {QT}, {VBT}, {OT})
  kernel.launch @g_mla_vb_{fmt("attn_v_b.weight")}[%st](%st, %o_pre, %vbw, %vo_pre) : [index](index, {QT}, {VBT}, {OT})
  check.expect.close actual(%vo_dec) expected(%evo) atol(1.0e-3) rtol(5.0e-3) nan(different) : {OT}
  %evop = check.file.read.npy path("{tag}_vop.npy") : {OT}
  check.expect.close actual(%vo_pre) expected(%evop) atol(1.0e-3) rtol(5.0e-3) nan(different) : {OT}
{"" if flash or ns.get("MLA_FLASH_ON") else f"  %efc = check.file.read.npy path(" + chr(34) + tag + "_fcount.npy" + chr(34) + ") : tensor<64xi32>" + chr(10) + "  check.expect.equal actual(%fcount) expected(%efc) : tensor<64xi32>" + chr(10)}  check.return
}}
""")
    chain(37, 20, 3, "mla_chain")          # short history, a span with padding
    chain(2100, 24, 0, "mla_chain_long")   # > 2048 keys: 64 decode splits of 33 keys
    if ns.get("MLA_FLASH_ON"):   # the flash prefill kernels (whole tiles + partial blocks) on runs of 9+ tokens
        chain(37, 20, 3, "mla_flash_part", flash=True)     # two partial blocks, padding
        chain(64, 48, 0, "mla_flash_whole", flash=True)    # three whole tiles
        chain(2100, 40, 0, "mla_flash_long", flash=True)   # partial + whole + partial past 2048 keys
    # the program's own launch sequence on real embedding rows (a 44-token prompt's worth)
    layer_case(ns, [0] + [int(x) for x in rng.integers(100, cfg["vocab"] - 1000, 43)])


def layer_case(ns, ids, tag="mla_layer0"):
    """Layer 0's attention exactly as the decode-shaped program launches it (t tokens of one prompt at positions 0..t-1,
    the 4-token matvecs), on the real embedding rows of `ids`: q_a, kv_a_mqa, rms, q_b, rope store, k_b, split + combine,
    v_b, attn_output; every intermediate buffer is checked against numpy (the first failing one names the culprit)."""
    from mv_rows import pick_r
    cfg, T, deq, raw, save, ty, out_cases = ns["cfg"], ns["T"], ns["deq"], ns["raw"], ns["save"], ns["ty"], ns["cases"]
    from gguf.quants import dequantize
    m = cfg["mla"]
    H, QH, KL, RD, NOPE, VD, QL = cfg["hidden"], cfg["heads"], m["kv_lora"], m["rope_dims"], m["nope_dims"], m["v_dims"], m["q_lora"]
    D, QB, eps, theta, scale = KL + RD, NOPE + RD, cfg["rms_eps"], cfg["rope_theta"], m["scale"]
    b = "blk.0."
    def W(n):
        t = T[b + n]
        return dequantize(np.asarray(t.data), t.tensor_type).reshape([int(x) for x in reversed(t.shape)]).astype(np.float64)
    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
    t = len(ids)
    E = deq("token_embd.weight").astype(np.float64)
    h = rms(E[ids], deq(b + "attn_norm.weight"))
    qa = rms(h @ W("attn_q_a.weight").T, W("attn_q_a_norm.weight"))
    kv = h @ W("attn_kv_a_mqa.weight").T
    qb = qa @ W("attn_q_b.weight").T
    q3 = qb.reshape(t, QH, QB)
    pos = np.arange(t)
    Q = np.concatenate([np.einsum("hrk,thk->thr", W("attn_k_b.weight"), q3[..., :NOPE]), rope_norm(q3[..., NOPE:], pos, theta)], -1)
    K = np.concatenate([rms(kv[:, :KL], W("attn_kv_a_norm.weight")), rope_norm(kv[:, KL:], pos, theta)], -1).astype(np.float16).astype(np.float64)
    s = np.einsum("thd,sd->hts", Q, K) * scale
    s[:, np.triu(np.ones((t, t), bool), 1)] = -np.inf
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    o = np.einsum("hts,sd->thd", p, K)                       # (t, QH, D): the tail dims too (V read from the whole row)
    if gen_mla_dec.on():   # the WMMA decode kernels write o[..., :KL] only (v_b reads those); the tail keeps its fill
        o = o.copy(); o[..., KL:] = 0.0
    vo = np.einsum("hvk,thk->thv", W("attn_v_b.weight"), o[..., :KL])
    y = vo.reshape(t, -1) @ W("attn_output.weight").T
    ctx = ((t + 15) // 16) * 16 + 64; blocks = ctx // 16
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)
    info = np.zeros((t, 4), np.int32); info[:, 1] = pos; info[:, 2] = 1
    f32 = lambda a: np.asarray(a, np.float32).reshape(t, -1)
    names = ["attn_q_a", "attn_kv_a_mqa", "attn_q_b", "attn_k_b", "attn_v_b", "attn_output"]
    for nm in names:
        save(f"{tag}_w_{nm}", raw(b + nm + ".weight"))
    for nn, a in [("h", f32(h)), ("qa", f32(qa)), ("kv", f32(kv)), ("qb", f32(qb)), ("q", f32(Q)), ("o", f32(o)), ("vo", f32(vo)), ("y", f32(y)),
                  ("info", info), ("orig", orig), ("qan", deq(b + "attn_q_a_norm.weight").astype(np.float32)), ("kvan", deq(b + "attn_kv_a_norm.weight").astype(np.float32))]:
        save(f"{tag}_{nn}", a)
    WT = {nm: ty(raw(b + nm + ".weight")) for nm in names}
    fmt = lambda nm: T[b + nm + ".weight"].tensor_type.name.lower()
    tg = 4 if t > 1 else 1
    def mv(nm, n, K, x, yb):
        R = pick_r(n, K, tg)
        kern = f"g_mv_{fmt(nm)}_k{K}" + (f"_t{tg}" if tg > 1 else "") + (f"_r{R}" if R != 16 else "")
        return (f"  %n_{yb} = check.literal value({n}) : index\n"
                f"  kernel.launch @{kern}[%n_{yb}, %t](%n_{yb}, %t, %acc, %{x}, %w_{nm}, %{yb}) : [index, index](index, index, i32, tensor<{t}x{K}xf32>, {WT[nm]}, tensor<{t}x{n}xf32>)\n")
    AR = f"tensor<{ctx * D}xf16>"
    S = SDEC
    PA, PM = f"tensor<{t * QH * S * D}xf32>", f"tensor<{t * QH * S * 2}xf32>"
    rd = lambda nn, shape: f'  %e_{nn} = check.file.read.npy path("{tag}_{nn}.npy") : tensor<{shape}xf32>\n'
    ex = lambda act, nn, shape, tol="atol(2.0e-3) rtol(1.0e-2)": f"  check.expect.close actual(%{act}) expected(%e_{nn}) {tol} nan(different) : tensor<{shape}xf32>\n"
    out_cases.append(f"""check.case public @{tag} {{
  %t = check.literal value({t}) : index
  %acc = check.literal value(0) : i32
  %layer = check.literal value(0) : index
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({ctx}) : index
""" + "".join(f'  %w_{nm} = check.file.read.npy path("{tag}_w_{nm}.npy") : {WT[nm]}\n' for nm in names) + f"""  %h = check.file.read.npy path("{tag}_h.npy") : tensor<{t}x{H}xf32>
  %info = check.file.read.npy path("{tag}_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("{tag}_orig.npy") : tensor<{10 + blocks}xi32>
  %qan = check.file.read.npy path("{tag}_qan.npy") : tensor<{QL}xf32>
  %kvan = check.file.read.npy path("{tag}_kvan.npy") : tensor<{KL}xf32>
""" + rd("qa", f"{t}x{QL}") + rd("kv", f"{t}x{D}") + rd("qb", f"{t}x{QH * QB}") + rd("q", f"{t}x{QH * D}") + rd("o", f"{t}x{QH * D}")
      + rd("vo", f"{t}x{QH * VD}") + rd("y", f"{t}x{H}") + f"""  %qa = check.generate.fill value(0.0) : tensor<{t}x{QL}xf32>
  %kv = check.generate.fill value(0.0) : tensor<{t}x{D}xf32>
  %qb = check.generate.fill value(0.0) : tensor<{t}x{QH * QB}xf32>
  %q = check.generate.fill value(0.0) : tensor<{t}x{QH * D}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{QH * D}xf32>
  %vo = check.generate.fill value(0.0) : tensor<{t}x{QH * VD}xf32>
  %y = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  %arena = check.generate.fill value(0.0) : {AR}
  %pacc = check.generate.fill value(0.0) : {PA}
  %pml = check.generate.fill value(0.0) : {PM}
""" + mv("attn_q_a", QL, H, "h", "qa") + mv("attn_kv_a_mqa", D, H, "h", "kv")
      + f"  kernel.launch @g_mla_rms_q[%t](%t, %qa, %qan) : [index](index, tensor<{t}x{QL}xf32>, tensor<{QL}xf32>)\n"
      + mv("attn_q_b", QH * QB, QL, "qa", "qb")
      + f"""  kernel.launch @g_mla_rope_store[%t](%t, %layer, %ctx, %pool, %qb, %kv, %kvan, %q, %info, %orig, %arena) : [index](index, index, index, index, tensor<{t}x{QH * QB}xf32>, tensor<{t}x{D}xf32>, tensor<{KL}xf32>, tensor<{t}x{QH * D}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, {AR})
  kernel.launch @g_mla_kb_{fmt("attn_k_b")}[%t](%t, %qb, %w_attn_k_b, %q) : [index](index, tensor<{t}x{QH * QB}xf32>, {WT["attn_k_b"]}, tensor<{t}x{QH * D}xf32>)
  kernel.launch @{DSPLIT}[%t](%t, %layer, %sliding, %ctx, %pool, %q, %info, %orig, %arena, %pacc, %pml) : [index](index, index, i32, index, index, tensor<{t}x{QH * D}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, {AR}, {PA}, {PM})
  kernel.launch @{DCOMB}[%t](%t, %sliding, %info, %pacc, %pml, %q, %o) : [index](index, i32, tensor<{t}x4xi32>, {PA}, {PM}, tensor<{t}x{QH * D}xf32>, tensor<{t}x{QH * D}xf32>)
  kernel.launch @g_mla_vb_{fmt("attn_v_b")}[%t](%t, %o, %w_attn_v_b, %vo) : [index](index, tensor<{t}x{QH * D}xf32>, {WT["attn_v_b"]}, tensor<{t}x{QH * VD}xf32>)
""" + mv("attn_output", H, QH * VD, "vo", "y")
      + ex("qa", "qa", f"{t}x{QL}") + ex("kv", "kv", f"{t}x{D}") + ex("qb", "qb", f"{t}x{QH * QB}") + ex("q", "q", f"{t}x{QH * D}")
      + ex("o", "o", f"{t}x{QH * D}") + ex("vo", "vo", f"{t}x{QH * VD}") + ex("y", "y", f"{t}x{H}") + "  check.return\n}\n")
