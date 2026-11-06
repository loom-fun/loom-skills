"""Kernel unit tests for a gen_llm.py build: numpy references from gguf dequantization.

usage: python make_kernel_tests.py CONFIG.json OUTDIR      (tools venv: gguf, numpy)
       run_kernel_tests.sh PACKAGE OUTDIR                  (loom-test on the GPU)
Covers g_embed, the decode matvecs (1- and 4-token variants; with ACT_Q8, the default, g_quant_q8 + the
int8-activation 1-/4-/8-/16-token variants for Q8_0/Q4_K/Q5_K/Q6_K) for one tensor of each (format, K), the QK-norm + RoPE kernel (incl. head_dim 256 / partial
RoPE), and for hybrid models the whole gated delta-net chain on a span that
continues a stored recurrent state (conv, history commit, q/k norms,
recurrence, gated norm, published state). Extend it
when you add a kernel: a kernel that passes here but breaks end-to-end is
usually a launch-shape/indexing assumption (e.g. one-wave rows), so test the
real shapes of the model, not toy ones.
"""
import json, sys, os, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_rows import pick_r, iq_r  # noqa: E402
ACT_Q8 = os.environ.get("ACT_Q8", "1") == "1"   # gen_llm.py's default: int8 activations for these formats
IQ_FMTS = {"q8_0", "q4_k", "q5_k", "q6_k"}


def q8_ref(xs):
    """llama.cpp quantize_q8_1 per 32-value group (d = amax/127 rounded to F16, q = round(x/d)): the dequantized
    activations the _q8 kernels contract with."""
    g = xs.reshape(xs.shape[0], -1, 32).astype(np.float32)
    d = np.abs(g).max(-1, keepdims=True) / np.float32(127)
    v = np.divide(g, d, out=np.zeros_like(g), where=d > 0)
    q = np.sign(v) * np.floor(np.abs(v) + 0.5)
    return (q * d.astype(np.float16).astype(np.float32)).reshape(xs.shape)


cfg = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
r = GGUFReader(cfg["gguf"]); T = {t.name: t for t in r.tensors}
def raw(name): return np.asarray(T[name].data).view(np.int8).reshape(-1)
def deq(name):
    t = T[name]; return dequantize(np.asarray(t.data), t.tensor_type).reshape(int(t.shape[1]), int(t.shape[0])) if len(t.shape) == 2 else np.asarray(t.data, dtype=np.float32)
H, D, QH, KVH, eps = cfg["hidden"], cfg["head_dim"], cfg["heads"], cfg["kv_heads"], cfg["rms_eps"]
rng = np.random.default_rng(0)
cases = []
def save(n, a): np.save(os.path.join(out, n + ".npy"), a)
def ty(a): return "tensor<" + "x".join(map(str, a.shape)) + "x" + {"float32": "f32", "int8": "i8", "int32": "i32", "float16": "f16"}[str(a.dtype)] + ">"
# 1. embed
ids = np.array([0, cfg['vocab'] - 1, cfg['vocab'] // 2, 1000], dtype=np.int32); t = len(ids)
# M-RoPE (cfg "mrope"): the rope kernel also reads the epoch metadata (one span, offset 0: rotary position =
# KV position, as in a text-only epoch) and the input buffer (unused without media)
MR = bool(cfg.get("mrope"))
if MR:
    save("mr_meta", np.array([1] + [0] * 15, np.int32))
def mr_decl(t):
    return (f"  %mr_meta = check.file.read.npy path(\"mr_meta.npy\") : tensor<16xi32>\n  %mr_ids = check.generate.fill value(0) : tensor<{t * 8}xi32>\n" if MR else "")
MR_ARGS = ", %mr_meta, %mr_ids" if MR else ""
def mr_types(t):
    return f", tensor<16xi32>, tensor<{t * 8}xi32>" if MR else ""
# image input (cfg "vision"): g_embed also takes the epoch metadata and token info; plain epochs have metadata word 2 = 0
VIS = bool(cfg.get("vision"))
VMETA = (f"\n  %meta = check.generate.fill value(0) : tensor<4xi32>\n  %tinfo = check.generate.fill value(0) : tensor<{t}x4xi32>" if VIS else "")
VARGS = ", %meta, %tinfo" if VIS else ""
VTYPES = f", tensor<4xi32>, tensor<{t}x4xi32>" if VIS else ""
E = deq("token_embd.weight"); w = deq("blk.0.attn_norm.weight")
x = E[ids]; h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
tab = raw("token_embd.weight")
for n, a in [("e_ids", ids), ("e_tab", tab), ("e_w", w.astype(np.float32)), ("e_x", x.astype(np.float32)), ("e_h", h.astype(np.float32))]: save(n, a)
cases.append(f"""check.case public @embed {{
  %t = check.literal value({t}) : index
  %ids = check.file.read.npy path("e_ids.npy") : {ty(ids)}
  %tab = check.file.read.npy path("e_tab.npy") : {ty(tab)}
  %w = check.file.read.npy path("e_w.npy") : tensor<{H}xf32>
  %ex = check.file.read.npy path("e_x.npy") : tensor<{t}x{H}xf32>
  %eh = check.file.read.npy path("e_h.npy") : tensor<{t}x{H}xf32>
  %x = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  %h = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>{VMETA}
  kernel.launch @g_embed[%t](%t, %ids{VARGS}, %tab, %w, %x, %h) : [index](index, {ty(ids)}{VTYPES}, {ty(tab)}, tensor<{H}xf32>, tensor<{t}x{H}xf32>, tensor<{t}x{H}xf32>)
  check.expect.close actual(%x) expected(%ex) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.expect.close actual(%h) expected(%eh) atol(1.0e-3) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.return
}}
""")
if VIS:   # a media epoch: 8-word records, token 1 carries a feature row (copied unscaled), all inputs live
    rng = np.random.default_rng(0); feat = rng.standard_normal(H).astype(np.float32)
    recs = np.zeros((t, 8), np.int32); recs[:, 0] = ids; recs[1, 1] = t * 32
    packet = np.concatenate([recs.reshape(-1), feat.view(np.int32)])
    meta = np.array([t, 0, 1, 0], np.int32); tinfo = np.zeros((t, 4), np.int32); tinfo[:, 2] = 1
    xm = x.astype(np.float32).copy()
    xm[1] = feat / np.sqrt((feat * feat).mean() + 1e-5) if cfg.get("embed_norm") else feat   # the weightless embedding norm applies to features too
    hm = xm / np.sqrt((xm * xm).mean(-1, keepdims=True) + eps) * w
    for n, a in [("m_pk", packet), ("m_meta", meta), ("m_info", tinfo), ("m_x", xm.astype(np.float32)), ("m_h", hm.astype(np.float32))]: save(n, a)
    cases.append(f"""check.case public @embed_media {{
  %t = check.literal value({t}) : index
  %pk = check.file.read.npy path("m_pk.npy") : {ty(packet)}
  %meta = check.file.read.npy path("m_meta.npy") : tensor<4xi32>
  %tinfo = check.file.read.npy path("m_info.npy") : tensor<{t}x4xi32>
  %tab = check.file.read.npy path("e_tab.npy") : {ty(tab)}
  %w = check.file.read.npy path("e_w.npy") : tensor<{H}xf32>
  %ex = check.file.read.npy path("m_x.npy") : tensor<{t}x{H}xf32>
  %eh = check.file.read.npy path("m_h.npy") : tensor<{t}x{H}xf32>
  %x = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  %h = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  kernel.launch @g_embed[%t](%t, %pk, %meta, %tinfo, %tab, %w, %x, %h) : [index](index, {ty(packet)}, tensor<4xi32>, tensor<{t}x4xi32>, {ty(tab)}, tensor<{H}xf32>, tensor<{t}x{H}xf32>, tensor<{t}x{H}xf32>)
  check.expect.close actual(%x) expected(%ex) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.expect.close actual(%h) expected(%eh) atol(1.0e-3) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.return
}}
""")
# 2. matvec per (tensor) for t=1 and t=4 variants
seen = set(); todo = []
VD = cfg["gdn"]["inner"] if cfg.get("gdn") else 0
for li in range(cfg["layers"]):
    for nm_, K in [("attn_q", H), ("attn_k", H), ("attn_v", H), ("ffn_down", cfg["ffn"][li] if isinstance(cfg["ffn"], list) else cfg["ffn"]), ("attn_output", QH * D), ("ffn_gate", H),
                   ("attn_qkv", H), ("attn_gate", H), ("ssm_alpha", H), ("ssm_out", VD)]:
        name = f"blk.{li}.{nm_}.weight"
        if name not in T:
            continue
        key = (T[name].tensor_type.name.lower(), K)
        if key not in seen: seen.add(key); todo.append((name, K))
for name, K in todo:
    fmt = T[name].tensor_type.name.lower(); W = deq(name); n = W.shape[0]
    for tg in (1, 4):
        if ACT_Q8 and fmt in IQ_FMTS:
            break
        xs = rng.standard_normal((tg, K)).astype(np.float32); y = xs @ W.T.astype(np.float64)
        tag = name.replace(".", "_") + f"_t{tg}"
        save(tag + "_w", raw(name)); save(tag + "_x", xs); save(tag + "_y", y.astype(np.float32))
        kern = f"g_mv_{fmt}_k{K}" + ("_t4" if tg == 4 else "")
        _R = pick_r(n, K, tg)   # the variant the model launches (gen_llm uses the same rule)
        if _R != 16: kern += f"_r{_R}"
        wt = ty(raw(name))
        cases.append(f"""check.case public @mv_{tag} {{
  %n = check.literal value({n}) : index
  %t = check.literal value({tg}) : index
  %acc = check.literal value(0) : i32
  %w = check.file.read.npy path("{tag}_w.npy") : {wt}
  %x = check.file.read.npy path("{tag}_x.npy") : tensor<{tg}x{K}xf32>
  %e = check.file.read.npy path("{tag}_y.npy") : tensor<{tg}x{n}xf32>
  %y = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  kernel.launch @{kern}[%n, %t](%n, %t, %acc, %x, %w, %y) : [index, index](index, index, i32, tensor<{tg}x{K}xf32>, {wt}, tensor<{tg}x{n}xf32>)
  check.expect.close actual(%y) expected(%e) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{tg}x{n}xf32>
  check.return
}}
""")
    if ACT_Q8 and fmt in IQ_FMTS:   # g_quant_q8 then the int8 matvec, at the 1-, 4- and 8-token variants
        for tg in (1, 4, 8, 16):
            xs = rng.standard_normal((tg, K)).astype(np.float32); y = q8_ref(xs).astype(np.float64) @ W.T.astype(np.float64)
            tag = name.replace(".", "_") + f"_q8_t{tg}"
            save(tag + "_w", raw(name)); save(tag + "_x", xs); save(tag + "_y", y.astype(np.float32))
            _R = iq_r(n, K, tg)   # gen_llm.py mv_name's rule
            kern = f"g_mv_{fmt}_k{K}" + (f"_t{tg}" if tg > 1 else "") + "_q8" + (f"_r{_R}" if _R != 16 else "")
            _R1 = iq_r(n, K, 1)
            k1 = f"g_mv_{fmt}_k{K}_q8" + (f"_r{_R1}" if _R1 != 16 else "")
            # verify == decode: the multi-token variant must round every token exactly as the 1-token kernel
            # (run here over the same tg tokens) does
            same = (f"""  %y1 = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  kernel.launch @{k1}[%n, %t](%n, %t, %acc, %xq, %w, %y1) : [index, index](index, index, i32, tensor<{tg}x{2 * K}xi8>, {ty(raw(name))}, tensor<{tg}x{n}xf32>)
  check.expect.close actual(%y) expected(%y1) atol(0.0) rtol(0.0) nan(different) : tensor<{tg}x{n}xf32>
""" if tg > 1 else "")
            wt = ty(raw(name))
            cases.append(f"""check.case public @mv_{tag} {{
  %n = check.literal value({n}) : index
  %t = check.literal value({tg}) : index
  %k = check.literal value({K}) : index
  %acc = check.literal value(0) : i32
  %w = check.file.read.npy path("{tag}_w.npy") : {wt}
  %x = check.file.read.npy path("{tag}_x.npy") : tensor<{tg}x{K}xf32>
  %e = check.file.read.npy path("{tag}_y.npy") : tensor<{tg}x{n}xf32>
  %xq = check.generate.fill value(0) : tensor<{tg}x{2 * K}xi8>
  %y = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  kernel.launch @g_quant_q8[%t, %k](%t, %k, %x, %xq) : [index, index](index, index, tensor<{tg}x{K}xf32>, tensor<{tg}x{2 * K}xi8>)
  kernel.launch @{kern}[%n, %t](%n, %t, %acc, %xq, %w, %y) : [index, index](index, index, i32, tensor<{tg}x{2 * K}xi8>, {wt}, tensor<{tg}x{n}xf32>)
  check.expect.close actual(%y) expected(%e) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{tg}x{n}xf32>
{same}  check.return
}}
""")
# 2b. int8 gate/up (swiglu) matvec: the 8/16-token variants (BFLY_PERM, 32 rows) equal the 1-token kernel bit for bit
_g = next((li for li in range(cfg["layers"]) if f"blk.{li}.ffn_gate.weight" in T and f"blk.{li}.ffn_up.weight" in T), None)
if ACT_Q8 and _g is not None:
    gn, un = f"blk.{_g}.ffn_gate.weight", f"blk.{_g}.ffn_up.weight"
    fmt = T[gn].tensor_type.name.lower(); n = int(T[gn].shape[1])
    if fmt in IQ_FMTS and T[un].tensor_type.name.lower() == fmt:
        save("sw_g", raw(gn)); save("sw_u", raw(un)); gt, ut = ty(raw(gn)), ty(raw(un))
        for tg in (8, 16):
            save(f"sw_x{tg}", rng.standard_normal((tg, H)).astype(np.float32))
            R1, Rt = iq_r(n, H, 1, True), iq_r(n, H, tg, True)
            k1 = f"g_mv_{fmt}_k{H}_swiglu_q8" + (f"_r{R1}" if R1 != 16 else "")
            kt = f"g_mv_{fmt}_k{H}_swiglu_t{tg}_q8" + (f"_r{Rt}" if Rt != 16 else "")
            sig = f"[index, index](index, index, i32, tensor<{tg}x{2 * H}xi8>, {gt}, {ut}, tensor<{tg}x{n}xf32>)"
            cases.append(f"""check.case public @mvsw_same_t{tg} {{
  %n = check.literal value({n}) : index
  %t = check.literal value({tg}) : index
  %k = check.literal value({H}) : index
  %acc = check.literal value(0) : i32
  %w = check.file.read.npy path("sw_g.npy") : {gt}
  %wu = check.file.read.npy path("sw_u.npy") : {ut}
  %x = check.file.read.npy path("sw_x{tg}.npy") : tensor<{tg}x{H}xf32>
  %xq = check.generate.fill value(0) : tensor<{tg}x{2 * H}xi8>
  %y = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  %y1 = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  kernel.launch @g_quant_q8[%t, %k](%t, %k, %x, %xq) : [index, index](index, index, tensor<{tg}x{H}xf32>, tensor<{tg}x{2 * H}xi8>)
  kernel.launch @{kt}[%n, %t](%n, %t, %acc, %xq, %w, %wu, %y) : {sig}
  kernel.launch @{k1}[%n, %t](%n, %t, %acc, %xq, %w, %wu, %y1) : {sig}
  check.expect.close actual(%y) expected(%y1) atol(0.0) rtol(0.0) nan(different) : tensor<{tg}x{n}xf32>
  check.return
}}
""")
# 3. qk norm + rope on q (live = 0: no KV store)
t = 4; pos = np.array([0, 1, 7, 300])
A0 = next(i for i in range(cfg["layers"]) if i not in set(cfg.get("recurrent_layers", [])))
RD = cfg.get("rope_dims") or D
qn = deq(f"blk.{A0}.attn_q_norm.weight") if cfg["qk_norm"] else np.ones(D, np.float32)
kn = deq(f"blk.{A0}.attn_k_norm.weight") if cfg["qk_norm"] else np.ones(D, np.float32)
q = rng.standard_normal((t, QH, D)).astype(np.float32); k = rng.standard_normal((t, KVH, D)).astype(np.float32); v = rng.standard_normal((t, KVH, D)).astype(np.float32)
def norm(a, w): return a / np.sqrt((a * a).mean(-1, keepdims=True) + eps) * w
def rope(a):
    inv = cfg["rope_theta"] ** (-np.arange(0, RD // 2) * 2.0 / RD); ang = pos[:, None] * inv[None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    if cfg["rope_type"] == "neox":
        a1, a2 = a[..., :RD // 2], a[..., RD // 2:RD]; return np.concatenate([a1 * c - a2 * s, a2 * c + a1 * s, a[..., RD:]], -1)
    a1, a2 = a[..., 0::2], a[..., 1::2]; o = np.empty_like(a); o[..., 0::2] = a1 * c - a2 * s; o[..., 1::2] = a2 * c + a1 * s; return o
qe = rope(norm(q, qn) if cfg["qk_norm"] else q)
info = np.zeros((t, 4), np.int32); info[:, 1] = pos
for n, a in [("r_q", q.reshape(t, -1)), ("r_k", k.reshape(t, -1)), ("r_v", v.reshape(t, -1)), ("r_qn", qn.astype(np.float32)), ("r_kn", kn.astype(np.float32)), ("r_qe", qe.reshape(t, -1).astype(np.float32)), ("r_info", info), ("r_orig", np.zeros(64, np.int32))]: save(n, a)
cases.append(f"""check.case public @qk_rope {{
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %rope = check.literal value(1) : i32
  %ctx = check.literal value(1024) : index
  %pool = check.literal value(1024) : index
  %q = check.file.read.npy path("r_q.npy") : tensor<{t}x{QH * D}xf32>
  %k = check.file.read.npy path("r_k.npy") : tensor<{t}x{KVH * D}xf32>
  %v = check.file.read.npy path("r_v.npy") : tensor<{t}x{KVH * D}xf32>
  %qn = check.file.read.npy path("r_qn.npy") : tensor<{D}xf32>
  %kn = check.file.read.npy path("r_kn.npy") : tensor<{D}xf32>
  %info = check.file.read.npy path("r_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("r_orig.npy") : tensor<64xi32>
  %qe = check.file.read.npy path("r_qe.npy") : tensor<{t}x{QH * D}xf32>
  %arena = check.generate.fill value(0.0) : tensor<1024xf16>
{mr_decl(t)}  kernel.launch @g_qk_rope_store[%t](%t, %layer, %rope, %ctx, %pool, %q, %k, %v, %qn, %kn, %info, %orig, %arena{MR_ARGS}) : [index](index, index, i32, index, index, tensor<{t}x{QH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{t}x4xi32>, tensor<64xi32>, tensor<1024xf16>{mr_types(t)})
  check.expect.close actual(%q) expected(%qe) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{QH * D}xf32>
  check.return
}}
""")
# 3a. output head (argmax over the vocabulary) for one selected span of 3 tokens
t = 3
head_name = "output.weight" if "output.weight" in T else "token_embd.weight"
Wh = deq(head_name).astype(np.float64)
hs = rng.standard_normal((t, H)).astype(np.float32)
HEAD_Q8 = ACT_Q8 and T[head_name].tensor_type.name.lower() in IQ_FMTS
logits = Wh @ (q8_ref(hs[-1:])[0] if HEAD_Q8 else hs[-1]).astype(np.float64)
meta = np.zeros(4 + 9, np.int32); meta[0] = 1; meta[4 + 0] = t; meta[4 + 6] = 1
order = np.argsort(-logits)
for n, a in [("hd_meta", meta), ("hd_h", hs), ("hd_w", raw(head_name)), ("hd_out", np.array([order[0]], np.int32))]:
    save(n, a)
CH = (cfg["vocab"] + 127) // 128
eos = set([cfg["eos_id"]] + cfg.get("extra_eos_ids", []))
print("head test: expected token", order[0], "logit gap to runner-up", float(logits[order[0]] - logits[order[1]]))
hwt = ty(raw(head_name))
cases.append(f"""check.case public @head_argmax {{
  %spans = check.literal value(1) : index
  %t = check.literal value({t}) : index
  %meta = check.file.read.npy path("hd_meta.npy") : tensor<13xi32>
  %h = check.file.read.npy path("hd_h.npy") : tensor<{t}x{H}xf32>
  %w = check.file.read.npy path("hd_w.npy") : {hwt}
  %expect = check.file.read.npy path("hd_out.npy") : tensor<1xi32>
  %partials = check.generate.fill value(0) : tensor<{CH * 2}xi32>
  %logits = check.generate.fill value(0.0) : tensor<{CH * 128}xf32>
  %out = check.generate.fill value(0) : tensor<1xi32>
""" + (f"""  %hk = check.literal value({H}) : index
  %hq = check.generate.fill value(0) : tensor<{t}x{2 * H}xi8>
  kernel.launch @g_quant_q8[%t, %hk](%t, %hk, %h, %hq) : [index, index](index, index, tensor<{t}x{H}xf32>, tensor<{t}x{2 * H}xi8>)
  kernel.launch @g_head_mv_q8[%spans](%spans, %t, %meta, %hq, %w, %partials, %logits) : [index](index, index, tensor<13xi32>, tensor<{t}x{2 * H}xi8>, {hwt}, tensor<{CH * 2}xi32>, tensor<{CH * 128}xf32>)""" if HEAD_Q8 else
      f"""  kernel.launch @g_head_mv[%spans](%spans, %t, %meta, %h, %w, %partials, %logits) : [index](index, index, tensor<13xi32>, tensor<{t}x{H}xf32>, {hwt}, tensor<{CH * 2}xi32>, tensor<{CH * 128}xf32>)""") + f"""
  kernel.launch @g_head_select[%spans](%spans, %meta, %partials, %logits, %out) : [index](index, tensor<13xi32>, tensor<{CH * 2}xi32>, tensor<{CH * 128}xf32>, tensor<1xi32>)
  check.expect.equal actual(%out) expected(%expect) : tensor<1xi32>
  check.return
}}
""")
# 3a'. multi-span int8 head: the 4/8/16-span kernels must give every span's logits and argmax partials bit-identical to
# the 1-span kernel (verification rounds like decode); 16 and 12 live spans (a partial span group)
if HEAD_Q8:
    for S in (16, 12):
        hm = np.zeros(4 + 9 * S, np.int32); hm[0] = S
        for j in range(S):
            hm[4 + 9 * j + 0] = 1; hm[4 + 9 * j + 4] = j; hm[4 + 9 * j + 6] = 1
        save(f"hdm_meta{S}", hm); save(f"hdm_h{S}", rng.standard_normal((S, H)).astype(np.float32))
        mt, ht = f"tensor<{hm.size}xi32>", f"tensor<{S}x{H}xf32>"
        def hl(k, tag):
            return (f"  %p{tag} = check.generate.fill value(0) : tensor<{S * CH * 2}xi32>\n  %l{tag} = check.generate.fill value(0.0) : tensor<{S * CH * 128}xf32>\n"
                    f"  kernel.launch @{k}[%spans](%spans, %t, %meta, %hq, %w, %p{tag}, %l{tag}) : [index](index, index, {mt}, tensor<{S}x{2 * H}xi8>, {hwt}, tensor<{S * CH * 2}xi32>, tensor<{S * CH * 128}xf32>)\n")
        variants = [("g_head_mv_t4_q8", "4"), ("g_head_mv_t8_q8", "8"), ("g_head_mv_t16_q8", "16")]
        cases.append(f"""check.case public @head_spans{S} {{
  %spans = check.literal value({S}) : index
  %t = check.literal value({S}) : index
  %meta = check.file.read.npy path("hdm_meta{S}.npy") : {mt}
  %h = check.file.read.npy path("hdm_h{S}.npy") : {ht}
  %w = check.file.read.npy path("hd_w.npy") : {hwt}
  %hk = check.literal value({H}) : index
  %hq = check.generate.fill value(0) : tensor<{S}x{2 * H}xi8>
  kernel.launch @g_quant_q8[%t, %hk](%t, %hk, %h, %hq) : [index, index](index, index, {ht}, tensor<{S}x{2 * H}xi8>)
""" + hl("g_head_mv_q8", "1") + "".join(hl(k, tag) for k, tag in variants) + "".join(
            f"  check.expect.close actual(%l{tag}) expected(%l1) atol(0.0) rtol(0.0) nan(different) : tensor<{S * CH * 128}xf32>\n"
            f"  check.expect.equal actual(%p{tag}) expected(%p1) : tensor<{S * CH * 2}xi32>\n" for _, tag in variants) + "  check.return\n}\n")
# 3b. attention chain: [q/gate split] -> qk norm + rope + paged KV store -> split-K attention -> combine (gate)
t, ctx = 4, 64
pos = np.arange(t)
gate_kind = cfg.get("attn_gate", False)
q = rng.standard_normal((t, QH, D)).astype(np.float32); k = rng.standard_normal((t, KVH, D)).astype(np.float32)
v = rng.standard_normal((t, KVH, D)).astype(np.float32); gate = rng.standard_normal((t, QH, D)).astype(np.float32)
qr = rope(norm(q, qn) if cfg["qk_norm"] else q); kr = rope(norm(k, kn) if cfg["qk_norm"] else k)
att = np.zeros((t, QH, D))
for hh in range(QH):
    kv = hh // (QH // KVH); sc = qr[:, hh] @ kr[:, kv].T / np.sqrt(D); sc[np.triu_indices(t, 1)] = -1e30
    pr = np.exp(sc - sc.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True); att[:, hh] = pr @ v[:, kv]
if gate_kind:
    att = att / (1 + np.exp(-gate))
blocks = (ctx + 15) // 16
orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)  # record: map origin 40; identity page map
info = np.zeros((t, 4), np.int32); info[:, 1] = pos; info[:, 2] = 1
S = 64
for n, a in [("a_q", q.reshape(t, -1)), ("a_k", k.reshape(t, -1)), ("a_v", v.reshape(t, -1)), ("a_g", gate.reshape(t, -1)),
             ("a_qg", np.stack([q, gate], 2).reshape(t, -1)), ("a_info", info), ("a_orig", orig), ("a_o", att.reshape(t, -1).astype(np.float32))]:
    save(n, a)
QD = QH * D
split_text = ""
if gate_kind == "fused_q":
    split_text = f"""  %qg = check.file.read.npy path("a_qg.npy") : tensor<{t}x{2 * QD}xf32>
  kernel.launch @g_split_qgate[%t](%t, %qg, %q, %g) : [index](index, tensor<{t}x{2 * QD}xf32>, tensor<{t}x{QD}xf32>, tensor<{t}x{QD}xf32>)
"""
cases.append(f"""check.case public @attention_chain {{
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %rope = check.literal value(1) : i32
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({ctx}) : index
  %q = {"check.generate.fill value(0.0)" if gate_kind == "fused_q" else 'check.file.read.npy path("a_q.npy")'} : tensor<{t}x{QD}xf32>
  %g = {"check.generate.fill value(0.0)" if gate_kind == "fused_q" else 'check.file.read.npy path("a_g.npy")'} : tensor<{t}x{QD}xf32>
  %k = check.file.read.npy path("a_k.npy") : tensor<{t}x{KVH * D}xf32>
  %v = check.file.read.npy path("a_v.npy") : tensor<{t}x{KVH * D}xf32>
  %qn = check.file.read.npy path("r_qn.npy") : tensor<{D}xf32>
  %kn = check.file.read.npy path("r_kn.npy") : tensor<{D}xf32>
  %info = check.file.read.npy path("a_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("a_orig.npy") : tensor<{10 + blocks}xi32>
  %eo = check.file.read.npy path("a_o.npy") : tensor<{t}x{QD}xf32>
  %arena = check.generate.fill value(0.0) : tensor<{2 * ctx * KVH * D}xf16>
  %pacc = check.generate.fill value(0.0) : tensor<{t * QH * S * D}xf32>
  %pml = check.generate.fill value(0.0) : tensor<{t * QH * S * 2}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{QD}xf32>
{split_text}{mr_decl(t)}  kernel.launch @g_qk_rope_store[%t](%t, %layer, %rope, %ctx, %pool, %q, %k, %v, %qn, %kn, %info, %orig, %arena{MR_ARGS}) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{2 * ctx * KVH * D}xf16>{mr_types(t)})
  kernel.launch @g_attention_split[%t](%t, %layer, %sliding, %ctx, %pool, %q, %info, %orig, %arena, %pacc, %pml) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{2 * ctx * KVH * D}xf16>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>)
  kernel.launch @g_attention_combine[%t](%t, %sliding, %info, %pacc, %pml, {"%g" if gate_kind else "%q"}, %o) : [index](index, i32, tensor<{t}x4xi32>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>, tensor<{t}x{QD}xf32>, tensor<{t}x{QD}xf32>)
  check.expect.close actual(%o) expected(%eo) atol(3.0e-3) rtol(3.0e-3) nan(different) : tensor<{t}x{QD}xf32>
  check.return
}}
""")
# 4. gated delta-net chain on one 5-token span at position 3 that continues the reader slot
if cfg.get("recurrent_layers"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_gdn import gdn_geometry
    G = gdn_geometry(cfg); HK, HV, DK, DV, K, C, VD = (G[k] for k in ("HK", "HV", "DK", "DV", "K", "C", "VD"))
    li = cfg["recurrent_layers"][0]; b = f"blk.{li}."
    conv_w = deq(b + "ssm_conv1d.weight").reshape(C, K).astype(np.float32)
    a_log = deq(b + "ssm_a").astype(np.float32); dt = deq(b + "ssm_dt.bias").astype(np.float32); nw = deq(b + "ssm_norm.weight").astype(np.float32)
    t, p0 = 40, 3   # over 32 tokens: the recurrence stores its outputs in 32-step batches
    qkv = rng.standard_normal((t, C)).astype(np.float32); hist0 = rng.standard_normal((K - 1, C)).astype(np.float32)
    S0 = (0.1 * rng.standard_normal((HV, DV, DK))).astype(np.float32)  # [head][b][a]
    alpha = rng.standard_normal((t, HV)).astype(np.float32); beta = rng.standard_normal((t, HV)).astype(np.float32)
    z = rng.standard_normal((t, VD)).astype(np.float32)
    full = np.concatenate([hist0, qkv], 0)
    conv = np.stack([(full[i:i + K] * conv_w.T).sum(0) for i in range(t)])
    silu = conv / (1 + np.exp(-conv)); hist1 = full[-(K - 1):]
    qs, ks, vs = silu[:, :HK * DK].reshape(t, HK, DK), silu[:, HK * DK:2 * HK * DK].reshape(t, HK, DK), silu[:, 2 * HK * DK:].reshape(t, HV, DV)
    qinv = 1 / np.sqrt((qs ** 2).sum(-1) + eps); kinv = 1 / np.sqrt((ks ** 2).sum(-1) + eps)
    inv = np.stack([qinv, kinv], -1).reshape(t, 2 * HK)
    S = S0.astype(np.float64).copy(); raw_o = np.zeros((t, HV, DV))
    for i in range(t):
        for h in range(HV):
            kh = h % HK; q = qs[i, kh] * qinv[i, kh] / np.sqrt(DK); k = ks[i, kh] * kinv[i, kh]
            g = np.log1p(np.exp(alpha[i, h] + dt[h])) * a_log[h]
            St = S[h] * np.exp(g)                      # [b][a]
            d = (vs[i, h] - St @ k) / (1 + np.exp(-beta[i, h]))
            St = St + np.outer(d, k); S[h] = St; raw_o[i, h] = St @ q
    o = raw_o / np.sqrt((raw_o ** 2).mean(-1, keepdims=True) + eps) * nw * (z.reshape(t, HV, DV) / (1 + np.exp(-z.reshape(t, HV, DV))))
    LB, SB = G["layer_bytes"], G["state_bytes"]
    arena = np.zeros(2 * LB // 4, np.float32)
    arena[:SB // 4] = S0.reshape(-1); arena[SB // 4:SB // 4 + (K - 1) * C] = hist0.reshape(-1)
    expect = arena.copy(); expect[LB // 4:LB // 4 + SB // 4] = S.astype(np.float32).reshape(-1)
    expect[LB // 4 + SB // 4:LB // 4 + SB // 4 + (K - 1) * C] = hist1.reshape(-1)
    meta = np.zeros(4 + 9, np.int32); meta[0] = 1; meta[4 + 0] = t; meta[4 + 1] = p0; meta[4 + 3] = 0; meta[4 + 4] = 0
    origins = np.zeros(5, np.int64); origins[0] = 0; origins[1] = LB; origins[3] = p0
    info = np.zeros((t, 4), np.int32); info[:, 1] = p0 + np.arange(t); info[:, 2] = 1
    for n, a in [("g_meta", meta), ("g_orig", origins.view(np.int32)), ("g_info", info), ("g_arena", arena), ("g_expect", expect),
                 ("g_qkv", qkv), ("g_conv_w", conv_w.reshape(-1)), ("g_silu", silu.astype(np.float32)), ("g_inv", inv.astype(np.float32)),
                 ("g_alpha", alpha), ("g_beta", beta), ("g_a", a_log), ("g_dt", dt), ("g_nw", nw), ("g_z", z),
                 ("g_raw", raw_o.reshape(t, VD).astype(np.float32)), ("g_o", o.reshape(t, VD).astype(np.float32))]:
        save(n, a)
    AT = f"tensor<{2 * LB // 4}xf32>"
    cases.append(f"""check.case public @gdn_chain {{
  %t = check.literal value({t}) : index
  %spans = check.literal value(1) : index
  %layer = check.literal value(0) : index
  %meta = check.file.read.npy path("g_meta.npy") : tensor<13xi32>
  %orig = check.file.read.npy path("g_orig.npy") : tensor<10xi32>
  %info = check.file.read.npy path("g_info.npy") : tensor<{t}x4xi32>
  %arena = check.file.read.npy path("g_arena.npy") : {AT}
  %expect = check.file.read.npy path("g_expect.npy") : {AT}
  %qkv = check.file.read.npy path("g_qkv.npy") : tensor<{t}x{C}xf32>
  %w = check.file.read.npy path("g_conv_w.npy") : tensor<{C * K}xf32>
  %esilu = check.file.read.npy path("g_silu.npy") : tensor<{t}x{C}xf32>
  %einv = check.file.read.npy path("g_inv.npy") : tensor<{t}x{2 * HK}xf32>
  %alpha = check.file.read.npy path("g_alpha.npy") : tensor<{t}x{HV}xf32>
  %beta = check.file.read.npy path("g_beta.npy") : tensor<{t}x{HV}xf32>
  %a = check.file.read.npy path("g_a.npy") : tensor<{HV}xf32>
  %dt = check.file.read.npy path("g_dt.npy") : tensor<{HV}xf32>
  %nw = check.file.read.npy path("g_nw.npy") : tensor<{DV}xf32>
  %z = check.file.read.npy path("g_z.npy") : tensor<{t}x{VD}xf32>
  %eraw = check.file.read.npy path("g_raw.npy") : tensor<{t}x{VD}xf32>
  %eo = check.file.read.npy path("g_o.npy") : tensor<{t}x{VD}xf32>
  %silu = check.generate.fill value(0.0) : tensor<{t}x{C}xf32>
  %inv = check.generate.fill value(0.0) : tensor<{t}x{2 * HK}xf32>
  %raw = check.generate.fill value(0.0) : tensor<{t}x{VD}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{VD}xf32>
  kernel.launch @g_gdn_conv[%t](%t, %spans, %layer, %meta, %info, %orig, %arena, %qkv, %w, %silu) : [index](index, index, index, tensor<13xi32>, tensor<{t}x4xi32>, tensor<10xi32>, {AT}, tensor<{t}x{C}xf32>, tensor<{C * K}xf32>, tensor<{t}x{C}xf32>)
  kernel.launch @g_gdn_conv_commit[%spans](%t, %spans, %layer, %meta, %orig, %arena, %qkv) : [index](index, index, index, tensor<13xi32>, tensor<10xi32>, {AT}, tensor<{t}x{C}xf32>)
  kernel.launch @g_gdn_qk_norm[%t](%t, %silu, %inv) : [index](index, tensor<{t}x{C}xf32>, tensor<{t}x{2 * HK}xf32>)
  kernel.launch @g_gdn_recurrence[%spans](%t, %spans, %layer, %meta, %orig, %arena, %silu, %inv, %alpha, %beta, %a, %dt, %raw) : [index](index, index, index, tensor<13xi32>, tensor<10xi32>, {AT}, tensor<{t}x{C}xf32>, tensor<{t}x{2 * HK}xf32>, tensor<{t}x{HV}xf32>, tensor<{t}x{HV}xf32>, tensor<{HV}xf32>, tensor<{HV}xf32>, tensor<{t}x{VD}xf32>)
  kernel.launch @g_gdn_norm_gate[%t](%t, %info, %raw, %nw, %z, %o) : [index](index, tensor<{t}x4xi32>, tensor<{t}x{VD}xf32>, tensor<{DV}xf32>, tensor<{t}x{VD}xf32>, tensor<{t}x{VD}xf32>)
  check.expect.close actual(%silu) expected(%esilu) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{C}xf32>
  check.expect.close actual(%inv) expected(%einv) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{2 * HK}xf32>
  check.expect.close actual(%raw) expected(%eraw) atol(1.0e-4) rtol(2.0e-3) nan(different) : tensor<{t}x{VD}xf32>
  check.expect.close actual(%o) expected(%eo) atol(1.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{VD}xf32>
  check.expect.close actual(%arena) expected(%expect) atol(1.0e-4) rtol(1.0e-3) nan(different) : {AT}
  check.return
}}
""")
# g_row_scale_f16 (the GEMM A operand): per-row power-of-two scale 2^-e, e = clamp(floor(log2 max|x|) - 13, 0, 100),
# x16 = f16(x * scale); rows of very different magnitude, width = the FFN width (the widest the model uses).
if os.environ.get("GEMM_TESTS", "1") == "1":
    rs_k = cfg["ffn"] if isinstance(cfg.get("ffn"), int) else H
    rs_x = (rng.standard_normal((3, rs_k)) * np.array([[1e-3], [1.0], [3e5]])).astype(np.float32)
    rs_e = np.clip(np.frexp(np.abs(rs_x).max(1))[1] - 1 - 13, 0, 100)
    rs_s = np.stack([2.0 ** -rs_e, 2.0 ** rs_e], 1).astype(np.float32)
    save("rs_x", rs_x); save("rs_s", rs_s); save("rs_h", (rs_x * rs_s[:, :1]).astype(np.float16))
    cases.append(f"""check.case public @row_scale_f16 {{
  %rows = check.literal value(3) : index
  %k = check.literal value({rs_k}) : index
  %x = check.file.read.npy path("rs_x.npy") : tensor<3x{rs_k}xf32>
  %es = check.file.read.npy path("rs_s.npy") : tensor<3x2xf32>
  %eh = check.file.read.npy path("rs_h.npy") : tensor<3x{rs_k}xf16>
  %s = check.generate.fill value(0.0) : tensor<3x2xf32>
  %h = check.generate.fill value(0.0) : tensor<3x{rs_k}xf16>
  kernel.launch @g_row_scale_f16[%rows](%rows, %k, %x, %s, %h) : [index](index, index, tensor<3x{rs_k}xf32>, tensor<3x2xf32>, tensor<3x{rs_k}xf16>)
  check.expect.close actual(%s) expected(%es) atol(0.0) rtol(0.0) nan(different) : tensor<3x2xf32>
  check.expect.close actual(%h) expected(%eh) atol(0.0) rtol(1.0e-3) nan(different) : tensor<3x{rs_k}xf16>
  check.return
}}
""")
# Prefill GEMMs (gen_gemm.py): y = (x16 . W^T) * inv per row, x16 the f16 row-scaled activations and inv the
# row's inverse scale (g_row_scale_f16's contract); swiglu: silu(gate) * up. One tensor per (format, K) the
# model's prefill uses: the 64-, 128- (_m128) and 256-row (_m256) kernels, two row tiles each.
def gemm_cases():
    seen_g = set()
    for li in range(cfg["layers"]):
        for nm_ in ("ffn_gate", "ffn_up", "ffn_down", "attn_qkv", "attn_gate", "attn_q", "attn_k", "attn_output", "ssm_out"):
            name = f"blk.{li}.{nm_}.weight"
            if name not in T:
                continue
            fmt = T[name].tensor_type.name.lower()
            up = f"blk.{li}.ffn_up.weight"
            sw = nm_ == "ffn_gate" and up in T and T[up].tensor_type == T[name].tensor_type
            if fmt not in ("q4_k", "q5_k", "q6_k", "q8_0") or (fmt, sw) in seen_g:
                continue
            seen_g.add((fmt, sw))
            yield name, fmt, sw
if os.environ.get("GEMM_TESTS", "1") == "1":
    for name, fmt, swiglu in gemm_cases():
        W = deq(name); n, K = W.shape
        n_use = min(n, 1024)                          # a slice of output rows keeps the reference cheap
        Wr = raw(name).reshape(n, -1)[:n_use].reshape(-1)
        Ws = W[:n_use].astype(np.float64)
        if swiglu:
            up = name.replace("ffn_gate", "ffn_up"); U = deq(up)[:n_use].astype(np.float64); Ur = raw(up).reshape(n, -1)[:n_use].reshape(-1)
        base_k = f"g_gemm_{fmt}_{'swiglu' if swiglu else 'store'}"
        for rows, kern in ((128, base_k), (256, base_k + "_m128"), (512, base_k + "_m256")):
            x16 = rng.standard_normal((rows, K)).astype(np.float16)
            inv = rng.uniform(0.25, 2.0, rows).astype(np.float32)
            sc = np.stack([np.ones(rows, np.float32), inv], 1)
            g = (x16.astype(np.float64) @ Ws.T) * inv[:, None]
            if swiglu:
                u = (x16.astype(np.float64) @ U.T) * inv[:, None]
                act = (0.5 * g * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g ** 3))) if cfg.get("ffn_act") == "gelu"
                       else g / (1 + np.exp(-g)))   # GEGLU ports (gemma4) swap the GEMM's silu for gelu-tanh
                yv = act * u
            else:
                yv = g
            tag = f"gm_{fmt}_{'sw' if swiglu else 'st'}_{rows}"
            save(tag + "_x", x16); save(tag + "_s", sc); save(tag + "_y", yv.astype(np.float32)); save(tag + "_w", Wr)
            wdecl = f'  %w = check.file.read.npy path("{tag}_w.npy") : {ty(Wr)}\n'
            wargs, wtys = "%w", ty(Wr)
            if swiglu:
                save(tag + "_u", Ur); wdecl += f'  %u = check.file.read.npy path("{tag}_u.npy") : {ty(Ur)}\n'; wargs += ", %u"; wtys += ", " + ty(Ur)
            tol = 3.0e-2 * float(np.abs(yv).std() + 1e-6)
            cases.append(f"""check.case public @{tag} {{
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({n_use}) : index
  %x = check.file.read.npy path("{tag}_x.npy") : tensor<{rows}x{K}xf16>
  %s = check.file.read.npy path("{tag}_s.npy") : tensor<{rows}x2xf32>
{wdecl}  %e = check.file.read.npy path("{tag}_y.npy") : tensor<{rows}x{n_use}xf32>
  %y = check.generate.fill value(0.0) : tensor<{rows}x{n_use}xf32>
  kernel.launch @{kern}[%rows, %n](%rows, %k, %n, %x, %s, {wargs}, %y) : [index, index](index, index, index, tensor<{rows}x{K}xf16>, tensor<{rows}x2xf32>, {wtys}, tensor<{rows}x{n_use}xf32>)
  check.expect.close actual(%y) expected(%e) atol({tol:.3e}) rtol(1.0e-2) nan(different) : tensor<{rows}x{n_use}xf32>
  check.return
}}
""")
# Prefill flash attention (gen_flash2.py / gen_flash2s.py): 64 uniform tokens at the end of a 1024-key context,
# paged arena filled directly (identity page map), layer 0, causal; q scaled so attention is sharp (a kernel that
# drops part of the score sum fails). Reference in float64 from f16 Q/K/V, times sigmoid(gate) when gated.
if os.environ.get("FLASH_TESTS", "1") == "1" and not cfg.get("attn_sinks") and cfg.get("sliding_window") in (None, 0):
    t, ctx = 64, 1024
    P0 = ctx - t; blocks = ctx // 16; pool = ctx
    kk = rng.standard_normal((ctx, KVH, D)).astype(np.float16); vv = rng.standard_normal((ctx, KVH, D)).astype(np.float16)
    arena = np.zeros((2, pool, KVH, D), np.float16); arena[0], arena[1] = kk, vv
    qq = (rng.standard_normal((t, QH, D)) * 2.5).astype(np.float32); gg = rng.standard_normal((t, QH, D)).astype(np.float32)
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)
    info = np.zeros((t, 4), np.int32); info[:, 1] = P0 + np.arange(t); info[:, 2] = 1
    o = np.zeros((t, QH, D)); qd = qq.astype(np.float16).astype(np.float64)
    for hh in range(QH):
        kv = hh // (QH // KVH); sc = qd[:, hh] @ kk[:, kv].astype(np.float64).T / np.sqrt(D)
        sc[np.arange(ctx)[None, :] > (P0 + np.arange(t))[:, None]] = -1e30
        pr = np.exp(sc - sc.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True); o[:, hh] = pr @ vv[:, kv].astype(np.float64)
    if cfg.get("attn_gate"):
        o = o / (1 + np.exp(-gg))
    QD = QH * D
    for n, a in [("f_arena", arena.reshape(-1)), ("f_q", qq.reshape(t, QD)), ("f_g", gg.reshape(t, QD)), ("f_orig", orig),
                 ("f_info", info), ("f_o", o.reshape(t, QD).astype(np.float32))]:
        save(n, a)
    AT = f"tensor<{2 * pool * KVH * D}xf16>"
    cases.append(f"""check.case public @flash_prefill {{
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({pool}) : index
  %q = check.file.read.npy path("f_q.npy") : tensor<{t}x{QD}xf32>
  %g = check.file.read.npy path("f_g.npy") : tensor<{t}x{QD}xf32>
  %info = check.file.read.npy path("f_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("f_orig.npy") : tensor<{10 + blocks}xi32>
  %arena = check.file.read.npy path("f_arena.npy") : {AT}
  %e = check.file.read.npy path("f_o.npy") : tensor<{t}x{QD}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{QD}xf32>
  kernel.launch @g_attention_flash2[%t](%t, %layer, %sliding, %ctx, %pool, %q, %g, %info, %orig, %arena, %o) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x{QD}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, {AT}, tensor<{t}x{QD}xf32>)
  check.expect.close actual(%o) expected(%e) atol(4.0e-3) rtol(4.0e-3) nan(different) : tensor<{t}x{QD}xf32>
  check.return
}}
""")
import re
# hrx-loom 0.1.0: float comparison is check.expect.approximate with the actual and expected types.
text = re.sub(r"check\.expect\.close (actual\(.*\)) : (.*)$", r"check.expect.approximate \1 : \2, \2", "\n".join(cases), flags=re.M)
open(os.path.join(out, "cases.loom-test"), "w", newline="\n").write(text)
print("wrote", len(cases), "cases")
