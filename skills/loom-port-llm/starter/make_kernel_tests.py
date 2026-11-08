"""Kernel unit tests for a gen_llm.py build: numpy references from gguf dequantization.

usage: python make_kernel_tests.py CONFIG.json OUTDIR      (tools venv: gguf, numpy)
       run_kernel_tests.sh PACKAGE OUTDIR                  (loom-test on the GPU)
Covers g_embed, the decode matvecs (1- and 4-token variants; with ACT_Q8, the default, g_quant_q8 + the
int8-activation 1-/4-/8-/16-token variants for formats.ICODES) for one tensor of each (format, K), the QK-norm + RoPE kernel (incl. head_dim 256 / partial
RoPE, q/k/v biases), and for hybrid models the whole gated delta-net chain on a span that
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
from mv_rows import pick_r, iq_r, flash2_deep_from  # noqa: E402
from gen_gemm import HELPER as GEMM_HELPER  # noqa: E402  formats with no 256-row (_m256) GEMM
ACT_Q8 = os.environ.get("ACT_Q8", "1") == "1"   # gen_llm.py's default: int8 activations for these formats
from formats import ICODES as IQ_FMTS  # noqa: E402  (gen_llm.py takes int8 activations for these)
import attn_rules  # noqa: E402


def q8_ref(xs):
    """llama.cpp quantize_q8_1 per 32-value group (d = amax/127 rounded to F16, q = round(x/d)): the dequantized
    activations the _q8 kernels contract with."""
    g = xs.reshape(xs.shape[0], -1, 32).astype(np.float32)
    d = np.abs(g).max(-1, keepdims=True) / np.float32(127)
    v = np.divide(g, d, out=np.zeros_like(g), where=d > 0)
    q = np.sign(v) * np.floor(np.abs(v) + 0.5)
    return (q * d.astype(np.float16).astype(np.float32)).reshape(xs.shape)


cfg = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
from formats import split_parts  # noqa: E402
T = {t.name: t for p in split_parts(cfg["gguf"]) for t in GGUFReader(p).tensors}
def raw(name): return np.asarray(T[name].data).view(np.int8).reshape(-1)
def deq(name):
    t = T[name]; return dequantize(np.asarray(t.data), t.tensor_type).reshape(int(t.shape[1]), int(t.shape[0])) if len(t.shape) == 2 else np.asarray(t.data, dtype=np.float32)
H, D, QH, KVH, eps = cfg["hidden"], cfg["head_dim"], cfg["heads"], cfg["kv_heads"], cfg["rms_eps"]
rng = np.random.default_rng(0)


def ffn_act(g):
    """The FFN gate activation of the reference: gelu-tanh for GEGLU (cfg ffn_act "gelu"), else SiLU (SwiGLU)."""
    if cfg.get("ffn_act") == "gelu":
        return 0.5 * g * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g ** 3)))
    return g / (1 + np.exp(-g))
cases = []
def save(n, a): np.save(os.path.join(out, n + ".npy"), a)
def tile_starts(info):
    """First tokens of the 16-token tiles g_attention_fallback_list gives to flash2: live, one row, consecutive
    positions, the first at a position multiple of 16. Every other token is the listed fallback's."""
    return [s for s in range(len(info) - 15)
            if info[s, 1] % 16 == 0 and (info[s:s + 16, 0] == info[s, 0]).all() and (info[s:s + 16, 2] != 0).all()
            and (info[s:s + 16, 1] == info[s, 1] + np.arange(16)).all()]
def covered_tokens(info):
    cov = np.zeros(len(info), bool)
    for s in tile_starts(info): cov[s:s + 16] = True
    return cov
def list_launch(tv, t, info, sfx=""):
    """Build the epoch's work lists from INFO (%tiles, %flist, %fcount, each + SFX); TV: the token count's SSA value."""
    return (f"  %tiles{sfx} = check.generate.fill value(0) : tensor<{t}xi32>\n  %flist{sfx} = check.generate.fill value(0) : tensor<{t}xi32>\n"
            f"  %fcount{sfx} = check.generate.fill value(0) : tensor<64xi32>\n"
            f"  kernel.launch @g_attention_fallback_list[{tv}]({tv}, {info}, %tiles{sfx}, %flist{sfx}, %fcount{sfx}) : "
            f"[index](index, tensor<{t}x4xi32>, tensor<{t}xi32>, tensor<{t}xi32>, tensor<64xi32>)\n")
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
x = E[ids]
if cfg.get("embed_norm"): x = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5)  # weightless embedding RMSNorm
if cfg.get("embed_scale"): x = x * cfg["embed_scale"]
h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
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
    # DeepStack: g_deepstack adds feature slice `level` of image inputs (tokens 1 and 3) to x and renormalizes h
    NDS = int((cfg.get("vision") or {}).get("deepstack") or 0)
    if NDS:
        level = NDS; drng = np.random.default_rng(7)
        drows = drng.standard_normal((2, (NDS + 1) * H)).astype(np.float32)
        drecs = np.zeros((t, 8), np.int32); drecs[:, 0] = ids; drecs[1, 1] = t * 32; drecs[3, 1] = t * 32 + (NDS + 1) * H * 4
        dpk = np.concatenate([drecs.reshape(-1), drows.reshape(-1).view(np.int32)])
        x0 = drng.standard_normal((t, H)).astype(np.float32); h0 = drng.standard_normal((t, H)).astype(np.float32)
        xd, hd = x0.copy(), h0.copy()
        for j, tok in enumerate((1, 3)):
            xd[tok] = x0[tok] + drows[j, level * H:(level + 1) * H]
            hd[tok] = xd[tok] / np.sqrt((xd[tok] * xd[tok]).mean() + eps) * w
        for n, a in [("d_pk", dpk), ("d_x0", x0), ("d_h0", h0), ("d_x", xd), ("d_h", hd.astype(np.float32))]: save(n, a)
        cases.append(f"""check.case public @deepstack_media {{
  %t = check.literal value({t}) : index
  %level = check.literal value({level}) : index
  %pk = check.file.read.npy path("d_pk.npy") : {ty(dpk)}
  %meta = check.file.read.npy path("m_meta.npy") : tensor<4xi32>
  %tinfo = check.file.read.npy path("m_info.npy") : tensor<{t}x4xi32>
  %w = check.file.read.npy path("e_w.npy") : tensor<{H}xf32>
  %x = check.file.read.npy path("d_x0.npy") : tensor<{t}x{H}xf32>
  %h = check.file.read.npy path("d_h0.npy") : tensor<{t}x{H}xf32>
  %ex = check.file.read.npy path("d_x.npy") : tensor<{t}x{H}xf32>
  %eh = check.file.read.npy path("d_h.npy") : tensor<{t}x{H}xf32>
  kernel.launch @g_deepstack[%t](%t, %level, %pk, %meta, %tinfo, %w, %x, %h) : [index](index, index, {ty(dpk)}, tensor<4xi32>, tensor<{t}x4xi32>, tensor<{H}xf32>, tensor<{t}x{H}xf32>, tensor<{t}x{H}xf32>)
  check.expect.close actual(%x) expected(%ex) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.expect.close actual(%h) expected(%eh) atol(1.0e-3) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.return
}}
""")
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
                   ("attn_qkv", H), ("attn_gate", H), ("ssm_alpha", H), ("ssm_out", VD),
                   ("ffn_gate_shexp", H), ("ffn_down_shexp", (cfg.get("moe") or {}).get("shared_ffn", 0))]:
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
# 2a. per-head attention gate (cfg attn_gate "per_head"): n = heads rows, run by gen_llm.lin_mv on f32 activations at
# every token count: the 1-token and 4-token kernels at the decode R, and the 4-token R=16 kernel prefill takes
if cfg.get("attn_gate") == "per_head":
    name = next(f"blk.{i}.attn_gate.weight" for i in range(cfg["layers"]) if f"blk.{i}.attn_gate.weight" in T)
    fmt = T[name].tensor_type.name.lower(); W = deq(name); n = W.shape[0]
    R1 = pick_r(n, H, 1)
    variants = [(1, R1), (4, R1)] + ([(4, 16)] if R1 != 16 and n % 16 == 0 else [])
    save("hg_w", raw(name)); wt = ty(raw(name))
    for tg, R in variants:
        xs = rng.standard_normal((tg, H)).astype(np.float32); y = xs @ W.T.astype(np.float64)
        tag = f"hg_t{tg}_r{R}"; save(tag + "_x", xs); save(tag + "_y", y.astype(np.float32))
        kern = f"g_mv_{fmt}_k{H}" + (f"_t{tg}" if tg > 1 else "") + (f"_r{R}" if R != 16 else "")
        cases.append(f"""check.case public @mv_{tag} {{
  %n = check.literal value({n}) : index
  %t = check.literal value({tg}) : index
  %acc = check.literal value(0) : i32
  %w = check.file.read.npy path("hg_w.npy") : {wt}
  %x = check.file.read.npy path("{tag}_x.npy") : tensor<{tg}x{H}xf32>
  %e = check.file.read.npy path("{tag}_y.npy") : tensor<{tg}x{n}xf32>
  %y = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  kernel.launch @{kern}[%n, %t](%n, %t, %acc, %x, %w, %y) : [index, index](index, index, i32, tensor<{tg}x{H}xf32>, {wt}, tensor<{tg}x{n}xf32>)
  check.expect.close actual(%y) expected(%e) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{tg}x{n}xf32>
  check.return
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
        # the 1-token kernel against numpy: act(gate) * up on the int8 activations (SwiGLU, or GEGLU for ffn_act gelu)
        xs = rng.standard_normal((1, H)).astype(np.float32); xq = q8_ref(xs).astype(np.float64)
        ysw = ffn_act(xq @ deq(gn).T.astype(np.float64)) * (xq @ deq(un).T.astype(np.float64))
        save("sw_x1", xs); save("sw_y1", ysw.astype(np.float32))
        R1 = iq_r(n, H, 1, True)
        k1 = f"g_mv_{fmt}_k{H}_swiglu_q8" + (f"_r{R1}" if R1 != 16 else "")
        cases.append(f"""check.case public @mvsw_ref_t1 {{
  %n = check.literal value({n}) : index
  %t = check.literal value(1) : index
  %k = check.literal value({H}) : index
  %acc = check.literal value(0) : i32
  %w = check.file.read.npy path("sw_g.npy") : {gt}
  %wu = check.file.read.npy path("sw_u.npy") : {ut}
  %x = check.file.read.npy path("sw_x1.npy") : tensor<1x{H}xf32>
  %e = check.file.read.npy path("sw_y1.npy") : tensor<1x{n}xf32>
  %xq = check.generate.fill value(0) : tensor<1x{2 * H}xi8>
  %y = check.generate.fill value(0.0) : tensor<1x{n}xf32>
  kernel.launch @g_quant_q8[%t, %k](%t, %k, %x, %xq) : [index, index](index, index, tensor<1x{H}xf32>, tensor<1x{2 * H}xi8>)
  kernel.launch @{k1}[%n, %t](%n, %t, %acc, %xq, %w, %wu, %y) : [index, index](index, index, i32, tensor<1x{2 * H}xi8>, {gt}, {ut}, tensor<1x{n}xf32>)
  check.expect.close actual(%y) expected(%e) atol(3.0e-3) rtol(3.0e-3) nan(different) : tensor<1x{n}xf32>
  check.return
}}
""")
# 2b'. g_silu_mul (layers whose ffn_gate and ffn_up formats differ: two projections, then act(gate) * up in place).
# GEGLU builds only: the SwiGLU kernel's bounds are proven from the model's launch (constant n), not as a test root.
if cfg.get("ffn_act") == "gelu" and any(f"blk.{li}.ffn_gate.weight" in T and f"blk.{li}.ffn_up.weight" in T
       and T[f"blk.{li}.ffn_gate.weight"].tensor_type != T[f"blk.{li}.ffn_up.weight"].tensor_type for li in range(cfg["layers"])):
    FFW = max(cfg["ffn"]) if isinstance(cfg["ffn"], list) else cfg["ffn"]
    fg = (2 * rng.standard_normal((3, FFW))).astype(np.float32); fu = rng.standard_normal((3, FFW)).astype(np.float32)
    save("sm_f", fg); save("sm_u", fu); save("sm_y", (ffn_act(fg.astype(np.float64)) * fu).astype(np.float32))
    cases.append(f"""check.case public @silu_mul {{
  %t = check.literal value(3) : index
  %n = check.literal value({FFW}) : index
  %f = check.file.read.npy path("sm_f.npy") : tensor<3x{FFW}xf32>
  %u = check.file.read.npy path("sm_u.npy") : tensor<3x{FFW}xf32>
  %e = check.file.read.npy path("sm_y.npy") : tensor<3x{FFW}xf32>
  kernel.launch @g_silu_mul[%t](%t, %n, %f, %u) : [index](index, index, tensor<3x{FFW}xf32>, tensor<3x{FFW}xf32>)
  check.expect.close actual(%f) expected(%e) atol(1.0e-3) rtol(1.0e-3) nan(different) : tensor<3x{FFW}xf32>
  check.return
}}
""")
# 2c. int8 pair matvecs (gen_llm PAIR_PROJ: attention q + gate, k + v of one format and shape): at 8 and 16 tokens
# each output equals the 1-token single-weight kernel on that weight bit for bit
_a = next((li for li in range(cfg["layers"]) if f"blk.{li}.attn_q.weight" in T), None)
for pa, pb in (("attn_q", "attn_gate"), ("attn_k", "attn_v")):
    if not ACT_Q8 or _a is None:
        break
    an, bn = f"blk.{_a}.{pa}.weight", f"blk.{_a}.{pb}.weight"
    if bn not in T or T[an].tensor_type != T[bn].tensor_type or tuple(T[an].shape) != tuple(T[bn].shape):
        continue
    fmt = T[an].tensor_type.name.lower(); n = int(T[an].shape[1])
    if fmt not in IQ_FMTS or H // 32 > 1024:
        continue
    save(f"pr_{pa}_a", raw(an)); save(f"pr_{pa}_b", raw(bn)); at, bt = ty(raw(an)), ty(raw(bn))
    for tg in (8, 16):
        save(f"pr_{pa}_x{tg}", rng.standard_normal((tg, H)).astype(np.float32))
        R1, Rt = iq_r(n, H, 1), iq_r(n, H, tg, 2)
        k1 = f"g_mv_{fmt}_k{H}_q8" + (f"_r{R1}" if R1 != 16 else "")
        kt = f"g_mv_{fmt}_k{H}_pair_t{tg}_q8" + (f"_r{Rt}" if Rt != 16 else "")
        s1 = f"[index, index](index, index, i32, tensor<{tg}x{2 * H}xi8>, {at}, tensor<{tg}x{n}xf32>)"
        cases.append(f"""check.case public @mvpair_{pa}_t{tg} {{
  %n = check.literal value({n}) : index
  %t = check.literal value({tg}) : index
  %k = check.literal value({H}) : index
  %acc = check.literal value(0) : i32
  %wa = check.file.read.npy path("pr_{pa}_a.npy") : {at}
  %wb = check.file.read.npy path("pr_{pa}_b.npy") : {bt}
  %x = check.file.read.npy path("pr_{pa}_x{tg}.npy") : tensor<{tg}x{H}xf32>
  %xq = check.generate.fill value(0) : tensor<{tg}x{2 * H}xi8>
  %ya = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  %yb = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  %ea = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  %eb = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  kernel.launch @g_quant_q8[%t, %k](%t, %k, %x, %xq) : [index, index](index, index, tensor<{tg}x{H}xf32>, tensor<{tg}x{2 * H}xi8>)
  kernel.launch @{kt}[%n, %t](%n, %t, %acc, %xq, %wa, %wb, %ya, %yb) : [index, index](index, index, i32, tensor<{tg}x{2 * H}xi8>, {at}, {bt}, tensor<{tg}x{n}xf32>, tensor<{tg}x{n}xf32>)
  kernel.launch @{k1}[%n, %t](%n, %t, %acc, %xq, %wa, %ea) : {s1}
  kernel.launch @{k1}[%n, %t](%n, %t, %acc, %xq, %wb, %eb) : {s1}
  check.expect.close actual(%ya) expected(%ea) atol(0.0) rtol(0.0) nan(different) : tensor<{tg}x{n}xf32>
  check.expect.close actual(%yb) expected(%eb) atol(0.0) rtol(0.0) nan(different) : tensor<{tg}x{n}xf32>
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
def yarn(inv, y=cfg.get("rope_scaling"), rd=RD, theta=cfg["rope_theta"]):
    """(frequencies, cos/sin scale) of llama.cpp's YaRN y (a rope_scaling block; gen_llm.yarn_constants) for a RoPE of
    rd rotated dims and base theta; unchanged without YaRN (y None or not yarn)."""
    if (y or {}).get("type") != "yarn":
        return inv, 1.0
    f, n0 = float(y["factor"]), float(y["orig_ctx"])
    dim = lambda rot: rd * np.log(n0 / (rot * 2 * np.pi)) / (2 * np.log(theta))
    low, high = max(0.0, np.floor(dim(y.get("beta_fast", 32.0)))), min(rd - 1.0, np.ceil(dim(y.get("beta_slow", 1.0))))
    ramp = np.clip((np.arange(len(inv)) - low) / max(0.001, high - low), 0, 1)
    return inv * (1 - (1 - 1 / f) * ramp), y.get("attn_factor", 1.0) * (1 + 0.1 * np.log(f))
def rope(a, RD=RD, theta=cfg["rope_theta"], factors=cfg.get("rope_freq_factors"), y=cfg.get("rope_scaling"), at=None):
    """RoPE of a (tokens x heads x D) at positions `at` (default: the global pos)."""
    inv = theta ** (-np.arange(0, RD // 2) * 2.0 / RD)
    if factors:   # llama3 rope_freqs: each pair's frequency divided by its factor
        inv = inv / np.asarray(factors[:RD // 2], np.float64)
    inv, msc = yarn(inv, y, RD, theta); ang = (pos if at is None else at)[:, None] * inv[None]
    c, s = np.cos(ang)[:, None] * msc, np.sin(ang)[:, None] * msc
    if cfg["rope_type"] == "neox":
        a1, a2 = a[..., :RD // 2], a[..., RD // 2:RD]; return np.concatenate([a1 * c - a2 * s, a2 * c + a1 * s, a[..., RD:]], -1)
    a1, a2 = a[..., 0::2], a[..., 1::2]; o = np.empty_like(a); o[..., 0::2] = a1 * c - a2 * s; o[..., 1::2] = a2 * c + a1 * s; return o
# attention projection biases (cfg "attn_bias", qwen2): g_qk_rope_store adds them to q/k/v before the norm and RoPE
BIAS = cfg.get("attn_bias") or []
BW = {"q": QH * D, "k": KVH * D, "v": KVH * D}
bq, bk, bv = (deq(f"blk.{A0}.attn_{p}.bias").astype(np.float32).reshape(-1, D) if p in BIAS else np.zeros((BW[p] // D, D), np.float32)
              for p in "qkv")
for p, b_ in zip("qkv", (bq, bk, bv)):
    if p in BIAS: save(f"b_{p}", b_.reshape(-1))
B_DECL = "".join(f'  %b{p} = check.file.read.npy path("b_{p}.npy") : tensor<{BW[p]}xf32>\n' for p in BIAS)
B_ARGS = "".join(f", %b{p}" for p in BIAS)
B_TYPES = "".join(f", tensor<{BW[p]}xf32>" for p in BIAS)
qe = rope(norm(q + bq, qn) if cfg["qk_norm"] else q + bq)
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
{B_DECL}{mr_decl(t)}  kernel.launch @g_qk_rope_store[%t](%t, %layer, %rope, %ctx, %pool, %q, %k, %v, %qn, %kn, %info, %orig, %arena{MR_ARGS}{B_ARGS}) : [index](index, index, i32, index, index, tensor<{t}x{QH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{t}x4xi32>, tensor<64xi32>, tensor<1024xf16>{mr_types(t)}{B_TYPES})
  check.expect.close actual(%q) expected(%qe) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{QH * D}xf32>
  check.return
}}
""")
# 3'. sliding layers with their own rotary dims / base at the same head_dim (gen_llm SWA_ROPE): g_qk_rope_store_swa
RD_SWA, TH_SWA = cfg.get("rope_dims_swa") or RD, cfg.get("rope_theta_swa") or cfg["rope_theta"]
if cfg.get("sliding_layers") and (cfg.get("head_dim_swa") or D) == D and (RD_SWA, TH_SWA) != (RD, cfg["rope_theta"]):
    qs = rope(norm(q + bq, qn) if cfg["qk_norm"] else q + bq, RD_SWA, TH_SWA, None, None)   # no frequency factors, no YaRN
    save("r_qs", qs.reshape(t, -1).astype(np.float32))
    cases.append(cases[-1].replace("@qk_rope {", "@qk_rope_swa {").replace("@g_qk_rope_store[", "@g_qk_rope_store_swa[")
                 .replace('path("r_qe.npy")', 'path("r_qs.npy")'))
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
qr = rope(norm(q + bq, qn) if cfg["qk_norm"] else q + bq); kr = rope(norm(k + bk, kn) if cfg["qk_norm"] else k + bk)
# the paged KV cache holds F16 K and V: with large biases (k up to |130|) their rounding moves the output past atol
kr, vs = (a.astype(np.float16).astype(np.float32) for a in (kr, v + bv))
att = np.zeros((t, QH, D))
sinks = deq(f"blk.{A0}.attn_sinks.weight") if cfg.get("attn_sinks") else np.zeros(QH, np.float32)
for hh in range(QH):
    kv = hh // (QH // KVH); sc = qr[:, hh] @ kr[:, kv].T / np.sqrt(D); sc[np.triu_indices(t, 1)] = -1e30
    if cfg.get("attn_sinks"):
        mx = np.maximum(sc.max(-1, keepdims=True), sinks[hh]); pr = np.exp(sc - mx); pr /= pr.sum(-1, keepdims=True) + np.exp(sinks[hh] - mx)
    else:
        pr = np.exp(sc - sc.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
    att[:, hh] = pr @ vs[:, kv]
gh = rng.standard_normal((t, QH)).astype(np.float32)   # per-head gate logits (attn_gate "per_head")
if gate_kind == "per_head":
    gate = np.repeat(gh[:, :, None], D, 2)
if gate_kind:
    att = att / (1 + np.exp(-gate))
blocks = (ctx + 15) // 16
orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)  # record: map origin 40; identity page map
info = np.zeros((t, 4), np.int32); info[:, 1] = pos; info[:, 2] = 1
S = 64
for n, a in [("a_q", q.reshape(t, -1)), ("a_k", k.reshape(t, -1)), ("a_v", v.reshape(t, -1)), ("a_g", gate.reshape(t, -1)),
             ("a_qg", np.stack([q, gate], 2).reshape(t, -1)), ("a_gh", gh), ("a_qkv", np.concatenate([q, k, v], 1).reshape(t, -1)), ("a_info", info), ("a_orig", orig), ("a_sinks", sinks.astype(np.float32)), ("a_o", att.reshape(t, -1).astype(np.float32))]:
    save(n, a)
QD = QH * D
split_text = ""
if gate_kind == "fused_q":
    split_text = f"""  %qg = check.file.read.npy path("a_qg.npy") : tensor<{t}x{2 * QD}xf32>
  kernel.launch @g_split_qgate[%t](%t, %qg, %q, %g) : [index](index, tensor<{t}x{2 * QD}xf32>, tensor<{t}x{QD}xf32>, tensor<{t}x{QD}xf32>)
"""
if gate_kind == "per_head":   # g_gate_heads broadcasts each head's logit over its D dims
    split_text += f"""  %gh = check.file.read.npy path("a_gh.npy") : tensor<{t}x{QH}xf32>
  kernel.launch @g_gate_heads[%t](%t, %gh, %g) : [index](index, tensor<{t}x{QH}xf32>, tensor<{t}x{QD}xf32>)
"""
QKV = cfg.get("attn_qkv", False)
if QKV:   # one fused [q | k | v] projection row per token: g_split_qkv fills q, k and v
    split_text += f"""  %qkv = check.file.read.npy path("a_qkv.npy") : tensor<{t}x{QD + 2 * KVH * D}xf32>
  kernel.launch @g_split_qkv[%t](%t, %qkv, %q, %k, %v) : [index](index, tensor<{t}x{QD + 2 * KVH * D}xf32>, tensor<{t}x{QD}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>)
"""
cases.append(f"""check.case public @attention_chain {{
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %rope = check.literal value(1) : i32
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({ctx}) : index
  %q = {"check.generate.fill value(0.0)" if gate_kind == "fused_q" or QKV else 'check.file.read.npy path("a_q.npy")'} : tensor<{t}x{QD}xf32>
  %g = {"check.generate.fill value(0.0)" if gate_kind in ("fused_q", "per_head") else 'check.file.read.npy path("a_g.npy")'} : tensor<{t}x{QD}xf32>
  %k = {"check.generate.fill value(0.0)" if QKV else 'check.file.read.npy path("a_k.npy")'} : tensor<{t}x{KVH * D}xf32>
  %v = {"check.generate.fill value(0.0)" if QKV else 'check.file.read.npy path("a_v.npy")'} : tensor<{t}x{KVH * D}xf32>
  %qn = check.file.read.npy path("r_qn.npy") : tensor<{D}xf32>
  %kn = check.file.read.npy path("r_kn.npy") : tensor<{D}xf32>
  %info = check.file.read.npy path("a_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("a_orig.npy") : tensor<{10 + blocks}xi32>
  %eo = check.file.read.npy path("a_o.npy") : tensor<{t}x{QD}xf32>
  %sinks = check.file.read.npy path("a_sinks.npy") : tensor<{QH}xf32>
  %arena = check.generate.fill value(0.0) : tensor<{2 * ctx * KVH * D}xf16>
  %pacc = check.generate.fill value(0.0) : tensor<{t * QH * S * D}xf32>
  %pml = check.generate.fill value(0.0) : tensor<{t * QH * S * 2}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{QD}xf32>
{split_text}{B_DECL}{mr_decl(t)}  kernel.launch @g_qk_rope_store[%t](%t, %layer, %rope, %ctx, %pool, %q, %k, %v, %qn, %kn, %info, %orig, %arena{MR_ARGS}{B_ARGS}) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{2 * ctx * KVH * D}xf16>{mr_types(t)}{B_TYPES})
  kernel.launch @g_attention_split[%t](%t, %layer, %sliding, %ctx, %pool, %q, %info, %orig, %arena, %pacc, %pml) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{2 * ctx * KVH * D}xf16>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>)
  kernel.launch @g_attention_combine[%t](%t, %sliding, %info, %pacc, %pml, {"%sinks" if cfg.get("attn_sinks") else "%g" if gate_kind else "%q"}, %o) : [index](index, i32, tensor<{t}x4xi32>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>, {f"tensor<{QH}xf32>" if cfg.get("attn_sinks") else f"tensor<{t}x{QD}xf32>"}, tensor<{t}x{QD}xf32>)
  check.expect.close actual(%o) expected(%eo) atol(3.0e-3) rtol(3.0e-3) nan(different) : tensor<{t}x{QD}xf32>
  check.return
}}
""")
# 3c. prefill attention on a span that CONTINUES stored KV: history positions [0, p0) are written
# by g_qk_rope_store, then a span at p0 (t tokens, the last `pad` rows not live) runs the prefill
# path (g_attention_flash2 on the listed 16-token tiles + the listed split/combine fallback for the
# rest) and, separately, the decode path (g_attention_split + combine). Both vs numpy, on a
# sliding and a non-sliding layer with that layer's real norms and sinks.
def prefill_cont_case(li, p0, t, pad, tag, qscale=1.0):
    sl = 1 if li in set(cfg.get("sliding_layers", [])) else 0
    rp = 0 if li in set(cfg.get("nope_layers", [])) else 1
    W = cfg.get("sliding_window") or 0
    qn_ = (deq(f"blk.{li}.attn_q_norm.weight") if cfg["qk_norm"] else np.ones(D, np.float32)) * np.float32(qscale)
    kn_ = deq(f"blk.{li}.attn_k_norm.weight") if cfg["qk_norm"] else np.ones(D, np.float32)
    snk = deq(f"blk.{li}.attn_sinks.weight").astype(np.float32) if cfg.get("attn_sinks") else np.zeros(QH, np.float32)
    n = p0 + t; ctx = ((n + 15) // 16) * 16 + 64; blocks = ctx // 16
    allpos = np.arange(n)
    qa = (rng.standard_normal((n, QH, D)) * (1.0 if cfg["qk_norm"] else qscale)).astype(np.float32); ka = rng.standard_normal((n, KVH, D)).astype(np.float32)
    va = rng.standard_normal((n, KVH, D)).astype(np.float32)
    qr_ = norm(qa, qn_) if cfg["qk_norm"] else qa; kr_ = norm(ka, kn_) if cfg["qk_norm"] else ka
    if rp: qr_, kr_ = rope(qr_, at=allpos), rope(kr_, at=allpos)
    kf = kr_.astype(np.float16).astype(np.float64); vf = va.astype(np.float16).astype(np.float64)
    live = np.ones(t, bool); live[t - pad:] = False
    att = np.zeros((t, QH, D))
    for i in range(t):
        if not live[i]: continue
        p = p0 + i; lo = max(0, p - (W - 1)) if (sl and W) else 0
        for hh in range(QH):
            kv = hh // (QH // KVH)
            sc = kf[lo:p + 1, kv] @ qr_[p, hh].astype(np.float64) / np.sqrt(D)
            mx = max(sc.max(), snk[hh]) if cfg.get("attn_sinks") else sc.max()
            pr = np.exp(sc - mx); den = pr.sum() + (np.exp(snk[hh] - mx) if cfg.get("attn_sinks") else 0.0)
            att[i, hh] = (pr @ vf[lo:p + 1, kv]) / den
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)
    hinfo = np.zeros((p0, 4), np.int32); hinfo[:, 1] = allpos[:p0]; hinfo[:, 2] = 1
    sinfo = np.zeros((t, 4), np.int32); sinfo[:, 1] = allpos[p0:]; sinfo[:, 2] = live
    att_f = np.where(covered_tokens(sinfo)[:, None, None], att, 0.0)   # flash2 writes the listed tiles only
    for nn, a in [("hq", qa[:p0].reshape(p0, -1)), ("hk", ka[:p0].reshape(p0, -1)), ("hv", va[:p0].reshape(p0, -1)), ("hinfo", hinfo),
                  ("sq", qa[p0:].reshape(t, -1)), ("sk", ka[p0:].reshape(t, -1)), ("sv", va[p0:].reshape(t, -1)), ("sinfo", sinfo),
                  ("orig", orig), ("qn", qn_.astype(np.float32)), ("kn", kn_.astype(np.float32)), ("sinks", snk), ("o", att.reshape(t, -1).astype(np.float32)), ("of", att_f.reshape(t, -1).astype(np.float32)),
                  # a used KV pool: positions this sequence never wrote hold another sequence's keys and values
                  ("arena", (rng.standard_normal(2 * ctx * KVH * D) * 4).astype(np.float16))]:
        save(f"{tag}_{nn}", a)
    print(f"{tag}: layer {li} sliding={sl} rope={rp} p0={p0} t={t} pad={pad} |o| max {np.abs(att).max():.4f} mean {np.abs(att).mean():.5f}")
    QD = QH * D; KD = KVH * D; AR = f"tensor<{2 * ctx * KD}xf16>"; OT = f"tensor<{t}x{QD}xf32>"
    S = 64; S16 = 16
    gt = f"tensor<{QH}xf32>" if cfg.get("attn_sinks") else OT
    gv = "%sinks" if cfg.get("attn_sinks") else "%sq"
    # projection biases (if the model has them) are zero here: g_qk_rope_store's bias add is checked by @qk_rope
    ZB_DECL = "".join(f"  %zb{p} = check.generate.fill value(0.0) : tensor<{BW[p]}xf32>\n" for p in BIAS)
    ZB_ARGS = "".join(f", %zb{p}" for p in BIAS)
    def st(tt, pre):
        return (f"  kernel.launch @g_qk_rope_store[%{pre}t](%{pre}t, %layer, %rope, %ctx, %pool, %{pre}q, %{pre}k, %{pre}v, %qn, %kn, %{pre}info, %orig, %arena{ZB_ARGS}) : "
                f"[index](index, index, i32, index, index, tensor<{tt}x{QD}xf32>, tensor<{tt}x{KD}xf32>, tensor<{tt}x{KD}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{tt}x4xi32>, tensor<{10 + blocks}xi32>, {AR}{B_TYPES})\n")
    cases.append(f"""check.case public @{tag} {{
  %ht = check.literal value({p0}) : index
  %st = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %rope = check.literal value({rp}) : i32
  %sliding = check.literal value({sl}) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({ctx}) : index
  %hq = check.file.read.npy path("{tag}_hq.npy") : tensor<{p0}x{QD}xf32>
  %hk = check.file.read.npy path("{tag}_hk.npy") : tensor<{p0}x{KD}xf32>
  %hv = check.file.read.npy path("{tag}_hv.npy") : tensor<{p0}x{KD}xf32>
  %hinfo = check.file.read.npy path("{tag}_hinfo.npy") : tensor<{p0}x4xi32>
  %sq = check.file.read.npy path("{tag}_sq.npy") : {OT}
  %sk = check.file.read.npy path("{tag}_sk.npy") : tensor<{t}x{KD}xf32>
  %sv = check.file.read.npy path("{tag}_sv.npy") : tensor<{t}x{KD}xf32>
  %sinfo = check.file.read.npy path("{tag}_sinfo.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("{tag}_orig.npy") : tensor<{10 + blocks}xi32>
  %qn = check.file.read.npy path("{tag}_qn.npy") : tensor<{D}xf32>
  %kn = check.file.read.npy path("{tag}_kn.npy") : tensor<{D}xf32>
  %sinks = check.file.read.npy path("{tag}_sinks.npy") : tensor<{QH}xf32>
  %eo = check.file.read.npy path("{tag}_o.npy") : {OT}
  %eof = check.file.read.npy path("{tag}_of.npy") : {OT}
  %arena = check.file.read.npy path("{tag}_arena.npy") : {AR}
  %pacc = check.generate.fill value(0.0) : tensor<{t * QH * S * D}xf32>
  %pml = check.generate.fill value(0.0) : tensor<{t * QH * S * 2}xf32>
  %o_pre = check.generate.fill value(0.0) : {OT}
  %o_dec = check.generate.fill value(0.0) : {OT}
{ZB_DECL}{st(p0, "h")}{st(t, "s")}{list_launch("%st", t, "%sinfo")}  kernel.launch @g_attention_flash2[%st](%st, %layer, %sliding, %ctx, %pool, %sq, {gv}, %sinfo, %tiles, %orig, %arena, %o_pre) : [index](index, index, i32, index, index, {OT}, {gt}, tensor<{t}x4xi32>, tensor<{t}xi32>, tensor<{10 + blocks}xi32>, {AR}, {OT})
  kernel.launch @g_attention_split[%st](%st, %layer, %sliding, %ctx, %pool, %sq, %sinfo, %orig, %arena, %pacc, %pml) : [index](index, index, i32, index, index, {OT}, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, {AR}, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>)
  kernel.launch @g_attention_combine[%st](%st, %sliding, %sinfo, %pacc, %pml, {gv}, %o_dec) : [index](index, i32, tensor<{t}x4xi32>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>, {gt}, {OT})
  check.expect.close actual(%o_dec) expected(%eo) atol(1.5e-3) rtol(3.0e-3) nan(different) : {OT}
  check.expect.close actual(%o_pre) expected(%eof) atol(2.5e-3) rtol(1.0e-2) nan(different) : {OT}
  check.return
}}
""")
if cfg.get("attn_sinks") and os.environ.get("PREFILL_CONT", "1") != "0":
    _sl = cfg.get("sliding_layers") or []
    _ns = [i for i in range(cfg["layers"]) if i not in set(_sl) and i not in set(cfg.get("recurrent_layers", []))]
    _p0 = max(300, (cfg.get("sliding_window") or 0) + 100)
    for _li in ([_sl[0]] if _sl else []) + _ns[:1]:
        prefill_cont_case(_li, _p0, 48, 0, f"pc_full_{_li}")     # 3 uniform tiles: flash2 only
        prefill_cont_case(_li, _p0, 48, 7, f"pc_part_{_li}")     # last tile partly padding: listed fallback
        prefill_cont_case(_li, _p0, 48, 0, f"pc_small_{_li}", 0.1)  # small scores: the sink dominates the softmax
        prefill_cont_case(_li, 64, 64, 60, f"pc_reuse_{_li}")    # a reused 64-token prefix, 4 new tokens in a 64-row epoch
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
# 4b. g_gdn_recurrence alone on one launch of two spans: a fresh 37-token span (position 0) and a 5-token span that
# continues its reader slot (odd lengths: partial output batches; each span's own reader/writer slots).
if cfg.get("recurrent_layers"):
    spans_ = [(37, 0, False), (5, 10, True)]; t = sum(L for L, _, _ in spans_); ns = len(spans_)
    x = rng.standard_normal((t, C)).astype(np.float32)
    qs, ks, vs = x[:, :HK * DK].reshape(t, HK, DK), x[:, HK * DK:2 * HK * DK].reshape(t, HK, DK), x[:, 2 * HK * DK:].reshape(t, HV, DV)
    qinv = 1 / np.sqrt((qs.astype(np.float64) ** 2).sum(-1) + eps); kinv = 1 / np.sqrt((ks.astype(np.float64) ** 2).sum(-1) + eps)
    inv = np.stack([qinv, kinv], -1).reshape(t, 2 * HK).astype(np.float32)
    alpha = rng.standard_normal((t, HV)).astype(np.float32); beta = rng.standard_normal((t, HV)).astype(np.float32)
    arena = np.zeros(2 * ns * LB // 4, np.float32); meta = np.zeros(4 + 9 * ns, np.int32); meta[0] = ns
    origins = np.zeros((ns, 5), np.int64); raw_o = np.zeros((t, HV, DV)); first = 0; finals = []
    for si, (L, p0, cont) in enumerate(spans_):
        rd, wr = 2 * si * LB, (2 * si + 1) * LB
        origins[si] = [rd, wr, 0, p0, 0]; meta[4 + 9 * si:4 + 9 * si + 5] = [L, p0, 0, si, first]
        S = np.zeros((HV, DV, DK))
        if cont:
            S0 = (0.3 * rng.standard_normal((HV, DV, DK))).astype(np.float32)
            arena[rd // 4:rd // 4 + SB // 4] = S0.reshape(-1); S = S0.astype(np.float64)
        for i in range(first, first + L):
            for h in range(HV):
                kh = h % HK; q = qs[i, kh] * qinv[i, kh] / np.sqrt(DK); k = ks[i, kh] * kinv[i, kh]
                St = S[h] * np.exp(np.log1p(np.exp(float(alpha[i, h]) + dt[h])) * a_log[h])
                d = (vs[i, h] - St @ k) / (1 + np.exp(-float(beta[i, h])))
                S[h] = St + np.outer(d, k); raw_o[i, h] = S[h] @ q
        finals.append((wr, S)); first += L
    expect = arena.copy()
    for wr, S in finals:
        expect[wr // 4:wr // 4 + SB // 4] = S.astype(np.float32).reshape(-1)
    for n, a in [("gr_meta", meta), ("gr_orig", origins.reshape(-1).view(np.int32)), ("gr_arena", arena), ("gr_expect", expect),
                 ("gr_x", x), ("gr_inv", inv), ("gr_alpha", alpha), ("gr_beta", beta), ("gr_raw", raw_o.reshape(t, VD).astype(np.float32))]:
        save(n, a)
    AT = f"tensor<{2 * ns * LB // 4}xf32>"; MT = f"tensor<{4 + 9 * ns}xi32>"; OT = f"tensor<{10 * ns}xi32>"
    cases.append(f"""check.case public @gdn_recurrence_spans {{
  %t = check.literal value({t}) : index
  %spans = check.literal value({ns}) : index
  %layer = check.literal value(0) : index
  %meta = check.file.read.npy path("gr_meta.npy") : {MT}
  %orig = check.file.read.npy path("gr_orig.npy") : {OT}
  %arena = check.file.read.npy path("gr_arena.npy") : {AT}
  %expect = check.file.read.npy path("gr_expect.npy") : {AT}
  %x = check.file.read.npy path("gr_x.npy") : tensor<{t}x{C}xf32>
  %inv = check.file.read.npy path("gr_inv.npy") : tensor<{t}x{2 * HK}xf32>
  %alpha = check.file.read.npy path("gr_alpha.npy") : tensor<{t}x{HV}xf32>
  %beta = check.file.read.npy path("gr_beta.npy") : tensor<{t}x{HV}xf32>
  %a = check.file.read.npy path("g_a.npy") : tensor<{HV}xf32>
  %dt = check.file.read.npy path("g_dt.npy") : tensor<{HV}xf32>
  %eraw = check.file.read.npy path("gr_raw.npy") : tensor<{t}x{VD}xf32>
  %raw = check.generate.fill value(0.0) : tensor<{t}x{VD}xf32>
  kernel.launch @g_gdn_recurrence[%spans](%t, %spans, %layer, %meta, %orig, %arena, %x, %inv, %alpha, %beta, %a, %dt, %raw) : [index](index, index, index, {MT}, {OT}, {AT}, tensor<{t}x{C}xf32>, tensor<{t}x{2 * HK}xf32>, tensor<{t}x{HV}xf32>, tensor<{t}x{HV}xf32>, tensor<{HV}xf32>, tensor<{HV}xf32>, tensor<{t}x{VD}xf32>)
  check.expect.close actual(%raw) expected(%eraw) atol(1.0e-4) rtol(2.0e-3) nan(different) : tensor<{t}x{VD}xf32>
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
# g_post_add_norm_rs[_live] (gen_llm fuse_post_add_row_scale): x += (rmsnorm(y) * w_post if post norms else y);
# h = rmsnorm(x) * w_next; then g_row_scale_f16's scale / f16 row of h. _live: a row with info word 2 = 0 gets scale 0
# and a zero f16 row (x and h are still updated). _live_lv (gen_llm live_row_skip): a padding row only gets scale 0.
if os.environ.get("GEMM_TESTS", "1") == "1":
    pa_y = (rng.standard_normal((3, H)) * np.array([[1e-2], [1.0], [30.0]])).astype(np.float32)
    pa_x = (rng.standard_normal((3, H)) * np.array([[1.0], [1e2], [1e-3]])).astype(np.float32)
    pa_wp = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
    pa_wn = (rng.standard_normal(H) * np.array([3e3])).astype(np.float32)
    def _rms(v, e): return v / np.sqrt((v.astype(np.float64) ** 2).mean(-1, keepdims=True) + e)
    pa_r = (_rms(pa_y, cfg.get("post_norm_eps", 1e-8)) * pa_wp) if cfg.get("post_norms") else pa_y
    pa_xe = (pa_x + pa_r).astype(np.float32)
    pa_h = (_rms(pa_xe, eps) * pa_wn).astype(np.float32)
    pa_e = np.clip(np.frexp(np.abs(pa_h).max(1))[1] - 1 - 13, 0, 100)
    pa_live = np.array([1, 0, 1])
    pa_info = np.zeros((3, 4), np.int32); pa_info[:, 2] = pa_live
    for live in (0, 1, 2):
        sc = np.stack([2.0 ** -pa_e, 2.0 ** pa_e], 1).astype(np.float32)
        if live:
            sc = sc * pa_live[:, None]
        h16 = (pa_h * sc[:, :1]).astype(np.float16)
        sfx, xs = ["", "_live", "_live_lv"][live], "_lv" if live == 2 else ""
        dead = pa_live[:, None] == 0 if live == 2 else np.zeros((3, 1), bool)
        for n, a in [("pa_y", pa_y), ("pa_x", pa_x), ("pa_wp", pa_wp), ("pa_wn", pa_wn), ("pa_xe" + xs, np.where(dead, pa_x, pa_xe)), ("pa_he" + xs, np.where(dead, 0, pa_h).astype(np.float32)),
                     ("pa_info", pa_info), ("pa_s" + sfx, sc.astype(np.float32)), ("pa_h16" + sfx, h16)]:
            save(n, a)
        info_decl = '  %info = check.file.read.npy path("pa_info.npy") : tensor<3x4xi32>\n' if live else ""
        cases.append(f"""check.case public @post_add_norm_rs{sfx} {{
  %rows = check.literal value(3) : index
  %y = check.file.read.npy path("pa_y.npy") : tensor<3x{H}xf32>
  %x = check.file.read.npy path("pa_x.npy") : tensor<3x{H}xf32>
  %wp = check.file.read.npy path("pa_wp.npy") : tensor<{H}xf32>
  %wn = check.file.read.npy path("pa_wn.npy") : tensor<{H}xf32>
  %ex = check.file.read.npy path("pa_xe{xs}.npy") : tensor<3x{H}xf32>
  %eh = check.file.read.npy path("pa_he{xs}.npy") : tensor<3x{H}xf32>
  %es = check.file.read.npy path("pa_s{sfx}.npy") : tensor<3x2xf32>
  %eh16 = check.file.read.npy path("pa_h16{sfx}.npy") : tensor<3x{H}xf16>
{info_decl}  %h = check.generate.fill value(0.0) : tensor<3x{H}xf32>
  %s = check.generate.fill value(0.0) : tensor<3x2xf32>
  %h16 = check.generate.fill value(0.0) : tensor<3x{H}xf16>
  kernel.launch @g_post_add_norm_rs{sfx}[%rows](%rows, {"%info, " if live else ""}%y, %wp, %x, %wn, %h, %s, %h16) : [index](index, {"tensor<3x4xi32>, " if live else ""}tensor<3x{H}xf32>, tensor<{H}xf32>, tensor<3x{H}xf32>, tensor<{H}xf32>, tensor<3x{H}xf32>, tensor<3x2xf32>, tensor<3x{H}xf16>)
  check.expect.close actual(%x) expected(%ex) atol(1.0e-5) rtol(1.0e-5) nan(different) : tensor<3x{H}xf32>
  check.expect.close actual(%h) expected(%eh) atol(1.0e-2) rtol(1.0e-4) nan(different) : tensor<3x{H}xf32>
  check.expect.close actual(%s) expected(%es) atol(0.0) rtol(0.0) nan(different) : tensor<3x2xf32>
  check.expect.close actual(%h16) expected(%eh16) atol(1.0e-3) rtol(1.0e-3) nan(different) : tensor<3x{H}xf16>
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
            if fmt not in IQ_FMTS or (fmt, sw) in seen_g:   # the formats with GEMM kernel tests (formats.ICODES)
                continue
            seen_g.add((fmt, sw))
            yield name, fmt, sw
if os.environ.get("GEMM_TESTS", "1") == "1":
    for name, fmt, swiglu in gemm_cases():
        W = deq(name); n, K = W.shape
        n_use = min(n, 1024 + n % 128)                # a slice of output rows keeps the reference cheap (an output
                                                      # width the 128-wide tiles do not divide keeps its remainder)
        Wr = raw(name).reshape(n, -1)[:n_use].reshape(-1)
        Ws = W[:n_use].astype(np.float64)
        if swiglu:
            up = name.replace("ffn_gate", "ffn_up"); U = deq(up)[:n_use].astype(np.float64); Ur = raw(up).reshape(n, -1)[:n_use].reshape(-1)
        base_k = f"g_gemm_{fmt}_{'swiglu' if swiglu else 'store'}"
        # gen_llm.linear launches the 128/256-row kernels only on output widths their 128-wide tiles divide
        big = n_use % (64 if swiglu else 128) == 0
        for rows, kern in ((128, base_k),) + (((256, base_k + "_m128"),) + (() if fmt in GEMM_HELPER else ((512, base_k + "_m256"),)) if big else ()):
            x16 = rng.standard_normal((rows, K)).astype(np.float16)
            inv = rng.uniform(0.25, 2.0, rows).astype(np.float32)
            sc = np.stack([np.ones(rows, np.float32), inv], 1)
            g = (x16.astype(np.float64) @ Ws.T) * inv[:, None]
            if swiglu:
                u = (x16.astype(np.float64) @ U.T) * inv[:, None]
                yv = ffn_act(g) * u   # GEGLU ports (gemma4) swap the GEMM's silu for gelu-tanh
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
# 3c. a verify span at long context: ATTN_MT tokens of one row at consecutive positions past ATTN_MT_FROM take the
# multi-token split-K path (one K/V load per key for the span); K/V written straight into the arena (identity pages).
# The model's attention scale and sinks are in the reference; a sliding layer (window W) gets its own case.
def attention_span_case(tag, sliding):
    W = cfg.get("sliding_window") or 0
    t = attn_rules.ATTN_MT
    p0 = max(attn_rules.ATTN_MT_FROM + 76, W + 100 if sliding else 0)
    ctx = (p0 + t + 15) // 16 * 16 + 64
    scale = cfg["attn_scale"] if cfg.get("attn_scale") is not None else 1 / np.sqrt(D)
    sinks = deq(f"blk.{A0}.attn_sinks.weight").astype(np.float32) if cfg.get("attn_sinks") else None
    kvk = (0.5 * rng.standard_normal((ctx, KVH, D))).astype(np.float16); kvv = rng.standard_normal((ctx, KVH, D)).astype(np.float16)
    kvv[p0 + 1:] *= 64   # keys after a token's own position: a causal leak (one span member's range for another) dominates
    q = rng.standard_normal((t, QH, D)).astype(np.float32); gate = rng.standard_normal((t, QH, D)).astype(np.float32)
    att = np.zeros((t, QH, D))
    for i in range(t):
        p = p0 + i; lo = p - (W - 1) if sliding else 0
        for hh in range(QH):
            kv = hh // (QH // KVH)
            sc = kvk[lo:p + 1, kv].astype(np.float64) @ q[i, hh] * scale
            mx = sc.max() if sinks is None else max(sc.max(), sinks[hh])
            pr = np.exp(sc - mx); den = pr.sum() + (0.0 if sinks is None else np.exp(sinks[hh] - mx))
            att[i, hh] = pr @ kvv[lo:p + 1, kv].astype(np.float64) / den
    if gate_kind and sinks is None:
        att = att / (1 + np.exp(-gate))
    blocks = ctx // 16
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)
    info = np.zeros((t, 4), np.int32); info[:, 1] = p0 + np.arange(t); info[:, 2] = 1
    for n, a in [("q", q.reshape(t, -1)), ("g", gate.reshape(t, -1)), ("info", info), ("orig", orig),
                 ("sinks", sinks if sinks is not None else np.zeros(QH, np.float32)),
                 ("arena", np.concatenate([kvk.reshape(-1), kvv.reshape(-1)])), ("o", att.reshape(t, -1).astype(np.float32))]:
        save(f"{tag}_{n}", a)
    S = 64
    gv, gt = ("%sinks", f"tensor<{QH}xf32>") if sinks is not None else ("%g" if gate_kind else "%q", f"tensor<{t}x{QD}xf32>")
    cases.append(f"""check.case public @{tag} {{
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value({int(sliding)}) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({ctx}) : index
  %q = check.file.read.npy path("{tag}_q.npy") : tensor<{t}x{QD}xf32>
  %g = check.file.read.npy path("{tag}_g.npy") : tensor<{t}x{QD}xf32>
  %sinks = check.file.read.npy path("{tag}_sinks.npy") : tensor<{QH}xf32>
  %info = check.file.read.npy path("{tag}_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("{tag}_orig.npy") : tensor<{10 + blocks}xi32>
  %arena = check.file.read.npy path("{tag}_arena.npy") : tensor<{2 * ctx * KVH * D}xf16>
  %eo = check.file.read.npy path("{tag}_o.npy") : tensor<{t}x{QD}xf32>
  %pacc = check.generate.fill value(0.0) : tensor<{t * QH * S * D}xf32>
  %pml = check.generate.fill value(0.0) : tensor<{t * QH * S * 2}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{QD}xf32>
  kernel.launch @g_attention_split[%t](%t, %layer, %sliding, %ctx, %pool, %q, %info, %orig, %arena, %pacc, %pml) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{2 * ctx * KVH * D}xf16>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>)
  kernel.launch @g_attention_combine[%t](%t, %sliding, %info, %pacc, %pml, {gv}, %o) : [index](index, i32, tensor<{t}x4xi32>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>, {gt}, tensor<{t}x{QD}xf32>)
  check.expect.close actual(%o) expected(%eo) atol(3.0e-3) rtol(3.0e-3) nan(different) : tensor<{t}x{QD}xf32>
  check.return
}}
""")


if attn_rules.multi_token(attn_rules.UNROLL):   # the condition gen_llm.py emits the multi-token path on
    attention_span_case("attention_span", False)
    if cfg.get("sliding_layers") and cfg.get("sliding_window"):
        attention_span_case("attention_span_sliding", True)

# 3a''. DFlash proposals' head (draft/prep_weights.py, PROPOSE_HEAD=q4_0): g_prep_<fmt>_q4_0 converts the head at load,
# then the int8 head runs over the Q4_0 copy. Logits vs numpy on llama.cpp's quantize_row_q4_0 of the same values (a
# code at a rounding boundary may differ: tolerance), and the 4/8/16-span variants bit-identical to 1 span.
HEAD_FMT = T[head_name].tensor_type.name.lower()
if (HEAD_Q8 and HEAD_FMT in ("q8_0", "bf16", "f16") and H % 32 == 0 and (cfg.get("drafter") or {}).get("gguf")
        and os.environ.get("PROPOSE_HEAD", "q4_0") == "q4_0"):
    blk = Wh.astype(np.float32).reshape(-1, 32)
    m = blk[np.arange(len(blk)), np.abs(blk).argmax(1)]
    d4 = m / np.float32(-8)
    id4 = np.divide(np.float32(1), d4, out=np.zeros_like(d4), where=d4 != 0)
    c4 = np.minimum(np.float32(15), np.floor(blk * id4[:, None] + np.float32(8.5)))
    W4 = ((c4 - 8) * d4.astype(np.float16).astype(np.float32)[:, None]).reshape(Wh.shape)
    del blk, c4
    S, nb = 8, Wh.size // 32
    hm = np.zeros(4 + 9 * S, np.int32); hm[0] = S
    for j in range(S):
        hm[4 + 9 * j + 0] = 1; hm[4 + 9 * j + 4] = j; hm[4 + 9 * j + 6] = 1
    hp = rng.standard_normal((S, H)).astype(np.float32)
    lp = (q8_ref(hp) @ W4.T).astype(np.float32)   # S x V
    del W4
    CHP = CH * 128
    lpad = np.full((S, CHP), -3.0e38, np.float32); lpad[:, :cfg["vocab"]] = lp
    save("ph_meta", hm); save("ph_h", hp); save("ph_l", lpad.reshape(-1))
    mt, ht, w4t = f"tensor<{hm.size}xi32>", f"tensor<{S}x{H}xf32>", f"tensor<{nb * 18}xi8>"
    # a code at a rounding boundary (the GPU's reciprocal / fma vs numpy) moves a logit by ~d x |activation|; a
    # layout or nibble-order error moves it by the logit's own scale
    tol = 3.0e-2 * float(np.abs(lp).max())
    def phl(k, tag):
        return (f"  %p{tag} = check.generate.fill value(0) : tensor<{S * CH * 2}xi32>\n  %l{tag} = check.generate.fill value(0.0) : tensor<{S * CHP}xf32>\n"
                f"  kernel.launch @{k}[%spans](%spans, %t, %meta, %hq, %w4, %p{tag}, %l{tag}) : [index](index, index, {mt}, tensor<{S}x{2 * H}xi8>, {w4t}, tensor<{S * CH * 2}xi32>, tensor<{S * CHP}xf32>)\n")
    pv = [("g_head_mv_t4_q8_p40", "4"), ("g_head_mv_t8_q8_p40", "8"), ("g_head_mv_t16_q8_p40", "16")]
    cases.append(f"""check.case public @propose_head_q4 {{
  %spans = check.literal value({S}) : index
  %t = check.literal value({S}) : index
  %b = check.literal value({nb}) : index
  %o = check.literal value(0) : i64
  %meta = check.file.read.npy path("ph_meta.npy") : {mt}
  %h = check.file.read.npy path("ph_h.npy") : {ht}
  %w = check.file.read.npy path("hd_w.npy") : {hwt}
  %le = check.file.read.npy path("ph_l.npy") : tensor<{S * CHP}xf32>
  %w4 = check.generate.fill value(0) : {w4t}
  kernel.launch @g_prep_{HEAD_FMT}_q4_0[%b](%b, %o, %w4, %w) : [index](index, i64, {w4t}, {hwt})
  %hk = check.literal value({H}) : index
  %hq = check.generate.fill value(0) : tensor<{S}x{2 * H}xi8>
  kernel.launch @g_quant_q8[%t, %hk](%t, %hk, %h, %hq) : [index, index](index, index, {ht}, tensor<{S}x{2 * H}xi8>)
""" + phl("g_head_mv_q8_p40", "1") + "".join(phl(k, tag) for k, tag in pv) + f"""  check.expect.close actual(%l1) expected(%le) atol({tol:.3e}) rtol(0.0) nan(different) : tensor<{S * CHP}xf32>
""" + "".join(f"  check.expect.close actual(%l{tag}) expected(%l1) atol(0.0) rtol(0.0) nan(different) : tensor<{S * CHP}xf32>\n"
              f"  check.expect.equal actual(%p{tag}) expected(%p1) : tensor<{S * CH * 2}xi32>\n" for _, tag in pv) + "  check.return\n}\n")
# Drafter matrices without an int8 path (a vendor BF16/F16 DFlash drafter used as-is, cfg["drafter"]["gguf"]):
# the f32-activation matvecs (1 and 4 tokens; gate/up pair) and the GEMMs (64-, 128- and 256-row tiles) the drafter's
# propose / warm programs launch, one tensor per (format, K), vs float64 numpy on the dequantized weights. Rows per
# workgroup as gen_llm.py picks them: mv_rows.pick_r / iq_r for outputs of the trunk's hidden width (gen_dflash's %nH:
# fc, attn_output, ffn_down), 16 for the drafter's own widths (%nDQ/%nDKV/%nDF are not in gen_llm's NVAL).
PREP = os.environ.get("DRAFTER_PREP", "1") != "0"   # gen_dflash.py: BF16/F16 drafter matrices become Q8_0 at load


def prep_cases(nm, up, fmt, K, n, W, U, tag0, seen_mv):
    """Load-time drafter preparation (draft/prep_weights.py): g_prep_<fmt>_q8_0 converts the file's BF16/F16
    bytes to Q8_0, then the drafter's int8 matvecs (1/4/8/16 tokens; gate/up as one swiglu launch) run on it.
    Reference: llama.cpp's Q8_0 rounding of the weights (q8_ref per row) times q8_ref(x). The 4/8/16-token variants
    must equal the 1-token kernel bit for bit (zero tolerance)."""
    if (fmt, K, bool(up)) in seen_mv or K // 32 > 1024:
        return
    seen_mv.add((fmt, K, bool(up)))
    B = n * K // 32
    Wq = q8_ref(W.astype(np.float32)).astype(np.float64); Uq = q8_ref(U.astype(np.float32)).astype(np.float64) if up else None
    names = [("w", nm)] + ([("u", up)] if up else [])
    decl, preps, wq_args, wq_tys = "", "", [], []
    for v, name in names:
        rw = np.asarray(DT[name].data).view(np.int8).reshape(-1)
        save(f"{tag0}_p{v}", rw)
        decl += (f'  %{v} = check.file.read.npy path("{tag0}_p{v}.npy") : tensor<{rw.size}xi8>\n'
                 f"  %{v}q = check.generate.fill value(0) : tensor<{B * 34}xi8>\n")
        preps += (f"  kernel.launch @g_prep_{fmt}_q8_0[%b](%b, %o, %{v}q, %{v}) : [index](index, i64, tensor<{B * 34}xi8>, "
                  f"tensor<{rw.size}xi8>)\n")
        wq_args.append(f"%{v}q"); wq_tys.append(f"tensor<{B * 34}xi8>")
    sw = "_swiglu" if up else ""
    sig = lambda tg: (f"[index, index](index, index, i32, tensor<{tg}x{2 * K}xi8>, {', '.join(wq_tys)}, tensor<{tg}x{n}xf32>)")
    nr = n if n == H else None   # the drafter's own widths take R = 16 (not in gen_llm's NVAL)
    R1 = iq_r(nr, K, 1, bool(up))
    k1 = f"g_mv_q8_0_k{K}{sw}_q8" + (f"_r{R1}" if R1 != 16 else "")
    for tg in (1, 4, 8, 16):
        xs = rng.standard_normal((tg, K)).astype(np.float32); xr = q8_ref(xs).astype(np.float64)
        y = ffn_act(xr @ Wq.T) * (xr @ Uq.T) if up else xr @ Wq.T
        tag = f"{tag0}_prep_t{tg}"; save(tag + "_x", xs); save(tag + "_y", y.astype(np.float32))
        Rt = iq_r(nr, K, tg, bool(up))
        kt = f"g_mv_q8_0_k{K}{sw}" + (f"_t{tg}" if tg > 1 else "") + "_q8" + (f"_r{Rt}" if Rt != 16 else "")
        same = (f"""  %y1 = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  kernel.launch @{k1}[%n, %t](%n, %t, %acc, %xq, {", ".join(wq_args)}, %y1) : {sig(tg)}
  check.expect.close actual(%y) expected(%y1) atol(0.0) rtol(0.0) nan(different) : tensor<{tg}x{n}xf32>
""" if tg > 1 else "")
        tol = 2.0e-3 * max(1.0, float(np.abs(y).std()))
        cases.append(f"""check.case public @mv_{tag} {{
  %n = check.literal value({n}) : index
  %t = check.literal value({tg}) : index
  %k = check.literal value({K}) : index
  %b = check.literal value({B}) : index
  %o = check.literal value(0) : i64
  %acc = check.literal value(0) : i32
{decl}  %x = check.file.read.npy path("{tag}_x.npy") : tensor<{tg}x{K}xf32>
  %e = check.file.read.npy path("{tag}_y.npy") : tensor<{tg}x{n}xf32>
  %xq = check.generate.fill value(0) : tensor<{tg}x{2 * K}xi8>
  %y = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
{preps}  kernel.launch @g_quant_q8[%t, %k](%t, %k, %x, %xq) : [index, index](index, index, tensor<{tg}x{K}xf32>, tensor<{tg}x{2 * K}xi8>)
  kernel.launch @{kt}[%n, %t](%n, %t, %acc, %xq, {", ".join(wq_args)}, %y) : {sig(tg)}
  check.expect.close actual(%y) expected(%e) atol({tol:.3e}) rtol(2.0e-3) nan(different) : tensor<{tg}x{n}xf32>
{same}  check.return
}}
""")


def drafter_cases():
    dg = (cfg.get("drafter") or {}).get("gguf")
    if not dg or not os.path.exists(dg) or os.environ.get("DRAFTER_TESTS", "1") != "1":
        return
    global DT
    DT = {t_.name: t_ for t_ in GGUFReader(dg).tensors}
    fmt_of = lambda nm: DT[nm].tensor_type.name.lower()
    if not any(n_.startswith("blk.0.") and fmt_of(n_) not in IQ_FMTS | {"f32"} for n_ in DT):
        return
    draw = lambda nm: np.asarray(DT[nm].data).view(np.int8).reshape(-1)
    ddeq = lambda nm: dequantize(np.asarray(DT[nm].data), DT[nm].tensor_type).reshape(int(DT[nm].shape[1]), int(DT[nm].shape[0]))
    names = ["fc.weight"] + [f"blk.0.{s}.weight" for s in ("attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_down")]
    seen_mv, seen_gm = set(), set()
    for nm in names:
        if nm not in DT or len(DT[nm].shape) != 2:
            continue
        fmt = fmt_of(nm)
        if fmt in IQ_FMTS | {"f32"}:
            continue
        K, n = int(DT[nm].shape[0]), int(DT[nm].shape[1])
        up = nm.replace("ffn_gate", "ffn_up")
        sw = nm != up and up in DT and fmt_of(up) == fmt
        W = ddeq(nm).astype(np.float64); U = ddeq(up).astype(np.float64) if sw else None
        tag0 = "dr_" + nm.replace(".weight", "").replace(".", "_")
        if PREP and fmt in ("bf16", "f16") and K % 32 == 0:
            prep_cases(nm, up if sw else None, fmt, K, n, W, U, tag0, seen_mv)
            continue
        save(tag0 + "_w", draw(nm)); wt = ty(draw(nm))
        wdecl = f'  %w = check.file.read.npy path("{tag0}_w.npy") : {wt}\n'; wargs, wtys = "%w", wt
        if sw:
            save(tag0 + "_u", draw(up)); wdecl += f'  %u = check.file.read.npy path("{tag0}_u.npy") : {ty(draw(up))}\n'
            wargs += ", %u"; wtys += ", " + ty(draw(up))
        if K // 32 <= 1024 and (fmt, K, sw) not in seen_mv:   # wider K runs mv_wide (not covered here)
            seen_mv.add((fmt, K, sw))
            for tg in (1, 4):
                xs = rng.standard_normal((tg, K)).astype(np.float32)
                y = ffn_act(xs @ W.T) * (xs @ U.T) if sw else xs @ W.T
                tag = f"{tag0}_t{tg}"; save(tag + "_x", xs); save(tag + "_y", y.astype(np.float32))
                _R = pick_r(n, K, tg) if n == H else 16
                kern = f"g_mv_{fmt}_k{K}" + ("_swiglu" if sw else "") + ("_t4" if tg == 4 else "") + (f"_r{_R}" if _R != 16 else "")
                cases.append(f"""check.case public @mv_{tag} {{
  %n = check.literal value({n}) : index
  %t = check.literal value({tg}) : index
  %acc = check.literal value(0) : i32
{wdecl}  %x = check.file.read.npy path("{tag}_x.npy") : tensor<{tg}x{K}xf32>
  %e = check.file.read.npy path("{tag}_y.npy") : tensor<{tg}x{n}xf32>
  %y = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  kernel.launch @{kern}[%n, %t](%n, %t, %acc, %x, {wargs}, %y) : [index, index](index, index, i32, tensor<{tg}x{K}xf32>, {wtys}, tensor<{tg}x{n}xf32>)
  check.expect.close actual(%y) expected(%e) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{tg}x{n}xf32>
  check.return
}}
""")
        if (fmt, sw) in seen_gm or os.environ.get("GEMM_TESTS", "1") != "1":
            continue
        seen_gm.add((fmt, sw))
        base_k = f"g_gemm_{fmt}_{'swiglu' if sw else 'store'}"
        for rows, kern in ((128, base_k), (256, base_k + "_m128"), (512, base_k + "_m256")):
            x16 = rng.standard_normal((rows, K)).astype(np.float16)
            inv = rng.uniform(0.25, 2.0, rows).astype(np.float32)
            sc = np.stack([np.ones(rows, np.float32), inv], 1)
            g = (x16.astype(np.float64) @ W.T) * inv[:, None]
            yv = ffn_act(g) * ((x16.astype(np.float64) @ U.T) * inv[:, None]) if sw else g
            tag = f"{tag0}_gm{rows}"
            save(tag + "_x", x16); save(tag + "_s", sc); save(tag + "_y", yv.astype(np.float32))
            tol = 3.0e-2 * float(np.abs(yv).std() + 1e-6)
            cases.append(f"""check.case public @{tag} {{
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({n}) : index
  %x = check.file.read.npy path("{tag}_x.npy") : tensor<{rows}x{K}xf16>
  %s = check.file.read.npy path("{tag}_s.npy") : tensor<{rows}x2xf32>
{wdecl}  %e = check.file.read.npy path("{tag}_y.npy") : tensor<{rows}x{n}xf32>
  %y = check.generate.fill value(0.0) : tensor<{rows}x{n}xf32>
  kernel.launch @{kern}[%rows, %n](%rows, %k, %n, %x, %s, {wargs}, %y) : [index, index](index, index, index, tensor<{rows}x{K}xf16>, tensor<{rows}x2xf32>, {wtys}, tensor<{rows}x{n}xf32>)
  check.expect.close actual(%y) expected(%e) atol({tol:.3e}) rtol(1.0e-2) nan(different) : tensor<{rows}x{n}xf32>
  check.return
}}
""")
drafter_cases()


def dflash_attention_case():
    """DFlash v1 draft attention, split-K (draft/gen_dflash.py: g_dfl_ctx_split + g_dfl_attention): three spans of
    three rows (context 701, 5 and 0 keys; 11, 16 and 4 block tokens, so 8-token chunks end mid-span) and a dead
    token, on permuted pages. Every block token attends its row's context [0, P) and the whole block. Context
    positions at or past P hold large values: a key read past the span's own context dominates the output."""
    dg = (cfg.get("drafter") or {}).get("gguf")
    if (cfg.get("drafter") or {}).get("kind") != "dflash" or not dg or not os.path.exists(dg):
        return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft"))
    import gen_dflash
    F = gen_dflash.config(dg)
    if not gen_dflash.split_enabled(F):
        return
    QH, KVH, D, S = F["heads"], F["kv_heads"], F["head_dim"], gen_dflash.DFL_SPLITS
    spans = [(701, 11), (5, 16), (0, 4)]   # (context P, block length)
    t = sum(n for _, n in spans) + 1
    nb = 48; pool = 16 * nb * len(spans); layer = min(1, F["layers"] - 1)
    perm = rng.permutation(nb * len(spans)).astype(np.int32)
    cache = rng.standard_normal((2 * layer + 2, pool, KVH, D)).astype(np.float16)
    q = rng.standard_normal((t, QH, D)).astype(np.float32)
    bk = rng.standard_normal((t, KVH, D)).astype(np.float32); bv = rng.standard_normal((t, KVH, D)).astype(np.float32)
    meta = np.zeros(4 + 9 * len(spans), np.int32); info = np.zeros((t, 4), np.int32)
    orig = np.zeros(10 * len(spans) + nb * len(spans), np.int32)
    o = np.zeros((t, QH, D))
    cur = 0
    for r, (P, n) in enumerate(spans):
        meta[4 + 9 * r:4 + 9 * r + 5] = [n, P, 0, 0, cur]
        info[cur:cur + n] = [[r, P + j, 1, r] for j in range(n)]
        orig[10 * r + 8] = 4 * (10 * len(spans) + nb * r)   # the row's page map (byte offset into origins)
        orig[10 * len(spans) + nb * r:10 * len(spans) + nb * (r + 1)] = perm[nb * r:nb * (r + 1)]
        slots = (perm[nb * r:nb * (r + 1), None] * 16 + np.arange(16)).reshape(-1)   # the row's positions -> pool slots
        cache[2 * layer + 1, slots[P:]] *= 64
        kc = cache[2 * layer, slots[:P]].astype(np.float64); vc = cache[2 * layer + 1, slots[:P]].astype(np.float64)
        for j in range(cur, cur + n):
            for h in range(QH):
                kv = h // (QH // KVH)
                keys = np.concatenate([kc[:, kv], bk[cur:cur + n, kv]]); vals = np.concatenate([vc[:, kv], bv[cur:cur + n, kv]])
                sc = keys @ q[j, h] * D ** -0.5; pr = np.exp(sc - sc.max())
                o[j, h] = pr @ vals / pr.sum()
        cur += n
    info[cur] = [0, 0, 0, 0]   # a dead token: no output
    tag = "dfl_attention"
    for nm, a in [("meta", meta), ("info", info), ("orig", orig), ("q", q.reshape(t, -1)), ("bk", bk.reshape(t, -1)),
                  ("bv", bv.reshape(t, -1)), ("cache", cache.reshape(-1)), ("o", o.reshape(t, -1).astype(np.float32))]:
        save(f"{tag}_{nm}", a)
    QD, KD = QH * D, KVH * D
    T = lambda n: f"tensor<{t}x{n}xf32>"
    tys = {"meta": ty(meta), "info": ty(info), "orig": ty(orig), "q": T(QD), "bk": T(KD), "bv": T(KD), "cache": f"tensor<{cache.size}xf16>"}
    PT, MT = f"tensor<{t * QH * S * D}xf32>", f"tensor<{t * QH * S * 2}xf32>"
    cases.append(f"""check.case public @{tag} {{
  %t = check.literal value({t}) : index
  %spans = check.literal value({len(spans)}) : index
  %layer = check.literal value({layer}) : index
  %ctx = check.literal value({16 * nb}) : index
  %pool = check.literal value({pool}) : index
""" + "".join(f'  %{nm} = check.file.read.npy path("{tag}_{nm}.npy") : {tt}\n' for nm, tt in tys.items()) + f"""  %eo = check.file.read.npy path("{tag}_o.npy") : {T(QD)}
  %pacc = check.generate.fill value(0.0) : {PT}
  %pml = check.generate.fill value(0.0) : {MT}
  %o = check.generate.fill value(0.0) : {T(QD)}
  kernel.launch @g_dfl_ctx_split[%t](%t, %spans, %layer, %ctx, %pool, %meta, %info, %q, %orig, %cache, %pacc, %pml) : [index](index, index, index, index, index, {tys["meta"]}, {tys["info"]}, {T(QD)}, {tys["orig"]}, {tys["cache"]}, {PT}, {MT})
  kernel.launch @g_dfl_attention[%t](%t, %spans, %layer, %ctx, %pool, %meta, %info, %q, %bk, %bv, %orig, %cache, %o, %pacc, %pml) : [index](index, index, index, index, index, {tys["meta"]}, {tys["info"]}, {T(QD)}, {T(KD)}, {T(KD)}, {tys["orig"]}, {tys["cache"]}, {T(QD)}, {PT}, {MT})
  check.expect.close actual(%o) expected(%eo) atol(2.0e-3) rtol(3.0e-3) nan(different) : {T(QD)}
  check.return
}}
""")


dflash_attention_case()
# Prefill flash attention (gen_flash2.py / gen_flash2s.py): 64 uniform tokens at the end of a 1024-key context,
# paged arena filled directly (identity page map), layer 0, causal; q scaled so attention is sharp (a kernel that
# drops part of the score sum fails). Reference in float64 from f16 Q/K/V, times sigmoid(gate) when gated.
WIN = cfg.get("sliding_window") or 0


def flash_cases(kern, D, KVH, width, scale, WIN, tag=""):
    """Cases for flash kernel KERN (head_dim D, KVH KV heads; WIDTH K or V values per arena position, more than
    KVH * D when a wider layer kind sets the stride; TAG names its files and cases)."""
    def flash_ref(q, kk, vv, pos, win=0):   # float64 reference: query p sees keys j <= p (and j > p - win)
        o = np.zeros(q.shape); j_ = np.arange(len(kk))[None, :]; p_ = np.asarray(pos)[:, None]
        hidden = (j_ > p_) | (j_ <= p_ - win) if win else j_ > p_
        for hh in range(QH):
            kv = hh // (QH // KVH); sc = q[:, hh] @ kk[:, kv].astype(np.float64).T * scale
            sc[hidden] = -1e30
            pr = np.exp(sc - sc.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True); o[:, hh] = pr @ vv[:, kv].astype(np.float64)
        return o
    gated = (lambda o, g: o / (1 + np.exp(-g))) if cfg.get("attn_gate") else (lambda o, g: o)

    def flash_case(name, sfx, t, pool, norig):
        sfx += tag
        AT = f"tensor<{2 * pool * width}xf16>"
        return f"""check.case public @{name} {{
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({pool}) : index
  %q = check.file.read.npy path("f_q{sfx}.npy") : tensor<{t}x{QD}xf32>
  %g = check.file.read.npy path("f_g{sfx}.npy") : tensor<{t}x{QD}xf32>
  %info = check.file.read.npy path("f_info{sfx}.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("f_orig{sfx}.npy") : tensor<{norig}xi32>
  %arena = check.file.read.npy path("f_arena{sfx}.npy") : {AT}
  %e = check.file.read.npy path("f_o{sfx}.npy") : tensor<{t}x{QD}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{QD}xf32>
{list_launch("%t", t, "%info")}  kernel.launch @{kern}[%t](%t, %layer, %sliding, %ctx, %pool, %q, %g, %info, %tiles, %orig, %arena, %o) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x{QD}xf32>, tensor<{t}x4xi32>, tensor<{t}xi32>, tensor<{norig}xi32>, {AT}, tensor<{t}x{QD}xf32>)
  check.expect.close actual(%o) expected(%e) atol(4.0e-3) rtol(4.0e-3) nan(different) : tensor<{t}x{QD}xf32>
  check.return
}}
"""
    t, ctx = 64, 1024
    P0 = ctx - t; blocks = ctx // 16; pool = ctx
    kk = rng.standard_normal((ctx, KVH, D)).astype(np.float16); vv = rng.standard_normal((ctx, KVH, D)).astype(np.float16)
    arena = np.zeros((2, pool, width), np.float16); arena[0, :, :KVH * D], arena[1, :, :KVH * D] = kk.reshape(ctx, -1), vv.reshape(ctx, -1)
    qq = (rng.standard_normal((t, QH, D)) * 2.5).astype(np.float32); gg = rng.standard_normal((t, QH, D)).astype(np.float32)
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)
    info = np.zeros((t, 4), np.int32); info[:, 1] = P0 + np.arange(t); info[:, 2] = 1
    qd = qq.astype(np.float16).astype(np.float64)
    o = gated(flash_ref(qd, kk, vv, info[:, 1]), gg)
    QD = QH * D
    ow = None
    if WIN:   # sliding layers: key j visible from position p when p - WIN < j <= p (llama.cpp LLAMA_SWA_TYPE_STANDARD)
        ow = gated(flash_ref(qd, kk, vv, info[:, 1], WIN), gg)
        save("f_ow" + tag, ow.reshape(t, QD).astype(np.float32))
    for n, a in [("f_arena", arena.reshape(-1)), ("f_q", qq.reshape(t, QD)), ("f_g", gg.reshape(t, QD)), ("f_orig", orig),
                 ("f_info", info), ("f_o", o.reshape(t, QD).astype(np.float32))]:
        save(n + tag, a)
    # Two rows in one prefill: row 1 at tokens 0-39 (positions 984-1023), row 0 (the keys above) at tokens 40-111
    # (positions 952-1023). flash2 must compute the tiles at positions 992 and 1008 of row 1 (tokens 8-39) and
    # 960-1008 of row 0 (tokens 48-111), whichever tiles share its workgroups (gen_flash2.py TS), and leave the
    # heads (tokens 0-7, 40-47: the listed fallback's) untouched.
    tm = 112; k2 = rng.standard_normal((2, ctx, KVH, D)).astype(np.float16)
    arena_m = np.zeros((2, 2 * pool, width), np.float16); arena_m[:, :pool] = arena; arena_m[:, pool:, :KVH * D] = k2.reshape(2, ctx, -1)
    orig_m = np.zeros(20 + 2 * blocks, np.int32); orig_m[8] = 80; orig_m[18] = 80 + 4 * blocks
    orig_m[20:] = np.arange(2 * blocks)
    row = (np.arange(tm) < 40).astype(np.int32)
    pos = np.where(row == 1, ctx - 40, ctx - tm) + np.arange(tm)
    info_m = np.zeros((tm, 4), np.int32); info_m[:, 0] = row; info_m[:, 1] = pos; info_m[:, 2] = 1
    qm = (rng.standard_normal((tm, QH, D)) * 2.5).astype(np.float32); gm = rng.standard_normal((tm, QH, D)).astype(np.float32)
    qmd = qm.astype(np.float16).astype(np.float64); om = np.zeros((tm, QH, D))
    for r, kr, vr in ((0, kk, vv), (1, k2[0], k2[1])):
        sel = row == r; om[sel] = gated(flash_ref(qmd[sel], kr, vr, pos[sel]), gm[sel])
    om[~covered_tokens(info_m)] = 0
    for n, a in [("f_arena_m", arena_m.reshape(-1)), ("f_q_m", qm.reshape(tm, QD)), ("f_g_m", gm.reshape(tm, QD)),
                 ("f_orig_m", orig_m), ("f_info_m", info_m), ("f_o_m", om.reshape(tm, QD).astype(np.float32))]:
        save(n + tag, a)
    cases.append(flash_case(f"flash_prefill_rows{tag}", "_m", tm, 2 * pool, len(orig_m)))
    cases.append(flash_case(f"flash_prefill{tag}", "", t, pool, len(orig)))
    if ow is not None:
        cases.append(cases[-1].replace(f"@flash_prefill{tag} {{", f"@flash_prefill_window{tag} {{").replace("%sliding = check.literal value(0)", "%sliding = check.literal value(1)")
                     .replace(f'path("f_o{tag}.npy")', f'path("f_ow{tag}.npy")'))


DEEP_CASES = []
if os.environ.get("FLASH_TESTS", "1") == "1" and not cfg.get("attn_sinks") and (not WIN or not cfg.get("head_dim_swa")):
    flash_cases("g_attention_flash2", D, KVH, KVH * D, D ** -0.5, WIN)
    DEEP = flash2_deep_from(cfg)
    # Epochs of several rows at chosen positions, over 3 rows' aliased 1024-key arenas (page b of a row maps to its
    # arena page b % 64, so key j of a row is that row's key j % 1024 and a position p sees key k (p - k) // 1024 + 1
    # times): the reference of a flash2 / listed fallback result at deep positions without a deep arena.
    SENT = 7.0
    ctx_a = 2 * 1024 + (max(DEEP, 1024) // 1024) * 1024; blk_a = ctx_a // 16; pool_a = 3 * 1024
    arena_a = rng.standard_normal((2, pool_a, KVH, D)).astype(np.float16)
    orig_a = np.zeros(30 + 3 * blk_a, np.int32)
    for r in range(3):
        orig_a[10 * r + 8] = 4 * (30 + r * blk_a)
        orig_a[30 + r * blk_a:30 + (r + 1) * blk_a] = r * 64 + np.arange(blk_a) % 64
    save("f_arena_a", arena_a.reshape(-1)); save("f_orig_a", orig_a)

    def epoch_info(spans):
        """info rows of an epoch: SPANS = [(row, first position, tokens)]; a row of None is dead padding."""
        rows = []
        for r, p, n in spans:
            rows += [(0, 0, 0)] * n if r is None else [(r, p + i, 1) for i in range(n)]
        pad = -len(rows) % 16
        return np.array(rows + [(0, 0, 0)] * pad, np.int32).reshape(-1, 3)

    def aliased_ref(q, info, win=0):
        """float64 attention (before the gate) of every live token of INFO (q: [t, QH, D]) over the aliased arenas."""
        o = np.zeros(q.shape)
        for i in np.flatnonzero(info[:, 2]):
            r, p = info[i, 0], info[i, 1]
            kk, vv = arena_a[0, r * 1024:(r + 1) * 1024], arena_a[1, r * 1024:(r + 1) * 1024]
            k_ = np.arange(1024)
            mult = (p - k_) // 1024 + 1                       # copies of key k at positions <= p
            if win: mult = np.maximum(0, (p - k_) // 1024 + 1) - np.maximum(0, (p - win - k_) // 1024 + 1)
            for hh in range(QH):
                kv = hh // (QH // KVH); sc = kk[:, kv].astype(np.float64) @ q[i, hh] * scale_a
                w = np.exp(sc - sc.max()) * mult
                o[i, hh] = (w / w.sum()) @ vv[:, kv].astype(np.float64)
        return o
    scale_a = D ** -0.5
    AA = f"tensor<{2 * pool_a * KVH * D}xf16>"; OA = f"tensor<{len(orig_a)}xi32>"

    def epoch_case(name, info3, q, g, expect, sliding=0, kerns=("g_attention_flash2",)):
        """A prefill epoch: the work lists, the flash kernels KERNS, and the output compared with EXPECT (flash-only
        rows: the sentinel / zero where the fallback's tokens are left alone)."""
        t = len(info3); QD = QH * D
        info = np.zeros((t, 4), np.int32); info[:, :3] = info3
        save(f"fa_info_{name}", info); save(f"fa_q_{name}", q.reshape(t, QD)); save(f"fa_g_{name}", g.reshape(t, QD))
        save(f"fa_e_{name}", expect.reshape(t, QD).astype(np.float32))
        OT = f"tensor<{t}x{QD}xf32>"
        launch = "".join(f"  kernel.launch @{k}[%t](%t, %layer, %sliding, %ctx, %pool, %q, %g, %info, %tiles, %orig, %arena, %o) : [index](index, "
                         f"index, i32, index, index, {OT}, {OT}, tensor<{t}x4xi32>, tensor<{t}xi32>, {OA}, {AA}, {OT})\n" for k in kerns)
        return f"""check.case public @{name} {{
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value({sliding}) : i32
  %ctx = check.literal value({ctx_a}) : index
  %pool = check.literal value({pool_a}) : index
  %q = check.file.read.npy path("fa_q_{name}.npy") : {OT}
  %g = check.file.read.npy path("fa_g_{name}.npy") : {OT}
  %info = check.file.read.npy path("fa_info_{name}.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("f_orig_a.npy") : {OA}
  %arena = check.file.read.npy path("f_arena_a.npy") : {AA}
  %e = check.file.read.npy path("fa_e_{name}.npy") : {OT}
  %o = check.generate.fill value({SENT}) : {OT}
{list_launch("%t", t, "%info")}{launch}  check.expect.close actual(%o) expected(%e) atol(4.0e-3) rtol(4.0e-3) nan(different) : {OT}
  check.return
}}
"""
    if DEEP:   # depth dispatch (gen_flash2.py): which kernel computes which tile, and that the pair is exact
        # Three rows in one prefill, positions chosen so the 16-token tiles (position // 16 = tile number T) are:
        # row 1: an 8-token head, tiles D0 .. D0 + 3 (two deep pairs, D0 = DEEP / 16 + 64), a 5-token tail;
        # row 2: a 3-token head, T0 - 2 .. T0 + 2 (T0 = DEEP / 16: the pair below it is one-tile, T0 / T0 + 1 deep,
        #   T0 + 2 alone with a 5-token tail);
        # row 0: tiles T0 + 65 (alone: its partner T0 + 64 is not in the span), T0 + 66, T0 + 67 (deep);
        # plus dead padding. The deep and one-tile kernels each write only their own pairs.
        T0 = DEEP // 16; assert T0 % 2 == 0
        spans = [(1, 16 * (T0 + 64) - 8, 8 + 64 + 5), (2, 16 * (T0 - 2) - 3, 3 + 80 + 5), (0, 16 * (T0 + 65), 48), (None, 0, 11)]
        info_d = epoch_info(spans)
        qd_ = (rng.standard_normal((len(info_d), QH, D)) * 2.5).astype(np.float32); gd_ = rng.standard_normal(qd_.shape).astype(np.float32)
        o_d = aliased_ref(qd_.astype(np.float16).astype(np.float64), info_d)
        if cfg.get("attn_gate"):
            o_d = o_d / (1 + np.exp(-gd_))
        starts = set(tile_starts(info_d)); deep_tok = np.zeros(len(info_d), bool); one_tok = np.zeros(len(info_d), bool)
        for s in starts:
            p = info_d[s, 1]; odd = (p // 16) % 2; partner = s - 16 if odd else s + 16
            pair = partner in starts and info_d[partner, 0] == info_d[s, 0] and info_d[partner, 1] == p + (-16 if odd else 16)
            (deep_tok if pair and p - 16 * odd >= DEEP else one_tok)[s:s + 16] = True
        assert deep_tok.sum() == 16 * 4 + 16 * 2 + 16 * 2 and one_tok.sum() == 16 * 3 + 16 * 1, (deep_tok.sum(), one_tok.sum())
        for name, kerns, mask in (("both", ("g_attention_flash2", "g_attention_flash2_deep"), deep_tok | one_tok),
                                  ("one", ("g_attention_flash2",), one_tok), ("deep", ("g_attention_flash2_deep",), deep_tok)):
            DEEP_CASES.append(epoch_case(f"flash_prefill_deep_{name}", info_d, qd_, gd_, np.where(mask[:, None, None], o_d, SENT), 0, kerns))

    # Batch invariance: one span (X, row 0) attended alone and packed at other offsets of an epoch between other rows'
    # spans must give bit-identical outputs, through the whole prefill path (work lists, flash2 and the listed split-K
    # fallback for X's head and tail). X's positions start at 5 mod 16, and at the deep boundary when there is one.
    COPY_ROWS = """
// Test helper: dst[r] = src[skip + r] for r < rows (F32 rows of `width`).
kernel.def export("test_copy_rows") @test_copy_rows(%rows: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%rows_arg: index, %width_arg: index, %skip_arg: index, %src_rows_arg: index, %src: buffer, %dst: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 8192)] : index
  %width = index.assume %width_arg [range(%width_arg, 1, 1048576)] : index
  %src_rows = index.assume %src_rows_arg [range(%src_rows_arg, 1, 8192)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c256 = index.constant 256 : index
  %src_view = buffer.view %src[%zero] : buffer -> view<[%src_rows]x[%width]xf32>
  %dst_view = buffer.view %dst[%zero] : buffer -> view<[%rows]x[%width]xf32>
  %skip = index.assume %skip_arg [range(%skip_arg, 0, 8191)] : index
  %src_row_value = index.add %skip, %row : index
  %src_row = index.assume %src_row_value [range(%src_row_value, 0, 8191), lt(%src_row_value, %src_rows)] : index
  scf.for %c = [%lane to %width step %c256] {
    %col = index.assume %c [lt(%c, %width)] : index
    %v = view.load %src_view[%src_row, %col] : view<[%src_rows]x[%width]xf32> -> f32
    view.store %v, %dst_view[%row, %col] : f32, view<[%rows]x[%width]xf32>
  }
  kernel.return
}
"""

    def invariance_case(sliding):
        n, QD = 100, QH * D
        p0 = (DEEP - 64 if DEEP else 900) + 5
        x_q = (rng.standard_normal((n, QH, D)) * 2.5).astype(np.float32); x_g = rng.standard_normal((n, QH, D)).astype(np.float32)
        packings = [(0, 0), (7, 14), (19, 1), (80, 0), (37, 21)]   # tokens of other rows before / after X
        text, ref, t_0 = "", None, None
        for k, (a, b) in enumerate(packings):
            info3 = epoch_info([(1, 333, a), (0, p0, n), (2, 5 + 16 * a, b)]); t = len(info3)
            info = np.zeros((t, 4), np.int32); info[:, :3] = info3
            q = (rng.standard_normal((t, QH, D)) * 2.5).astype(np.float32); g = rng.standard_normal((t, QH, D)).astype(np.float32)
            q[a:a + n], g[a:a + n] = x_q, x_g
            if k == 0:
                ref = aliased_ref(q.astype(np.float16).astype(np.float64), info, WIN if sliding else 0)[:n]
                ref = ref / (1 + np.exp(-x_g)) if cfg.get("attn_gate") else ref
                save(f"fi_ref{sliding}", ref.reshape(n, QD).astype(np.float32))
            for nm, arr in (("info", info), ("q", q.reshape(t, QD)), ("g", g.reshape(t, QD))):
                save(f"fi_{nm}{sliding}_{k}", arr)
            OT = f"tensor<{t}x{QD}xf32>"; IT = f"tensor<{t}x4xi32>"; XT = f"tensor<{n}x{QD}xf32>"; PT = f"tensor<{t * QH * 16 * D}xf32>"; MT = f"tensor<{t * QH * 16 * 2}xf32>"
            text += f"""  %t{k} = check.literal value({t}) : index
  %skip{k} = check.literal value({a}) : index
  %q{k} = check.file.read.npy path("fi_q{sliding}_{k}.npy") : {OT}
  %g{k} = check.file.read.npy path("fi_g{sliding}_{k}.npy") : {OT}
  %info{k} = check.file.read.npy path("fi_info{sliding}_{k}.npy") : {IT}
  %o{k} = check.generate.fill value(0.0) : {OT}
  %pacc{k} = check.generate.fill value(0.0) : {PT}
  %pml{k} = check.generate.fill value(0.0) : {MT}
  %x{k} = check.generate.fill value(0.0) : {XT}
{list_launch(f"%t{k}", t, f"%info{k}", k)}  kernel.launch @g_attention_flash2[%t{k}](%t{k}, %layer, %sliding, %ctx, %pool, %q{k}, %g{k}, %info{k}, %tiles{k}, %orig, %arena, %o{k}) : [index](index, index, i32, index, index, {OT}, {OT}, {IT}, tensor<{t}xi32>, {OA}, {AA}, {OT})
  kernel.launch @g16_attention_split_listed[%fcount{k}](%t{k}, %flist{k}, %layer, %sliding, %ctx, %pool, %q{k}, %info{k}, %orig, %arena, %pacc{k}, %pml{k}) : [tensor<64xi32>](index, tensor<{t}xi32>, index, i32, index, index, {OT}, {IT}, {OA}, {AA}, {PT}, {MT})
  kernel.launch @g16_attention_combine_listed[%fcount{k}](%t{k}, %flist{k}, %sliding, %info{k}, %pacc{k}, %pml{k}, %g{k}, %o{k}) : [tensor<64xi32>](index, tensor<{t}xi32>, i32, {IT}, {PT}, {MT}, {OT}, {OT})
  kernel.launch @test_copy_rows[%nx](%nx, %wd, %skip{k}, %t{k}, %o{k}, %x{k}) : [index](index, index, index, index, {OT}, {XT})
"""
            if k:
                text += f"  check.expect.close actual(%x{k}) expected(%x0) atol(0.0) rtol(0.0) nan(different) : {XT}\n"
        XT = f"tensor<{n}x{QD}xf32>"
        check = f"  %ref = check.file.read.npy path(\"fi_ref{sliding}.npy\") : {XT}\n  check.expect.close actual(%x0) expected(%ref) atol(4.0e-3) rtol(4.0e-3) nan(different) : {XT}\n" if not sliding else ""
        return f"""check.case public @flash_batch_invariant{'_window' if sliding else ''} {{
  %layer = check.literal value(0) : index
  %sliding = check.literal value({sliding}) : i32
  %ctx = check.literal value({ctx_a}) : index
  %pool = check.literal value({pool_a}) : index
  %nx = check.literal value({n}) : index
  %wd = check.literal value({QD}) : index
  %orig = check.file.read.npy path("f_orig_a.npy") : {OA}
  %arena = check.file.read.npy path("f_arena_a.npy") : {AA}
{text}{check}  check.return
}}
"""
    cases.append(COPY_ROWS)
    cases += [invariance_case(s) for s in ((0, 1) if WIN else (0,))]

# 6. MoE (gen_moe.py) on real tensors, one layer per expert-down format: the decode chain (router -> top-k ->
# gate/up -> down -> combine, plus the prefill router with padding tokens) and the prefill chain (prefill router ->
# bucket by expert -> grouped WMMA gate/up GEMM -> tile row scale -> grouped down GEMM -> combine of the sorted rows),
# both with the shared experts' output already in y (scaled by the Qwen shared-expert gate when the model has one).
if cfg.get("moe"):
    M = cfg["moe"]; E_, KU, FE = M["experts"], M["used"], M["ffn"]; R = int(os.environ.get("MOE_R", "8"))
    SHG, ACC = bool(M.get("shared_gate")), bool(M.get("shared_ffn"))
    # Gemma 4: gate and up fused in one tensor (each expert's gate rows, then its up rows), GEGLU, and a per-expert
    # output scale folded into the combine weights (g_moe_wscale, gemma4 example)
    FUSED, DSC = bool(M.get("gate_up_fused")), bool(M.get("down_scale"))
    GN, UN = ("ffn_gate_up_exps.weight",) * 2 if FUSED else ("ffn_gate_exps.weight", "ffn_up_exps.weight")
    def deq3(name):
        tt = T[name]; return dequantize(np.asarray(tt.data), tt.tensor_type).reshape(E_, int(tt.shape[1]), int(tt.shape[0]))
    def f32w(name):   # f32 or bf16 tensor -> (f32 values, the raw array the kernel reads)
        tt = T[name]; bf = tt.tensor_type.name == "BF16"
        v = (np.asarray(tt.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32) if bf else np.asarray(tt.data, np.float32)
        return v.reshape(-1), (np.asarray(tt.data).view(np.int8).reshape(-1) if bf else v.reshape(-1).astype(np.float32))
    def route_ref(hx, Wr, bias):
        logit = hx.astype(np.float64) @ Wr.T.astype(np.float64)
        if M["gating"] == 3:   # SOFTMAX_WEIGHT: the router bias is part of the logits; weights = softmax over the k picked
            logit = logit + bias; prob = logit
        elif M["gating"] == 2:
            prob = 1 / (1 + np.exp(-logit))
        else:
            ex = np.exp(logit - logit.max(-1, keepdims=True)); prob = ex / ex.sum(-1, keepdims=True)
        sel = prob + (bias if M["gating"] != 3 else 0)
        ids = np.argsort(-sel, -1, kind="stable")[:, :KU].astype(np.int32)
        wts = np.take_along_axis(prob, ids, -1)
        if M["gating"] == 3:
            wts = np.exp(wts - wts.max(-1, keepdims=True)); wts = wts / wts.sum(-1, keepdims=True)
        if M["norm_weights"]:
            wts = wts / np.maximum(wts.sum(-1, keepdims=True), 6.103515625e-05)
        ss = np.sort(sel, -1)[:, ::-1]
        rel = (ss[:, KU - 1] - ss[:, KU]) / np.maximum(np.abs(ss[:, KU - 1]), 1e-30)   # k-th vs (k+1)-th, relative
        return ids, wts, rel
    def clear_tokens(t):
        """t random rows whose k-th and (k+1)-th scores differ by > 0.2 %: f32 kernels pick the same experts as the
        f64 reference (softmax over 256 experts leaves near-ties in many random rows)."""
        hx = rng.standard_normal((40 * t, H)).astype(np.float32)
        _, _, rel = route_ref(hx, Wr, bias)
        keep = np.flatnonzero(rel > 2e-3)[:t]
        assert len(keep) == t, f"only {len(keep)} of {40 * t} random rows have a clear top-{KU}"
        return hx[keep]
    EB = bool(M.get("expert_bias")); OAI = M.get("act") == "swiglu_oai"
    def f16_rows(x):   # g_row_scale_f16: each row scaled by a power of two into F16 range, rounded to F16, scaled back
        e = np.clip(np.frexp(np.abs(x).max(-1, keepdims=True))[1] - 1 - 13, 0, 100).astype(np.float64)
        return (x * 2.0 ** -e).astype(np.float16).astype(np.float64) * 2.0 ** e
    def experts_ref(hx, ids, wts, Wg, Wu, Wd, bg=None, bu=None, bd=None, f16=False):
        """f16: the prefill GEMMs' activation precision (hidden rows and expert activations rounded to F16 by the row
        scale kernels), so the reference differs from the kernels only by summation order, not by the rounding the
        design chooses (a cancelling sum of large expert outputs otherwise shows that rounding as an error)."""
        y = np.zeros((len(hx), H))
        if f16: hx = f16_rows(hx.astype(np.float64))
        for i in range(len(hx)):
            for k in range(KU):
                e = ids[i, k]; g = Wg[e] @ hx[i]; u = Wu[e] @ hx[i]
                if EB: g, u = g + bg[e], u + bu[e]
                if OAI:   # ggml_swiglu_oai: x = min(g, limit), y = clamp(u, -limit, limit), x sigmoid(alpha x) (y + 1)
                    lim, al = M.get("swiglu_limit", 7.0), M.get("swiglu_alpha", 1.702)
                    xg = np.minimum(g, lim); a = xg / (1 + np.exp(-al * xg)) * (np.clip(u, -lim, lim) + 1)
                else:
                    a = ffn_act(g) * u
                if f16: a = f16_rows(a)
                y[i] += wts[i, k] * (Wd[e] @ a + (bd[e] if EB else 0))
        return y
    done = set()
    for li in range(M["leading_dense"], cfg["layers"]):
        b = f"blk.{li}."; fd = T[b + "ffn_down_exps.weight"].tensor_type.name.lower(); fg = T[b + GN].tensor_type.name.lower()
        fu = T[b + UN].tensor_type.name.lower(); gun = (fg if fg == fu else f"{fg}_{fu}") + ("_fused" if FUSED else "")   # gate+up formats (gen_moe)
        if (fd, gun) in done: continue
        done.add((fd, gun))
        Wr, wr_raw = f32w(b + "ffn_gate_inp.weight"); Wr = Wr.reshape(E_, H)
        bias = (np.asarray(T[b + "exp_probs_b.bias"].data, np.float32) if M["probs_bias"] else
                np.asarray(T[b + "ffn_gate_inp.bias"].data, np.float32) if M.get("router_bias") else np.zeros(E_, np.float32))
        ebias = {p: np.asarray(T[f"{b}ffn_{p}_exps.bias"].data, np.float32).reshape(E_, -1) for p in ("gate", "up", "down")} if EB else {}
        Wg, Wu, Wd = deq3(b + GN), deq3(b + UN), deq3(b + "ffn_down_exps.weight")
        if FUSED:
            Wg, Wu = Wg[:, :FE], Wu[:, FE:]
        dsc = np.asarray(T[b + "ffn_down_exps.scale"].data, np.float32).reshape(-1) if DSC else np.ones(E_, np.float32)
        if SHG:
            Wsg, wsg_raw = f32w(b + "ffn_gate_inp_shexp.weight"); sty = T[b + "ffn_gate_inp_shexp.weight"].tensor_type.name.lower()
        tag = f"m{li}"
        for n, a in ([(tag + "_wr", wr_raw), (tag + "_b", bias), (tag + "_wg", raw(b + GN)),
                      (tag + "_wu", raw(b + UN)), (tag + "_wd", raw(b + "ffn_down_exps.weight"))] + ([(tag + "_wsg", wsg_raw)] if SHG else [])
                     + ([(tag + "_dsc", dsc)] if DSC else []) + [(f"{tag}_b{p[0]}", v.reshape(-1)) for p, v in ebias.items()]):
            save(n, a)
        def chain(t, sfx):
            """Inputs + reference of a t-token chain: h, y0 (shared output in y), ids/wts, the expected y."""
            hx = clear_tokens(t)
            ids, wts, rel = route_ref(hx, Wr, bias)
            y0 = (rng.standard_normal((t, H)) * 0.1).astype(np.float32) if ACC else np.zeros((t, H), np.float32)
            sgv = 1 / (1 + np.exp(-(hx.astype(np.float64) @ Wsg.astype(np.float64)))) if SHG else np.ones(t)
            wts_s = wts * dsc[ids]   # the combine weights after g_moe_wscale (DSC)
            yref = sgv[:, None] * y0 + experts_ref(hx, ids, wts_s, Wg, Wu, Wd, ebias.get("gate"), ebias.get("up"), ebias.get("down"), f16=sfx == "_p")
            for n, a in [("_h", hx), ("_y0", y0), ("_ids", ids), ("_wts", wts.astype(np.float32)), ("_y", yref.astype(np.float32)),
                         ("_sg", sgv.astype(np.float32))] + ([("_wts_s", wts_s.astype(np.float32))] if DSC else []):
                save(tag + sfx + n, a)
            print(f"moe test layer {li}{sfx}: t={t} ids[0] {ids[0].tolist()} min relative score gap k/k+1 {rel.min():.3g}")
            return ids, wts
        wr_t, wsg_t = ty(wr_raw), (ty(wsg_raw) if SHG else "")
        gt, dt_ = ty(raw(b + GN)), ty(raw(b + "ffn_down_exps.weight"))
        ut = ty(raw(b + UN))
        EW = "%ewts_s" if DSC else "%ewts"   # the routed weights after wsc
        # DSC: g_moe_wscale scales the routed weights by the experts' down scales before the experts run
        wsc = lambda t: f"""  kernel.launch @g_moe_wscale[%t](%t, %ids, %wts, %dsc) : [index](index, tensor<{t}x{KU}xi32>, tensor<{t}x{KU}xf32>, tensor<{E_}xf32>)
""" if DSC else ""
        def common(t, sfx):
            return f"""  %t = check.literal value({t}) : index
  %ta = check.literal value({t * KU}) : index
  %nFE = check.literal value({FE}) : index
  %nH = check.literal value({H}) : index
  %h = check.file.read.npy path("{tag}{sfx}_h.npy") : tensor<{t}x{H}xf32>
  %wr = check.file.read.npy path("{tag}_wr.npy") : {wr_t}
  %b = check.file.read.npy path("{tag}_b.npy") : tensor<{E_}xf32>
  %eids = check.file.read.npy path("{tag}{sfx}_ids.npy") : tensor<{t}x{KU}xi32>
  %ewts = check.file.read.npy path("{tag}{sfx}_wts.npy") : tensor<{t}x{KU}xf32>
""" + (f"""  %ewts_s = check.file.read.npy path("{tag}{sfx}_wts_s.npy") : tensor<{t}x{KU}xf32>
  %dsc = check.file.read.npy path("{tag}_dsc.npy") : tensor<{E_}xf32>
""" if DSC else "") + f"""  %wg = check.file.read.npy path("{tag}_wg.npy") : {gt}
  %wu = check.file.read.npy path("{tag}_wu.npy") : {ut}
  %wd = check.file.read.npy path("{tag}_wd.npy") : {dt_}
  %ey = check.file.read.npy path("{tag}{sfx}_y.npy") : tensor<{t}x{H}xf32>
  %y = check.file.read.npy path("{tag}{sfx}_y0.npy") : tensor<{t}x{H}xf32>
  %ids = check.generate.fill value(0) : tensor<{t}x{KU}xi32>
  %wts = check.generate.fill value(0.0) : tensor<{t}x{KU}xf32>
""" + "".join(f"""  %b{p} = check.file.read.npy path("{tag}_b{p}.npy") : tensor<{E_ * (H if p == "d" else FE)}xf32>
""" for p in ("g", "u", "d") if EB) + (f"""  %wsg = check.file.read.npy path("{tag}_wsg.npy") : {wsg_t}
  %esg = check.file.read.npy path("{tag}{sfx}_sg.npy") : tensor<{t}xf32>
  %sg = check.generate.fill value(0.0) : tensor<{t}xf32>
  kernel.launch @g_moe_shared_gate_{sty}[%t](%t, %h, %wsg, %sg) : [index](index, tensor<{t}x{H}xf32>, {wsg_t}, tensor<{t}xf32>)
  check.expect.close actual(%sg) expected(%esg) atol(1.0e-5) rtol(1.0e-4) nan(different) : tensor<{t}xf32>
""" if SHG else "")
        SGA, SGT = ("%sg, ", f"tensor<{t}xf32>, ") if SHG else ("", "")
        GUB, GUBT = (", %bg, %bu", f", tensor<{E_ * FE}xf32>, tensor<{E_ * FE}xf32>") if EB else ("", "")
        DNB, DNBT = (", %bd", f", tensor<{E_ * H}xf32>") if EB else ("", "")
        # decode chain: 6 tokens (> the prefill router's tokens per workgroup: a full group plus a partial one)
        t = 6; ids, wts = chain(t, "")
        SGT = f"tensor<{t}xf32>, " if SHG else ""
        info_l = np.zeros((t, 4), np.int32); info_l[:, 2] = 1; info_l[2, 2] = 0; info_l[4:, 2] = 0
        save(tag + "_info", info_l); save(tag + "_ids_l", np.where(np.arange(t)[:, None] >= 4, 0, ids).astype(np.int32))
        save(tag + "_wts_l", np.where(np.arange(t)[:, None] >= 4, 0, wts).astype(np.float32))
        cases.append(f"""check.case public @moe_chain_{li} {{
{common(t, "")}  %act = check.generate.fill value(0.0) : tensor<{t * KU}x{FE}xf32>
  %ys = check.generate.fill value(0.0) : tensor<{t * KU}x{H}xf32>
  kernel.launch @g_moe_route[%t](%t, %h, %wr, %b, %ids, %wts) : [index](index, tensor<{t}x{H}xf32>, {wr_t}, tensor<{E_}xf32>, tensor<{t}x{KU}xi32>, tensor<{t}x{KU}xf32>)
  %ids_tr = check.generate.fill value(0) : tensor<{t}x{KU}xi32>
  %wts_tr = check.generate.fill value(0.0) : tensor<{t}x{KU}xf32>
  kernel.launch @g_moe_route_tr[%t](%t, %h, %wr, %b, %ids_tr, %wts_tr) : [index](index, tensor<{t}x{H}xf32>, {wr_t}, tensor<{E_}xf32>, tensor<{t}x{KU}xi32>, tensor<{t}x{KU}xf32>)
  %info_l = check.file.read.npy path("{tag}_info.npy") : tensor<{t}x4xi32>
  %eids_l = check.file.read.npy path("{tag}_ids_l.npy") : tensor<{t}x{KU}xi32>
  %ewts_l = check.file.read.npy path("{tag}_wts_l.npy") : tensor<{t}x{KU}xf32>
  %ids_l = check.generate.fill value(0) : tensor<{t}x{KU}xi32>
  %wts_l = check.generate.fill value(0.0) : tensor<{t}x{KU}xf32>
  kernel.launch @g_moe_route_tr_live[%t](%t, %h, %wr, %b, %ids_l, %wts_l, %info_l) : [index](index, tensor<{t}x{H}xf32>, {wr_t}, tensor<{E_}xf32>, tensor<{t}x{KU}xi32>, tensor<{t}x{KU}xf32>, tensor<{t}x4xi32>)
  check.expect.equal actual(%ids_l) expected(%eids_l) : tensor<{t}x{KU}xi32>
  check.expect.close actual(%wts_l) expected(%ewts_l) atol(1.0e-5) rtol(1.0e-4) nan(different) : tensor<{t}x{KU}xf32>
{wsc(t)}  kernel.launch @g_moe_gu_{gun}_k{H}_r{R}[%nFE, %ta](%nFE, %ta, %ids, %h, %wg, %wu{GUB}, %act) : [index, index](index, index, tensor<{t}x{KU}xi32>, tensor<{t}x{H}xf32>, {gt}, {ut}{GUBT}, tensor<{t * KU}x{FE}xf32>)
  kernel.launch @g_moe_dn_{fd}_k{FE}_r{R}[%nH, %ta](%nH, %ta, %ids, %act, %wd{DNB}, %ys) : [index, index](index, index, tensor<{t}x{KU}xi32>, tensor<{t * KU}x{FE}xf32>, {dt_}{DNBT}, tensor<{t * KU}x{H}xf32>)
  kernel.launch @g_moe_combine[%t](%t, %ys, %wts, {SGA}%y) : [index](index, tensor<{t * KU}x{H}xf32>, tensor<{t}x{KU}xf32>, {SGT}tensor<{t}x{H}xf32>)
  check.expect.equal actual(%ids) expected(%eids) : tensor<{t}x{KU}xi32>
  check.expect.equal actual(%ids_tr) expected(%eids) : tensor<{t}x{KU}xi32>
  check.expect.close actual(%wts_tr) expected(%ewts) atol(1.0e-5) rtol(1.0e-4) nan(different) : tensor<{t}x{KU}xf32>
  check.expect.close actual(%wts) expected({EW}) atol(1.0e-5) rtol(1.0e-4) nan(different) : tensor<{t}x{KU}xf32>
  check.expect.close actual(%y) expected(%ey) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.return
}}
""")
        # prefill chain: 70 live tokens (two 64-row GEMM tiles' worth of rows per busy expert, partial tiles)
        t = 70; chain(t, "_p")
        SGT = f"tensor<{t}xf32>, " if SHG else ""
        ta = t * KU; mt = ta // 64 + E_; mr = mt * 64
        save(tag + "_p_info", np.tile(np.array([0, 0, 1, 0], np.int32), (t, 1)))
        cases.append(f"""check.case public @moe_prefill_{li} {{
{common(t, "_p")}  %info = check.file.read.npy path("{tag}_p_info.npy") : tensor<{t}x4xi32>
  %max_tiles = check.literal value({mt}) : index
  %max_rows = check.literal value({mr}) : index
  %s_h = check.generate.fill value(0.0) : tensor<{t}x2xf32>
  %h16 = check.generate.fill value(0.0) : tensor<{t}x{H}xf16>
  %tile_info = check.generate.fill value(0) : tensor<{mt}x2xi32>
  %rowmap = check.generate.fill value(0) : tensor<{mr}xi32>
  %pos = check.generate.fill value(0) : tensor<{ta}xi32>
  %act_sorted = check.generate.fill value(0.0) : tensor<{mr}x{FE}xf32>
  %s_rows = check.generate.fill value(0.0) : tensor<{mr}x2xf32>
  %act16 = check.generate.fill value(0.0) : tensor<{mr}x{FE}xf16>
  %ys_sorted = check.generate.fill value(0.0) : tensor<{mr}x{H}xf32>
  kernel.launch @g_row_scale_f16[%t](%t, %nH, %h, %s_h, %h16) : [index](index, index, tensor<{t}x{H}xf32>, tensor<{t}x2xf32>, tensor<{t}x{H}xf16>)
  kernel.launch @g_moe_route_tr_live[%t](%t, %h, %wr, %b, %ids, %wts, %info) : [index](index, tensor<{t}x{H}xf32>, {wr_t}, tensor<{E_}xf32>, tensor<{t}x{KU}xi32>, tensor<{t}x{KU}xf32>, tensor<{t}x4xi32>)
{wsc(t)}  kernel.launch @g_moe_bucket[%t](%t, %max_tiles, %info, %ids, %tile_info, %rowmap, %pos) : [index](index, index, tensor<{t}x4xi32>, tensor<{t}x{KU}xi32>, tensor<{mt}x2xi32>, tensor<{mr}xi32>, tensor<{ta}xi32>)
  kernel.launch @g_moe_ggu_{gun}_k{H}[%max_rows, %nFE](%max_rows, %nH, %nFE, %t, %tile_info, %rowmap, %h16, %s_h, %wg, %wu{GUB}, %act_sorted) : [index, index](index, index, index, index, tensor<{mt}x2xi32>, tensor<{mr}xi32>, tensor<{t}x{H}xf16>, tensor<{t}x2xf32>, {gt}, {ut}{GUBT}, tensor<{mr}x{FE}xf32>)
  kernel.launch @g_row_scale_f16_tiles[%max_rows](%max_rows, %nFE, %act_sorted, %s_rows, %act16, %tile_info) : [index](index, index, tensor<{mr}x{FE}xf32>, tensor<{mr}x2xf32>, tensor<{mr}x{FE}xf16>, tensor<{mt}x2xi32>)
  kernel.launch @g_moe_gdn_{fd}_k{FE}[%max_rows, %nH](%max_rows, %nFE, %nH, %max_rows, %tile_info, %rowmap, %act16, %s_rows, %wd{DNB}, %ys_sorted) : [index, index](index, index, index, index, tensor<{mt}x2xi32>, tensor<{mr}xi32>, tensor<{mr}x{FE}xf16>, tensor<{mr}x2xf32>, {dt_}{DNBT}, tensor<{mr}x{H}xf32>)
  kernel.launch @g_moe_combine_sorted[%t](%t, %max_rows, %ys_sorted, %pos, %wts, {SGA}%y) : [index](index, index, tensor<{mr}x{H}xf32>, tensor<{ta}xi32>, tensor<{t}x{KU}xf32>, {SGT}tensor<{t}x{H}xf32>)
  check.expect.equal actual(%ids) expected(%eids) : tensor<{t}x{KU}xi32>
  check.expect.close actual(%wts) expected({EW}) atol(1.0e-5) rtol(1.0e-4) nan(different) : tensor<{t}x{KU}xf32>
  check.expect.close actual(%y) expected(%ey) atol(1.0e-2) rtol(2.0e-2) nan(different) : tensor<{t}x{H}xf32>
  check.return
}}
""")
    # g_row_scale_f16_tiles: g_row_scale_f16 over bucketed expert rows; rows of an unused tile (expert -1) untouched
    rt_x = (rng.standard_normal((128, FE)) * np.where(np.arange(128) % 3 == 0, 1e3, 1e-2)[:, None]).astype(np.float32)
    rt_e = np.clip(np.frexp(np.abs(rt_x).max(1))[1] - 1 - 13, 0, 100)
    rt_s = np.stack([2.0 ** -rt_e, 2.0 ** rt_e], 1).astype(np.float32); rt_s[64:] = 0
    rt_h = (rt_x * rt_s[:, :1]).astype(np.float16)
    save("rt_x", rt_x); save("rt_s", rt_s); save("rt_h", rt_h); save("rt_ti", np.array([[3, 64], [-1, 0]], np.int32))
    cases.append(f"""check.case public @row_scale_f16_tiles {{
  %rows = check.literal value(128) : index
  %k = check.literal value({FE}) : index
  %x = check.file.read.npy path("rt_x.npy") : tensor<128x{FE}xf32>
  %ti = check.file.read.npy path("rt_ti.npy") : tensor<2x2xi32>
  %es = check.file.read.npy path("rt_s.npy") : tensor<128x2xf32>
  %eh = check.file.read.npy path("rt_h.npy") : tensor<128x{FE}xf16>
  %s = check.generate.fill value(0.0) : tensor<128x2xf32>
  %h = check.generate.fill value(0.0) : tensor<128x{FE}xf16>
  kernel.launch @g_row_scale_f16_tiles[%rows](%rows, %k, %x, %s, %h, %ti) : [index](index, index, tensor<128x{FE}xf32>, tensor<128x2xf32>, tensor<128x{FE}xf16>, tensor<2x2xi32>)
  check.expect.close actual(%s) expected(%es) atol(0.0) rtol(0.0) nan(different) : tensor<128x2xf32>
  check.expect.close actual(%h) expected(%eh) atol(0.0) rtol(1.0e-3) nan(different) : tensor<128x{FE}xf16>
  check.return
}}
""")
    # g_moe_combine_sorted alone: padding tokens (every pos -1) and single dead assignments read no expert row
    t, cr = 37, 512
    c_pos = rng.permutation(cr)[:t * KU].astype(np.int32).reshape(t, KU); c_pos[[0, 9]] = -1; c_pos[12, KU - 1] = -1
    c_w, c_sg = rng.random((t, KU)).astype(np.float32), rng.random(t).astype(np.float32)
    c_ys, c_y0 = rng.standard_normal((cr, H)).astype(np.float32), rng.standard_normal((t, H)).astype(np.float32)
    c_e = (c_w[..., None] * np.where(c_pos[..., None] >= 0, c_ys[np.maximum(c_pos, 0)], 0.0)).sum(1)
    c_e += ((c_sg[:, None] if SHG else 1) * c_y0) if ACC else 0
    for n, a in (("pos", c_pos.reshape(-1)), ("w", c_w), ("sg", c_sg), ("ys", c_ys), ("y0", c_y0), ("e", c_e.astype(np.float32))):
        save("cs_" + n, a)
    YT = f"tensor<{t}x{H}xf32>"
    cases.append(f"""check.case public @moe_combine_sorted {{
  %t = check.literal value({t}) : index
  %rows = check.literal value({cr}) : index
  %pos = check.file.read.npy path("cs_pos.npy") : tensor<{t * KU}xi32>
  %wts = check.file.read.npy path("cs_w.npy") : tensor<{t}x{KU}xf32>
  %sg = check.file.read.npy path("cs_sg.npy") : tensor<{t}xf32>
  %ys = check.file.read.npy path("cs_ys.npy") : tensor<{cr}x{H}xf32>
  %e = check.file.read.npy path("cs_e.npy") : {YT}
  %y = {f'check.file.read.npy path("cs_y0.npy") : {YT}' if ACC else f"check.generate.fill value(3.0) : {YT}"}
  kernel.launch @g_moe_combine_sorted[%t](%t, %rows, %ys, %pos, %wts, {"%sg, " if SHG else ""}%y) : [index](index, index, tensor<{cr}x{H}xf32>, tensor<{t * KU}xi32>, tensor<{t}x{KU}xf32>, {f"tensor<{t}xf32>, " if SHG else ""}{YT})
  check.expect.close actual(%y) expected(%e) atol(1.0e-4) rtol(1.0e-4) nan(different) : {YT}
  check.return
}}
""")
import re
if flash2_deep_from(cfg):   # gen_llm.py launches the deep pairs' kernel after every one-tile prefill flash launch
    cases = [re.sub(r"^( *)kernel\.launch @g_attention_flash2\[(.*)$", r"\g<0>\n\1kernel.launch @g_attention_flash2_deep[\2", c,
                    flags=re.M) for c in cases] + DEEP_CASES   # these launch the two kernels as they test them
# hrx-loom 0.1.0: float comparison is check.expect.approximate with the actual and expected types.
text = re.sub(r"check\.expect\.close (actual\(.*\)) : (.*)$", r"check.expect.approximate \1 : \2, \2", "\n".join(cases), flags=re.M)
open(os.path.join(out, "cases.loom-test"), "w", newline="\n").write(text)
print("wrote", len(cases), "cases")
