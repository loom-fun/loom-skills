"""Kernel unit tests for a gen_llm.py build: numpy references from gguf dequantization.

usage: python make_kernel_tests.py CONFIG.json OUTDIR      (tools venv: gguf, numpy)
       run_kernel_tests.sh PACKAGE OUTDIR                  (loom-test on the GPU)
Covers g_embed, the decode matvecs (1- and 4-token variants) for one tensor of
each (format, K), the QK-norm + RoPE kernel (incl. head_dim 256 / partial
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
  %h = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  kernel.launch @g_embed[%t](%t, %ids, %tab, %w, %x, %h) : [index](index, {ty(ids)}, {ty(tab)}, tensor<{H}xf32>, tensor<{t}x{H}xf32>, tensor<{t}x{H}xf32>)
  check.expect.close actual(%x) expected(%ex) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.expect.close actual(%h) expected(%eh) atol(1.0e-3) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.return
}}
""")
# 1b. stop-token restore in g_embed (gen_llm.stop_restore; needs LLM_TOKENIZER like build.py passes):
# a fed service terminal becomes stop_after_calls before a tool-result opener, else stop_after_answer.
_ch = cfg.get("chat") or {}
if os.environ.get("LLM_TOKENIZER") and _ch.get("stop_after_calls") and _ch.get("stop_after_answer") and cfg.get("extra_eos_ids"):
    _added = {a["content"]: a["id"] for a in json.load(open(os.environ["LLM_TOKENIZER"], encoding="utf-8"))["added_tokens"]}
    _rest = _ch.get("tool_item_prefix", "")[len(_ch["stop_after_calls"]):]
    _fo = max((a for a in _added if _rest.startswith(a)), key=len, default=None)
    if _fo and _ch["stop_after_calls"] in _added and _ch["stop_after_answer"] in _added:
        _e, _sc, _sa, _f = cfg["eos_id"], _added[_ch["stop_after_calls"]], _added[_ch["stop_after_answer"]], _added[_fo]
        ids = np.array([_e, _f, 1000, _e, 1000, _e], dtype=np.int32); t = len(ids)
        want = np.array([_sc, _f, 1000, _sa, 1000, _sa], dtype=np.int32)
        x = E[want]; h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
        for n, a in [("sr_ids", ids), ("sr_x", x.astype(np.float32)), ("sr_h", h.astype(np.float32))]: save(n, a)
        cases.append(f"""check.case public @embed_stop_restore {{
  %t = check.literal value({t}) : index
  %ids = check.file.read.npy path("sr_ids.npy") : {ty(ids)}
  %tab = check.file.read.npy path("e_tab.npy") : {ty(tab)}
  %w = check.file.read.npy path("e_w.npy") : tensor<{H}xf32>
  %ex = check.file.read.npy path("sr_x.npy") : tensor<{t}x{H}xf32>
  %eh = check.file.read.npy path("sr_h.npy") : tensor<{t}x{H}xf32>
  %x = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  %h = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  kernel.launch @g_embed[%t](%t, %ids, %tab, %w, %x, %h) : [index](index, {ty(ids)}, {ty(tab)}, tensor<{H}xf32>, tensor<{t}x{H}xf32>, tensor<{t}x{H}xf32>)
  check.expect.close actual(%x) expected(%ex) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.expect.close actual(%h) expected(%eh) atol(1.0e-3) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.return
}}
""")
# 2. matvec per (tensor) for t=1 and t=4 variants
seen = set(); todo = []
VD = cfg["gdn"]["inner"] if cfg.get("gdn") else 0
for li in range(cfg["layers"]):
    for nm_, K in [("attn_q", H), ("attn_k", H), ("attn_v", H), ("ffn_down", cfg["ffn"]), ("attn_output", QH * D), ("ffn_gate", H),
                   ("attn_qkv", H), ("attn_gate", H), ("ssm_alpha", H), ("ssm_out", VD)]:
        name = f"blk.{li}.{nm_}.weight"
        if name not in T:
            continue
        key = (T[name].tensor_type.name.lower(), K)
        if key not in seen: seen.add(key); todo.append((name, K))
for name, K in todo:
    fmt = T[name].tensor_type.name.lower(); W = deq(name); n = W.shape[0]
    for tg in (1, 4):
        xs = rng.standard_normal((tg, K)).astype(np.float32); y = xs @ W.T.astype(np.float64)
        tag = name.replace(".", "_") + f"_t{tg}"
        save(tag + "_w", raw(name)); save(tag + "_x", xs); save(tag + "_y", y.astype(np.float32))
        kern = f"g_mv_{fmt}_k{K}" + ("_t4" if tg == 4 else "")
        # adaptive rows per workgroup (same rule as gen_llm.pick_r): test the variant the model launches
        _tgt = (640 if K >= 8192 else int(os.environ.get("MV_WG1", "4096"))) if (tg == 1 or "MV_WG4" not in os.environ) else int(os.environ["MV_WG4"]); _R = 16
        while _R > 1 and n // _R < _tgt and n % (_R // 2) == 0: _R //= 2
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
  kernel.launch @g_qk_rope_store[%t](%t, %layer, %rope, %ctx, %pool, %q, %k, %v, %qn, %kn, %info, %orig, %arena) : [index](index, index, i32, index, index, tensor<{t}x{QH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{t}x4xi32>, tensor<64xi32>, tensor<1024xf16>)
  check.expect.close actual(%q) expected(%qe) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{QH * D}xf32>
  check.return
}}
""")
# 3a. output head (argmax over the vocabulary) for one selected span of 3 tokens
t = 3
head_name = "output.weight" if "output.weight" in T else "token_embd.weight"
Wh = deq(head_name).astype(np.float64)
hs = rng.standard_normal((t, H)).astype(np.float32)
logits = Wh @ hs[-1].astype(np.float64)
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
  kernel.launch @g_head_mv[%spans](%spans, %t, %meta, %h, %w, %partials, %logits) : [index](index, index, tensor<13xi32>, tensor<{t}x{H}xf32>, {hwt}, tensor<{CH * 2}xi32>, tensor<{CH * 128}xf32>)
  kernel.launch @g_head_select[%spans](%spans, %meta, %partials, %logits, %out) : [index](index, tensor<13xi32>, tensor<{CH * 2}xi32>, tensor<{CH * 128}xf32>, tensor<1xi32>)
  check.expect.equal actual(%out) expected(%expect) : tensor<1xi32>
  check.return
}}
""")
# 3b. attention chain: [q/gate split] -> qk norm + rope + paged KV store -> split-K attention -> combine (gate)
t, ctx = 4, 64
pos = np.arange(t)
gate_kind = cfg.get("attn_gate", False)
q = rng.standard_normal((t, QH, D)).astype(np.float32); k = rng.standard_normal((t, KVH, D)).astype(np.float32)
v = rng.standard_normal((t, KVH, D)).astype(np.float32); gate = rng.standard_normal((t, QH, D)).astype(np.float32)
qr = rope(norm(q, qn) if cfg["qk_norm"] else q); kr = rope(norm(k, kn) if cfg["qk_norm"] else k)
att = np.zeros((t, QH, D))
sinks = deq(f"blk.{A0}.attn_sinks.weight") if cfg.get("attn_sinks") else np.zeros(QH, np.float32)
for hh in range(QH):
    kv = hh // (QH // KVH); sc = qr[:, hh] @ kr[:, kv].T / np.sqrt(D); sc[np.triu_indices(t, 1)] = -1e30
    if cfg.get("attn_sinks"):
        mx = np.maximum(sc.max(-1, keepdims=True), sinks[hh]); pr = np.exp(sc - mx); pr /= pr.sum(-1, keepdims=True) + np.exp(sinks[hh] - mx)
    else:
        pr = np.exp(sc - sc.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
    att[:, hh] = pr @ v[:, kv]
if gate_kind:
    att = att / (1 + np.exp(-gate))
blocks = (ctx + 15) // 16
orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)  # record: map origin 40; identity page map
info = np.zeros((t, 4), np.int32); info[:, 1] = pos; info[:, 2] = 1
S = 64
for n, a in [("a_q", q.reshape(t, -1)), ("a_k", k.reshape(t, -1)), ("a_v", v.reshape(t, -1)), ("a_g", gate.reshape(t, -1)),
             ("a_qg", np.stack([q, gate], 2).reshape(t, -1)), ("a_info", info), ("a_orig", orig), ("a_sinks", sinks.astype(np.float32)), ("a_o", att.reshape(t, -1).astype(np.float32))]:
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
  %sinks = check.file.read.npy path("a_sinks.npy") : tensor<{QH}xf32>
  %arena = check.generate.fill value(0.0) : tensor<{2 * ctx * KVH * D}xf16>
  %pacc = check.generate.fill value(0.0) : tensor<{t * QH * S * D}xf32>
  %pml = check.generate.fill value(0.0) : tensor<{t * QH * S * 2}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{QD}xf32>
{split_text}  kernel.launch @g_qk_rope_store[%t](%t, %layer, %rope, %ctx, %pool, %q, %k, %v, %qn, %kn, %info, %orig, %arena) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{2 * ctx * KVH * D}xf16>)
  kernel.launch @g_attention_split[%t](%t, %layer, %sliding, %ctx, %pool, %q, %info, %orig, %arena, %pacc, %pml) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{2 * ctx * KVH * D}xf16>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>)
  kernel.launch @g_attention_combine[%t](%t, %sliding, %info, %pacc, %pml, {"%sinks" if cfg.get("attn_sinks") else "%g" if gate_kind else "%q"}, %o) : [index](index, i32, tensor<{t}x4xi32>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>, {f"tensor<{QH}xf32>" if cfg.get("attn_sinks") else f"tensor<{t}x{QD}xf32>"}, tensor<{t}x{QD}xf32>)
  check.expect.close actual(%o) expected(%eo) atol(3.0e-3) rtol(3.0e-3) nan(different) : tensor<{t}x{QD}xf32>
  check.return
}}
""")
# 3c. prefill attention on a span that CONTINUES stored KV: history positions [0, p0) are written
# by g_qk_rope_store, then a span at p0 (t tokens, the last `pad` rows not live) runs the prefill
# path (g_attention_flash2 on uniform 16-token tiles + the listed split/combine fallback for the
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
    def rope_p(a, p):
        inv = cfg["rope_theta"] ** (-np.arange(0, RD // 2) * 2.0 / RD); ang = p[:, None] * inv[None]
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        if cfg["rope_type"] == "neox":
            a1, a2 = a[..., :RD // 2], a[..., RD // 2:RD]; return np.concatenate([a1 * c - a2 * s, a2 * c + a1 * s, a[..., RD:]], -1)
        a1, a2 = a[..., 0::2], a[..., 1::2]; o = np.empty_like(a); o[..., 0::2] = a1 * c - a2 * s; o[..., 1::2] = a2 * c + a1 * s; return o
    qr_ = norm(qa, qn_) if cfg["qk_norm"] else qa; kr_ = norm(ka, kn_) if cfg["qk_norm"] else ka
    if rp: qr_, kr_ = rope_p(qr_, allpos), rope_p(kr_, allpos)
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
    att_f = att.copy()
    for tl in range(t // 16):
        if not live[tl * 16:tl * 16 + 16].all(): att_f[tl * 16:tl * 16 + 16] = 0
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)
    hinfo = np.zeros((p0, 4), np.int32); hinfo[:, 1] = allpos[:p0]; hinfo[:, 2] = 1
    sinfo = np.zeros((t, 4), np.int32); sinfo[:, 1] = allpos[p0:]; sinfo[:, 2] = live
    for nn, a in [("hq", qa[:p0].reshape(p0, -1)), ("hk", ka[:p0].reshape(p0, -1)), ("hv", va[:p0].reshape(p0, -1)), ("hinfo", hinfo),
                  ("sq", qa[p0:].reshape(t, -1)), ("sk", ka[p0:].reshape(t, -1)), ("sv", va[p0:].reshape(t, -1)), ("sinfo", sinfo),
                  ("orig", orig), ("qn", qn_.astype(np.float32)), ("kn", kn_.astype(np.float32)), ("sinks", snk), ("o", att.reshape(t, -1).astype(np.float32)), ("of", att_f.reshape(t, -1).astype(np.float32))]:
        save(f"{tag}_{nn}", a)
    print(f"{tag}: layer {li} sliding={sl} rope={rp} p0={p0} t={t} pad={pad} |o| max {np.abs(att).max():.4f} mean {np.abs(att).mean():.5f}")
    QD = QH * D; KD = KVH * D; AR = f"tensor<{2 * ctx * KD}xf16>"; OT = f"tensor<{t}x{QD}xf32>"
    S = 64; S16 = 16
    gt = f"tensor<{QH}xf32>" if cfg.get("attn_sinks") else OT
    gv = "%sinks" if cfg.get("attn_sinks") else "%sq"
    def st(tt, pre):
        return (f"  kernel.launch @g_qk_rope_store[%{pre}t](%{pre}t, %layer, %rope, %ctx, %pool, %{pre}q, %{pre}k, %{pre}v, %qn, %kn, %{pre}info, %orig, %arena) : "
                f"[index](index, index, i32, index, index, tensor<{tt}x{QD}xf32>, tensor<{tt}x{KD}xf32>, tensor<{tt}x{KD}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{tt}x4xi32>, tensor<{10 + blocks}xi32>, {AR})\n")
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
  %arena = check.generate.fill value(0.0) : {AR}
  %pacc = check.generate.fill value(0.0) : tensor<{t * QH * S * D}xf32>
  %pml = check.generate.fill value(0.0) : tensor<{t * QH * S * 2}xf32>
  %o_pre = check.generate.fill value(0.0) : {OT}
  %o_dec = check.generate.fill value(0.0) : {OT}
{st(p0, "h")}{st(t, "s")}  kernel.launch @g_attention_flash2[%st](%st, %layer, %sliding, %ctx, %pool, %sq, {gv}, %sinfo, %orig, %arena, %o_pre) : [index](index, index, i32, index, index, {OT}, {gt}, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, {AR}, {OT})
  kernel.launch @g_attention_split[%st](%st, %layer, %sliding, %ctx, %pool, %sq, %sinfo, %orig, %arena, %pacc, %pml) : [index](index, index, i32, index, index, {OT}, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, {AR}, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>)
  kernel.launch @g_attention_combine[%st](%st, %sliding, %sinfo, %pacc, %pml, {gv}, %o_dec) : [index](index, i32, tensor<{t}x4xi32>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>, {gt}, {OT})
  check.expect.close actual(%o_dec) expected(%eo) atol(1.5e-3) rtol(3.0e-3) nan(different) : {OT}
  check.expect.close actual(%o_pre) expected(%eof) atol(2.5e-3) rtol(1.0e-2) nan(different) : {OT}
  check.return
}}
""")
if os.environ.get("PREFILL_CONT", "1") != "0":
    _sl = cfg.get("sliding_layers") or []
    _ns = [i for i in range(cfg["layers"]) if i not in set(_sl) and i not in set(cfg.get("recurrent_layers", []))]
    _p0 = max(300, (cfg.get("sliding_window") or 0) + 100)
    for _li in ([_sl[0]] if _sl else []) + _ns[:1]:
        prefill_cont_case(_li, _p0, 48, 0, f"pc_full_{_li}")     # 3 uniform tiles: flash2 only
        prefill_cont_case(_li, _p0, 48, 7, f"pc_part_{_li}")     # last tile partly padding: listed fallback
        prefill_cont_case(_li, _p0, 48, 0, f"pc_small_{_li}", 0.1)  # small scores: the sink dominates the softmax
# 4. gated delta-net chain on one 5-token span at position 3 that continues the reader slot
if cfg.get("recurrent_layers"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_gdn import gdn_geometry
    G = gdn_geometry(cfg); HK, HV, DK, DV, K, C, VD = (G[k] for k in ("HK", "HV", "DK", "DV", "K", "C", "VD"))
    li = cfg["recurrent_layers"][0]; b = f"blk.{li}."
    conv_w = deq(b + "ssm_conv1d.weight").reshape(C, K).astype(np.float32)
    a_log = deq(b + "ssm_a").astype(np.float32); dt = deq(b + "ssm_dt.bias").astype(np.float32); nw = deq(b + "ssm_norm.weight").astype(np.float32)
    t, p0 = 5, 3
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
# 6. MoE chain on real tensors (one Q4_K-down and one Q6_K-down layer): router -> top-k -> gate/up -> down -> combine
if cfg.get("moe"):
    M = cfg["moe"]; E_, KU, FE = M["experts"], M["used"], M["ffn"]; R = int(os.environ.get("MOE_R", "8"))
    def deq3(name):
        tt = T[name]; return dequantize(np.asarray(tt.data), tt.tensor_type).reshape(E_, int(tt.shape[1]), int(tt.shape[0]))
    done = set()
    for li in range(M["leading_dense"], cfg["layers"]):
        b = f"blk.{li}."; fd = T[b + "ffn_down_exps.weight"].tensor_type.name.lower(); fg = T[b + "ffn_gate_exps.weight"].tensor_type.name.lower()
        if fd in done: continue
        done.add(fd)
        t = 3
        hx = rng.standard_normal((t, H)).astype(np.float32)
        tr = T[b + "ffn_gate_inp.weight"]; rbf = tr.tensor_type.name == "BF16"
        Wr = ((np.asarray(tr.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32) if rbf else np.asarray(tr.data, np.float32)).reshape(E_, H)
        wr_raw = np.asarray(tr.data).view(np.int8).reshape(-1) if rbf else Wr.reshape(-1)
        bias = np.asarray(T[b + "exp_probs_b.bias"].data, np.float32) if M["probs_bias"] else np.zeros(E_, np.float32)
        logit = hx.astype(np.float64) @ Wr.T.astype(np.float64); prob = 1 / (1 + np.exp(-logit)); sel = prob + bias
        ids = np.argsort(-sel, -1, kind="stable")[:, :KU].astype(np.int32)
        wts = np.take_along_axis(prob, ids, -1); wts = wts / np.maximum(wts.sum(-1, keepdims=True), 6.103515625e-05)
        Wg, Wu, Wd = deq3(b + "ffn_gate_exps.weight"), deq3(b + "ffn_up_exps.weight"), deq3(b + "ffn_down_exps.weight")
        yref = np.zeros((t, H))
        for i in range(t):
            for k in range(KU):
                e = ids[i, k]; g = Wg[e] @ hx[i]; u = Wu[e] @ hx[i]; a = g / (1 + np.exp(-g)) * u
                yref[i] += wts[i, k] * (Wd[e] @ a)
        tag = f"m{li}"
        for n, a in [(tag + "_h", hx), (tag + "_wr", wr_raw), (tag + "_b", bias), (tag + "_ids", ids), (tag + "_wts", wts.astype(np.float32)),
                     (tag + "_wg", raw(b + "ffn_gate_exps.weight")), (tag + "_wu", raw(b + "ffn_up_exps.weight")), (tag + "_wd", raw(b + "ffn_down_exps.weight")),
                     (tag + "_y", yref.astype(np.float32))]:
            save(n, a)
        print(f"moe test layer {li}: ids", ids.tolist(), "min score gap k/k+1", float(np.min(np.sort(sel, -1)[:, ::-1][:, KU - 1] - np.sort(sel, -1)[:, ::-1][:, KU])))
        gt, dt_ = ty(raw(b + "ffn_gate_exps.weight")), ty(raw(b + "ffn_down_exps.weight"))
        cases.append(f"""check.case public @moe_chain_{li} {{
  %t = check.literal value({t}) : index
  %ta = check.literal value({t * KU}) : index
  %nFE = check.literal value({FE}) : index
  %nH = check.literal value({H}) : index
  %h = check.file.read.npy path("{tag}_h.npy") : tensor<{t}x{H}xf32>
  %wr = check.file.read.npy path("{tag}_wr.npy") : {ty(wr_raw)}
  %b = check.file.read.npy path("{tag}_b.npy") : tensor<{E_}xf32>
  %eids = check.file.read.npy path("{tag}_ids.npy") : tensor<{t}x{KU}xi32>
  %ewts = check.file.read.npy path("{tag}_wts.npy") : tensor<{t}x{KU}xf32>
  %wg = check.file.read.npy path("{tag}_wg.npy") : {gt}
  %wu = check.file.read.npy path("{tag}_wu.npy") : {gt}
  %wd = check.file.read.npy path("{tag}_wd.npy") : {dt_}
  %ey = check.file.read.npy path("{tag}_y.npy") : tensor<{t}x{H}xf32>
  %ids = check.generate.fill value(0) : tensor<{t}x{KU}xi32>
  %wts = check.generate.fill value(0.0) : tensor<{t}x{KU}xf32>
  %act = check.generate.fill value(0.0) : tensor<{t * KU}x{FE}xf32>
  %ys = check.generate.fill value(0.0) : tensor<{t * KU}x{H}xf32>
  %y = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  kernel.launch @g_moe_route[%t](%t, %h, %wr, %b, %ids, %wts) : [index](index, tensor<{t}x{H}xf32>, {ty(wr_raw)}, tensor<{E_}xf32>, tensor<{t}x{KU}xi32>, tensor<{t}x{KU}xf32>)
  kernel.launch @g_moe_gu_{fg}_k{H}_r{R}[%nFE, %ta](%nFE, %ta, %ids, %h, %wg, %wu, %act) : [index, index](index, index, tensor<{t}x{KU}xi32>, tensor<{t}x{H}xf32>, {gt}, {gt}, tensor<{t * KU}x{FE}xf32>)
  kernel.launch @g_moe_dn_{fd}_k{FE}_r{R}[%nH, %ta](%nH, %ta, %ids, %act, %wd, %ys) : [index, index](index, index, tensor<{t}x{KU}xi32>, tensor<{t * KU}x{FE}xf32>, {dt_}, tensor<{t * KU}x{H}xf32>)
  kernel.launch @g_moe_combine[%t](%t, %ys, %wts, %y) : [index](index, tensor<{t * KU}x{H}xf32>, tensor<{t}x{KU}xf32>, tensor<{t}x{H}xf32>)
  check.expect.equal actual(%ids) expected(%eids) : tensor<{t}x{KU}xi32>
  check.expect.close actual(%wts) expected(%ewts) atol(1.0e-5) rtol(1.0e-4) nan(different) : tensor<{t}x{KU}xf32>
  check.expect.close actual(%y) expected(%ey) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.return
}}
""")
import re
# hrx-loom 0.1.0: float comparison is check.expect.approximate with the actual and expected types.
text = re.sub(r"check\.expect\.close (actual\(.*\)) : (.*)$", r"check.expect.approximate \1 : \2, \2", "\n".join(cases), flags=re.M)
open(os.path.join(out, "cases.loom-test"), "w", newline="\n").write(text)
print("wrote", len(cases), "cases")
