"""Vision encoder plan for a GGUF mmproj: config from metadata + an ordered list of kernel launches.
Backends render the plan into a loom-test check.case (whole-encoder test) or a command program (serving)."""
import math, gguf, numpy as np
import gen_vision as G

FAMILIES = ("gemma4v", "qwen3vl_merger", "muse-glimmer")

def _kv(r, k, default=None):
    f = r.fields.get(k)
    if f is None: return default
    if f.types[0] == gguf.GGUFValueType.STRING: return bytes(f.parts[f.data[0]]).decode()
    v = [f.parts[i].tolist()[0] for i in f.data]
    return v[0] if len(v) == 1 else v

GGML_FMT = {"F16": "f16", "BF16": "bf16", "F32": "f32"}

def config(path):
    r = gguf.GGUFReader(path)
    fam = _kv(r, "clip.vision.projector_type") or _kv(r, "clip.projector_type")
    if fam not in FAMILIES:
        raise SystemExit(f"STOP: vision projector '{fam}' has no encoder plan yet (see loom-port-llm/reference/vision.md)")
    tens = {t.name: dict(type=t.tensor_type.name, shape=[int(x) for x in t.shape], bytes=int(t.n_bytes)) for t in r.tensors
            if not t.name.startswith(("a.", "mm.a."))}
    E = _kv(r, "clip.vision.embedding_length"); NH = _kv(r, "clip.vision.attention.head_count")
    c = dict(family=fam, patch=_kv(r, "clip.vision.patch_size"), E=E, NH=NH, D=E // NH,
             ffn=_kv(r, "clip.vision.feed_forward_length"), layers=_kv(r, "clip.vision.block_count"),
             eps=_kv(r, "clip.vision.attention.layer_norm_epsilon", 1e-6), out=_kv(r, "clip.vision.projection_dim"),
             mean=_kv(r, "clip.vision.image_mean"), std=_kv(r, "clip.vision.image_std"), tensors=tens)
    c["merge"] = {"gemma4v": 3, "qwen3vl_merger": 2, "muse-glimmer": 2}[fam]
    c["merge"] = _kv(r, "clip.vision.spatial_merge_size", c["merge"]) if fam != "gemma4v" else _kv(r, "clip.vision.projector.scale_factor", 3)
    c["fmt"] = GGML_FMT[tens["v.blk.0.ffn_up.weight"]["type"]]
    c["pos_n"] = tens["v.position_embd.weight"]["shape"][1]
    if fam != "gemma4v": c["pos_n"] = int(round(math.sqrt(c["pos_n"])))
    c["out"] = tens[{"gemma4v": "mm.input_projection.weight", "qwen3vl_merger": "mm.2.weight", "muse-glimmer": "mm.2.weight"}[fam]]["shape"][1]
    # DeepStack (Qwen3-VL): ViT layers whose merged output goes through its own merger (v.deepstack.N.*); each
    # output row is the main projection followed by one projection per such layer (llama.cpp concatenates them
    # along the feature dimension), and the text model adds slice k+1 to its hidden state after layer k.
    flags = _kv(r, "clip.vision.is_deepstack_layers")
    c["deepstack"] = [i for i, f in enumerate(flags if isinstance(flags, list) else [flags] if flags is not None else []) if f]
    missing = [i for i in c["deepstack"] if f"v.deepstack.{i}.fc2.weight" not in tens]
    if missing:
        raise SystemExit(f"STOP: deepstack layers {missing} have no v.deepstack.N.* tensors")
    c["out_row"] = c["out"] * (1 + len(c["deepstack"]))
    return c

def fmt_of(c, name):
    return GGML_FMT[c["tensors"][name]["type"]]

class Plan:
    """Ordered launches over named buffers. Buffers: dict name -> (rows_expr, width, dtype). Weights by tensor name."""
    def __init__(s, c):
        s.c = c; s.kernels = {}; s.ops = []
    def k(s, made):
        name, src = made; s.kernels[name] = src; return name
    def launch(s, kern, grid, args):
        s.ops.append((kern, grid, args))

def build(c):
    """Returns Plan. Runtime scalars: %T (tokens), %Tp (padded), %gw, %gh, %M (merged tokens), %Mp (padded),
    %n_pix (h*w*3). Buffers: pix u8 -> pixf, h, x, q, k, v, o, a, f, g, xs, x16, sc, m_in, m1, m2, out."""
    p = Plan(c); fam = c["family"]; E, NH, D, FF, P = c["E"], c["NH"], c["D"], c["ffn"], c["patch"]
    W = lambda n: ("W", n)          # weight reference
    B = lambda n: ("B", n)          # activation buffer
    S = lambda n: ("S", n)          # runtime scalar
    rs = p.k(G.row_scale())
    def lin(xbuf, K, N, wname, ybuf, rows, act="none", bias=None, clamp_pref=None):
        src = xbuf
        if clamp_pref and (clamp_pref + ".input_min") in c["tensors"]:
            cl = p.k(G.clamp()); p.launch(cl, [S(f"{rows}*{K}")], [S(f"{rows}*{K}"), B(xbuf), W(clamp_pref + ".input_min"), W(clamp_pref + ".input_max"), B("xs")]); src = "xs"
        p.launch(rs, [S(rows + "p")], [S(rows + "p"), S(str(K)), B(src), B("sc"), B("x16")])
        gm = p.k(G.gemm(fmt_of(c, wname), K, N))
        p.launch(gm, [S(rows + "p"), S(str(N))], [S(rows + "p"), S(str(K)), S(str(N)), B("x16"), B("sc"), W(wname), B(ybuf)])
        if bias is not None and bias in c["tensors"]:
            ba = p.k(G.bias_act(N, act)); p.launch(ba, [S(rows)], [S(rows), B(ybuf), W(bias), B(ybuf)])
        elif act != "none":
            ew = p.k(G.ew(act)); p.launch(ew, [S(f"{rows}*{N}")], [S(f"{rows}*{N}"), B(ybuf), B(ybuf)])
        if clamp_pref and (clamp_pref + ".output_min") in c["tensors"]:
            cl = p.k(G.clamp()); p.launch(cl, [S(f"{rows}*{N}")], [S(f"{rows}*{N}"), B(ybuf), W(clamp_pref + ".output_min"), W(clamp_pref + ".output_max"), B(ybuf)])
    def norm(xbuf, ybuf, rows, width, mode, w=None, b=None):
        nk = p.k(G.norm(width, mode, c["eps"]))
        p.launch(nk, [S(rows)], [S(rows), B(xbuf)] + ([W(w)] if w else []) + ([W(b)] if b else []) + [B(ybuf)])
    def place(xbuf, rows, k):   # [rows, out] -> slice k of the [rows, out_row] output
        pk_ = p.k(G.place_cols(c["out"], c["out_row"], k * c["out"])); p.launch(pk_, [S(rows)], [S(rows), B(xbuf), B("out")])
    def add(xbuf, ybuf, rows, width):
        ak = p.k(G.ew("add")); p.launch(ak, [S(f"{rows}*{width}")], [S(f"{rows}*{width}"), B(xbuf), B(ybuf), B(xbuf)])
    # --- pixels + patch embedding + positions ---
    order = "block2" if fam == "qwen3vl_merger" else "raster"
    pk = p.k(G.pixels(c["mean"], c["std"], scale2=(fam == "gemma4v")))
    p.launch(pk, [S("n_pix")], [S("n_pix"), B("pix"), B("pixf")])
    dual = "v.patch_embd.weight.1" in c["tensors"]; bias = "v.patch_embd.bias" in c["tensors"]
    pe = p.k(G.patch_embed(P, E, fmt_of(c, "v.patch_embd.weight"), order, dual, bias))
    p.launch(pe, [S("T")], [S("T"), S("gw"), S("img_w"), B("pixf"), W("v.patch_embd.weight")] + ([W("v.patch_embd.weight.1")] if dual else []) + ([W("v.patch_embd.bias")] if bias else []) + [B("h")])
    if fam == "gemma4v":
        pk2 = p.k(G.pos_xy(E, c["pos_n"])); p.launch(pk2, [S("T")], [S("T"), S("gw"), W("v.position_embd.weight"), B("h")])
    else:
        pk2 = p.k(G.pos_interp(E, c["pos_n"], order, align_corners=(fam == "qwen3vl_merger")))
        p.launch(pk2, [S("T")], [S("T"), S("gw"), S("gh"), W("v.position_embd.weight"), B("h")])
    if "v.pre_ln.weight" in c["tensors"]:
        norm("h", "h", "T", E, "ln", "v.pre_ln.weight", "v.pre_ln.bias")
    scale = 1.0 if fam == "gemma4v" else D ** -0.5
    for i in range(c["layers"]):
        b = f"v.blk.{i}."
        if fam == "gemma4v":
            norm("h", "x", "T", E, "rmsw", b + "ln1.weight")
            for nm, buf in (("attn_q", "q"), ("attn_k", "k"), ("attn_v", "v")):
                lin("x", E, E, b + nm + ".weight", buf, "T", clamp_pref=b + nm)
            norm("q", "q", "T*NH", D, "rmsw", b + "attn_q_norm.weight"); norm("k", "k", "T*NH", D, "rmsw", b + "attn_k_norm.weight")
            norm("v", "v", "T*NH", D, "rms")
            for buf in ("q", "k"):
                rk = p.k(G.rope("gemma_halves", D, NH, E, 0, order)); p.launch(rk, [S("T")], [S("T"), S("gw"), B(buf)])
            ak = p.k(G.attention(D, NH, E, 0, 0, 0, scale)); p.launch(ak, [S("T")], [S("T"), S("gw"), B("q"), B("k"), B("v"), B("o")])
            lin("o", E, E, b + "attn_out.weight", "a", "T", clamp_pref=b + "attn_out")
            norm("a", "a", "T", E, "rmsw", b + "attn_post_norm.weight"); add("h", "a", "T", E)
            norm("h", "x", "T", E, "rmsw", b + "ln2.weight")
            lin("x", E, FF, b + "ffn_gate.weight", "g", "T", clamp_pref=b + "ffn_gate")
            lin("x", E, FF, b + "ffn_up.weight", "f", "T", clamp_pref=b + "ffn_up")
            gk = p.k(G.ew("gate_gelu_tanh")); p.launch(gk, [S(f"T*{FF}")], [S(f"T*{FF}"), B("g"), B("f"), B("f")])
            lin("f", FF, E, b + "ffn_down.weight", "a", "T", clamp_pref=b + "ffn_down")
            norm("a", "a", "T", E, "rmsw", b + "ffn_post_norm.weight"); add("h", "a", "T", E)
        else:
            norm("h", "x", "T", E, "ln", b + "ln1.weight", b + "ln1.bias")
            if fam == "qwen3vl_merger":
                lin("x", E, 3 * E, b + "attn_qkv.weight", "q", "T", bias=b + "attn_qkv.bias")
                for off in (0, E):
                    rk = p.k(G.rope("qwen_vision", D, NH, 3 * E, off, order)); p.launch(rk, [S("T")], [S("T"), S("gw"), B("q")])
                ak = p.k(G.attention(D, NH, 3 * E, 0, E, 2 * E, scale)); p.launch(ak, [S("T")], [S("T"), S("gw"), B("q"), B("q"), B("q"), B("o")])
            else:
                for nm, buf in (("attn_q", "q"), ("attn_k", "k"), ("attn_v", "v")):
                    lin("x", E, E, b + nm + ".weight", buf, "T", bias=b + nm + ".bias")
                for buf in ("q", "k"):
                    rk = p.k(G.rope("normal_halves", D, NH, E, 0, order)); p.launch(rk, [S("T")], [S("T"), S("gw"), B(buf)])
                glob = (i == c["layers"] - 1) or ((i + 1) % 4 == 0)
                ak = p.k(G.attention(D, NH, E, 0, 0, 0, scale, window=0 if glob else c["pos_n"]))
                p.launch(ak, [S("T")], [S("T"), S("gw"), B("q"), B("k"), B("v"), B("o")])
            lin("o", E, E, b + "attn_out.weight", "a", "T", bias=b + "attn_out.bias"); add("h", "a", "T", E)
            norm("h", "x", "T", E, "ln", b + "ln2.weight", b + "ln2.bias")
            act = "gelu_tanh" if fam == "qwen3vl_merger" else "gelu_erf"
            lin("x", E, FF, b + "ffn_up.weight", "f", "T", act=act, bias=b + "ffn_up.bias")
            lin("f", FF, E, b + "ffn_down.weight", "a", "T", bias=b + "ffn_down.bias"); add("h", "a", "T", E)
        if i in c["deepstack"]:   # this layer's merged output through its own merger, into output slice k + 1
            d = f"v.deepstack.{i}."; k = c["deepstack"].index(i) + 1; wm = E * c["merge"] ** 2
            norm("h", "x", "M", wm, "ln", d + "norm.weight", d + "norm.bias")
            lin("x", wm, c["tensors"][d + "fc1.weight"]["shape"][1], d + "fc1.weight", "m1", "M", act="gelu_tanh", bias=d + "fc1.bias")
            lin("m1", c["tensors"][d + "fc1.weight"]["shape"][1], c["out"], d + "fc2.weight", "dso", "M", bias=d + "fc2.bias")
            place("dso", "M", k)
    # --- merge + projector ---
    m = c["merge"]
    if fam == "gemma4v":
        pk3 = p.k(G.avgpool(E, m, math.sqrt(E))); p.launch(pk3, [S("M")], [S("M"), S("gw"), S("T"), B("h"), B("m_in")])
        if "v.std_bias" in c["tensors"]:
            raise SystemExit("STOP: gemma4v std_bias/std_scale not implemented yet")
        norm("m_in", "m_in", "M", E, "rms")
        lin("m_in", E, c["out"], "mm.input_projection.weight", "out", "M", clamp_pref="mm.input_projection")
    elif fam == "qwen3vl_merger":
        norm("h", "x", "T", E, "ln", "v.post_ln.weight", "v.post_ln.bias")
        hid = c["tensors"]["mm.0.weight"]["shape"][1]
        lin("x", 4 * E, hid, "mm.0.weight", "m1", "M", act="gelu_tanh", bias="mm.0.bias")
        if c["deepstack"]:
            lin("m1", hid, c["out"], "mm.2.weight", "dso", "M", bias="mm.2.bias"); place("dso", "M", 0)
        else:
            lin("m1", hid, c["out"], "mm.2.weight", "out", "M", bias="mm.2.bias")
    else:
        norm("h", "x", "T", E, "ln", "v.post_ln.weight", "v.post_ln.bias")
        sk = p.k(G.shuffle(E, m)); p.launch(sk, [S("M")], [S("M"), S("gw"), S("T"), B("x"), B("m_in")])
        h1 = c["tensors"]["mm.0.weight"]["shape"][1]; h2 = c["tensors"]["mm.1.weight"]["shape"][1]
        lin("m_in", E * m * m, h1, "mm.0.weight", "m1", "M", act="gelu_erf")
        lin("m1", h1, h2, "mm.1.weight", "m2", "M", act="gelu_erf")
        lin("m2", h2, c["out"], "mm.2.weight", "out", "M")
    return p

def buffers(c, T, M):
    """Element counts (f32 unless noted) per buffer for T tokens / M merged tokens (rows padded)."""
    Tp = -(-T // 256) * 256; Mp = -(-M // 64) * 64
    return {n: (Tp if r == "Tp" else Mp) * w for n, (r, w, _) in buffer_exprs(c).items()}

def buffer_exprs(c):
    """name -> (rows variable 'Tp' | 'Mp', width in elements, bytes per element) for the command program."""
    E, FF = c["E"], c["ffn"]
    mats = ["mm.0.weight", "mm.1.weight"] + [f"v.deepstack.{i}.fc1.weight" for i in c.get("deepstack", [])]
    hid = max([c["tensors"][n]["shape"][1] for n in mats if n in c["tensors"]] + [1])
    wide = max(FF, 3 * E, E * c["merge"] ** 2, hid, c["out"])
    xw = 4 * E if c["family"] == "qwen3vl_merger" else E
    b = dict(h=("Tp", E, 4), x=("Tp", xw, 4), q=("Tp", 3 * E, 4), k=("Tp", E, 4), v=("Tp", E, 4), o=("Tp", E, 4), a=("Tp", E, 4),
             f=("Tp", FF, 4), g=("Tp", FF, 4), xs=("Tp", wide, 4), x16=("Tp", wide, 2), sc=("Tp", 2, 4),
             m_in=("Mp", E * c["merge"] ** 2, 4), m1=("Mp", hid, 4), m2=("Mp", hid, 4), out=("Mp", c.get("out_row", c["out"]), 4))
    if c.get("deepstack"):   # one projection's rows before they are placed in the wide output row
        b["dso"] = ("Mp", c["out"], 4)
    return b
