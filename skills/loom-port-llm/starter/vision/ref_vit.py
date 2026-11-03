"""NumPy reference ViT encoders for GGUF mmproj files (development aid; checked against llama-mtmd-debug).
usage: ref_vit.py MMPROJ.gguf rainbow|gray N   -> prints projected sum/edges"""
import sys, math, numpy as np, gguf

def load(path):
    r = gguf.GGUFReader(path); W = {}; KV = {}
    for t in r.tensors:
        a = np.array(t.data)
        if t.tensor_type.name == "BF16":
            a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        W[t.name] = a.astype(np.float32).reshape([int(x) for x in reversed(t.shape)]) if a.ndim == 1 or True else a
    for k, f in r.fields.items():
        if not k.startswith("clip."): continue
        try:
            KV[k] = bytes(f.parts[f.data[0]]).decode() if f.types[0] == gguf.GGUFValueType.STRING else (
                f.parts[f.data[0]].tolist()[0] if len(f.data) == 1 else [f.parts[i].tolist()[0] for i in f.data])
        except Exception: pass
    return W, KV

TRACE = {}
ROUND_ACT = None   # e.g. bf16 rounding of matmul inputs, to mimic llama.cpp's GPU matmuls
def bf16(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
def f16(x): return x.astype(np.float16).astype(np.float32)
def tr(name, x):
    TRACE[name] = x
    return x

def rms(x, w=None, eps=1e-6):
    y = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps).astype(np.float32)
    return y * w if w is not None else y

def ln(x, w, b, eps):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

def gelu_tanh(x): return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
def gelu_erf(x):
    from math import erf; return 0.5 * x * (1 + np.vectorize(erf)(x / math.sqrt(2)))

def rope_neox(x, pos, n_dims, theta):
    # x [..., n_pos, D]; rotate the first n_dims dims with neox pairs (i, i + n_dims/2)
    h = n_dims // 2
    inv = theta ** (-np.arange(h, dtype=np.float64) * 2 / n_dims)
    ang = pos[:, None].astype(np.float64) * inv[None]
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    a, b = x[..., :h], x[..., h:n_dims]
    out = x.copy(); out[..., :h] = a * c - b * s; out[..., h:n_dims] = a * s + b * c
    return out

class Lin:
    def __init__(s, W, name):
        s.w = W[name + ".weight"]; s.b = W.get(name + ".bias")
        s.clamp = (W.get(name + ".input_min"), W.get(name + ".input_max"), W.get(name + ".output_min"), W.get(name + ".output_max"))
    def __call__(s, x):
        if s.clamp[0] is not None: x = np.clip(x, s.clamp[0].item(), s.clamp[1].item())
        if ROUND_ACT and s.w.ndim == 2: x = ROUND_ACT(x)
        y = x @ s.w.T
        if s.b is not None: y = y + s.b
        if s.clamp[2] is not None: y = np.clip(y, s.clamp[2].item(), s.clamp[3].item())
        return y

def gemma4v(W, KV, img):
    """img: float32 [H, W, 3] in [0,1] (already at the processed size)."""
    P = KV["clip.vision.patch_size"]; E = KV["clip.vision.embedding_length"]; NH = KV["clip.vision.attention.head_count"]
    eps = KV.get("clip.vision.attention.layer_norm_epsilon", 1e-6); D = E // NH; merge = 3
    x = img * 2 - 1; Hh, Ww = x.shape[0] // P, x.shape[1] // P
    patches = x.reshape(Hh, P, Ww, P, 3).transpose(0, 2, 4, 1, 3).reshape(Hh * Ww, 3 * P * P)   # (c, ky, kx)
    pw = W["v.patch_embd.weight"].reshape(E, 3 * P * P)
    h = patches @ pw.T
    px = np.tile(np.arange(Ww), Hh); py = np.repeat(np.arange(Hh), Ww)
    pe = W["v.position_embd.weight"]                    # [2, 10240, E]
    h = h + pe[0][px] + pe[1][py]
    nb = len({k.split(".")[2] for k in W if k.startswith("v.blk.")})
    for i in range(nb):
        p = f"v.blk.{i}."
        cur = rms(h, W[p + "ln1.weight"], eps)
        q = Lin(W, p + "attn_q")(cur).reshape(-1, NH, D); k = Lin(W, p + "attn_k")(cur).reshape(-1, NH, D); v = Lin(W, p + "attn_v")(cur).reshape(-1, NH, D)
        q = rms(q, W[p + "attn_q_norm.weight"], eps); k = rms(k, W[p + "attn_k_norm.weight"], eps); v = rms(v, None, eps)
        q, k, v = (t.transpose(1, 0, 2) for t in (q, k, v))           # [NH, n, D]
        def pos2d(t):
            t = t.copy(); t[..., :D // 2] = rope_neox(t[..., :D // 2], px, D // 2, 100.0)
            t[..., D // 2:] = rope_neox(t[..., D // 2:], py, D // 2, 100.0); return t
        q, k = pos2d(q), pos2d(k)
        s = q @ k.transpose(0, 2, 1); s = s - s.max(-1, keepdims=True); a = np.exp(s); a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(-1, E)
        o = Lin(W, p + "attn_out")(o)
        o = rms(o, W[p + "attn_post_norm.weight"], eps); h = h + o
        cur = rms(h, W[p + "ln2.weight"], eps)
        f = gelu_tanh(Lin(W, p + "ffn_gate")(cur)) * Lin(W, p + "ffn_up")(cur)
        f = Lin(W, p + "ffn_down")(f); f = rms(f, W[p + "ffn_post_norm.weight"], eps); h = h + f
    g = h.reshape(Hh, Ww, E)
    oy, ox = Hh // merge, Ww // merge
    g = g[:oy * merge, :ox * merge].reshape(oy, merge, ox, merge, E).mean((1, 3)).reshape(oy * ox, E) * math.sqrt(E)
    if "v.std_bias" in W: g = (g - W["v.std_bias"]) * W["v.std_scale"]
    g = rms(g, None, eps)
    return Lin(W, "mm.input_projection")(g)

def synth(kind, n):
    if kind == "gray": return np.full((n, n, 3), 0.5, np.float32)
    img = np.zeros((n, n, 3), np.float32); cx = cy = n / 2.0; md = math.sqrt(cx * cx + cy * cy)
    for y in range(n):
        for x in range(n):
            dx, dy = x - cx, y - cy; hue = math.atan2(dy, dx) / (2 * 3.14159265)
            hue += 1 if hue < 0 else 0; sat = min(math.sqrt(dx * dx + dy * dy) / md, 1.0)
            h6 = hue * 6; i6 = int(h6); f = h6 - i6; p = 1 - sat; q = 1 - sat * f; t = 1 - sat * (1 - f)
            img[y, x] = [(1, t, p), (q, 1, p), (p, 1, t), (p, q, 1), (t, p, 1), (1, p, q)][i6 % 6]
    return img

def bilinear_ac(grid, oh, ow):
    """ggml_interpolate bilinear + align_corners: grid [n, n, E] -> [oh, ow, E]."""
    n = grid.shape[0]
    def coords(o):
        if o == 1: return np.zeros(1)
        return np.arange(o) * (n - 1) / (o - 1)
    ys, xs = coords(oh), coords(ow)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, n - 1); x1 = np.minimum(x0 + 1, n - 1); fy = (ys - y0)[:, None, None]; fx = (xs - x0)[None, :, None]
    g = grid
    return ((g[y0][:, x0] * (1 - fx) + g[y0][:, x1] * fx) * (1 - fy) + (g[y1][:, x0] * (1 - fx) + g[y1][:, x1] * fx) * fy).astype(np.float32)

def qwen3vl(W, KV, img, normalized=False):
    """img: float32 [H, W, 3]; normalized=False applies (x - mean) / std as the preprocessor does."""
    P = KV["clip.vision.patch_size"]; E = KV["clip.vision.embedding_length"]; NH = KV["clip.vision.attention.head_count"]
    eps = KV.get("clip.vision.attention.layer_norm_epsilon", 1e-6); D = E // NH; m = KV.get("clip.vision.spatial_merge_size", 2)
    x = img if normalized else (img - np.array(KV["clip.vision.image_mean"], np.float32)) / np.array(KV["clip.vision.image_std"], np.float32)
    Hh, Ww = x.shape[0] // P, x.shape[1] // P
    patches = x.reshape(Hh, P, Ww, P, 3).transpose(0, 2, 4, 1, 3).reshape(Hh, Ww, 3 * P * P)
    pw = W["v.patch_embd.weight"].reshape(E, -1) + W["v.patch_embd.weight.1"].reshape(E, -1)
    h = patches @ pw.T + W["v.patch_embd.bias"]                       # [Hh, Ww, E]
    n = int(round(math.sqrt(W["v.position_embd.weight"].shape[0])))
    h = h + bilinear_ac(W["v.position_embd.weight"].reshape(n, n, E), Hh, Ww)
    # block order: (ty, tx, dy, dx)
    order = [(ty * m + dy, tx * m + dx) for ty in range(Hh // m) for tx in range(Ww // m) for dy in range(m) for dx in range(m)]
    ys = np.array([o[0] for o in order]); xs = np.array([o[1] for o in order])
    h = h[ys, xs]
    j = np.arange(D // 2); sec = D // 4
    inv = np.where(j < sec, 10000.0 ** (-(j * 2.0) / (D // 2)), 10000.0 ** (-((j - sec) * 2.0) / (D // 2)))
    ang = np.where(j[None] < sec, ys[:, None], xs[:, None]) * inv[None]
    c, s_ = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    def rope(t):  # t [NH, n, D]; pairs (j, j + D/2)
        a, b = t[..., :D // 2], t[..., D // 2:]
        return np.concatenate([a * c - b * s_, a * s_ + b * c], -1)
    nb = len({k.split(".")[2] for k in W if k.startswith("v.blk.")})
    for i in range(nb):
        p = f"v.blk.{i}."
        cur = ln(h, W[p + "ln1.weight"], W[p + "ln1.bias"], eps)
        qkv = Lin(W, p + "attn_qkv")(cur)
        q, k, v = (qkv[:, j * E:(j + 1) * E].reshape(-1, NH, D).transpose(1, 0, 2) for j in range(3))
        q, k = rope(q), rope(k)
        sc = q @ k.transpose(0, 2, 1) / math.sqrt(D); sc -= sc.max(-1, keepdims=True); a = np.exp(sc); a /= a.sum(-1, keepdims=True)
        h = h + Lin(W, p + "attn_out")((a @ v).transpose(1, 0, 2).reshape(-1, E))
        cur = ln(h, W[p + "ln2.weight"], W[p + "ln2.bias"], eps)
        h = h + Lin(W, p + "ffn_down")(gelu_tanh(Lin(W, p + "ffn_up")(cur)))
    h = ln(h, W["v.post_ln.weight"], W["v.post_ln.bias"], eps).reshape(-1, E * m * m)
    return Lin(W, "mm.2")(gelu_tanh(Lin(W, "mm.0")(h)))

def muse_glimmer(W, KV, img, normalized=False):
    P = KV["clip.vision.patch_size"]; E = KV["clip.vision.embedding_length"]; NH = KV["clip.vision.attention.head_count"]
    eps = KV.get("clip.vision.attention.layer_norm_epsilon", 1e-5); D = E // NH; m = KV.get("clip.vision.spatial_merge_size", 2)
    x = img if normalized else (img - np.array(KV["clip.vision.image_mean"], np.float32)) / np.array(KV["clip.vision.image_std"], np.float32)
    Hh, Ww = x.shape[0] // P, x.shape[1] // P
    patches = x.reshape(Hh, P, Ww, P, 3).transpose(0, 2, 4, 1, 3).reshape(Hh, Ww, 3 * P * P)
    h = patches @ W["v.patch_embd.weight"].reshape(E, -1).T
    if "v.patch_embd.bias" in W: h = h + W["v.patch_embd.bias"]
    n = int(round(math.sqrt(W["v.position_embd.weight"].shape[0])))
    pe = W["v.position_embd.weight"].reshape(n, n, E)
    h = h + (pe if (Hh, Ww) == (n, n) else bilinear_hp(pe, Hh, Ww))
    h = tr('after_posemb', h.reshape(Hh * Ww, E))
    ys = np.repeat(np.arange(Hh), Ww); xs = np.tile(np.arange(Ww), Hh)
    win = (ys // n) * 100000 + (xs // n); same = win[:, None] == win[None, :]
    h = tr("pre_ln", ln(h, W["v.pre_ln.weight"], W["v.pre_ln.bias"], eps))
    half = D // 2; i = np.arange(half // 2); inv = 10000.0 ** (-2.0 * i / half)
    def rope_norm(t, pos):  # adjacent pairs within t[..., :half]
        ang = (pos[:, None] * inv[None]); c, s_ = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        a, b = t[..., 0::2], t[..., 1::2]; out = np.empty_like(t); out[..., 0::2] = a * c - b * s_; out[..., 1::2] = a * s_ + b * c; return out
    def rope(t):
        return np.concatenate([rope_norm(t[..., :half], xs + 1.0), rope_norm(t[..., half:], ys + 1.0)], -1)
    nb = len({k.split(".")[2] for k in W if k.startswith("v.blk.")}); sf = 4
    for li in range(nb):
        p = f"v.blk.{li}."
        cur = ln(h, W[p + "ln1.weight"], W[p + "ln1.bias"], eps)
        q, k, v = (Lin(W, p + nm)(cur).reshape(-1, NH, D).transpose(1, 0, 2) for nm in ("attn_q", "attn_k", "attn_v"))
        tr(f"Qcur-{li}", q); q, k = rope(q), rope(k); tr(f"Qcur_pos-{li}", q)
        sc = q @ k.transpose(0, 2, 1) / math.sqrt(D)
        if not (li == nb - 1 or (li + 1) % sf == 0): sc = np.where(same[None], sc, -np.inf)
        sc -= sc.max(-1, keepdims=True); a = np.exp(sc); a /= a.sum(-1, keepdims=True)
        h = tr(f"ffn_inp-{li}", h + tr(f"attn_out-{li}", Lin(W, p + "attn_out")((a @ v).transpose(1, 0, 2).reshape(-1, E))))
        cur = tr(f"ffn_inp_normed-{li}", ln(h, W[p + "ln2.weight"], W[p + "ln2.bias"], eps))
        u = tr(f"ffn_up_b-{li}", Lin(W, p + "ffn_up")(cur)); g_ = tr(f"ffn_gelu_erf-{li}", gelu_erf(u))
        h = tr(f"layer_out-{li}", h + tr(f"ffn_out-{li}", Lin(W, p + "ffn_down")(g_)))
    h = ln(h, W["v.post_ln.weight"], W["v.post_ln.bias"], eps).reshape(Hh, Ww, E)
    # pixel shuffle: out[c * m*m + s] (channel-outer)
    g = tr("encoder_out", h.reshape(Hh // m, m, Ww // m, m, E).transpose(0, 2, 4, 1, 3).reshape(-1, E * m * m))
    g = gelu_erf(Lin(W, "mm.0")(g)); g = gelu_erf(Lin(W, "mm.1")(g)); return Lin(W, "mm.2")(g)

def bilinear_hp(grid, oh, ow):
    """ggml_interpolate bilinear without align_corners (half-pixel centres, clamped)."""
    n = grid.shape[0]
    def coords(o): return np.clip((np.arange(o) + 0.5) * n / o - 0.5, 0, n - 1)
    ys, xs = coords(oh), coords(ow)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int); y1 = np.minimum(y0 + 1, n - 1); x1 = np.minimum(x0 + 1, n - 1)
    fy = (ys - y0)[:, None, None]; fx = (xs - x0)[None, :, None]; g = grid
    return ((g[y0][:, x0] * (1 - fx) + g[y0][:, x1] * fx) * (1 - fy) + (g[y1][:, x0] * (1 - fx) + g[y1][:, x1] * fx) * fy).astype(np.float32)

ENCODERS = {"muse-glimmer": lambda W, KV, img: muse_glimmer(W, KV, img, normalized=True), "gemma4v": gemma4v, "qwen3vl_merger": lambda W, KV, img: qwen3vl(W, KV, img, normalized=True)}

if __name__ == "__main__":
    W, KV = load(sys.argv[1]); pt = KV.get("clip.vision.projector_type") or KV.get("clip.projector_type")
    out = ENCODERS[pt](W, KV, synth(sys.argv[2], int(sys.argv[3])))
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    print(pt, out.shape, "sum =", out.sum()); print(out[:4, :3], out[:4, -3:])
