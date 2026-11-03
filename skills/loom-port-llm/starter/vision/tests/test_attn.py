import numpy as np, sys, math
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import gen_vision as G, vk_common as V, ref_vit as R
from test_embed_util import order_idx
d = __import__("os").environ.get("VISION_TEST_DIR", __import__("os").path.join(__import__("tempfile").gettempdir(), "loom_vision_tests")); __import__("os").makedirs(d, exist_ok=True); rng = np.random.default_rng(3); src = []; cases = []
def rope_ref(mode, x, ys, xs, D, NH, O):
    theta, off = G.ROPE_MODES[mode]; out = x.copy()
    for (lo, hi, ax, i, nd) in G.rope_pairs(mode, D):
        pos = (ys if ax == "y" else xs) + off; ang = pos * theta ** (-2.0 * i / nd)
        for h in range(NH):
            a = x[:, O + h * D + lo]; b = x[:, O + h * D + hi]
            out[:, O + h * D + lo] = a * np.cos(ang) - b * np.sin(ang); out[:, O + h * D + hi] = a * np.sin(ang) + b * np.cos(ang)
    return out
for mode, D, NH, S, O, order in [("gemma_halves", 64, 12, 768, 0, "raster"), ("qwen_vision", 64, 12, 2304, 768, "block2"), ("normal_halves", 96, 16, 1536, 0, "raster")]:
    nm, s = G.rope(mode, D, NH, S, O, order); src.append(s)
    gh, gw = 6, 8; n = gh * gw; x = rng.standard_normal((n, S)).astype(np.float32); ys, xs = order_idx(order, gh, gw)
    # check the gemma/qwen/glimmer ref functions agree with the pair table
    ref = rope_ref(mode, x, ys, xs, D, NH, O)
    c = V.Case(f"t_{nm}", d); tn = c.lit(n); g = c.lit(gw); xa = c.arr(x); c.launch(nm, [tn], [tn, g, xa]); c.expect(xa, ref, 1e-4, 1e-4); cases.append(c)
# cross-check the pair tables against ref_vit's rope implementations
gh, gw = 6, 8; ys, xs = order_idx("raster", gh, gw); D = 64
t = rng.standard_normal((2, gh * gw, D)).astype(np.float32)
a = t.copy(); a[..., :32] = R.rope_neox(t[..., :32], xs, 32, 100.0); a[..., 32:] = R.rope_neox(t[..., 32:], ys, 32, 100.0)
b = rope_ref("gemma_halves", t.reshape(2 * gh * gw, D) if False else t[0], ys, xs, D, 1, 0)
print("gemma pair table vs ref_vit:", np.abs(a[0] - b).max())
def attn_ref(q, k, v, scale, mask=None):
    s = q @ k.T * scale
    if mask is not None: s = np.where(mask, s, -np.inf)
    s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True); return p @ v
for D, NH, S, offs, scale, win, gh, gw in [(64, 12, 768, (0, 0, 0), 1.0, 0, 7, 9), (64, 12, 2304, (0, 768, 1536), 0.125, 0, 6, 8), (96, 4, 384, (0, 0, 0), 96 ** -0.5, 4, 10, 9)]:
    sep = offs == (0, 0, 0); nm, s = G.attention(D, NH, S, *offs, scale, window=win); src.append(s)
    n = gh * gw; E = D * NH
    if sep: q, k, v = (rng.standard_normal((n, S)).astype(np.float32) for _ in range(3))
    else: q = k = v = rng.standard_normal((n, S)).astype(np.float32)
    ys, xs = order_idx("raster", gh, gw); mask = None
    if win: wid = (ys // win) * 1000 + xs // win; mask = wid[:, None] == wid[None, :]
    ref = np.concatenate([attn_ref(q[:, offs[0] + h * D: offs[0] + (h + 1) * D], k[:, offs[1] + h * D: offs[1] + (h + 1) * D], v[:, offs[2] + h * D: offs[2] + (h + 1) * D], scale, mask) for h in range(NH)], 1)
    c = V.Case(f"t_{nm}", d); tn = c.lit(n); g = c.lit(gw); qa = c.arr(q)
    ka, va = (c.arr(k), c.arr(v)) if sep else (qa, qa); o = c.out((n, E))
    c.launch(nm, [tn], [tn, g, qa, ka, va, o]); c.expect(o, ref, 1e-4, 1e-3); cases.append(c)
E = 128; nm, s = G.avgpool(E, 3, math.sqrt(E)); src.append(s); gh, gw = 6, 9; x = rng.standard_normal((gh * gw, E)).astype(np.float32)
ref = x.reshape(2, 3, 3, 3, E).mean((1, 3)).reshape(-1, E) * math.sqrt(E)
c = V.Case(f"t_{nm}", d); on = c.lit(6); g = c.lit(gw); tn = c.lit(gh * gw); y = c.out((6, E)); c.launch(nm, [on], [on, g, tn, c.arr(x), y]); c.expect(y, ref, 1e-4, 1e-4); cases.append(c)
nm, s = G.shuffle(E, 2); src.append(s); gh, gw = 4, 6; x = rng.standard_normal((gh * gw, E)).astype(np.float32)
ref = x.reshape(2, 2, 3, 2, E).transpose(0, 2, 4, 1, 3).reshape(6, E * 4)
c = V.Case(f"t_{nm}", d); on = c.lit(6); g = c.lit(gw); tn = c.lit(gh * gw); y = c.out((6, E * 4)); c.launch(nm, [on], [on, g, tn, c.arr(x), y]); c.expect(y, ref, 1e-6, 1e-6); cases.append(c)
V.run("\n".join(src), cases, d)
