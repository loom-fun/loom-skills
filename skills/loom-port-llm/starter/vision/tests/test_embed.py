import numpy as np, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import gen_vision as G, vk_common as V, ref_vit as R
d = __import__("os").environ.get("VISION_TEST_DIR", __import__("os").path.join(__import__("tempfile").gettempdir(), "loom_vision_tests")); __import__("os").makedirs(d, exist_ok=True); rng = np.random.default_rng(2); src = []; cases = []
def order_idx(order, gh, gw):
    if order == "raster": return np.repeat(np.arange(gh), gw), np.tile(np.arange(gw), gh)
    o = [(ty * 2 + dy, tx * 2 + dx) for ty in range(gh // 2) for tx in range(gw // 2) for dy in range(2) for dx in range(2)]
    return np.array([a for a, b in o]), np.array([b for a, b in o])
for P, E, fmt, order, dual, bias in [(16, 768, "f16", "raster", False, False), (16, 768, "f16", "block2", True, True), (14, 1536, "f32", "raster", False, False)]:
    nm, s = G.patch_embed(P, E, fmt, order, dual, bias); src.append(s)
    gh, gw = 4, 6; H, W = gh * P, gw * P; img = rng.standard_normal((H, W, 3)).astype(np.float32)
    w = (rng.standard_normal((E, 3, P, P)) * 0.05).astype(np.float32); w2 = (rng.standard_normal((E, 3, P, P)) * 0.05).astype(np.float32); b = rng.standard_normal(E).astype(np.float32)
    if fmt == "f16": w = w.astype(np.float16).astype(np.float32); w2 = w2.astype(np.float16).astype(np.float32)
    raw = lambda a: (a.astype(np.float16) if fmt == "f16" else a).reshape(-1).view(np.int8)
    patches = img.reshape(gh, P, gw, P, 3).transpose(0, 2, 4, 1, 3).reshape(gh, gw, -1)
    ys, xs = order_idx(order, gh, gw); ref = patches[ys, xs] @ (w + (w2 if dual else 0)).reshape(E, -1).T + (b if bias else 0)
    c = V.Case(f"t_{nm}", d); n = c.lit(gh * gw); g = c.lit(gw); iw = c.lit(W)
    args = [n, g, iw, c.arr(img.reshape(-1)), c.arr(raw(w))] + ([c.arr(raw(w2))] if dual else []) + ([c.arr(b)] if bias else [])
    y = c.out((gh * gw, E)); c.launch(nm, [n], args + [y]); c.expect(y, ref, 1e-3, 1e-3); cases.append(c)
nm, s = G.pixels([0.5] * 3, [0.5] * 3); src.append(s)
u8 = rng.integers(0, 256, 300).astype(np.uint8); c = V.Case(f"t_{nm}", d); n = c.lit(300); y = c.out((300,))
c.launch(nm, [n], [n, c.arr(u8), y]); c.expect(y, (u8 / 255.0 - 0.5) / 0.5, 1e-5, 1e-5); cases.append(c)
nm, s = G.pixels([0.0] * 3, [1.0] * 3, True); src.append(s)
c = V.Case(f"t_{nm}", d); n = c.lit(300); y = c.out((300,)); c.launch(nm, [n], [n, c.arr(u8), y]); c.expect(y, u8 / 255.0 * 2 - 1, 1e-5, 1e-5); cases.append(c)
E = 768; npos = 64; nm, s = G.pos_xy(E, npos); src.append(s)
gh, gw = 5, 7; tab = rng.standard_normal((2, npos, E)).astype(np.float32); y0 = rng.standard_normal((gh * gw, E)).astype(np.float32)
ys, xs = order_idx("raster", gh, gw); c = V.Case(f"t_{nm}", d); n = c.lit(gh * gw); g = c.lit(gw); ya = c.arr(y0)
c.launch(nm, [n], [n, g, c.arr(tab), ya]); c.expect(ya, y0 + tab[0][xs] + tab[1][ys], 1e-5, 1e-5); cases.append(c)
for order, ac, nn, gh, gw in [("block2", True, 48, 6, 10), ("raster", False, 32, 36, 30), ("raster", False, 32, 32, 32)]:
    E = 64; nm, s = G.pos_interp(E, nn, order, ac); src.append(s)
    grid = rng.standard_normal((nn, nn, E)).astype(np.float32); y0 = rng.standard_normal((gh * gw, E)).astype(np.float32)
    full = (R.bilinear_ac if ac else R.bilinear_hp)(grid, gh, gw); ys, xs = order_idx(order, gh, gw)
    c = V.Case(f"t_{nm}_{gh}", d); n = c.lit(gh * gw); g = c.lit(gw); h = c.lit(gh); ya = c.arr(y0)
    c.launch(nm, [n], [n, g, h, c.arr(grid), ya]); c.expect(ya, y0 + full[ys, xs], 1e-4, 1e-4); cases.append(c)
V.run("\n".join(dict.fromkeys(src)), cases, d)
