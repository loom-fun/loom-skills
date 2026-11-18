"""Kernel-test cases read off the generated kernels themselves: per-row norms, the MoE router / combine / input kernels and the
Gemma 4 assistant drafter's kernels (draft/gen_assist.py). Each case is made only when the package's llm.loom defines the
kernel, and its sizes and constants (widths, eps, scales, RoPE step, head grouping) are parsed from that kernel's text, so a
case tests the kernel as the package generated it, against a float64 numpy reference of the operation it states.

cases(lib, save, rng, ks) -> [check.case text]."""
import re

import numpy as np

from ktest_spec import _Case, _def
import ktest_spec


def _const(text, name):
    m = re.search(r"^\s*%" + re.escape(name) + r" = (?:index|scalar)\.constant (-?[0-9.e+-]+) : \w+$", text, re.M)
    return float(m.group(1)) if m else None


def _view(text, var):
    """The static extents of the view a kernel makes of buffer VAR (dynamic [%x] extents omitted)."""
    m = re.search(r"buffer\.view %" + re.escape(var) + r"\[[^\]]*\] : buffer -> view<([^>]*)>", text)
    if not m:
        return None
    return [int(d) for d in m.group(1).split("x")[:-1] if d.isdigit()]


def _rms(v, eps):
    v = v.astype(np.float64)
    return v / np.sqrt((v * v).mean(-1, keepdims=True) + eps)


def post_add_norm_case(save, rng, k, scaled):
    """g_post_add_norm[_s]: x += rmsnorm(y) * w_post (post norms; else x += y), x *= the layer scale for _s; h = rmsnorm(x) * w_next."""
    H = int(_const(k, "count"))
    peps, eps = _const(k, "post_eps"), _const(k, "eps")
    if peps is None or eps is None or "%w_post_view" not in k:
        return None
    T = 3
    y = (rng.standard_normal((T, H)) * np.array([[1e-2], [1.0], [30.0]])).astype(np.float32)
    x = (rng.standard_normal((T, H)) * np.array([[1.0], [1e2], [1e-3]])).astype(np.float32)
    wp = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
    wn = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
    ls = np.array([0.37], np.float32)
    post = "vector.mulf %yv, %yinvs" in k   # gen_llm POST_NORM: the branch output is normalized before the add, else added as it is
    xe = x + (_rms(y, peps) * wp if post else y)
    if scaled:
        xe = xe * ls[0]
    he = _rms(xe, eps) * wn
    name = "g_post_add_norm_s" if scaled else "g_post_add_norm"
    c = _Case("pan_" + ("s" if scaled else "plain"), save)
    args = [c.lit("t", T), c.inp("y", y), c.inp("wp", wp), c.out("x", xe.astype(np.float32), init=x, close="1.0e-5"), c.inp("wn", wn),
            c.out("h", he.astype(np.float32), close="1.0e-4")]
    if scaled:
        args.append(c.inp("ls", ls))
    c.launch(name, ["%t"], args)
    return c.text()


def moe_inputs_case(save, rng, k):
    """g_moe_inputs: h2 = rmsnorm(x) * w_pre; hr = rmsnorm(x) * router scale * w_router."""
    H, eps, rs = int(_const(k, "count")), _const(k, "xi_eps"), _const(k, "rs")
    if None in (eps, rs):
        return None
    T = 3
    x = (rng.standard_normal((T, H)) * np.array([[1e-2], [1.0], [20.0]])).astype(np.float32)
    wpre = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
    wr = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
    xn = _rms(x, eps)
    c = _Case("moe_inputs", save)
    args = [c.lit("t", T), c.inp("x", x), c.inp("wpre", wpre), c.inp("wr", wr),
            c.out("h2", (xn * wpre).astype(np.float32), close="1.0e-5"), c.out("hr", (xn * rs * wr).astype(np.float32), close="1.0e-5")]
    c.launch("g_moe_inputs", ["%t"], args)
    return c.text()


def moe_route_case(save, rng, k):
    """g_moe_route (softmax gating, no bias): logits = x . w_e; probabilities softmax over the experts; the top KU by probability
    (ties: the lower expert id) in order, weights = their probabilities over the sum of the selected ones."""
    if "%lden" not in k or re.search(r"buffer\.view %b\[", k):
        return None
    HG, E, KU = int(_const(k, "cHG")), int(_const(k, "cE")), int(_const(k, "cKU"))
    H = HG * 32
    T = 3
    x = rng.standard_normal((T, H)).astype(np.float32)
    w = (rng.standard_normal((E, H)) / np.sqrt(H)).astype(np.float32)
    lg = x.astype(np.float64) @ w.astype(np.float64).T
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ids = np.argsort(-p, axis=1, kind="stable")[:, :KU].astype(np.int32)
    sel = np.take_along_axis(p, ids, 1)
    wts = (sel / np.maximum(sel.sum(-1, keepdims=True), 6.103515625e-05)).astype(np.float32)
    c = _Case("moe_route", save)
    args = [c.lit("t", T), c.inp("x", x), c.inp("w", w), c.inp("b", np.zeros(E, np.float32)), c.out("ids", ids), c.out("wts", wts, close="1.0e-5")]
    c.launch("g_moe_route", ["%t"], args)
    return c.text()


def moe_wscale_case(save, rng, k):
    """g_moe_wscale: each routed weight times its expert's down scale (ids clamped to the experts)."""
    KU = int(_const(k, "ku"))
    E = _view(k, "scale")[0]
    T = 3
    ids = rng.integers(0, E, (T, KU)).astype(np.int32)
    ids[0, 0], ids[1, 1] = -1, E + 5
    wts = rng.random((T, KU)).astype(np.float32)
    sc = (0.5 + rng.random(E)).astype(np.float32)
    e = (wts * sc[np.clip(ids, 0, E - 1)]).astype(np.float32)
    c = _Case("moe_wscale", save)
    c.launch("g_moe_wscale", ["%t"], [c.lit("t", T), c.inp("ids", ids), c.out("wts", e, init=wts, close="1.0e-6"), c.inp("sc", sc)])
    return c.text()


def moe_combine_case(save, rng, k):
    """g_moe_combine: y = sum_k wts[k] * ys[k] (the experts' outputs)."""
    KU, W = _view(k, "ys")
    T = 3
    ys = rng.standard_normal((T, KU, W)).astype(np.float32)
    wts = rng.random((T, KU)).astype(np.float32)
    y = (wts[:, :, None].astype(np.float64) * ys).sum(1).astype(np.float32)
    c = _Case("moe_combine", save)
    c.launch("g_moe_combine", ["%t"], [c.lit("t", T), c.inp("ys", ys), c.inp("wts", wts), c.out("y", y, close="1.0e-5")])
    return c.text()


def as_argmax_case(save, rng, k):
    V = int(_const(k, "cv"))
    T = 2
    lg = rng.standard_normal((T, V)).astype(np.float32)
    lg[1, 7] = lg[1, V - 3] = lg[1].max() + 1.0   # a tie: the lower index
    c = _Case("as_argmax", save)
    c.launch("g_as_argmax", ["%t"], [c.lit("t", T), c.inp("lg", lg), c.out("o", lg.argmax(1).astype(np.int32))])
    return c.text()


def as_embed_case(save, rng, k):
    """g_as_embed: the token's Q4_0 row of the table (ids clamped to the vocabulary), times the embedding scale."""
    if "@g_q4_0_values" not in k:
        return None
    G, RB, scale = int(_const(k, "groups")), int(_const(k, "row_bytes")), _const(k, "scale_s")
    if RB != G * 18:
        return None
    N = 12
    d = (rng.random((N, G)) * 0.02 + 0.001).astype(np.float16)
    q = rng.integers(0, 256, (N, G, 16)).astype(np.uint8)
    blocks = np.concatenate([d.view(np.uint8).reshape(N, G, 2), q], -1)
    table = blocks.reshape(-1).view(np.int8)
    lo, hi = (q & 15).astype(np.float64) - 8, (q >> 4).astype(np.float64) - 8
    vals = np.concatenate([lo, hi], -1) * d.astype(np.float64)[..., None]   # [N, G, 32]: low nibbles are columns 0-15
    ids = np.array([3, -2, 11, 0], np.int32)
    x = (vals[np.clip(ids, 0, None)].reshape(len(ids), G * 32) * scale).astype(np.float32)
    c = _Case("as_embed", save)
    c.launch("g_as_embed", ["%t"], [c.lit("t", len(ids)), c.inp("ids", ids), c.inp("tab", table), c.out("x", x, close="1.0e-5")])
    return c.text()


def _slot(ah, ga, gt):
    return (ah // ga) * gt + ah % ga


def as_gather_case(save, rng, k, name):
    Q, D = _view(k, "ot")
    ga, gt = int(_const(k, "ga")), int(_const(k, "gt"))
    T = 2
    ot = rng.standard_normal((T, Q, D)).astype(np.float32)
    o = np.stack([ot[:, _slot(h, ga, gt)] for h in range(Q)], 1)
    c = _Case("as_" + name[5:], save)
    c.launch(name, ["%t"], [c.lit("t", T), c.inp("ot", ot), c.out("o", o)])
    return c.text()


def as_visible_case(save, rng, k):
    T, rnd = 5, 1
    info = np.array([[2, 0, 1, 0], [2, 1, 1, 0], [2, 2, 1, 0], [3, 9, 0, 1], [3, 7, 1, 1]], np.int32)
    vis = info[:, 1] - (rnd + 1)
    out = np.stack([info[:, 0], np.maximum(vis, 0), ((info[:, 2] != 0) & (vis >= 0)).astype(np.int32), info[:, 3]], 1).astype(np.int32)
    c = _Case("as_visible", save)
    c.launch("g_as_visible", ["%t"], [c.lit("t", T), c.lit("r", rnd), c.inp("info", info), c.out("o", out)])
    return c.text()


def as_norm_cases(save, rng, k):
    W, eps = int(_const(k, "width")), _const(k, "eps")
    T = 3
    a = (rng.standard_normal((T, W)) * np.array([[1e-2], [1.0], [20.0]])).astype(np.float32)
    w = (1 + 0.1 * rng.standard_normal(W)).astype(np.float32)
    b = rng.standard_normal((T, W)).astype(np.float32)
    s = np.array([0.6], np.float32)
    out = []
    for add, scaled in ((1, 1), (0, 0)):
        e = _rms(a, eps) * w + (b if add else 0)
        e = e * (s[0] if scaled else 1.0)
        c = _Case(f"as_norm_{add}{scaled}", save)
        c.lines.append(f"  %add = check.literal value({add}) : i32")
        c.lines.append(f"  %scaled = check.literal value({scaled}) : i32")
        args = [c.lit("t", T), ("%add", "i32"), ("%scaled", "i32"), c.inp("a", a), c.inp("w", w), c.inp("b", b), c.inp("s", s),
                c.out("o", e.astype(np.float32), close="1.0e-5")]
        c.launch("g_as_norm", ["%t"], args)
        out.append(c.text())
    return out


def as_qrope_case(save, rng, k, name):
    """g_as_qrope_d<D>: per-head rmsnorm(q) * w, RoPE at the token's position (pair i rotates by pos * exp(i * step), pairs past
    the rotated count unrotated), stored at the head's grouped slot."""
    Q, D = _view(k, "q")
    eps, step, npairs = _const(k, "eps"), _const(k, "step"), int(_const(k, "npairs"))
    ga, gt = int(_const(k, "ga")), int(_const(k, "gt"))
    half = D // 2
    T = 3
    q = rng.standard_normal((T, Q, D)).astype(np.float32)
    w = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    pos = np.array([0, 5, 37], np.int32)
    info = np.zeros((T, 4), np.int32)
    info[:, 1] = pos
    n = _rms(q, eps) * w
    a, b = n[..., :half], n[..., half:]
    i = np.arange(half)
    ang = pos[:, None].astype(np.float64) * np.exp(i * step)[None, :]
    ang = np.where(i[None, :] < npairs, ang, 0.0)[:, None, :]
    ra, rb = a * np.cos(ang) - b * np.sin(ang), b * np.cos(ang) + a * np.sin(ang)
    r = np.concatenate([ra, rb], -1)
    qt = np.zeros_like(r)
    for h in range(Q):
        qt[:, _slot(h, ga, gt)] = r[:, h]
    c = _Case("as_" + name[5:], save)
    c.launch(name, ["%t"], [c.lit("t", T), c.inp("q", q), c.inp("w", w), c.inp("info", info), c.out("qt", qt.astype(np.float32), close="2.0e-3")])
    return c.text()


def cases(lib, save, rng, ks=""):
    if not lib:
        return []
    ktest_spec.KS = ks
    out = []

    def add(x):
        if isinstance(x, list):
            out.extend(x)
        elif x:
            out.append(x)
    for name, fn in (("g_post_add_norm", lambda k: post_add_norm_case(save, rng, k, False)),
                     ("g_post_add_norm_s", lambda k: post_add_norm_case(save, rng, k, True)),
                     ("g_moe_inputs", lambda k: moe_inputs_case(save, rng, k)),
                     ("g_moe_route", lambda k: moe_route_case(save, rng, k)),
                     ("g_moe_wscale", lambda k: moe_wscale_case(save, rng, k)),
                     ("g_moe_combine", lambda k: moe_combine_case(save, rng, k)),
                     ("g_as_argmax", lambda k: as_argmax_case(save, rng, k)),
                     ("g_as_embed", lambda k: as_embed_case(save, rng, k)),
                     ("g_as_visible", lambda k: as_visible_case(save, rng, k)),
                     ("g_as_norm", lambda k: as_norm_cases(save, rng, k))):
        k = _def(lib, name)
        if k:
            add(fn(k))
    for name in sorted(set(re.findall(r'export\("(g_as_gather_d\d+)"\)', lib))):
        add(as_gather_case(save, rng, _def(lib, name), name))
    for name in sorted(set(re.findall(r'export\("(g_as_qrope_d\d+)"\)', lib))):
        add(as_qrope_case(save, rng, _def(lib, name), name))
    return out
