#!/usr/bin/env python3
"""Generate the mla_xpf kbench suite: MLA (deepseek2, JoyAI-LLM-Flash) PREFILL attention of one 4096-token chunk (or a shorter
whole prompt), ABSORBED vs EXPANDED.
base = the absorbed path the deepseek2 example builds today (base.loom, extracted from the generated program: g_mla_kbm (q_abs =
k_b q_nope), g_attention_flash2 (576-dim keys / 512-dim values from the latent row), g_mla_vbm (vo = v_b o)); the other variants =
gen_mla_x.py (expanded: g_mla_xlist once, then per layer g_mla_xdeq + g_mla_xexp + g_mla_xflash; the direct-grid _d twins, counts
from the host) with the environment of VARIANTS.

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]
Timing (bench*.loom): one row's chunk at the end of a 1k / 4k / 8k / 32k / 131k context (layers over distinct KV planes), random
latent rows and valid random IQ4 weights.
Correctness (test.loom): every variant's vo against a float64 numpy reference of the expanded form (dequantized weights), with
permuted pages, a second row too short to expand (its tiles must come back in tiles[]), the expected work lists; the base path is
checked against the same reference (it validates the reference)."""
import json, os, re, shutil, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "../../../.."))
HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
QH, D, PB, KL, QB, VD = 32, 576, 1152, 512, 192, 128
XCAP = 131072
VARIANTS = {"x": {}, "xg": {"MLA_XF": "global"},   # x: LDS-staged flash (default); xg: fragments straight from the scratch
            "t": {"MLA_XCAP": "4096"}, "tg": {"MLA_XCAP": "4096", "MLA_XF": "global"}}   # small-scratch twins for the staged tests


def set_xcap(variant):
    global XCAP
    XCAP = int(next((VARIANTS[x]["MLA_XCAP"] for x in variant.split("+") if x in VARIANTS and "MLA_XCAP" in VARIANTS[x]), 131072))
CASES = [   # tag, context end (chunk = last t positions), tokens, layers
    ("1k", 1024, 1024, 8), ("4k", 4096, 4096, 4), ("8k", 8192, 4096, 4), ("32k", 32768, 4096, 2), ("131k", 131072, 4096, 1)]
OUT = None
_KIT = None
KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float64)


def kit():
    global _KIT
    if _KIT is None:
        _KIT = tempfile.mkdtemp() + "/kit"
        subprocess.run([os.path.join(ROOT, "skills/loom-port-llm/reference/apply_example.sh"), "deepseek2", _KIT], check=True, stdout=subprocess.DEVNULL)
    return _KIT


def cfg():
    return json.load(open(os.path.join(ROOT, "skills/loom-port-llm/reference/deepseek2-example/cfg.json")))


def base_text():
    p = os.path.join(HERE, "base.loom")
    return open(p if os.path.exists(p) else os.path.join(OUT, "base.loom")).read()


def x_text(variant):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLA_X")}
    env.update({k: v for x in variant.split("+") for k, v in VARIANTS[x].items()})
    code = (f"import json,sys; sys.path.insert(0, {kit()!r}); import gen_mla_x; c = json.load(open({os.path.join(ROOT, 'skills/loom-port-llm/reference/deepseek2-example/cfg.json')!r})); "
            "print(gen_mla_x.kernels(c, 39, 'iq4_nl', 'iq4_xs', 72, 272, 18, 136, tests=True))")
    return subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True, text=True).stdout


def kernels(variant, tag):
    """Kernel text of VARIANT (helpers included) with every symbol suffixed by TAG (one file holds several cases)."""
    b = base_text()
    t = b if variant == "base" else b + "\n" + x_text(variant)
    s = tag.replace("_", "")
    for f in set(re.findall(r"func\.def inline @(\w+)\(", t)) | set(re.findall(r"global\.rodata\.def @(\w+) ", t)):
        t = re.sub(rf"@{f}\b", f"@{f}_{s}", t)
    for k in set(re.findall(r'kernel\.def export\("(\w+)"\)', t)):
        t = t.replace(f'export("{k}") @{k}(', f'export("k_{k}_{s}") @k_{k}_{s}(')
    return t


def save(name, a):
    np.save(os.path.join(OUT, name + ".npy"), a)


# ---- IQ4 weights: valid random blocks and their numpy dequantization ----
def iq4_nl_rand(rng, nblocks, d):
    b = np.zeros((nblocks, 18), np.uint8)
    b[:, 0:2] = np.full(nblocks, d, np.float16).view(np.uint8).reshape(nblocks, 2)
    b[:, 2:] = rng.integers(0, 256, (nblocks, 16), dtype=np.uint8)
    return b


def iq4_nl_deq(b):
    d = b[:, 0:2].copy().view(np.float16).astype(np.float64).reshape(-1)
    qs = b[:, 2:]
    v = np.concatenate([KV[qs & 15], KV[qs >> 4]], 1)
    return v * d[:, None]   # (nblocks, 32)


def iq4_xs_rand(rng, nblocks, d):
    b = np.zeros((nblocks, 136), np.uint8)
    b[:, 0:2] = np.full(nblocks, d, np.float16).view(np.uint8).reshape(nblocks, 2)
    b[:, 2:] = rng.integers(0, 256, (nblocks, 134), dtype=np.uint8)
    return b


def iq4_xs_deq(b):
    n = len(b)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float64).reshape(-1)
    sh = b[:, 2].astype(np.int64) | (b[:, 3].astype(np.int64) << 8)
    sl = b[:, 4:8].astype(np.int64)
    out = np.zeros((n, 256))
    for ib in range(8):
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32)
        qs = b[:, 8 + 16 * ib: 24 + 16 * ib]
        out[:, 32 * ib:32 * ib + 16] = KV[qs & 15] * dl[:, None]
        out[:, 32 * ib + 16:32 * ib + 32] = KV[qs >> 4] * dl[:, None]
    return out


def weights(rng):
    """k_b (32 heads x 512 latent rows x 128 dims, IQ4_NL) and v_b (32 x 128 dims x 512 latent, IQ4_XS) bytes + dequantized
    (QH, KL, 128) / (QH, 128, KL)."""
    kb = iq4_nl_rand(rng, QH * KL * 4, 0.002)
    vb = iq4_xs_rand(rng, QH * VD * 2, 0.0002)
    return (kb.reshape(-1).view(np.int8), vb.reshape(-1).view(np.int8),
            iq4_nl_deq(kb).reshape(QH, KL, 128), iq4_xs_deq(vb).reshape(QH, VD, KL))


def origins(maps):
    R = len(maps)
    o = [np.zeros(10 * R, np.int32)]
    off = 40 * R
    for r, m in enumerate(maps):
        o[0][10 * r + 8] = off
        o.append(np.asarray(m, np.int32)); off += 4 * len(m)
    return np.concatenate(o)


def tile_list(t, info):
    """g_attention_fallback_list's whole-tile list for runs that start at a multiple of 16 (tiles[0] = count, then the first
    token of each tile; the partial list is not used here)."""
    tl = np.zeros(t, np.int32)
    n = 0
    for i in range(0, t, 16):
        r, p, lv = info[i, 0], info[i, 1], info[i, 2]
        if p % 16 == 0 and all(info[i + k, 0] == r and info[i + k, 1] == p + k and info[i + k, 2] for k in range(16)):
            tl[1 + n] = i; n += 1
    tl[0] = n
    return tl


def xlists(info, tiles, xmin=16):
    """numpy mirror of g_mla_xlist: (tiles after, xq, xb, nq, nb)."""
    n = tiles[0]
    segs, cur = [], []
    for i in range(n):
        tok = tiles[1 + i]
        r, p = info[tok, 0], info[tok, 1]
        if cur and r == cur[-1][1] and p == cur[-1][2] + 16 and tok == cur[-1][0] + 16:
            cur.append((tok, r, p))
        else:
            if cur: segs.append(cur)
            cur = [(tok, r, p)]
    if cur: segs.append(cur)
    out = tiles.copy(); xq, xb = [], []
    base = nc = 0
    for sg in segs:
        nk = sg[-1][2] + 16
        nk64 = (nk + 63) // 64 * 64
        if base + nk64 <= XCAP and len(sg) >= xmin:
            for q in range(0, len(sg), 4):
                xq.append((sg[q][0], min(4, len(sg) - q), base, sg[0][1]))
            for k0 in range(0, nk64, 64):
                xb.append((sg[0][1], k0, base + k0, nk))
            base += nk64
        else:
            for tok, _, _ in sg:
                out[1 + nc] = tok; nc += 1
    out[0] = nc
    return out, np.array(xq, np.int32).reshape(-1, 4), np.array(xb, np.int32).reshape(-1, 4)


def make_inputs(pfx, rng, rows, toks, layers, ctx, keys_of=None, garbage=1000.0):
    """rows: key count per row (positions 0..n-1 written in every layer); toks: (row, position) per token, live. Arena pages
    permuted; positions past a row's keys hold finite garbage."""
    blocks = ctx // 16
    nb = [(n + 15) // 16 for n in rows]
    pool_blocks = sum(nb) + 2
    pool = pool_blocks * 16
    pages = rng.permutation(pool_blocks)
    arena = np.full((layers, pool, D), garbage, np.float16)
    lat, maps, used = [], [], 0
    for n, b in zip(rows, nb):
        m = np.zeros(blocks, np.int32); m[:b] = pages[used:used + b]; used += b; maps.append(m)
        kk = rng.uniform(-1, 1, (layers, n, D)).astype(np.float16)
        lat.append(kk)
        for j in range(b):
            seg = kk[:, 16 * j:16 * j + 16]
            arena[:, m[j] * 16:m[j] * 16 + seg.shape[1]] = seg
    t = len(toks)
    info = np.zeros((t, 4), np.int32)
    for i, (r, p) in enumerate(toks): info[i] = (r, p, 1, 0)
    qb = rng.standard_normal((t, QH * QB)).astype(np.float32)
    q = np.zeros((t, QH * D), np.float32)
    q.reshape(t, QH, D)[:, :, KL:] = rng.standard_normal((t, QH, 64)).astype(np.float32)
    save(f"{pfx}_arena", arena.reshape(-1).view(np.int8)); save(f"{pfx}_orig", origins(maps)); save(f"{pfx}_info", info)
    save(f"{pfx}_qb", qb); save(f"{pfx}_q", q)
    return dict(t=t, info=info, qb=qb, q=q, lat=lat, pool=pool, alen=layers * pool * PB, olen=len(origins(maps)), ctx=ctx)


def reference(inp, kbd, vbd, layer):
    """vo (t, QH*128) of the expanded form in float64 for every live token (all tokens here)."""
    m = cfg()["mla"]
    t, info = inp["t"], inp["info"]
    qb = inp["qb"].reshape(t, QH, QB).astype(np.float64)
    qpe = inp["q"].reshape(t, QH, D)[:, :, KL:].astype(np.float64)
    vo = np.zeros((t, QH, VD))
    for r, lat in enumerate(inp["lat"]):
        sel = np.nonzero(info[:, 0] == r)[0]
        if not len(sel): continue
        c = lat[layer].astype(np.float64)                      # (n, 576)
        kn = c[:, :KL] @ kbd                                   # (QH, n, 128)
        vv = c[:, :KL] @ vbd.transpose(0, 2, 1)                # (QH, n, 128)
        pos = info[sel, 1]
        n = int(pos.max()) + 1
        mask = np.arange(n)[None, :] <= pos[:, None]           # (s, n)
        spe = qpe[sel].transpose(1, 0, 2) @ c[:n, KL:].T       # (QH, s, n)
        for h in range(QH):
            s = (qb[sel, h, :128] @ kn[h, :n].T + spe[h]) * m["scale"]
            s = np.where(mask, s, -np.inf)
            e = np.exp(s - s.max(-1, keepdims=True)); e /= e.sum(-1, keepdims=True)
            vo[sel, h] = e @ vv[h, :n]
    return vo.reshape(t, QH * VD)


def T(n, ty):
    return f"tensor<{n}x{ty}>"


def decls(pfx, inp, x=True):
    t = inp["t"]
    s = [f"""  %t = check.literal value({t}) : index
  %c1x = check.literal value(1) : index
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({inp["ctx"]}) : index
  %pool = check.literal value({inp["pool"]}) : index
  %qb = check.file.read.npy path("{pfx}_qb.npy") : tensor<{t}x{QH * QB}xf32>
  %q = check.file.read.npy path("{pfx}_q.npy") : tensor<{t}x{QH * D}xf32>
  %info = check.file.read.npy path("{pfx}_info.npy") : tensor<{t}x4xi32>
  %tiles = check.file.read.npy path("{pfx}_tiles.npy") : tensor<{t}xi32>
  %orig = check.file.read.npy path("{pfx}_orig.npy") : {T(inp["olen"], "i32")}
  %arena = check.file.read.npy path("{pfx}_arena.npy") : {T(inp["alen"], "i8")}
  %kb = check.file.read.npy path("w_kb.npy") : {T(QH * KL * 72, "i8")}
  %vb = check.file.read.npy path("w_vb.npy") : {T(QH * VD * 272, "i8")}
  %vo = check.generate.fill value(0.0) : tensor<{t}x{QH * VD}xf32>"""]
    if x:
        s.append(f"""  %xq = check.generate.fill value(0) : tensor<512x4xi32>
  %xb = check.generate.fill value(0) : tensor<{XCAP // 64}x4xi32>
  %xcnt = check.generate.fill value(0) : tensor<64xi32>
  %xskip = check.generate.fill value(0) : tensor<128xi32>
  %xw = check.generate.fill value(0.0) : tensor<{QH * 256 * 512}xf16>
  %xk = check.generate.fill value(0.0) : tensor<{QH * XCAP * 128}xf16>
  %xv = check.generate.fill value(0.0) : tensor<{QH * XCAP * 128}xf16>
  %xpe = check.generate.fill value(0.0) : tensor<{XCAP * 64}xf16>""")
    else:
        s.append(f"""  %o = check.generate.fill value(0.0) : tensor<{t}x{QH * D}xf32>""")
    return "\n".join(s) + "\n"


def launches(variant, s, inp, layers, nq, nb, layer0=0):
    t = inp["t"]
    TQB, TQ, TI, TT, TO, TA = (f"tensor<{t}x{QH * QB}xf32>", f"tensor<{t}x{QH * D}xf32>", f"tensor<{t}x4xi32>", f"tensor<{t}xi32>",
                               T(inp["olen"], "i32"), T(inp["alen"], "i8"))
    TKB, TVB, TVO = T(QH * KL * 72, "i8"), T(QH * VD * 272, "i8"), f"tensor<{t}x{QH * VD}xf32>"
    L = []
    if variant == "base":
        for l in range(layer0, layer0 + layers):
            L.append(f"""  %l{l} = check.literal value({l}) : index
  kernel.launch @k_g_mla_kbm_iq4_nl_{s}[%t](%t, %qb, %kb, %q) : [index](index, {TQB}, {TKB}, {TQ})
  kernel.launch @k_g_attention_flash2_{s}[%t](%t, %l{l}, %sliding, %ctx, %pool, %q, %q, %info, %tiles, %orig, %arena, %o) : [index](index, index, i32, index, index, {TQ}, {TQ}, {TI}, {TT}, {TO}, {TA}, {TQ})
  kernel.launch @k_g_mla_vbm_iq4_xs_{s}[%t](%t, %o, %vb, %vo) : [index](index, {TQ}, {TVB}, {TVO})""")
        return "\n".join(L) + "\n"
    TXQ, TXB, TXC, TXW, TXK, TXP = ("tensor<512x4xi32>", f"tensor<{XCAP // 64}x4xi32>", "tensor<64xi32>", T(QH * 256 * 512, "f16"),
                                    T(QH * XCAP * 128, "f16"), T(XCAP * 64, "f16"))
    L.append(f"""  %nq = check.literal value({nq}) : index
  %nb = check.literal value({nb}) : index
  kernel.launch @k_g_mla_xlist_{s}[%t](%t, %info, %tiles, %xq, %xb, %xcnt, %xskip) : [index](index, {TI}, {TT}, {TXQ}, {TXB}, {TXC}, tensor<128xi32>)""")
    for l in range(layer0, layer0 + layers):
        L.append(f"""  %l{l} = check.literal value({l}) : index
  kernel.launch @k_g_mla_xdeq_{s}[%c1x](%c1x, %kb, %vb, %xw) : [index](index, {TKB}, {TVB}, {TXW})
  kernel.launch @k_g_mla_xexp_d_{s}[%nb](%t, %l{l}, %ctx, %pool, %xb, %orig, %arena, %xw, %xk, %xv, %xpe) : [index](index, index, index, index, {TXB}, {TO}, {TA}, {TXW}, {TXK}, {TXK}, {TXP})
  kernel.launch @k_g_mla_xflash_d_{s}[%nq](%t, %xq, %info, %qb, %q, %xk, %xv, %xpe, %vo) : [index](index, {TXQ}, {TI}, {TQB}, {TQ}, {TXK}, {TXK}, {TXP}, {TVO})""")
    return "\n".join(L) + "\n"


def bench_inputs(rng):
    res = {}
    for tag, ctx, t, layers in CASES:
        inp = make_inputs(f"b{tag}", rng, [ctx], [(0, ctx - t + i) for i in range(t)], layers, ctx)
        inp.pop("lat")
        tiles = tile_list(t, inp["info"])
        save(f"b{tag}_tiles", tiles)
        _, xq, xb = xlists(inp["info"], tiles)
        res[tag] = (inp, len(xq), len(xb))
    return res


def bench(variant, res):
    out = []
    set_xcap(variant)
    for tag, ctx, t, layers in CASES:
        inp, nq, nb = res[tag]
        s = f"{tag}{variant.replace('+', '')}"
        out.append(kernels(variant, s))
        out.append(f"check.case public @c{tag} {{\n" + decls(f"b{tag}", inp, variant != "base") + launches(variant, s, inp, layers, nq, nb)
                   + f"  check.return\n}}\ncheck.benchmark<@c{tag}> @bm_{tag}\n")
    return out


TESTS = [   # name, rows (keys per row), tokens (row, position), layers, layer, ctx
    ("tA", [480, 48], [(0, 160 + i) for i in range(320)] + [(1, i) for i in range(48)], 2, 1, 512),
    ("tB", [40000], [(0, 40000 - 256 + i) for i in range(256)], 1, 0, 40960),
]


def tests(variants, kbd, vbd):
    rng = np.random.default_rng(7)
    out = []
    for name, rows, toks, layers, layer, ctx in TESTS:
        set_xcap("x")
        inp = make_inputs(name, rng, rows, toks, layers, ctx)
        t = inp["t"]
        tiles = tile_list(t, inp["info"])
        save(f"{name}_tiles", tiles)
        after, xq, xb = xlists(inp["info"], tiles)
        ref = reference(inp, kbd, vbd, layer)
        expanded = np.zeros(t, bool)
        for tok, nt, _, _ in xq:
            expanded[tok:tok + 16 * nt] = True
        for v in variants:
            set_xcap(v)
            after, xq, xb = xlists(inp["info"], tiles)
            s = f"{name}{v.replace('+', '')}"
            e = ref.copy()
            if v != "base":
                e[~expanded] = 0.0     # rows of absorbed tiles: not written by the expanded kernels
            save(f"{s}_vo", e.astype(np.float32))
            chk = [f'  %e = check.file.read.npy path("{s}_vo.npy") : tensor<{t}x{QH * VD}xf32>',
                   f"  check.expect.approximate actual(%vo) expected(%e) atol(5.0e-3) rtol(1.0e-2) nan(different) : tensor<{t}x{QH * VD}xf32>, tensor<{t}x{QH * VD}xf32>"]
            if v != "base":
                xc = np.zeros(64, np.int32); xc[0], xc[1] = len(xq), len(xb)
                xqe = np.zeros((512, 4), np.int32); xqe[:len(xq)] = xq
                xbe = np.zeros((XCAP // 64, 4), np.int32); xbe[:len(xb)] = xb
                save(f"{s}_tiles_e", after); save(f"{s}_xcnt_e", xc); save(f"{s}_xq_e", xqe); save(f"{s}_xb_e", xbe)
                chk += [f'  %et = check.file.read.npy path("{s}_tiles_e.npy") : tensor<{t}xi32>',
                        f"  check.expect.equal actual(%tiles) expected(%et) : tensor<{t}xi32>",
                        f'  %ec = check.file.read.npy path("{s}_xcnt_e.npy") : tensor<64xi32>',
                        "  check.expect.equal actual(%xcnt) expected(%ec) : tensor<64xi32>",
                        f'  %eq = check.file.read.npy path("{s}_xq_e.npy") : tensor<512x4xi32>',
                        "  check.expect.equal actual(%xq) expected(%eq) : tensor<512x4xi32>",
                        f'  %eb = check.file.read.npy path("{s}_xb_e.npy") : tensor<{XCAP // 64}x4xi32>',
                        f"  check.expect.equal actual(%xb) expected(%eb) : tensor<{XCAP // 64}x4xi32>"]
            out.append(kernels(v, s))
            out.append(f"check.case public @{s} {{\n" + decls(name, inp, v != "base") + launches(v, s, inp, 1, len(xq), len(xb), layer0=layer)
                       + "\n".join(chk) + "\n  check.return\n}\n")
    return out


def staged(v, kbd, vbd, kb_bytes):
    """Per-stage cases of a small-scratch variant (one file each: a device fault in one stage does not hide the others):
    1 g_mla_xlist (lists), 2 g_mla_xdeq (W), 3 g_mla_xexp_d (scratch of the used rows), 4 the whole chain (vo)."""
    set_xcap(v)
    rng = np.random.default_rng(11)
    name, rows, toks, layers, layer, ctx = TESTS[0]
    pfx = f"s{v}"
    inp = make_inputs(pfx, rng, rows, toks, layers, ctx)
    t = inp["t"]
    tiles = tile_list(t, inp["info"]); save(f"{pfx}_tiles", tiles)
    after, xq, xb = xlists(inp["info"], tiles)
    xc = np.zeros(64, np.int32); xc[0], xc[1] = len(xq), len(xb)
    xqe = np.zeros((512, 4), np.int32); xqe[:len(xq)] = xq
    xbe = np.zeros((XCAP // 64, 4), np.int32); xbe[:len(xb)] = xb
    save(f"{pfx}_tiles_e", after); save(f"{pfx}_xcnt_e", xc); save(f"{pfx}_xq_e", xqe); save(f"{pfx}_xb_e", xbe)
    W = np.zeros((QH, 256, KL)); W[:, :128] = kbd.transpose(0, 2, 1); W[:, 128:] = vbd
    save(f"{pfx}_w_e", W.reshape(-1).astype(np.float16))
    # expansion of row 0 (the expanded one): rows 0 .. nk64-1 of the scratch, zero past nk
    c = inp["lat"][0][layer].astype(np.float64)
    nk = int(xb[0][3]); n64 = len(xb) * 64
    W16 = W.astype(np.float16).astype(np.float64)
    xk = np.zeros((QH, XCAP, 128)); xv = np.zeros((QH, 128, XCAP)); xpe = np.zeros((XCAP, 64))
    xk[:, :nk] = c[:nk, :KL] @ W16[:, :128].transpose(0, 2, 1)
    xv[:, :, :nk] = (c[:nk, :KL] @ W16[:, 128:].transpose(0, 2, 1)).transpose(0, 2, 1)
    xpe[:nk] = c[:nk, KL:]
    save(f"{pfx}_xk_e", xk.reshape(-1).astype(np.float16)); save(f"{pfx}_xv_e", xv.reshape(-1).astype(np.float16)); save(f"{pfx}_xpe_e", xpe.reshape(-1).astype(np.float16))
    ref = reference(inp, kbd, vbd, layer)
    expanded = np.zeros(t, bool)
    for tok, nt, _, _ in xq:
        expanded[tok:tok + 16 * nt] = True
    ref[~expanded] = 0.0
    save(f"{pfx}_vo_e", ref.astype(np.float32))
    s = f"{pfx}"
    txt = kernels(v, s)
    TXB, TXK, TXP, TXW = f"tensor<{XCAP // 64}x4xi32>", T(QH * XCAP * 128, "f16"), T(XCAP * 64, "f16"), T(QH * 256 * 512, "f16")
    TO, TA = T(inp["olen"], "i32"), T(inp["alen"], "i8")
    rd = lambda var, f, ty: f'  %{var} = check.file.read.npy path("{pfx}_{f}.npy") : {ty}\n'
    files = {}
    files["1"] = (f"check.case public @{s}_list {{\n" + decls(pfx, inp) + f"""  kernel.launch @k_g_mla_xlist_{s}[%t](%t, %info, %tiles, %xq, %xb, %xcnt, %xskip) : [index](index, tensor<{t}x4xi32>, tensor<{t}xi32>, tensor<512x4xi32>, {TXB}, tensor<64xi32>, tensor<128xi32>)
""" + rd("et", "tiles_e", f"tensor<{t}xi32>") + rd("ec", "xcnt_e", "tensor<64xi32>") + rd("eq", "xq_e", "tensor<512x4xi32>") + rd("eb", "xb_e", TXB)
                  + f"""  check.expect.equal actual(%tiles) expected(%et) : tensor<{t}xi32>
  check.expect.equal actual(%xcnt) expected(%ec) : tensor<64xi32>
  check.expect.equal actual(%xq) expected(%eq) : tensor<512x4xi32>
  check.expect.equal actual(%xb) expected(%eb) : {TXB}
  check.return
}}
""")
    files["2"] = (f"check.case public @{s}_deq {{\n" + decls(pfx, inp) + f"""  kernel.launch @k_g_mla_xdeq_{s}[%c1x](%c1x, %kb, %vb, %xw) : [index](index, {T(QH * KL * 72, "i8")}, {T(QH * VD * 272, "i8")}, {TXW})
""" + rd("ew", "w_e", TXW) + f"""  check.expect.approximate actual(%xw) expected(%ew) atol(1.0e-3) rtol(2.0e-3) nan(different) : {TXW}, {TXW}
  check.return
}}
""")
    files["3"] = (f"check.case public @{s}_exp {{\n" + decls(pfx, inp) + rd("xbi", "xb_e", TXB) + f"""  %nb = check.literal value({len(xb)}) : index
  %lay = check.literal value({layer}) : index
  kernel.launch @k_g_mla_xdeq_{s}[%c1x](%c1x, %kb, %vb, %xw) : [index](index, {T(QH * KL * 72, "i8")}, {T(QH * VD * 272, "i8")}, {TXW})
  kernel.launch @k_g_mla_xexp_d_{s}[%nb](%t, %lay, %ctx, %pool, %xbi, %orig, %arena, %xw, %xk, %xv, %xpe) : [index](index, index, index, index, {TXB}, {TO}, {TA}, {TXW}, {TXK}, {TXK}, {TXP})
""" + rd("ek", "xk_e", TXK) + rd("ev", "xv_e", TXK) + rd("ep", "xpe_e", TXP) + f"""  check.expect.approximate actual(%xpe) expected(%ep) atol(0.0) rtol(0.0) nan(different) : {TXP}, {TXP}
  check.expect.approximate actual(%xk) expected(%ek) atol(2.0e-2) rtol(1.0e-2) nan(different) : {TXK}, {TXK}
  check.expect.approximate actual(%xv) expected(%ev) atol(2.0e-2) rtol(1.0e-2) nan(different) : {TXK}, {TXK}
  check.return
}}
""")
    files["4"] = (f"check.case public @{s}_all {{\n" + decls(pfx, inp) + launches(v, s, inp, 1, len(xq), len(xb), layer0=layer) + rd("e", "vo_e", f"tensor<{t}x{QH * VD}xf32>")
                  + f"""  check.expect.approximate actual(%vo) expected(%e) atol(5.0e-3) rtol(1.0e-2) nan(different) : tensor<{t}x{QH * VD}xf32>, tensor<{t}x{QH * VD}xf32>
  check.return
}}
""")
    TQB, TQ, TI = f"tensor<{t}x{QH * QB}xf32>", f"tensor<{t}x{QH * D}xf32>", f"tensor<{t}x4xi32>"
    files["5"] = (f"check.case public @{s}_flash {{\n" + decls(pfx, inp) + rd("xqi", "xq_e", "tensor<512x4xi32>") + rd("xki", "xk_e", TXK)
                  + rd("xvi", "xv_e", TXK) + rd("xpi", "xpe_e", TXP) + f"""  %nq = check.literal value({len(xq)}) : index
  kernel.launch @k_g_mla_xflash_d_{s}[%nq](%t, %xqi, %info, %qb, %q, %xki, %xvi, %xpi, %vo) : [index](index, tensor<512x4xi32>, {TI}, {TQB}, {TQ}, {TXK}, {TXK}, {TXP}, tensor<{t}x{QH * VD}xf32>)
""" + rd("e", "vo_e", f"tensor<{t}x{QH * VD}xf32>") + f"""  check.expect.approximate actual(%vo) expected(%e) atol(5.0e-3) rtol(1.0e-2) nan(different) : tensor<{t}x{QH * VD}xf32>, tensor<{t}x{QH * VD}xf32>
  check.return
}}
""")
    for k, body in files.items():
        open(os.path.join(OUT, f"test_{v}{k}.loom"), "w").write(HEAD + txt + "\n" + body)


if __name__ == "__main__" and len(sys.argv) > 2:
    OUT = sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    vs = sys.argv[3:] or ["base", "x"]
    for v in vs:
        if v != "base" and not all(x in VARIANTS for x in v.split("+")):
            raise SystemExit("unknown variant " + v)
    kb, vb, kbd, vbd = weights(np.random.default_rng(3))
    save("w_kb", kb); save("w_vb", vb)
    res = bench_inputs(np.random.default_rng(1))
    for v in vs:
        open(os.path.join(OUT, "bench%s.loom" % ("" if v == "base" else "_" + v.replace("+", "_"))), "w").write(HEAD + "\n".join(bench(v, res)))
    for v in ("t", "tg"):
        staged(v, kbd, vbd, kb)
    open(os.path.join(OUT, "test.loom"), "w").write(HEAD + "\n".join(tests(vs, kbd, vbd)))
    if _KIT:
        shutil.rmtree(os.path.dirname(_KIT), ignore_errors=True)
