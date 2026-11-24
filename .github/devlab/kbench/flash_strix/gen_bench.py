#!/usr/bin/env python3
"""Generate the flash_strix kbench suite: the gfx11 whole-tile prefill flash kernel of bartowski-Qwen3.8-27B (head_dim 256, 24 query heads,
4 KV heads, output gate) as gen_flash2s.py builds it for Strix Halo, at a 1024-token prefill chunk at the end of an 8k and a 32k context.

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   VARIANT: base (bench.loom) or a name of VARIANTS (bench_<name>.loom: its environment for
gen_flash2s.py; '+'-joined names merge, FLASH2_G11 lists concatenate). Writes bench*.loom (timing: k_fa8k / k_fa32k, random K/V/Q, identity
page map) and test.loom (check.cases of the base and every non-ablation variant against a float64 numpy reference: 2048-key context,
128 tokens, permuted page map, sharp scores; run by devlab_kbench.sh's "tests" step with loom-test)."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "../../../.."))
STARTER = os.path.join(ROOT, "skills", "loom-port-llm", "starter")
MODEL = "bartowski-Qwen3.8-27B-loom"
T = 1024
if len(sys.argv) > 3 and sys.argv[3] == "qwen35":
    MODEL = "unsloth-Qwen3.5-35B-A3B-loom"
    T = 4096
    del sys.argv[3]
CONFIG = os.path.join(ROOT, "fleet", MODEL, "llm_config.json")
import numpy as np

HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
CFG = json.load(open(CONFIG))
QH, KVH, D = CFG["heads"], CFG["kv_heads"], CFG["head_dim"]
QD, R = QH * D, KVH * D
CTXS = [("8k", 8192), ("32k", 32768)]
VARIANTS = {   # name -> environment of gen_flash2s.py
    "wave0": {}, "wave4": {}, "wave8": {},
    "lazy": {"FLASH2_G11": "lazy"}, "kvh": {"FLASH2_G11": "kvh"}, "vtf": {"FLASH2_G11": "vtf"}, "hot": {"FLASH2_G11": "hot"},
    "pf": {"FLASH2_G11": "pf"}, "vp": {"FLASH2_G11": "vp"}, "kt16": {"FLASH2_KT256": "16"},
    "db": {"FLASH2_KT256": "16", "FLASH2_G11": "db"},
    "b2": {"FLASH2_DB_AT": "b2"},          # with db: stores after the S exchange barrier
    "ng2": {"FLASH2_NG256": "2"}, "ng3": {"FLASH2_NG256": "3"},
    "ngx": {"FLASH2_NGX": "1"},   # with ng2/ng3: a tile's head groups adjacent in dispatch order
    "kv0": {"FLASH2_G11": "abl=nokld"},
    "trim": {"FLASH2_G11": "lazy,kvh,vtf,hot"},
    "pfu": {"FLASH2_G11": "pfu"}, "bcast": {"FLASH2_ENTRY": "bcast"},
    "tnl": {"FLASH2_G11": "kvh,vtf,hot"},   # trim without lazy (lazy spills on gfx11)
    "mt2": {"FLASH2_KT256": "16", "FLASH2_MT256": "2"},   # two 16-token query tiles per workgroup (flash2_mt.py)
    "mdb": {"FLASH2_KT256": "16", "FLASH2_MT256": "2", "FLASH2_G11": "db"},   # mt2 + double-buffered staging
    "vp4": {"FLASH2_VTPAD": "4"},   # V^T row pad 4 halves (LDS room for mdb)
    "nostv": {"FLASH2_G11": "abl=nostv"}, "nokld": {"FLASH2_G11": "abl=nokld"}, "nobar": {"FLASH2_G11": "abl=nobar"},
    "noqk": {"FLASH2_G11": "abl=noqk"}, "nopv": {"FLASH2_G11": "abl=nopv"}, "nosm": {"FLASH2_G11": "abl=nosm"},
}
OUT = None


def env_of(v):
    e = {}
    for x in v.split("+") if v else []:
        for k, val in VARIANTS[x].items():
            e[k] = (e[k] + "," + val) if k in e and k == "FLASH2_G11" else val
    return e


def kvo(name):
    """gen_llm.py's paged-KV offset function at this model's bytes per position, renamed for kernel NAME."""
    src = open(os.path.join(STARTER, "gen_llm.py")).read()
    i = src.index("func.def inline @g_kv_offset(")
    j = src.index("\n}\n", i) + 3
    t = src[i:j].replace("%position_bytes = index.constant 512 : offset", f"%position_bytes = index.constant {KVH * D * 2} : offset")
    return t.replace("@g_kv_offset(", f"@{name}_kvo(")


def kernel(variant, name):
    """The whole-tile gfx11 kernel text of VARIANT ('' = base), renamed NAME."""
    if variant.startswith("wave"):
        sys.path.insert(0, STARTER)
        from wave_gfx11 import wave_kernel
        return wave_kernel(CFG, int(variant[4:])).replace('export("g_attention_flash2") @g_attention_flash2(', f'export("{name}") @{name}(')
    env = dict(os.environ, LLM_CONFIG=CONFIG, LOOM_RDNA4="0", FLASH2_OUT=os.path.join(OUT, f".fl_{name}.loom"), **env_of(variant))
    subprocess.run([sys.executable, os.path.join(STARTER, "gen_flash2s.py")], check=True, env=env)
    t = open(env["FLASH2_OUT"]).read(); os.remove(env["FLASH2_OUT"])
    t = t[:t.index('kernel.def export("g_attention_flash2_partial")')]
    t = t.replace('export("g_attention_flash2") @g_attention_flash2(', f'export("{name}") @{name}(')
    return t.replace("@g_kv_offset(", f"@{name}_kvo(")


def save(name, a):
    np.save(os.path.join(OUT, name + ".npy"), a)


def inputs(pfx, ctx, t, rng, perm=False, q_scale=1.0):
    """Arena (layer 0: K plane, V plane), origins (one row: page map at byte 40), info, tile list for T tokens at positions ctx - t .. ctx - 1."""
    blocks = ctx // 16
    kk = rng.uniform(-1, 1, (ctx, KVH, D)).astype(np.float16); vv = rng.uniform(-1, 1, (ctx, KVH, D)).astype(np.float16)
    pages = rng.permutation(blocks) if perm else np.arange(blocks)
    arena = np.zeros((2, ctx, R), np.float16)
    for b in range(blocks):
        arena[0, pages[b] * 16:pages[b] * 16 + 16] = kk[b * 16:b * 16 + 16].reshape(16, R)
        arena[1, pages[b] * 16:pages[b] * 16 + 16] = vv[b * 16:b * 16 + 16].reshape(16, R)
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = pages
    info = np.zeros((t, 4), np.int32); info[:, 1] = ctx - t + np.arange(t); info[:, 2] = 1
    tiles = np.zeros(t, np.int32); tiles[0] = t // 16; tiles[1:1 + t // 16] = 16 * np.arange(t // 16)
    q = (rng.uniform(-1, 1, (t, QH, D)) * q_scale).astype(np.float32)
    save(f"{pfx}_arena", arena.reshape(-1)); save(f"{pfx}_orig", orig); save(f"{pfx}_info", info); save(f"{pfx}_tiles", tiles)
    save(f"{pfx}_q", q.reshape(t, QD))
    return kk, vv, q, info


def launch(name, pfx, ctx, t, gate, pool=None, olen=None):
    pool = pool or ctx
    AT, OT = f"tensor<{2 * pool * R}xf16>", f"tensor<{olen or 10 + ctx // 16}xi32>"
    return f"""  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({pool}) : index
  %q = check.file.read.npy path("{pfx}_q.npy") : tensor<{t}x{QD}xf32>
  %g = {gate} : tensor<{t}x{QD}xf32>
  %info = check.file.read.npy path("{pfx}_info.npy") : tensor<{t}x4xi32>
  %tiles = check.file.read.npy path("{pfx}_tiles.npy") : tensor<{t}xi32>
  %orig = check.file.read.npy path("{pfx}_orig.npy") : {OT}
  %arena = check.file.read.npy path("{pfx}_arena.npy") : {AT}
  %o = check.generate.fill value(0.0) : tensor<{t}x{QD}xf32>
  kernel.launch @{name}[%t](%t, %layer, %sliding, %ctx, %pool, %q, %g, %info, %tiles, %orig, %arena, %o) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x{QD}xf32>, tensor<{t}x4xi32>, tensor<{t}xi32>, {OT}, {AT}, tensor<{t}x{QD}xf32>)
"""


def bench(variant):
    out = []
    for tag, ctx in CTXS:
        name = f"k_fa{tag}"
        out.append(kvo(name))
        out.append(kernel(variant, name))
        out.append(f"check.case public @{name}_case {{\n" + launch(name, f"b{tag}", ctx, T, "check.generate.fill value(4.0)") + f"  check.return\n}}\ncheck.benchmark<@{name}_case> @{name}_bm\n")
    return out


def reference(kk, vv, q, pos):
    o = np.zeros(q.shape)
    qd = q.astype(np.float16).astype(np.float64)
    j = np.arange(len(kk))[None, :]
    for h in range(QH):
        kv = h // (QH // KVH)
        sc = qd[:, h] @ kk[:, kv].astype(np.float64).T * D ** -0.5
        sc[j > pos[:, None]] = -1e30
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        o[:, h] = p @ vv[:, kv].astype(np.float64)
    return o


def inputs_rows(pfx, rng, q_scale=6.0):
    """Two sequences (paged-KV rows) of 256 keys in one 512-position pool (permuted pages); queries: row 0 positions 192..255 (4 tiles),
    row 1 positions 208..255 (3 tiles), tiles listed interleaved (r0, r1, r0, r1, r0, r1, r0: 7 listed, an odd count) so that
    consecutive listed tiles belong to different rows. Returns the expected output (before the gate) per token."""
    ctx, nb = 256, 16
    pages = rng.permutation(2 * nb)
    kk = rng.uniform(-1, 1, (2, ctx, KVH, D)).astype(np.float16); vv = rng.uniform(-1, 1, (2, ctx, KVH, D)).astype(np.float16)
    arena = np.zeros((2, 2 * ctx, R), np.float16)
    for r in range(2):
        for b in range(nb):
            ph = pages[r * nb + b]
            arena[0, ph * 16:ph * 16 + 16] = kk[r, b * 16:b * 16 + 16].reshape(16, R)
            arena[1, ph * 16:ph * 16 + 16] = vv[r, b * 16:b * 16 + 16].reshape(16, R)
    orig = np.zeros(20 + 2 * nb, np.int32); orig[8] = 80; orig[18] = 80 + 4 * nb
    orig[20:20 + nb] = pages[:nb]; orig[20 + nb:] = pages[nb:]
    spans = [(0, 192, 64), (1, 208, 48)]
    t = sum(n for _, _, n in spans)
    info = np.zeros((t, 4), np.int32); first = {}
    i = 0
    for r, p0, n in spans:
        first[r] = [i + 16 * k for k in range(n // 16)]
        info[i:i + n, 0] = r; info[i:i + n, 1] = p0 + np.arange(n); info[i:i + n, 2] = 1
        i += n
    order = [first[0][0], first[1][0], first[0][1], first[1][1], first[0][2], first[1][2], first[0][3]]
    tiles = np.zeros(t, np.int32); tiles[0] = len(order); tiles[1:1 + len(order)] = order
    q = (rng.uniform(-1, 1, (t, QH, D)) * q_scale).astype(np.float32)
    save(f"{pfx}_arena", arena.reshape(-1)); save(f"{pfx}_orig", orig); save(f"{pfx}_info", info); save(f"{pfx}_tiles", tiles)
    save(f"{pfx}_q", q.reshape(t, QD))
    ref = np.zeros((t, QH, D))
    for r, p0, n in spans:
        sel = info[:, 0] == r
        ref[sel] = reference(kk[r], vv[r], q[sel], info[sel, 1])
    return ref, t, len(orig)


def tests(tested):
    rng = np.random.default_rng(3)
    out = []
    ctx, t = 2048, 128
    kc, vc, qc, ic = inputs("tC", 64, 64, rng, perm=True, q_scale=6.0)
    gate = rng.standard_normal((64, QD)).astype(np.float32); save("tC_g", gate)
    save("tC_o", (reference(kc, vc, qc, ic[:, 1]).reshape(64, QD) / (1 + np.exp(-gate.astype(np.float64)))).astype(np.float32))
    kk, vv, q, info = inputs("tA", ctx, t, rng, perm=True, q_scale=6.0)
    gate = rng.standard_normal((t, QD)).astype(np.float32); save("tA_g", gate)
    save("tA_o", (reference(kk, vv, q, info[:, 1]).reshape(t, QD) / (1 + np.exp(-gate.astype(np.float64)))).astype(np.float32))
    ref_b, tb, olen_b = inputs_rows("tB", rng)
    gate = rng.standard_normal((tb, QD)).astype(np.float32); save("tB_g", gate)
    save("tB_o", (ref_b.reshape(tb, QD) / (1 + np.exp(-gate.astype(np.float64)))).astype(np.float32))
    for v in [""] + tested:
        for c, cx, tc in [("tA", ctx, t), ("tC", 64, 64), ("tB", 256, tb)]:
            name = f"t_{(v or 'base').replace('+', '_')}_{c}"
            out.append(kvo(name))
            out.append(kernel(v, name))
            kw = dict(pool=512, olen=olen_b) if c == "tB" else {}
            out.append(f"check.case public @{name}_case {{\n" + launch(name, c, cx, tc, f'check.file.read.npy path("{c}_g.npy")', **kw)
                       + f'  %e = check.file.read.npy path("{c}_o.npy") : tensor<{tc}x{QD}xf32>\n'
                       + f"  check.expect.approximate actual(%o) expected(%e) atol(4.0e-3) rtol(4.0e-3) nan(different) : tensor<{tc}x{QD}xf32>, tensor<{tc}x{QD}xf32>\n  check.return\n}}\n")
    return out


if __name__ == "__main__" and len(sys.argv) > 2:
    OUT = sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    for tag, ctx in CTXS:
        inputs(f"b{tag}", ctx, T, rng)
    vs = sys.argv[3:] or ["base"]
    for v in vs:
        if v != "base" and not all(x in VARIANTS for x in v.split("+")):
            raise SystemExit("unknown variant " + v)
        v = "" if v == "base" else v
        open(os.path.join(OUT, "bench%s.loom" % ("_" + v if v else "")), "w").write(HEAD + "\n".join(bench(v)))
    tested = [v for v in vs if v != "base" and not any("abl=" in VARIANTS[x].get("FLASH2_G11", "") for x in v.split("+"))]   # ablations compute garbage
    open(os.path.join(OUT, "test.loom"), "w").write(HEAD + "\n".join(tests(tested)))
