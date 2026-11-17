#!/usr/bin/env python3
"""Generate the flash_r9700 kbench suite: the gfx12 whole-tile prefill flash kernel of the AMD MXFP4 Qwen3.8-27B card (head_dim 256, 24 query
heads, 4 KV heads, output gate) as gen_flash2s.py builds it, at a 1024-token prefill chunk at the end of an 8k and a 32k context.

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   VARIANT: base (bench.loom: the key-split loop, FLASH2_B=2, without prefill.attn.flash2_b2_fm), a
name of VARIANTS (bench_<name>.loom: "B" picks the loop, FLASH2_B=6 flash2_g6.py with FLASH2_G6_OPTS, else flash2_b2.py with FLASH2_B2_OPTS;
fmp = what prefill.attn.flash2_b2_fm builds; a '+'-joined list merges options, "abl" ablations concatenate) or mm<K>[l<KB>][w<W>]
(a WMMA throughput microbenchmark: K independent accumulator chains, KB of workgroup memory, W waves per workgroup).
Writes bench*.loom (timing: k_fa8k / k_fa32k, random K/V/Q, identity page map) and test.loom (check.cases of every TESTED variant and the base
against a float64 numpy reference: 2048-key context, 128 tokens, permuted page map, sharp scores; a tile whose later key outweighs page 0 and its
own page by more than 2^15 (the g6 fast path's fallback); run by devlab_kbench.sh's "tests" step with loom-test)."""
import json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "../../../.."))
STARTER = os.path.join(ROOT, "skills", "loom-port-llm", "starter")
CONFIG = os.path.join(ROOT, "fleet", "amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom", "llm_config.json")
import numpy as np

HEAD = "amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n\n"
CFG = json.load(open(CONFIG))
QH, KVH, D = CFG["heads"], CFG["kv_heads"], CFG["head_dim"]
QD, R = QH * D, KVH * D
T = 1024
CTXS = [("8k", 8192), ("32k", 32768)]
VARIANTS = {   # name -> flash2_g6.py options (FLASH2_G6_OPTS)
    "g6": {},
    "g6o0": {"opq": 0},
    "g6q8": {"qlds": 8},
    "g6slow": {"fast": 0},
    "g6kv0": {"abl": "kv0"},
    "g6noqk": {"abl": "noqk"},
    "g6nopv": {"abl": "nopv"},
    "g6noexp": {"abl": "noexp"},
    # key-split loop (flash2_b2.py, FLASH2_B=2 unless "B" says otherwise; options FLASH2_B2_OPTS)
    "b3": {"B": "3"},
    "kv0": {"B": "2", "abl": "kv0"},
    "noqk": {"B": "2", "abl": "noqk"},
    "nopv": {"B": "2", "abl": "nopv"},
    "nobar": {"B": "2", "abl": "nobar"},
    "k0": {"B": "2", "abl": "k0"},
    "v0": {"B": "2", "abl": "v0"},
    "noexp": {"B": "2", "abl": "noexp"},
    "vtop": {"B": "2", "vpf": "top"},
    "vmid": {"B": "2", "vpf": "mid"},
    "vtop2": {"B": "2", "vpf": "top2"},
    "e2": {"B": "2", "e2": 1},
    "wg3qs": {"B": "2", "wg": 3, "qshare": 1},   # timing only (the three pairs read one Q^T region)
    "sacc2": {"B": "2", "sacc": 2},
    "fm": {"B": "2", "fmax": 1, "vpf": "top2", "e2": 1},   # fixed reference m0 (first + diagonal page), running-max fallback above m0 + 15
    "fmt": {"B": "2", "fmax": 1, "vpf": "top", "e2": 1},
    "fmp": {"B": "2", "fmax": 1, "vpf": "top2", "e2": 1, "peel": 1},   # + unmasked loop below the diagonal
    "ovl": {"B": "2", "ovl": 1},   # with fmax: overflow tracked by the per-lane block sum
}
TESTED = ["fmp"]


def opts_of(v):
    o = {}
    for x in v.split("+"):
        for k, val in VARIANTS[x].items():
            o[k] = (o[k] + "," + val) if k == "abl" and k in o else val
    return o


def kernel(variant, name):
    """The whole-tile kernel text of VARIANT ('' = base), renamed NAME."""
    o = opts_of(variant) if variant else {}
    b = o.pop("B", "6" if variant else "2")
    env = dict(os.environ, LLM_CONFIG=CONFIG, FLASH2_ARCH="rdna4", FLASH2_TRIM="1", FLASH2_B=b,
               FLASH2_G6_OPTS=json.dumps(o) if b == "6" else "{}", FLASH2_B2_OPTS=json.dumps(o) if b != "6" else "{}",
               FLASH2_OUT=os.path.join(OUT, f".fl_{name}.loom"))
    subprocess.run([sys.executable, os.path.join(STARTER, "gen_flash2s.py")], check=True, env=env)
    t = open(env["FLASH2_OUT"]).read(); os.remove(env["FLASH2_OUT"])
    t = t[:t.index('kernel.def target(@rdna4) export("g_attention_flash2_partial_rdna4")')]
    return t.replace('export("g_attention_flash2_rdna4") @g_attention_flash2_rdna4(', f'export("{name}") @{name}(')


def micro(k, lds=0, waves=4):
    """WMMA throughput microbenchmark (variant mm<K>[l<KB>]): 4096 workgroups of WAVES waves, each wave 1024 x 16 f16 WMMAs spread round-robin
    over K independent accumulators (K = 1: one dependent chain); LDS KB of workgroup memory (touched once) limits the waves per SIMD."""
    acc = ", ".join(f"%a{j} = %init : vector<8xf32>" for j in range(k))
    ty = ", ".join(["vector<8xf32>"] * k)
    body, cur = [], [f"%a{j}" for j in range(k)]
    for i in range(16):
        j = i % k
        body.append(f"      %m{i} = vector.mma %fa, %fb, {cur[j]} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
        cur[j] = f"%m{i}"
    tot = cur[0]
    fin = ""
    for j in range(1, k):
        fin += f"  %t{j} = vector.addf {tot}, %r{j} : vector<8xf32>\n"
        tot = f"%t{j}"
    res = ", ".join(f"%r{j}" for j in range(k)).replace("%r0", "%r0")
    ldst = (f"  %lb = index.constant {lds * 1024} : offset\n  %lt = buffer.alloca<workgroup> align(16) %lb : buffer\n"
            f"  %lv = buffer.view %lt[%origin] : buffer -> view<{lds * 256}xf32>\n  %z1 = scalar.constant 0.0 : f32\n  view.store %z1, %lv[%lane] : f32, view<{lds * 256}xf32>\n") if lds else ""
    t = f"""kernel.def target(@rdna4) export("k_mm") @k_mm(%n: index) {{
  %c1 = index.constant 1 : index
  %cw = index.constant 4096 : index
  %cl = index.constant {32 * waves} : index
  kernel.launch.config workgroups(%cw, %c1, %c1) workgroup_size(%cl, %c1, %c1) : index
}} launch(%n_arg: index, %a: buffer, %o: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, 1, 65536)] : index
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %av = buffer.view %a[%origin] : buffer -> view<32x16xf16>
  %ah = vector.load %av[%lane, %c0] : view<32x16xf16> -> vector<16xf16>
  %a8 = vector.slice %ah[0] : vector<16xf16> -> vector<8xf16>
  %b8 = vector.slice %ah[8] : vector<16xf16> -> vector<8xf16>
  %fa = vector.fragment<lhs> %a8 shape [%c16, %c16] : vector<8xf16>
  %fb = vector.fragment<rhs> %b8 shape [%c16, %c16] : vector<8xf16>
{ldst}  %z = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %z shape [%c16, %c16] : vector<8xf32>
  {", ".join(f"%r{j}" for j in range(k))} = scf.for %i = [%c0 to %n step %c1]({acc}) -> ({ty}) {{
{chr(10).join(body)}
      scf.yield {", ".join(cur)} : {ty}
  }}
{fin.replace(cur[0], "%r0", 1) if False else ""}"""
    tot = "%r0"
    for j in range(1, k):
        t += f"  %t{j} = vector.addf {tot}, %r{j} : vector<8xf32>\n"
        tot = f"%t{j}"
    t += f"""  %ov = buffer.view %o[%origin] : buffer -> view<32x8xf32>
  vector.store {tot}, %ov[%lane, %c0] : vector<8xf32>, view<32x8xf32>
  kernel.return
}}
check.case public @k_mm_case {{
  %n = check.literal value(1024) : index
  %a = check.generate.fill value(0.01) : tensor<512xf16>
  %o = check.generate.fill value(0.0) : tensor<256xf32>
  kernel.launch @k_mm[%n](%n, %a, %o) : [index](index, tensor<512xf16>, tensor<256xf32>)
  check.return
}}
check.benchmark<@k_mm_case> @k_mm_bm
"""
    return t


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


def launch(name, pfx, ctx, t, gate):
    AT, OT = f"tensor<{2 * ctx * R}xf16>", f"tensor<{10 + ctx // 16}xi32>"
    return f"""  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({ctx}) : index
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


def tests():
    rng = np.random.default_rng(3)
    out = []
    ctx, t = 2048, 128
    cases = []
    # the first tiles of a row: the diagonal page is page 0 (pg_diag 0) or follows it directly (an empty middle loop)
    kc, vc, qc, ic = inputs("tC", 64, 64, rng, perm=True, q_scale=6.0)
    gate = rng.standard_normal((64, QD)).astype(np.float32); save("tC_g", gate)
    save("tC_o", (reference(kc, vc, qc, ic[:, 1]).reshape(64, QD) / (1 + np.exp(-gate.astype(np.float64)))).astype(np.float32))
    kk, vv, q, info = inputs("tA", ctx, t, rng, perm=True, q_scale=6.0)
    gate = rng.standard_normal((t, QD)).astype(np.float32); save("tA_g", gate)
    save("tA_o", (reference(kk, vv, q, info[:, 1]).reshape(t, QD) / (1 + np.exp(-gate.astype(np.float64)))).astype(np.float32))
    cases.append("tA")
    # overflow: key 1000 (page 62: neither page 0 nor a tile's own page) aligned with the queries of KV group 0's heads: its score exceeds every
    # key of page 0 and of the tile's own page by ~27 nats (> 15 in log2 units: the g6 fast path must fall back)
    kk, vv, q, info = inputs("tB", ctx, t, rng, perm=True, q_scale=1.0)
    for h in range(QH // KVH):
        q[:, h] = 0.2 * rng.uniform(-1, 1, (t, D)) + np.float32(4.0) * np.sign(kk[1000, 0].astype(np.float32))[None, :]
    save("tB_q", q.reshape(t, QD))
    gate = rng.standard_normal((t, QD)).astype(np.float32); save("tB_g", gate)
    save("tB_o", (reference(kk, vv, q, info[:, 1]).reshape(t, QD) / (1 + np.exp(-gate.astype(np.float64)))).astype(np.float32))
    cases.append("tB")
    for v in [""] + TESTED:
        for c, cx, tc in [(c, ctx, t) for c in cases] + [("tC", 64, 64)]:
            name = f"t_{(v or 'base').replace('+', '_')}_{c}"
            out.append(kernel(v, name))
            out.append(f"check.case public @{name}_case {{\n" + launch(name, c, cx, tc, f'check.file.read.npy path("{c}_g.npy")')
                       + f'  %e = check.file.read.npy path("{c}_o.npy") : tensor<{tc}x{QD}xf32>\n'
                       + f"  check.expect.close actual(%o) expected(%e) atol(4.0e-3) rtol(4.0e-3) nan(different) : tensor<{tc}x{QD}xf32>\n  check.return\n}}\n")
    return out


if __name__ == "__main__" and len(sys.argv) > 2:
    OUT = sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    for tag, ctx in CTXS:
        inputs(f"b{tag}", ctx, T, rng)
    for v in (sys.argv[3:] or ["base"]):
        if v != "base" and not re.fullmatch(r"mm\d+(l\d+)?(w\d+)?", v) and not all(x in VARIANTS for x in v.split("+")):
            raise SystemExit("unknown variant " + v)
        m = re.fullmatch(r"mm(\d+)(?:l(\d+))?(?:w(\d+))?", v)
        if m:
            open(os.path.join(OUT, f"bench_{v}.loom"), "w").write(HEAD + micro(int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 4)))
            continue
        v = "" if v == "base" else v
        open(os.path.join(OUT, "bench%s.loom" % ("_" + v if v else "")), "w").write(HEAD + "\n".join(bench(v)))
    text = re.sub(r"check\.expect\.close (actual\(.*\)) : (.*)$", r"check.expect.approximate \1 : \2, \2", HEAD + "\n".join(tests()), flags=re.M)
    open(os.path.join(OUT, "test.loom"), "w").write(text)
