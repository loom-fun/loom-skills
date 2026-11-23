#!/usr/bin/env python3
"""JoyAI-LLM-Flash (deepseek2, 256 experts top-8, expert FFN 768, hidden 2048, IQ4_XS experts) prefill MoE grouped GEMMs as the
deepseek2 example kit generates them (gen_moe.grouped_gemm over gen_gemm.gemm), at a 1024- and a 4096-token chunk.

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   (devlab_kbench.sh "gen"; VARIANT in VARIANTS, '+'-joined names merge; base = bench.loom)
Writes bench*.loom (timing: k_ggu_<T> gate/up swiglu, k_gdn_<T> down; the same names in every file) and test.loom (every
non-base variant against v1 on the same inputs: exact, except variants named x* (reordered sums: 1e-3); plus a
negative control that must fail). Inputs: random valid IQ4_XS blocks (d 0.5e-3..2e-3, random 6-bit scales and codes), random f16
activations, a skewed top-8 routing (per-expert weight lognormal sigma 0.6) bucketed exactly as g_moe_bucket does."""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H, F, E, KU, TILE = 2048, 768, 256, 8, 64
TS = [int(x) for x in os.environ.get("JOY_MOE_T", "1024,4096").split(",")]
IQ4 = {"GEMM_IQ4_NATIVE": "1", "GEMM_IQ4_F16LUT": "1", "GEMM_IQ4_G32": "1", "GEMM_M256_G32": "1"}
VARIANTS = {   # name -> environment of the kit generators
    "base": {},
    "v1": IQ4,
    "v2": dict(IQ4, GEMM_T64_G32="1"),
}
VARIANTS.update(json.loads(os.environ.get("JOY_MOE_VARIANTS", "{}")))
BLOCK = 136   # IQ4_XS bytes per 256 values


def env_of(v):
    e = {}
    for x in v.split("+"):
        e.update(VARIANTS[x])
    return e


def iq4xs(rng, rows, k):
    nb = rows * (k // 256)
    b = np.empty((nb, BLOCK), np.uint8)
    b[:, 0:2] = rng.uniform(5e-4, 2e-3, nb).astype("<f2").view(np.uint8).reshape(nb, 2)
    b[:, 2:] = rng.integers(0, 256, (nb, BLOCK - 2), dtype=np.uint8)
    return b.reshape(-1).view(np.int8)


def routing(rng, T):
    """ids [T, KU] (distinct experts per token, skewed popularity) -> tile_info [max_tiles, 2], rowmap [max_rows] as g_moe_bucket."""
    p = np.exp(rng.normal(0, 0.6, E)); p /= p.sum()
    ids = np.stack([rng.choice(E, KU, replace=False, p=p) for _ in range(T)])
    max_tiles = (T * KU + TILE - 1) // TILE + E
    ti = np.zeros((max_tiles, 2), np.int32); ti[:, 0] = -1
    rm = np.full(max_tiles * TILE, -1, np.int32)
    tile = 0
    for e in range(E):
        a = np.nonzero(ids.reshape(-1) == e)[0]
        for j in range(0, len(a), TILE):
            n = min(TILE, len(a) - j)
            ti[tile] = (e, n)
            rm[tile * TILE: tile * TILE + n] = a[j:j + n]
            tile += 1
    return ti, rm, max_tiles


def emit(kit, variant, out):
    """kernel texts of VARIANT (run in a child process: the generators read their switches at import)."""
    env = dict(os.environ, LOOM_RDNA4="0", **env_of(variant))
    code = f"""
import sys; sys.path.insert(0, {kit!r})
import gen_gemm, gen_moe
gu = gen_moe.grouped_gemm("iq4_xs", "swiglu", {F}, {H}, {E}, {KU}, gen_gemm.gemm)
dn = gen_moe.grouped_gemm("iq4_xs", "store", {H}, {F}, {E}, {KU}, gen_gemm.gemm)
open({out!r}, "w").write(gu + "\\n" + dn)
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    return open(out).read()


def helpers(kit):
    return open(os.path.join(kit, "quant_values.loom")).read()


def main():
    root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    variants = sys.argv[3:] or ["base", "v1", "v2"]
    os.makedirs(out, exist_ok=True)
    kit = os.path.join(out, "kit")
    if not os.path.isdir(kit):
        subprocess.run(["bash", os.path.join(root, "skills/loom-port-llm/reference/apply_example.sh"), "deepseek2", kit], check=True, stdout=subprocess.DEVNULL)
    rng = np.random.default_rng(1361)
    os.makedirs(os.path.join(out, "w"), exist_ok=True)
    wb = {"gate": (E * F, H), "up": (E * F, H), "down": (E * H, F)}
    for n, (r, k) in wb.items():
        p = os.path.join(out, "w", f"{n}.npy")
        if not os.path.exists(p):
            np.save(p, iq4xs(rng, r, k))
    nbytes = {n: r * (k // 256) * BLOCK for n, (r, k) in wb.items()}
    shapes = {}
    for T in TS:
        ti, rm, mt = routing(rng, T)
        mr = mt * TILE
        np.save(os.path.join(out, "w", f"ti{T}.npy"), ti)
        np.save(os.path.join(out, "w", f"rm{T}.npy"), rm)
        np.save(os.path.join(out, "w", f"x{T}.npy"), rng.uniform(-1, 1, (T, H)).astype(np.float16))
        np.save(os.path.join(out, "w", f"a{T}.npy"), rng.uniform(-1, 1, (mr, F)).astype(np.float16))
        np.save(os.path.join(out, "w", f"s{T}.npy"), np.tile(rng.uniform(0.5, 2, (T, 1)), (1, 2)).astype(np.float32))
        np.save(os.path.join(out, "w", f"sa{T}.npy"), np.tile(rng.uniform(0.5, 2, (mr, 1)), (1, 2)).astype(np.float32))
        shapes[T] = (mt, mr)
        print(f"T={T}: max_tiles {mt}, used tiles {(ti[:, 0] >= 0).sum()}, rows/expert mean {T * KU / E:.1f}")

    def case(T, kn_gu, kn_dn, ret=("y",), extra=""):
        mt, mr = shapes[T]
        return f"""  %ti = check.file.read.npy path("w/ti{T}.npy") : tensor<{mt}x2xi32>
  %rm = check.file.read.npy path("w/rm{T}.npy") : tensor<{mr}xi32>
  %x = check.file.read.npy path("w/x{T}.npy") : tensor<{T}x{H}xf16>
  %s = check.file.read.npy path("w/s{T}.npy") : tensor<{T}x2xf32>
  %a = check.file.read.npy path("w/a{T}.npy") : tensor<{mr}x{F}xf16>
  %sa = check.file.read.npy path("w/sa{T}.npy") : tensor<{mr}x2xf32>
  %wg = check.file.read.npy path("w/gate.npy") : tensor<{nbytes['gate']}xi8>
  %wu = check.file.read.npy path("w/up.npy") : tensor<{nbytes['up']}xi8>
  %wd = check.file.read.npy path("w/down.npy") : tensor<{nbytes['down']}xi8>
  %rows = check.literal value({mr}) : index
  %xrows = check.literal value({T}) : index
  %kh = check.literal value({H}) : index
  %nf = check.literal value({F}) : index
"""

    def launches(T, gu, dn, yg, yd):
        mt, mr = shapes[T]
        s = ""
        if gu:
            s += (f"  kernel.launch @{gu}[%rows, %nf](%rows, %kh, %nf, %xrows, %ti, %rm, %x, %s, %wg, %wu, {yg}) : [index, index](index, index, index, index, "
                  f"tensor<{mt}x2xi32>, tensor<{mr}xi32>, tensor<{T}x{H}xf16>, tensor<{T}x2xf32>, tensor<{nbytes['gate']}xi8>, tensor<{nbytes['up']}xi8>, tensor<{mr}x{F}xf32>)\n")
        if dn:
            s += (f"  kernel.launch @{dn}[%rows, %kh](%rows, %nf, %kh, %rows, %ti, %rm, %a, %sa, %wd, {yd}) : [index, index](index, index, index, index, "
                  f"tensor<{mt}x2xi32>, tensor<{mr}xi32>, tensor<{mr}x{F}xf16>, tensor<{mr}x2xf32>, tensor<{nbytes['down']}xi8>, tensor<{mr}x{H}xf32>)\n")
        return s

    texts = {}
    for v in variants:
        t = emit(kit, v, os.path.join(out, f".k_{v}.loom"))
        t = t.replace('export("g_moe_ggu_iq4_xs_k2048") @g_moe_ggu_iq4_xs_k2048(', f'export("k_ggu_{v}") @k_ggu_{v}(')
        t = t.replace('export("g_moe_gdn_iq4_xs_k768") @g_moe_gdn_iq4_xs_k768(', f'export("k_gdn_{v}") @k_gdn_{v}(')
        assert f"@k_ggu_{v}(" in t and f"@k_gdn_{v}(" in t, v
        texts[v] = t
        parts = [helpers(kit)]
        for T in TS:   # one copy per shape so each timing row has its own kernel name
            mt, mr = shapes[T]
            parts.append(t.replace(f"k_ggu_{v}", f"k_ggu_{T}").replace(f"k_gdn_{v}", f"k_gdn_{T}"))
            for kind in ("ggu", "gdn"):
                parts.append(f"check.case public @b_{kind}_{T}_case {{\n" + case(T, None, None) +
                             f"  %yg = check.generate.fill value(0.0) : tensor<{mr}x{F}xf32>\n  %yd = check.generate.fill value(0.0) : tensor<{mr}x{H}xf32>\n" +
                             launches(T, f"k_ggu_{T}" if kind == "ggu" else None, f"k_gdn_{T}" if kind == "gdn" else None, "%yg", "%yd") +
                             f"  check.return\n}}\ncheck.benchmark<@b_{kind}_{T}_case> @b_{kind}_{T}_bm\n")
        fn = "bench.loom" if v == "base" else f"bench_{v}.loom"
        open(os.path.join(out, fn), "w").write("\n".join(parts))
    # correctness: every non-base variant vs v1 (exact for v1-derived), base vs v1 within f16 rounding of the dequant
    ref = "v1" if "v1" in texts else variants[0]
    tparts = [helpers(kit)] + [texts[v] for v in variants]
    T = TS[0]
    mt, mr = shapes[T]
    for v in variants:
        if v in (ref, "base"):   # base (the value helper) rounds differently; it is covered by the model's kernel tests
            continue
        tol = "atol(0.0) rtol(0.0)" if not v.startswith("x") else "atol(1.0e-3) rtol(1.0e-3)"   # x*: variants that reorder sums
        tparts.append(f"check.case public @t_{v} {{\n" + case(T, None, None) +
                      f"  %g0 = check.generate.fill value(0.0) : tensor<{mr}x{F}xf32>\n  %d0 = check.generate.fill value(0.0) : tensor<{mr}x{H}xf32>\n"
                      f"  %g1 = check.generate.fill value(0.0) : tensor<{mr}x{F}xf32>\n  %d1 = check.generate.fill value(0.0) : tensor<{mr}x{H}xf32>\n" +
                      launches(T, f"k_ggu_{ref}", f"k_gdn_{ref}", "%g0", "%d0") + launches(T, f"k_ggu_{v}", f"k_gdn_{v}", "%g1", "%d1") +
                      f"  check.expect.approximate actual(%g1) expected(%g0) {tol} nan(different) : tensor<{mr}x{F}xf32>, tensor<{mr}x{F}xf32>\n"
                      f"  check.expect.approximate actual(%d1) expected(%d0) {tol} nan(different) : tensor<{mr}x{H}xf32>, tensor<{mr}x{H}xf32>\n  check.return\n}}\n")
    open(os.path.join(out, "test.loom"), "w").write("\n".join(tparts))
    neg = [helpers(kit), texts[ref], f"check.case public @t_negative {{\n" + case(T, None, None) +
           f"  %g0 = check.generate.fill value(0.0) : tensor<{mr}x{F}xf32>\n  %d0 = check.generate.fill value(0.0) : tensor<{mr}x{H}xf32>\n" +
           launches(T, f"k_ggu_{ref}", None, "%g0", "%d0") +
           f"  %z = check.generate.fill value(0.0) : tensor<{mr}x{F}xf32>\n"
           f"  check.expect.approximate actual(%g0) expected(%z) atol(0.0) rtol(0.0) nan(different) : tensor<{mr}x{F}xf32>, tensor<{mr}x{F}xf32>\n  check.return\n}}\n"]
    open(os.path.join(out, "negative.loom"), "w").write("\n".join(neg))


if __name__ == "__main__":
    main()
