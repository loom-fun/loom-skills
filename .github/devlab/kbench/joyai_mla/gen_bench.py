#!/usr/bin/env python3
"""JoyAI-LLM-Flash (deepseek2 MLA: 32 heads, kv_lora 512, nope 128, v 128, q head 192, latent row 576) prefill k_b / v_b
per-head WMMA products as the deepseek2 example kit generates them (gen_mla.bmm / bmm2), at 1024- and 4096-token epochs.

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   (devlab_kbench.sh "gen"; base = bench.loom)
k_kbm_<T>: y[t, h, 0:512] = W_kb[h] (512 x 128 iq4_nl) . q[t, h, 0:128]; k_vbm_<T>: y[t, h, 0:128] = W_vb[h] (128 x 512 iq4_xs) . o[t, h, 0:512].
test.loom: every non-base variant against base on the same inputs, exact; plus a negative control."""
import json, os, subprocess, sys
import numpy as np

H, KL, NOPE, VD, QB, D = 32, 512, 128, 128, 192, 576
TS = [int(x) for x in os.environ.get("JOY_MLA_T", "1024,4096").split(",")]
VARIANTS = {"base": {}, "b2": {"MLA_BMM2": "1"}}
VARIANTS.update(json.loads(os.environ.get("JOY_MLA_VARIANTS", "{}")))
FMTB = {"iq4_nl": (18, 32), "iq4_xs": (136, 256)}


def blocks(rng, fmt, rows, k):
    bb, sb = FMTB[fmt]
    nb = rows * (k // sb)
    b = rng.integers(0, 256, (nb, bb), dtype=np.uint8)
    b[:, 0:2] = rng.uniform(5e-4, 2e-3, nb).astype("<f2").view(np.uint8).reshape(nb, 2)
    return b.reshape(-1).view(np.int8)


def emit(kit, variant, out):
    env = dict(os.environ, LOOM_RDNA4="0", **VARIANTS[variant])
    code = f"""
import sys; sys.path.insert(0, {kit!r})
import gen_mla
rb = lambda bb, sb, K: K // sb * bb
t = gen_mla._bmm()("g_mla_kbm_iq4_nl", "iq4_nl", {NOPE}, {KL}, {H}, {QB}, {D}, 18, 32, rb(18, 32, {NOPE}))
t += gen_mla._bmm()("g_mla_vbm_iq4_xs", "iq4_xs", {KL}, {VD}, {H}, {D}, {VD}, 136, 256, rb(136, 256, {KL}))
open({out!r}, "w").write(t)
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    return open(out).read()


def main():
    root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    variants = sys.argv[3:] or ["base", "b2"]
    os.makedirs(os.path.join(out, "w"), exist_ok=True)
    kit = os.path.join(out, "kit")
    if not os.path.isdir(kit):
        subprocess.run(["bash", os.path.join(root, "skills/loom-port-llm/reference/apply_example.sh"), "deepseek2", kit], check=True, stdout=subprocess.DEVNULL)
    helpers = open(os.path.join(kit, "quant_values.loom")).read()
    rng = np.random.default_rng(1361)
    wkb = blocks(rng, "iq4_nl", H * KL, NOPE)
    wvb = blocks(rng, "iq4_xs", H * VD, KL)
    np.save(os.path.join(out, "w", "wkb.npy"), wkb)
    np.save(os.path.join(out, "w", "wvb.npy"), wvb)
    for T in TS:
        np.save(os.path.join(out, "w", f"q{T}.npy"), rng.uniform(-1, 1, (T, H, QB)).astype(np.float32))
        np.save(os.path.join(out, "w", f"o{T}.npy"), rng.uniform(-1, 1, (T, H, D)).astype(np.float32))

    def case(T):
        return f"""  %q = check.file.read.npy path("w/q{T}.npy") : tensor<{T}x{H}x{QB}xf32>
  %o = check.file.read.npy path("w/o{T}.npy") : tensor<{T}x{H}x{D}xf32>
  %wkb = check.file.read.npy path("w/wkb.npy") : tensor<{wkb.size}xi8>
  %wvb = check.file.read.npy path("w/wvb.npy") : tensor<{wvb.size}xi8>
  %t = check.literal value({T}) : index
"""

    def launch(T, kind, kn, y):
        if kind == "kbm":
            return (f"  kernel.launch @{kn}[%t](%t, %q, %wkb, {y}) : [index](index, tensor<{T}x{H}x{QB}xf32>, tensor<{wkb.size}xi8>, "
                    f"tensor<{T}x{H}x{D}xf32>)\n")
        return (f"  kernel.launch @{kn}[%t](%t, %o, %wvb, {y}) : [index](index, tensor<{T}x{H}x{D}xf32>, tensor<{wvb.size}xi8>, "
                f"tensor<{T}x{H}x{VD}xf32>)\n")
    ys = {"kbm": lambda T: f"tensor<{T}x{H}x{D}xf32>", "vbm": lambda T: f"tensor<{T}x{H}x{VD}xf32>"}
    texts = {}
    for v in variants:
        t = emit(kit, v, os.path.join(out, f".k_{v}.loom"))
        t = t.replace("g_mla_kbm_iq4_nl", f"k_kbm_{v}").replace("g_mla_vbm_iq4_xs", f"k_vbm_{v}")
        texts[v] = t
        parts = [helpers]
        for T in TS:
            parts.append(t.replace(f"k_kbm_{v}", f"k_kbm_{T}").replace(f"k_vbm_{v}", f"k_vbm_{T}"))
            for kind in ("kbm", "vbm"):
                parts.append(f"check.case public @b_{kind}_{T}_case {{\n" + case(T) + f"  %y = check.generate.fill value(0.0) : {ys[kind](T)}\n"
                             + launch(T, kind, f"k_{kind}_{T}", "%y") + f"  check.return\n}}\ncheck.benchmark<@b_{kind}_{T}_case> @b_{kind}_{T}_bm\n")
        open(os.path.join(out, "bench.loom" if v == "base" else f"bench_{v}.loom"), "w").write("\n".join(parts))
    T = TS[0]
    tp = [helpers] + [texts[v] for v in variants]
    for v in variants:
        if v == "base":
            continue
        body = case(T)
        for kind in ("kbm", "vbm"):
            body += (f"  %{kind}0 = check.generate.fill value(0.0) : {ys[kind](T)}\n  %{kind}1 = check.generate.fill value(0.0) : {ys[kind](T)}\n"
                     + launch(T, kind, f"k_{kind}_base", f"%{kind}0") + launch(T, kind, f"k_{kind}_{v}", f"%{kind}1")
                     + f"  check.expect.approximate actual(%{kind}1) expected(%{kind}0) atol(0.0) rtol(0.0) nan(different) : {ys[kind](T)}, {ys[kind](T)}\n")
        tp.append(f"check.case public @t_{v} {{\n{body}  check.return\n}}\n")
    open(os.path.join(out, "test.loom"), "w").write("\n".join(tp))
    neg = [helpers, texts["base"], f"check.case public @t_negative {{\n" + case(T) + f"  %y = check.generate.fill value(0.0) : {ys['vbm'](T)}\n"
           + launch(T, "vbm", "k_vbm_base", "%y") + f"  %z = check.generate.fill value(0.0) : {ys['vbm'](T)}\n"
           f"  check.expect.approximate actual(%y) expected(%z) atol(0.0) rtol(0.0) nan(different) : {ys['vbm'](T)}, {ys['vbm'](T)}\n  check.return\n}}\n"]
    open(os.path.join(out, "negative.loom"), "w").write("\n".join(neg))


if __name__ == "__main__":
    main()
