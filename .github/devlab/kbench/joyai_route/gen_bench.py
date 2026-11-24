#!/usr/bin/env python3
"""JoyAI-LLM-Flash prefill router kernels (gen_moe: g_moe_route_tr_live / _tg_live / _wm_live) on a 2197-token prompt in the
4096-token epoch (1899 padding tokens), hidden 2048, 256 experts, top-8 sigmoid + bias, normalized, scale 2.5.

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   (devlab_kbench.sh "gen")
Writes bench_<v>.loom (case c4k: one launch) and test.loom (tr / tg / wm ids vs tr's on the live tokens; timing probes *nk
(no top-k) untested)."""
import os, subprocess, sys
import numpy as np

H, E, KU, T, LIVE = 2048, 256, 8, 4096, 2196   # live count a multiple of 4: tr computes whole 4-token groups
VARIANTS = {"tr": ({}, "g_moe_route_tr_live"), "tg": ({"MOE_ROUTE_TG": "1"}, "g_moe_route_tg_live"),
            "wm": ({"MOE_ROUTE_WM": "1"}, "g_moe_route_wm_live"),
            "tgnk": ({"MOE_ROUTE_TG": "1", "MOE_ROUTE_NOTOPK": "1"}, "g_moe_route_tg_live"),
            "wmnk": ({"MOE_ROUTE_WM": "1", "MOE_ROUTE_NOTOPK": "1"}, "g_moe_route_wm_live")}
HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"


def emit(kit, v, out):
    env = dict(os.environ, LOOM_RDNA4="0", **VARIANTS[v][0])
    fn = {"tr": "route_prefill_live", "tg": "route_prefill_tg", "wm": "route_prefill_wm"}[v[:2]]
    code = f"""
import sys; sys.path.insert(0, {kit!r})
import gen_moe
open({out!r}, "w").write(gen_moe.{fn}({H}, {E}, {KU}, 2, True, True, "f32", scale=2.5))
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    t = open(out).read()
    k = VARIANTS[v][1]
    return t.replace(f"@{k}(", f"@{k}_{v}(").replace(f'export("{k}")', f'export("{k}_{v}")'), f"{k}_{v}"


def main():
    root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    vs = sys.argv[3:] or list(VARIANTS)
    os.makedirs(out, exist_ok=True)
    kit = os.path.join(out, "kit")
    if not os.path.isdir(kit):
        subprocess.run(["bash", os.path.join(root, "skills/loom-port-llm/reference/apply_example.sh"), "deepseek2", kit], check=True, stdout=subprocess.DEVNULL)
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((T, H)) * 0.5).astype(np.float32); x[LIVE:] = np.nan   # padding rows: garbage
    w = (rng.standard_normal((E * H,)) * 0.02).astype(np.float32)
    b = (rng.standard_normal(E) * 0.01).astype(np.float32)
    info = np.zeros((T, 4), np.int32); info[:LIVE, 1] = np.arange(LIVE); info[:LIVE, 2] = 1
    for n, a in (("x", x), ("w", w), ("b", b), ("info", info)):
        np.save(os.path.join(out, f"r_{n}.npy"), a)
    decl = f"""  %t = check.literal value({T}) : index
  %x = check.file.read.npy path("r_x.npy") : tensor<{T}x{H}xf32>
  %w = check.file.read.npy path("r_w.npy") : tensor<{E * H}xf32>
  %b = check.file.read.npy path("r_b.npy") : tensor<{E}xf32>
  %info = check.file.read.npy path("r_info.npy") : tensor<{T}x4xi32>
"""
    sig = f"[index](index, tensor<{T}x{H}xf32>, tensor<{E * H}xf32>, tensor<{E}xf32>, tensor<{T}x{KU}xi32>, tensor<{T}x{KU}xf32>, tensor<{T}x4xi32>)"
    texts, names = {}, {}
    for v in set(vs) | {"tr"}:
        texts[v], names[v] = emit(kit, v, os.path.join(out, f"k_{v}.loom"))
    for v in vs:
        open(os.path.join(out, f"bench_{v}.loom"), "w").write(HEAD + texts[v] + f"""
check.case public @c4k {{
{decl}  %ids = check.generate.fill value(0) : tensor<{T}x{KU}xi32>
  %wts = check.generate.fill value(0.0) : tensor<{T}x{KU}xf32>
  kernel.launch @{names[v]}[%t](%t, %x, %w, %b, %ids, %wts, %info) : {sig}
  check.return
}}
check.benchmark<@c4k> @bm_c4k
""")
    tv = [v for v in vs if not v.endswith("nk") and v != "tr"]
    body = HEAD + texts["tr"] + "".join(texts[v] for v in tv)
    for v in tv:   # ids of the live tokens equal tr's (padding tokens: tr writes groups with a live token, tg/wm write none)
        body += f"""
check.case public @t_{v} {{
{decl}  %ids0 = check.generate.fill value(0) : tensor<{T}x{KU}xi32>
  %wts0 = check.generate.fill value(0.0) : tensor<{T}x{KU}xf32>
  %ids = check.generate.fill value(0) : tensor<{T}x{KU}xi32>
  %wts = check.generate.fill value(0.0) : tensor<{T}x{KU}xf32>
  kernel.launch @{names['tr']}[%t](%t, %x, %w, %b, %ids0, %wts0, %info) : {sig}
  kernel.launch @{names[v]}[%t](%t, %x, %w, %b, %ids, %wts, %info) : {sig}
  check.expect.equal actual(%ids) expected(%ids0) : tensor<{T}x{KU}xi32>
  check.expect.close actual(%wts) expected(%wts0) atol(1.0e-5) rtol(1.0e-4) nan(different) : tensor<{T}x{KU}xf32>
  check.return
}}
"""
    open(os.path.join(out, "test.loom"), "w").write(body)


if __name__ == "__main__":
    main()
