"""Whole-encoder test: plan -> check.case with real mmproj weights; compares to ref_vit. usage: test_encoder.py MMPROJ H W [atol]"""
import sys, os, numpy as np, gguf, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import encoder as EN, vk_common as V, ref_vit as R
path, H, Wd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = EN.config(path); p = EN.build(c); P = c["patch"]; gh, gw = H // P, Wd // P; T = gh * gw; M = T // c["merge"] ** 2
d = os.environ.get("VISION_TEST_DIR", os.path.join(__import__("tempfile").gettempdir(), "loom_vision_tests")) + "/encoder"; os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(5)
# smooth synthetic image (u8)
yy, xx = np.mgrid[0:H, 0:Wd]; img = np.stack([(127 + 120 * np.sin(xx / 7.0)), (127 + 120 * np.cos(yy / 5.0)), ((xx + yy) % 256)], -1).astype(np.uint8)
vals = dict(T=T, Tp=-(-T // 256) * 256, M=M, Mp=-(-M // 64) * 64, gw=gw, gh=gh, img_w=Wd, n_pix=H * Wd * 3, NH=c["NH"])
def ev(expr):
    return int(eval(expr.replace("Tp", str(vals["Tp"])).replace("Mp", str(vals["Mp"])), {}, {k: v for k, v in vals.items()}))
r = gguf.GGUFReader(path); raw = {t.name: np.array(t.data).reshape(-1).view(np.int8) for t in r.tensors}
cs = V.Case("t_encoder_" + c["family"].replace("-", "_"), d)
bufs = {n: cs.out((sz,), "f16" if n == "x16" else "f32") for n, sz in EN.buffers(c, T, M).items()}
bufs["pix"] = cs.arr(img.reshape(-1)); bufs["pixf"] = cs.out((H * Wd * 3,))
wts = {}; lits = {}
def arg(a):
    kind, n = a
    if kind == "B": return bufs[n]
    if kind == "W":
        if n not in wts: wts[n] = cs.arr(np.ascontiguousarray(raw[n]))
        return wts[n]
    v = ev(n)
    if v not in lits: lits[v] = cs.lit(v)
    return lits[v]
for kern, grid, args in p.ops:
    cs.launch(kern, [arg(g) for g in grid], [arg(a) for a in args])
pre = img.astype(np.float32) / 255.0
ref = R.ENCODERS[c["family"]].__wrapped__(None) if False else None
W_, KV = R.load(path)
fn = {"gemma4v": lambda: R.gemma4v(W_, KV, pre), "qwen3vl_merger": lambda: R.qwen3vl(W_, KV, pre), "muse-glimmer": lambda: R.muse_glimmer(W_, KV, pre)}[c["family"]]
ref = fn().astype(np.float32)
out_rows = np.zeros((vals["Mp"], c["out"]), np.float32); out_rows[:M] = ref
# compare only the first M rows: write expected padded with actual-agnostic tolerance -> compare a view instead
cs.lines.append(f"  %outv = check.view {bufs['out'][0]} : {bufs['out'][1]}") if False else None
e = cs.arr(np.ascontiguousarray(out_rows.reshape(-1)[: M * c["out"]]))
cs.lines.append(f"  %got = check.slice {bufs['out'][0]} [0] [{M * c['out']}] : {bufs['out'][1]} -> tensor<{M * c['out']}xf32>") if False else None
np.save(os.path.join(d, "ref.npy"), ref)
atol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
RTOL = sys.argv[5] if len(sys.argv) > 5 else "5.0e-2"
print("kernels:", len(p.kernels), "launches:", len(p.ops), "T", T, "M", M, "ref |max|", np.abs(ref).max())
# expectation over the full padded buffer: rows >= M are don't-care, so copy them from... use a mask trick: expected = ref padded with 0 and compare
# only possible if padded rows are 0: they are (merged rows beyond M are never written: buffers start at 0)
full = np.zeros(EN.buffers(c, T, M)["out"], np.float32); full[: M * c["out"]] = ref.reshape(-1)
ea = cs.arr(full)
cs.lines.append(f'  check.file.write.npy value({bufs["out"][0]}) path("got.npy") mode(always) : {bufs["out"][1]}')
cs.lines.append(f"  check.expect.approximate actual({bufs['out'][0]}) expected({ea[0]}) atol({atol:e}) rtol({RTOL}) nan(different) : {bufs['out'][1]}, {ea[1]}")
ok = V.run("\n".join(p.kernels.values()), [cs], d)

got = np.load(os.path.join(d, "got.npy"))[: M * c["out"]].reshape(M, c["out"])
err = np.abs(got - ref); rel = err / (np.abs(ref) + 1e-3)
print("max abs err", err.max(), "mean abs err", err.mean(), "ref mean |x|", np.abs(ref).mean(), "frac(err>0.1)", (err > 0.1).mean(),
      "row-cos min", min(float(np.dot(got[i], ref[i]) / np.linalg.norm(got[i]) / np.linalg.norm(ref[i])) for i in range(M)))
