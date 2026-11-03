import numpy as np, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import gen_vision as G, vk_common as V
d = __import__("os").environ.get("VISION_TEST_DIR", __import__("os").path.join(__import__("tempfile").gettempdir(), "loom_vision_tests")); __import__("os").makedirs(d, exist_ok=True); rng = np.random.default_rng(1); src = []; cases = []
rs, s = G.row_scale(); src.append(s)
for fmt, (M, K, N) in [("f16", (128, 768, 2304)), ("bf16", (64, 1536, 8960)), ("f32", (64, 64, 128))]:
    nm, s = G.gemm(fmt, K, N); src.append(s)
    x = (rng.standard_normal((M, K)) * (100 if fmt == "bf16" else 1)).astype(np.float32); x[M - 7:] = 0   # padded rows
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    if fmt == "f16": wb = w.astype(np.float16); wf = wb.astype(np.float32); raw = wb.view(np.int8)
    elif fmt == "bf16":
        import ref_vit as R; wf = R.bf16(w); raw = (wf.view(np.uint32) >> 16).astype(np.uint16).view(np.int8)
    else: wf = w; raw = w.view(np.int8)
    ref = x @ wf.T
    c = V.Case(f"t_{nm}", d); m = c.lit(M); k = c.lit(K); n = c.lit(N)
    xa = c.arr(x); sc = c.out((M, 2)); x16 = c.out((M, K), "f16"); wa = c.arr(raw.reshape(-1)); y = c.out((M, N))
    c.launch(rs, [m], [m, k, xa, sc, x16]); c.launch(nm, [m, n], [m, k, n, x16, sc, wa, y])
    c.expect(y, ref, 2e-2 * np.abs(ref).max() / 10, 2e-2); cases.append(c)
V.run("\n".join(src), cases, d)
