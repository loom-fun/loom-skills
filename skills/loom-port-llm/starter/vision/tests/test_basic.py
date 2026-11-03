import numpy as np, math, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import gen_vision as G, vk_common as V, ref_vit as R
d = __import__("os").environ.get("VISION_TEST_DIR", __import__("os").path.join(__import__("tempfile").gettempdir(), "loom_vision_tests")); __import__("os").makedirs(d, exist_ok=True); rng = np.random.default_rng(0); src = []; cases = []
for W, mode in [(768, "ln"), (1536, "rmsw"), (64, "rms"), (96, "rmsw")]:
    nm, s = G.norm(W, mode, 1e-6); src.append(s)
    rows = 37; x = rng.standard_normal((rows, W)).astype(np.float32) * 3 + 1; w = rng.standard_normal(W).astype(np.float32); b = rng.standard_normal(W).astype(np.float32)
    ref = R.ln(x, w, b, 1e-6) if mode == "ln" else R.rms(x, w if mode == "rmsw" else None, 1e-6)
    c = V.Case(f"t_{nm}", d); r = c.lit(rows); xa = c.arr(x); args = [r, xa] + ([c.arr(w), c.arr(b)] if mode == "ln" else [c.arr(w)] if mode == "rmsw" else []); y = c.out((rows, W))
    c.launch(nm, [r], args + [y]); c.expect(y, ref, 1e-4, 1e-4); cases.append(c)
for op in ["add", "gelu_tanh", "gelu_erf", "gate_gelu_tanh"]:
    nm, s = G.ew(op); src.append(s); n = 1000
    x = (rng.standard_normal(n) * 4).astype(np.float32); b = rng.standard_normal(n).astype(np.float32)
    ref = {"add": x + b, "gelu_tanh": R.gelu_tanh(x), "gelu_erf": R.gelu_erf(x), "gate_gelu_tanh": R.gelu_tanh(x) * b}[op]
    c = V.Case(f"t_{nm}", d); nn = c.lit(n); args = [nn, c.arr(x)] + ([c.arr(b)] if G.EW_OPS[op][2] else []); y = c.out((n,))
    c.launch(nm, [nn], args + [y]); c.expect(y, ref, 1e-5, 1e-5); cases.append(c)
for N, act in [(3072, "gelu_erf"), (768, "none")]:
    nm, s = G.bias_act(N, act); src.append(s); rows = 5
    x = rng.standard_normal((rows, N)).astype(np.float32) * 3; b = rng.standard_normal(N).astype(np.float32)
    ref = {"none": x + b, "gelu_erf": R.gelu_erf(x + b)}[act]
    c = V.Case(f"t_{nm}", d); r = c.lit(rows); y = c.out((rows, N)); c.launch(nm, [r], [r, c.arr(x), c.arr(b), y]); c.expect(y, ref, 1e-5, 1e-5); cases.append(c)
V.run("\n".join(src), cases, d)
