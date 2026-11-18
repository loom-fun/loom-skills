import numpy as np, sys, json, subprocess, os
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import gen_vision as G, vk_common as V, preprocess as PP
d = __import__("os").environ.get("VISION_TEST_DIR", __import__("os").path.join(__import__("tempfile").gettempdir(), "loom_vision_tests")); __import__("os").makedirs(d, exist_ok=True); rng = np.random.default_rng(4); src = []; cases = []
img = rng.integers(0, 256, (37, 53, 3)).astype(np.uint8)
smooth = np.clip(np.cumsum(rng.integers(-9, 10, (37, 53, 3)), 1) + 128, 0, 255).astype(np.uint8)
for filt in ["bicubic", "lanczos"]:
    for axis, (oh, ow) in [("h", (37, 120)), ("h", (37, 20)), ("v", (90, 53)), ("v", (13, 53))]:
        nm, s = G.resize_pass(filt, axis, 3)
        if s not in src: src.append(s)
        for tag, im in [("noise", img), ("smooth", smooth)]:
            ref = PP._pass(im, 1 if axis == "h" else 0, ow if axis == "h" else oh, filt)
            c = V.Case(f"t_{nm}_{oh}x{ow}_{tag}", d); cnt = c.lit(oh * ow); inn = c.lit(im.shape[1] if axis == "h" else im.shape[0])
            outn = c.lit(ow if axis == "h" else oh); oth = c.lit(1 if axis == "h" else im.shape[1])
            y = c.out((oh * ow * 3,), "i8"); c.launch(nm, [cnt], [cnt, inn, outn, oth, c.arr(im.reshape(-1)), y])
            # integer outputs as i8: compare via float of the uint8 view -> use expect on the int bits with atol 1
            e = c.arr(ref.reshape(-1)); c.lines.append(f"  check.expect.approximate actual({y[0]}) expected({e[0]}) atol(1) rtol(0) : {y[1]}, {e[1]}") if False else None
            c.lines.append(f"  check.expect.equal actual({y[0]}) expected({e[0]}) : {y[1]}")
            cases.append(c)
V.run("\n".join(src), cases, d)
