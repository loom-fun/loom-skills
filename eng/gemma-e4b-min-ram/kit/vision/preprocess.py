"""llama.cpp (mtmd) image preprocessing, per projector family: target size + Pillow-compatible resample.
Reference for the Loom image programs. gemma4v: plain resize to the target size, no letterbox padding, as in
upstream llama.cpp 7bd18083 (lemonade b1368); older builds (b1355) letterbox, so compare against b1368 or newer."""
import math, numpy as np

# family -> (align = patch * merge, min_tokens, max_tokens, filter, round_half_even)
FAMILY = {
    "gemma4v":        dict(merge=3, tokens=(70, 1120), filt="bicubic", half_even=False),
    "qwen3vl_merger": dict(merge=2, tokens=(8, 4096),  filt="bicubic", half_even=True),
    "qwen2.5vl_merger": dict(merge=2, tokens=(8, 4096), filt="bicubic", half_even=True),
    "muse-glimmer":   dict(merge=2, tokens=(1, 4096),  filt="lanczos", half_even=False),
}

def target_size(w, h, patch, merge, tmin, tmax, half_even):
    f = patch * merge; area = f * f; mn, mx = tmin * area, tmax * area
    def rnd(x):
        q = np.float32(x) / np.float32(f)
        return int(np.rint(q) if half_even else math.floor(q + 0.5)) * f   # std::round = half away from zero (q > 0)
    wb, hb = max(f, rnd(w)), max(f, rnd(h))
    if hb * wb > mx:
        beta = np.float32(math.sqrt(np.float32(h) * w / mx))
        hb = max(f, int(math.floor(np.float32(h / beta) / f)) * f); wb = max(f, int(math.floor(np.float32(w / beta) / f)) * f)
    elif hb * wb < mn:
        beta = np.float32(math.sqrt(np.float32(mn) / (np.float32(h) * w)))
        hb = int(math.ceil(np.float32(h * beta) / f)) * f; wb = int(math.ceil(np.float32(w * beta) / f)) * f
    return wb, hb

def _filter(name):
    if name == "lanczos":
        sinc = lambda v: 1.0 if v == 0 else math.sin(math.pi * v) / (math.pi * v)
        return 3.0, lambda x: sinc(x) * sinc(x / 3.0) if -3.0 <= x < 3.0 else 0.0
    if name == "bilinear":
        return 1.0, lambda x: max(0.0, 1.0 - abs(x))
    a = -0.5
    def cubic(x):
        x = abs(x)
        if x < 1: return ((a + 2) * x - (a + 3)) * x * x + 1
        if x < 2: return (((x - 5) * x + 8) * x - 4) * a
        return 0.0
    return 2.0, cubic

def weights(n_in, n_out, filt):
    """Pillow precompute_weights: returns (xmin[n_out], xcnt[n_out], ksize, int32 weights[n_out, ksize])."""
    support0, fn = _filter(filt)
    scale = n_in / n_out; fs = max(scale, 1.0); support = support0 * fs; ksize = int(math.ceil(support)) * 2 + 1
    xmin = np.zeros(n_out, np.int32); xcnt = np.zeros(n_out, np.int32); W = np.zeros((n_out, ksize), np.int32)
    for xx in range(n_out):
        c = (xx + 0.5) * scale; lo = max(0, int(c - support + 0.5)); hi = min(n_in, int(c + support + 0.5)); cnt = hi - lo
        w = [fn((x + lo - c + 0.5) / fs) for x in range(cnt)]; s = sum(w)
        if s != 0: w = [v / s for v in w]
        for x, v in enumerate(w): W[xx, x] = int(v * (1 << 22) + (-0.5 if v < 0 else 0.5))
        xmin[xx], xcnt[xx] = lo, cnt
    return xmin, xcnt, ksize, W

def _pass(img, axis, n_out, filt):
    """img uint8 [H, W, 3]; resample along axis (1 = horizontal, 0 = vertical) with Pillow fixed point."""
    n_in = img.shape[axis]; xmin, xcnt, k, W = weights(n_in, n_out, filt)
    src = np.moveaxis(img, axis, 0).astype(np.int64); out = np.empty((n_out,) + src.shape[1:], np.uint8)
    for o in range(n_out):
        acc = np.full(src.shape[1:], 1 << 21, np.int64)
        for x in range(xcnt[o]): acc += src[xmin[o] + x] * int(W[o, x])
        out[o] = np.clip(acc >> 22, 0, 255)
    return np.moveaxis(out, 0, axis)

def resize(img, w, h, filt):
    if img.shape[1] != w: img = _pass(img, 1, w, filt)
    if img.shape[0] != h: img = _pass(img, 0, h, filt)
    return img

def preprocess(img_u8, family, patch):
    """uint8 [H, W, 3] -> float32 [h, w, 3] in [0, 1] at the processed size (normalization is the encoder's)."""
    F = FAMILY[family]; w, h = target_size(img_u8.shape[1], img_u8.shape[0], patch, F["merge"], *F["tokens"], F["half_even"])
    return resize(img_u8, w, h, F["filt"]).astype(np.float32) / 255.0
