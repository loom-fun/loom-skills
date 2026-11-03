"""Header-driven resize kernels (the encoder program's first two launches) vs preprocess.py (llama.cpp's
Pillow-compatible resample). Coefficients are computed in F32 on the device, so a few pixels may differ by 1."""
import os, sys, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gen_component as GC, vk_common as V, preprocess as PP
d = os.environ.get("VISION_TEST_DIR", os.path.join(tempfile.gettempdir(), "loom_vision_tests")); os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(7); ok = True
for filt in ["bicubic", "lanczos"]:
    (hn, hs), (vn, vs) = GC.resize_kernels(filt); cases = []; refs = []
    for (sw, sh, dw, dh, cin) in [(800, 500, 816, 480, 3), (640, 480, 624, 480, 4), (300, 1200, 288, 1152, 3)]:
        img = rng.integers(0, 256, (sh, sw, cin)).astype(np.uint8)
        off = 64 + 160
        payload = np.zeros(off + img.size, np.uint8); payload[off:] = img.reshape(-1)
        hdr = np.zeros(16, np.int32); hdr[:6] = [sw, sh, cin, off, dw, dh]; payload[:64] = hdr.view(np.uint8)
        refs.append((f"got_{filt}_{sw}.npy", PP.resize(np.ascontiguousarray(img[..., :3]), dw, dh, filt)))
        c = V.Case(f"t_resize_{filt}_{sw}x{sh}_{cin}", d)
        ow = c.lit(dw); mr = c.lit(4096); oh = c.lit(dh)
        pa = c.arr(payload); wide = c.out((4096 * dw * 3,), "i8"); pix = c.out((dh * dw * 3,), "i8")
        c.launch(hn, [ow, mr], [ow, mr, pa, wide]); c.launch(vn, [oh, ow], [oh, ow, pa, wide, pix])
        c.lines.append(f'  check.file.write.npy value({pix[0]}) path("{refs[-1][0]}") mode(always) : {pix[1]}'); cases.append(c)
    V.run(hs + "\n" + vs, cases, d)
    for fn, ref in refs:
        got = np.load(os.path.join(d, fn)).view(np.uint8).reshape(ref.shape).astype(int)
        diff = np.abs(got - ref.astype(int)); good = diff.max() <= 1 and (diff > 0).mean() < 0.01
        print(("PASS" if good else "FAIL"), fn, "max diff", diff.max(), "fraction differing", round(float((diff > 0).mean()), 5)); ok &= bool(good)
sys.exit(0 if ok else 1)
