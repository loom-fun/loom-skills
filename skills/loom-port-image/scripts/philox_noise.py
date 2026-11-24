#!/usr/bin/env python3
"""NumPy reproduction of the Loom initial-noise draw (motif/tensor/random.loom), so a diffusers oracle starts from Loom's latents.

usage: philox_noise.py --self-test
       philox_noise.py --seed S --size WxH --layout {flux2,zimage,flat} [--out noise.npy]

The kernel draws Philox4x32-10 over counter i = (i, 0, 0, 0) with key (seed_lo, seed_hi), turns each
word pair into two normals with Box-Muller (half-bin offset, open-interval clamp, angle in turns) and
rounds F32 -> BF16. Counter i fills elements 4i..4i+3 of the card's packed latent, row-major.

Layouts (what the request program packs and what the diffusers pipeline accepts as `latents=`):
  flux2   packed [tokens, 128], tokens = (H/16)*(W/16)   -> (1, 128, H/16, W/16)   (Flux2KleinPipeline)
  zimage  patch tokens [tokens, 64], feature (ph*2+pw)*16+c -> (1, 16, H/8, W/8)  (ZImagePipeline)
  flat    the raw draw, 4*counters values; pass --counters
A new architecture needs its own layout: read the card's request program for the packing.

The kernel uses fast-math F32 log/cos/sin; this file uses float64 transcendentals then rounds, so a
handful of BF16 values can differ by one ulp. Report the exact-mismatch count, do not gate on zero.
"""
import argparse
import sys

import numpy as np

M_LEFT = np.uint64(0xD2511F53)
M_RIGHT = np.uint64(0xCD9E8D57)
W0 = np.uint32(0x9E3779B9)
W1 = np.uint32(0xBB67AE85)
MASK32 = np.uint64(0xFFFFFFFF)
SHIFT = np.uint64(32)


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Vectorized Philox4x32-10 over uint32 arrays; returns four uint32 arrays."""
    a, b, c, d = (np.asarray(x, dtype=np.uint32) for x in (c0, c1, c2, c3))
    key0, key1 = np.uint32(k0 & 0xFFFFFFFF), np.uint32(k1 & 0xFFFFFFFF)
    with np.errstate(over="ignore"):
        for _ in range(10):
            left = a.astype(np.uint64) * M_LEFT
            right = c.astype(np.uint64) * M_RIGHT
            a, b, c, d = ((right >> SHIFT).astype(np.uint32) ^ b ^ key0, (right & MASK32).astype(np.uint32),
                          (left >> SHIFT).astype(np.uint32) ^ d ^ key1, (left & MASK32).astype(np.uint32))
            key0 = np.uint32(key0 + W0)
            key1 = np.uint32(key1 + W1)
    return a, b, c, d


def normal_pair(radius_word, angle_word):
    """F32 Box-Muller with Loom's half-bin offset, open-interval clamp and angle in turns."""
    scale = np.float32(2.3283064365386963e-10)
    half = np.float32(0.5)
    radial = np.minimum((radius_word.astype(np.float32) + half) * scale, np.float32(0.99999994039535522))
    angle = (angle_word.astype(np.float32) + half) * scale
    radius = np.sqrt(np.float32(-2.0) * np.log(radial.astype(np.float64)).astype(np.float32))
    turns = 2.0 * np.pi * angle.astype(np.float64)
    return (radius * np.cos(turns).astype(np.float32)).astype(np.float32), \
           (radius * np.sin(turns).astype(np.float32)).astype(np.float32)


def round_bf16(x):
    """Round F32 to BF16 (nearest-even), returned as F32 holding BF16 values."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    with np.errstate(over="ignore"):
        return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def loom_noise(seed, n_counters):
    """4*n_counters BF16-valued standard normals, in kernel output order."""
    seed &= (1 << 64) - 1
    counter = np.arange(n_counters, dtype=np.uint32)
    zero = np.zeros_like(counter)
    a, b, c, d = philox4x32_10(counter, zero, zero, zero, seed & 0xFFFFFFFF, seed >> 32)
    n0, n1 = normal_pair(a, b)
    n2, n3 = normal_pair(c, d)
    return round_bf16(np.stack([n0, n1, n2, n3], axis=1).reshape(-1))


def _grid(height, width):
    if height % 16 or width % 16:
        raise ValueError("height and width must be multiples of 16")
    return height // 16, width // 16


def flux2_latents(seed, height, width):
    """(1, 128, H/16, W/16) from the packed [tokens, 128] draw."""
    h, w = _grid(height, width)
    return loom_noise(seed, h * w * 32).reshape(h, w, 128).transpose(2, 0, 1)[None].copy()


def zimage_latents(seed, height, width):
    """(1, 16, H/8, W/8) from [tokens, 64] patch tokens, feature f = (ph*2 + pw)*16 + c."""
    h, w = _grid(height, width)
    tokens = loom_noise(seed, h * w * 16).reshape(h, w, 2, 2, 16)
    return tokens.transpose(4, 0, 2, 1, 3).reshape(16, 2 * h, 2 * w)[None].copy()


LAYOUTS = {"flux2": flux2_latents, "zimage": zimage_latents}

# Random123 known-answer vectors for philox4x32_10.
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


def self_test():
    ok = True
    for ctr, key, want in KAT:
        got = tuple(int(w[0]) for w in philox4x32_10(*[np.array([v], np.uint32) for v in ctr], *key))
        ok &= got == want
        print(f"philox KAT ctr={ctr[0]:08x}.. key={key[0]:08x}: {'ok' if got == want else 'FAIL'}")
    z = loom_noise(42, 1 << 18)
    moments = abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01 and bool(np.isfinite(z).all())
    print(f"normals n={z.size} mean={z.mean():+.4f} std={z.std():.4f}: {'ok' if moments else 'FAIL'}")
    return bool(ok and moments)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--size", default="1024x1024", help="WxH")
    ap.add_argument("--layout", choices=[*LAYOUTS, "flat"], default="flux2")
    ap.add_argument("--counters", type=int, help="flat layout: number of Philox counters")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.self_test or args.seed is None:
        sys.exit(0 if self_test() else 1)
    width, height = (int(v) for v in args.size.lower().split("x"))
    if args.layout == "flat":
        if not args.counters:
            ap.error("--layout flat needs --counters")
        noise = loom_noise(args.seed, args.counters)
    else:
        noise = LAYOUTS[args.layout](args.seed, height, width)
    out = args.out or f"noise_{args.layout}_s{args.seed}_{width}x{height}.npy"
    np.save(out, noise)
    print(f"{out}: shape={noise.shape}")


if __name__ == "__main__":
    main()
