#!/usr/bin/env python3
"""Report the E8M0 block-exponent range of a Quark/OCP MXFP4 safetensors checkpoint (every `*.weight_scale` U8 tensor), and whether the
gfx12 fp8 prefill GEMM (gen_mxfp4s.py) decodes every block exactly (107 <= e <= 132).

usage: scan_mxfp4s_exponents.py SRC [--json OUT]
SRC: a local .safetensors file, or HF_REPO[@REV] (reads the header and every scale plane by HTTP range, ~0.8 GB for a 27B model).
Prints per-range statistics; the exit code is 1 when a block is outside the exact range (the rows / tensors that need the exact fallback are listed)."""
import json, struct, sys, os, urllib.request
try:
    import numpy as np
except ImportError:
    np = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_mxfp4s import MXFP4_F8_E_MIN, MXFP4_F8_E_MAX


class Src:
    def __init__(self, src):
        self.local = os.path.exists(src)
        if self.local:
            self.f = open(src, "rb")
        else:
            repo, _, rev = src.partition("@")
            self.url = f"https://huggingface.co/{repo}/resolve/{rev or 'main'}/model.safetensors"

    def read(self, off, n):
        if self.local:
            self.f.seek(off); return self.f.read(n)
        for attempt in range(4):
            try:
                r = urllib.request.Request(self.url, headers={"Range": f"bytes={off}-{off + n - 1}"})
                with urllib.request.urlopen(r, timeout=120) as u:
                    return u.read()
            except Exception:
                if attempt == 3: raise


def scan(src):
    s = Src(src)
    hl = struct.unpack("<Q", s.read(0, 8))[0]
    hdr = json.loads(s.read(8, hl)); base = 8 + hl
    rows = []
    for name, t in hdr.items():
        if not name.endswith(".weight_scale") or t["dtype"] != "U8":
            continue
        a, b = t["data_offsets"]
        e = np.frombuffer(s.read(base + a, b - a), np.uint8).reshape(t["shape"])
        out = (e < MXFP4_F8_E_MIN) | (e > MXFP4_F8_E_MAX)
        rows.append({"tensor": name, "shape": t["shape"], "min": int(e.min()), "max": int(e.max()), "outside": int(out.sum()),
                     "rows_outside": int(out.any(1).sum()), "hist": np.bincount(e.ravel(), minlength=256).tolist()})
    return rows


if __name__ == "__main__":
    rows = scan(sys.argv[1])
    lo, hi = min(r["min"] for r in rows), max(r["max"] for r in rows)
    tot = sum(sum(r["hist"]) for r in rows); out = sum(r["outside"] for r in rows)
    print(f"{len(rows)} scale planes, {tot} blocks, e8m0 {lo}..{hi}; exact range {MXFP4_F8_E_MIN}..{MXFP4_F8_E_MAX}; {out} blocks outside")
    for r in rows:
        if r["outside"]:
            print(f"  {r['tensor']} {r['shape']}: e {r['min']}..{r['max']}, {r['outside']} blocks in {r['rows_outside']} rows outside")
    if "--json" in sys.argv:
        json.dump(rows, open(sys.argv[sys.argv.index("--json") + 1], "w"))
    sys.exit(1 if out else 0)
