"""Pick the GGUF to port from a Hugging Face repo, without downloading any weights.

usage: pick_gguf.py ORG/REPO [--gtt-gib N]   -> prints the chosen file on stdout, the reasoning on stderr

Candidates (single-file, not mmproj/dflash/imatrix) are ranked by quant preference
(Q4_K_M > Q4_K_S > Q5_K_M > Q6_K > Q8_0, then unsloth UD-*_K_XL mixes). For each, only the
header is fetched (HTTP range request into a sparse file) to read the tensor formats:
  - the first candidate whose formats the starter kit has kernels for (F32 norms + Q4_K/Q5_K/
    Q6_K/Q8_0 matrices) and that FITS is chosen;
  - "fits" = 2 x file + 8 GiB <= min(GTT, 60 % of RAM): the gates run llama.cpp and Loom side by
    side, and a model too big for the machine can hard-crash it (the loom skill's hard rules).
If none is fully supported, the fitting candidate with the fewest unsupported formats is chosen
(BF16/F16 count as minor: a small value helper), and those formats are named (WARNING on stderr).
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
import urllib.request

SUPPORTED = {"F32", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}
PREF = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0",
        "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"]


def gtt_bytes():
    gtt = 61 << 30
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total"):
        try:
            gtt = int(open(p).read())
            break
        except (OSError, ValueError):
            pass
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        gtt = min(gtt, int(0.6 * ram))
    except (ValueError, OSError, AttributeError):
        pass
    return gtt


def quant_of(name):
    n = name.upper()
    for q in sorted(PREF, key=len, reverse=True):
        if re.search(r"(^|[-_.])" + re.escape(q) + r"\.GGUF$", n):
            return q
    return None


def header_formats(repo, path, size):
    """Tensor formats from the GGUF header: fetch the first bytes into a sparse file of the full size."""
    from gguf import GGUFReader
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
        for n in (16 << 20, 64 << 20):
            req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"})
            data = urllib.request.urlopen(req, timeout=120).read()
            f.seek(0); f.write(data); f.truncate(size); f.flush()
            try:
                r = GGUFReader(f.name)
                return {t.tensor_type.name for t in r.tensors}
            except Exception:  # header longer than n bytes (huge vocab): try a bigger range
                continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--gtt-gib", type=float)
    a = ap.parse_args()
    gtt = a.gtt_gib * 2**30 if a.gtt_gib else gtt_bytes()
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{a.repo}/tree/main?recursive=true", timeout=60))
    files = [(f["path"], f.get("size", 0)) for f in tree if f["path"].lower().endswith(".gguf")
             and not re.search(r"mmproj|dflash|imatrix|-of-\d", f["path"], re.I)]
    cands = sorted(((PREF.index(quant_of(p)), p, s) for p, s in files if quant_of(p)), key=lambda x: x[0])
    if not cands:
        sys.exit(f"no single-file GGUF with a known quant ({', '.join(PREF)}) in {a.repo}: pass REPO:FILE")
    options = []
    for rank, p, s in cands:
        fits = 2 * s + (8 << 30) <= gtt
        if not fits:
            print(f"skip {p}: {s / 2**30:.1f} GiB does not fit twice in the {gtt / 2**30:.0f} GiB budget "
                  "(gates run llama.cpp and Loom side by side)", file=sys.stderr)
            continue
        fmts = header_formats(a.repo, p, s)
        print(f"{p}: {s / 2**30:.1f} GiB, formats {sorted(fmts) if fmts else '?'}", file=sys.stderr)
        if fmts and fmts <= SUPPORTED:
            print(p)
            return
        major = sorted((fmts or {"?"}) - SUPPORTED - {"F16", "BF16"})
        minor = sorted((fmts or set()) & {"F16", "BF16"})
        options.append(((len(major), len(minor), rank), p, major + minor))
    if not options:
        sys.exit(f"no candidate in {a.repo} fits the GPU memory budget twice; pick a smaller quant (REPO:FILE)")
    _, p, missing = min(options)
    print(f"WARNING: no candidate uses only starter formats; chose {p}. Needs dequant helpers for "
          f"{missing} (loom-port-llm: 'New quant format')", file=sys.stderr)
    print(p)


if __name__ == "__main__":
    main()
