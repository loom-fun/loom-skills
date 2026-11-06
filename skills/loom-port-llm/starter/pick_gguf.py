"""Pick the GGUF to port from a Hugging Face repo, preferring what is already in the HF cache.

usage: pick_gguf.py ORG/REPO [--gtt-gib N] [--file FILE]
  -> prints "FILE<TAB>CACHED_PATH" on stdout (CACHED_PATH empty when the file must be downloaded), reasons on stderr

Quant tier: Q4 by default (Q4_K_M, UD-Q4_K_XL, Q4_K_S, ...: what Loom ports are built and benchmarked with on Strix
Halo; Q4_K_M itself is the default, the others are only for repos without one); QUANT_TIER=Q5|Q6|Q8 asks for a
bigger one. The picker never leaves the tier on its own: a heavier quant decodes
proportionally slower on this bandwidth-bound machine and doubles the download (an earlier run silently fell back to
Q8_0 because every Q4 file used formats the kit lacked). Within the tier:
  1. the preference order (Q4_K_M first: a cached UD-Q4_K_XL never beats Q4_K_M; another run took that detour), then
     the HF cache (any revision; a ported model is never downloaded twice). When the chosen file would have to be
     downloaded but the same quant of the same model is cached under another repo (bartowski/ next to unsloth/),
     STOP and name that repo: port it instead of downloading. Then
  2. files whose tensor formats the starter decodes (F32 + Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/IQ4_NL/IQ4_XS/Q3_K/IQ3_S; read
     from the local file, or from the header by HTTP range request), plus what a reference example for the file's
     architecture decodes (reference/*/formats.json, e.g. gemma4: BF16), then the preference order;
  and the file must FIT: 2 x file + 8 GiB <= min(GTT, 60 % of RAM) - LOOM_RESERVED_GIB (the loom skill's hard rules).
No supported file in the tier: STOP (exit 1) naming the missing formats (add them: loom-port-llm 'New quant format').
Same-tier files of this model cached under other repos (e.g. bartowski/ next to unsloth/) are listed as a NOTE.
--file FILE: only resolve FILE (cache lookup), no ranking.
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
import urllib.request

SUPPORTED = {"F32", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "IQ4_NL", "IQ4_XS", "Q3_K", "IQ3_S"}
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
    # LOOM_RESERVED_GIB: memory another resident server already holds (weaver's agent model).
    try:
        reserved = float(os.environ.get("LOOM_RESERVED_GIB") or 0)
    except ValueError:
        sys.exit(f"LOOM_RESERVED_GIB={os.environ['LOOM_RESERVED_GIB']!r} is not a number (GiB)")
    return gtt - int(reserved * 2**30)


def quant_of(name):
    n = name.upper()
    for q in sorted(PREF, key=len, reverse=True):
        if re.search(r"(^|[-_.])" + re.escape(q) + r"\.GGUF$", n):
            return q
    return None


def _arch(reader):
    try:
        f = reader.fields["general.architecture"]
        return bytes(f.parts[f.data[0]]).decode()
    except Exception:
        return None


def example_formats():
    """{arch: {"decodes", "lacks", "hint"}} from reference/*/formats.json: a model with a reference example is
    ported with that example's kit, which decodes some formats the starter doesn't (gemma4: BF16) and lacks some
    in its own kernels (gemma4: IQ4_NL per-layer embeddings)."""
    out = {}
    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference")
    for f in glob.glob(os.path.join(ref, "*", "formats.json")):
        try:
            d = json.load(open(f))
            out[d["arch"]] = {"decodes": set(d.get("decodes", [])), "lacks": set(d.get("lacks", [])),
                              "hint": d.get("hint", "")}
        except (OSError, ValueError, KeyError):
            pass
    return out


def header_formats(repo, path, size):
    """Tensor formats (+ "arch:<architecture>") from the GGUF header: fetch the first bytes into a sparse file
    of the full size."""
    from gguf import GGUFReader
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
        for n in (16 << 20, 64 << 20):
            req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"})
            data = urllib.request.urlopen(req, timeout=120).read()
            f.seek(0); f.write(data); f.truncate(size); f.flush()
            try:
                r = GGUFReader(f.name)
                return {t.tensor_type.name for t in r.tensors} | {f"arch:{_arch(r)}"}
            except Exception:  # header longer than n bytes (huge vocab): try a bigger range
                continue
    return None


TIERS = {"Q4": ["Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "UD-Q4_K_M"], "Q5": ["Q5_K_M", "UD-Q5_K_XL", "Q5_K_S"],
         "Q6": ["Q6_K", "UD-Q6_K_XL"], "Q8": ["Q8_0", "UD-Q8_K_XL"]}


def cache_root():
    return os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def cached_files(repo):
    """{file name: path} of this repo's GGUFs present in the HF cache, any revision (newest snapshot first)."""
    out = {}
    snaps = sorted(glob.glob(os.path.join(cache_root(), "models--" + repo.replace("/", "--"), "snapshots", "*")),
                   key=os.path.getmtime, reverse=True)
    for snap in snaps:
        for f in glob.glob(os.path.join(snap, "**", "*.gguf"), recursive=True):
            rel = os.path.relpath(f, snap)
            if rel not in out and os.path.exists(f) and os.path.getsize(f) > (1 << 20):   # resolves (not a dangling link)
                out[rel] = f
    return out


def other_cached(repo, tier):
    """Same-tier GGUFs of the same model name cached under other repos."""
    name = re.sub(r"(?i)[-_](gguf|mtp-gguf)$", "", repo.split("/")[-1]).lower()
    hits = []
    for d in glob.glob(os.path.join(cache_root(), "models--*")):
        r = os.path.basename(d)[len("models--"):].replace("--", "/", 1)
        if r == repo or name not in r.lower():
            continue
        for f, p in cached_files(r).items():
            if quant_of(f) in TIERS[tier] and not re.search(r"mmproj|dflash", f, re.I):
                hits.append(f"{r}:{f}")
    return hits


def local_formats(path):
    from gguf import GGUFReader
    try:
        r = GGUFReader(path)
        return {t.tensor_type.name for t in r.tensors} | {f"arch:{_arch(r)}"}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--gtt-gib", type=float)
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    cached = cached_files(a.repo)
    if a.file:
        print(f"{a.file}\t{cached.get(a.file, '')}")
        return
    tier = os.environ.get("QUANT_TIER", "Q4").upper()
    if tier not in TIERS:
        sys.exit(f"QUANT_TIER must be one of {sorted(TIERS)}")
    gtt = a.gtt_gib * 2**30 if a.gtt_gib else gtt_bytes()
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{a.repo}/tree/main?recursive=true", timeout=60))
    files = {f["path"]: f.get("size", 0) for f in tree if f["path"].lower().endswith(".gguf")
             and not re.search(r"mmproj|dflash|imatrix|-of-\d", f["path"], re.I)}
    for f, p in cached.items():   # cached files of older revisions count too
        if not re.search(r"mmproj|dflash|imatrix|-of-\d", f, re.I):
            files.setdefault(f, os.path.getsize(p))
    cands = [(p, s) for p, s in files.items() if quant_of(p) in TIERS[tier]]
    if not cands:
        sys.exit(f"no single-file {tier} GGUF ({', '.join(TIERS[tier])}) in {a.repo}: set QUANT_TIER or pass REPO:FILE")
    cands.sort(key=lambda c: (TIERS[tier].index(quant_of(c[0])), c[0] not in cached))
    others = other_cached(a.repo, tier)
    best = cands[0][0]
    same = [o for o in others if quant_of(os.path.basename(o.split(":")[-1])) == quant_of(best)]
    if best not in cached and same:
        sys.exit(f"STOP: {best} is not cached for {a.repo}, but the same quant is cached at {', '.join(same)}: "
                 "port that repo (no download), or pass REPO:FILE to choose explicitly")
    if others and not any(c[0] in cached for c in cands):
        print(f"NOTE: not cached for {a.repo}, but cached elsewhere at {tier}: {', '.join(others)} "
              "(port that repo to reuse it)", file=sys.stderr)
    missing = {}
    extra = example_formats()   # e.g. gemma4: + BF16, - IQ4_NL (what its reference example's kernels decode)
    hints = set()
    for p, s in cands:
        if 2 * s + (8 << 30) > gtt:
            print(f"skip {p}: {s / 2**30:.1f} GiB does not fit twice in the {gtt / 2**30:.0f} GiB budget", file=sys.stderr)
            continue
        found = local_formats(cached[p]) if p in cached else header_formats(a.repo, p, s)
        arch = next((x[5:] for x in found or () if x.startswith("arch:")), None)
        fmts = {x for x in found or () if not x.startswith("arch:")}
        ex = extra.get(arch, {})
        ok = (SUPPORTED | ex.get("decodes", set())) - ex.get("lacks", set())
        if ex.get("hint"):
            hints.add(ex["hint"])
        print(f"{p}: {s / 2**30:.1f} GiB, {'cached' if p in cached else 'to download'}, arch {arch}, "
              f"formats {sorted(fmts) if fmts else '?'}", file=sys.stderr)
        if fmts and fmts <= ok:
            print(f"{p}\t{cached.get(p, '')}")
            return
        missing[p] = sorted((fmts or {"?"}) - ok)
    if not missing:
        sys.exit(f"no {tier} GGUF in {a.repo} fits the GPU memory budget twice")
    sys.exit(f"STOP: no {tier} GGUF in {a.repo} uses only formats the starter decodes: "
             + "; ".join(f"{p} needs {m}" for p, m in missing.items())
             + ". Add the formats (loom-port-llm 'New quant format'); a bigger tier is only by request (QUANT_TIER)."
             + "".join(f" Note: {h}." for h in sorted(hints)))


if __name__ == "__main__":
    main()
