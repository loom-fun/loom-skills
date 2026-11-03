"""Finds and downloads a DFlash/DFlash2 drafter GGUF for a ported model (port.sh runs it after inspect).

usage: find_drafter.py CFG.json TARGET_REPO [--download] [--repo ORG/NAME[:FILE]]
Prints DRAFTER_RESULT {"gguf": path | null, "repo": ..., "file": ..., "reason": ...} and, with --download, records
cfg["drafter"] = {"gguf", "repo", "file"} in CFG.json.

Candidates are HF repos named after the target's base model with "DFlash" in the name and GGUF files (z-lab, the
DFlash authors, first; then by downloads). A candidate is accepted only when its header (HTTP range read; nothing
large is downloaded until one passes) says: arch dflash, the target's hidden size, tap layers inside the target's
trunk, the target's vocabulary size, a mask token, and tensor formats the starter's kernels support. Preferred
quant: Q8_0, Q6_K, Q5_K_M, Q4_K_M. Whether the drafter is worth using is decided later by measurement
(draft/tune.py): a drafter that does not speed decoding up is not selected.
"""
import argparse
import json
import os
import re
import struct
import sys
import urllib.request

SUPPORTED = {"F32", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}
QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K", "Q4_K_M", "Q4_K"]
TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
         13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
         22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}


def header(repo, path, nbytes=24_000_000):
    """GGUF metadata and tensor infos from the first bytes of a remote file."""
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"}), timeout=60).read()
    o = 4
    def rd(fmt):
        nonlocal o
        v = struct.unpack_from(fmt, data, o); o += struct.calcsize(fmt); return v[0]
    def rs():
        nonlocal o
        n = rd("<Q"); s = data[o:o + n].decode("utf8", "replace"); o += n; return s
    sizes = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    def val(t, keep=True):
        if t == 8:
            return rs()
        if t == 9:
            et, n = rd("<I"), rd("<Q")
            v = [val(et) for _ in range(n)]
            return v if n <= 64 else {"len": n}
        return rd(sizes[t])
    rd("<I"); nt = rd("<Q"); nk = rd("<Q")
    kv = {}
    for _ in range(nk):
        k = rs(); t = rd("<I"); kv[k] = val(t)
    tensors = {}
    for _ in range(nt):
        n = rs(); nd = rd("<I"); dims = [rd("<Q") for _ in range(nd)]; tt = rd("<I"); rd("<Q")
        tensors[n] = (TYPES.get(tt, str(tt)), dims)
    return kv, tensors


def base_name(cfg, target_repo):
    name = (cfg.get("name") or "").strip()
    repo = target_repo.split("/")[-1]
    repo = re.sub(r"(?i)[-_]?(gguf|mtp|instruct-gguf)$", "", repo)
    repo = re.sub(r"(?i)[-_]?gguf$", "", repo)
    # prefer the repo name (it carries size/variant), fall back to general.name
    return repo or name.replace(" ", "-")


def candidates(base):
    from huggingface_hub import HfApi
    api = HfApi()
    seen, out = set(), []
    for q in (f"{base}-DFlash", f"{base} DFlash", base.split("-Instruct")[0] + "-DFlash"):
        for m in api.list_models(search=q, limit=50, sort="downloads"):
            if m.id in seen or "dflash" not in m.id.lower():
                continue
            seen.add(m.id)
            out.append(m)
    def key(m):
        return (0 if m.id.lower().startswith("z-lab/") else 1, -(m.downloads or 0))
    return sorted(out, key=key)


def pick_file(api, repo):
    files = [s.rfilename for s in api.model_info(repo).siblings if s.rfilename.endswith(".gguf") and not re.search(r"-of-\d", s.rfilename)]
    def rank(f):
        u = f.upper()
        for i, q in enumerate(QUANT_ORDER):
            if q in u:
                return i
        return len(QUANT_ORDER)
    files = [f for f in files if rank(f) < len(QUANT_ORDER)]
    return sorted(files, key=rank)


def check(cfg, kv, tensors):
    a = lambda k, d=None: kv.get(f"dflash.{k}", d)
    if kv.get("general.architecture") != "dflash":
        return "not arch dflash"
    if a("embedding_length") != cfg["hidden"]:
        return f"hidden {a('embedding_length')} != {cfg['hidden']}"
    taps = a("target_layers") or []
    if not taps or max(taps) > cfg["layers"] or min(taps) < 1:
        return f"tap layers {taps} outside the {cfg['layers']}-layer trunk"
    if kv.get("tokenizer.ggml.mask_token_id") is None:
        return "no mask token"
    toks = kv.get("tokenizer.ggml.tokens")
    n = toks.get("len") if isinstance(toks, dict) else (len(toks) if toks else None)
    if n is not None and abs(n - cfg["vocab"]) > 0 and n > cfg["vocab"]:
        return f"vocabulary {n} > target {cfg['vocab']}"
    bad = sorted({t for t, _ in tensors.values()} - SUPPORTED)
    if bad:
        return f"tensor formats {bad} not supported"
    if (a("attention.key_length") or 0) not in (64, 128, 256):
        return f"head_dim {a('attention.key_length')}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("target_repo")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--repo", default=None, help="use this drafter repo[:file] instead of searching")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    result = {"gguf": None, "repo": None, "file": None, "reason": "no DFlash drafter found"}
    tried = []
    if a.repo:
        r, _, f = a.repo.partition(":")
        repos = [(r, [f] if f else pick_file(api, r))]
    else:
        base = base_name(cfg, a.target_repo)
        repos = [(m.id, None) for m in candidates(base)][:8]
    for repo, files in repos:
        try:
            files = files or pick_file(api, repo)
        except Exception as e:  # noqa: BLE001
            tried.append(f"{repo}: {e}"); continue
        for f in files[:3]:
            try:
                kv, tensors = header(repo, f)
            except Exception as e:  # noqa: BLE001
                tried.append(f"{repo}/{f}: header {e}"); continue
            why = check(cfg, kv, tensors)
            if why:
                tried.append(f"{repo}/{f}: {why}"); continue
            result = {"gguf": None, "repo": repo, "file": f, "reason": "ok",
                      "kind": "dflash2" if "blk.0.attn_conv_proj.weight" in tensors else "dflash",
                      "block_size": kv.get("dflash.block_size")}
            break
        if result["repo"]:
            break
    if result["repo"] and a.download:
        result["gguf"] = hf_hub_download(result["repo"], result["file"])
        cfg["drafter"] = {"gguf": result["gguf"], "repo": result["repo"], "file": result["file"]}
        json.dump(cfg, open(a.cfg, "w"), indent=1)
    for t in tried:
        print("skip", t)
    print("DRAFTER_RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
