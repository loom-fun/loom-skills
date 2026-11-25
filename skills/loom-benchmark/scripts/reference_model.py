"""The GGUF the gates' llama.cpp reference loads for a package (reference_models.json).

Most packages serve a GGUF llama.cpp loads as is, and the gates use it. A package whose GGUF is in a layout mainline
llama.cpp cannot read (Inkling-2-Mini: Unsloth's compact inkling2 layout) has an entry keyed by its checkpoint: the
publisher's mainline-layout file of the same tensors at the same revision, pinned by sha256, and the oldest llama.cpp
release that serves it correctly.

usage: reference_model.py PKG --cache DIR [--tag TAG]
  prints the local path of the reference GGUF (downloaded into DIR once, its sha256 checked every time), or nothing
  when the package has no entry (the gates use the package's own GGUF); exit 1 when the entry needs a newer llama.cpp
  than TAG (lemonade-sdk/llamacpp-rocm bNNNN), the download fails or the bytes do not match.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "reference_models.json")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def package_checkpoint(pkg):
    """(repository@revision, gguf file name) of the checkpoint the package serves, or (None, None)."""
    model, cfg = (read_json(os.path.join(pkg, f)) for f in ("model.json", "llm_config.json"))
    ck = (model.get("origins") or {}).get("checkpoint") or {}
    if ck.get("kind") != "hf" or not ck.get("repository") or not cfg.get("gguf"):
        return None, None
    return f"{ck['repository']}@{ck.get('revision') or ''}", os.path.basename(cfg["gguf"])


def lookup(pkg, table=TABLE):
    checkpoint, gguf = package_checkpoint(pkg)
    if not checkpoint:
        return None
    for entry in read_json(table)["models"]:
        if entry["checkpoint"] == checkpoint and entry["gguf"] == gguf:
            return entry
    return None


def release_number(tag):
    m = re.fullmatch(r"b(\d+)", tag or "")
    return int(m.group(1)) if m else None


def check_tag(entry, tag):
    """None when TAG serves the entry, else why not (an unknown tag is refused: fail closed)."""
    need = entry.get("llamacpp_min")
    if not need:
        return None
    have = release_number(tag)
    if have is None or have < release_number(need):
        return f"the reference needs llama.cpp ROCm {need} or newer, the job runs {tag!r} ({entry['why']})"
    return None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(entry, cache, base=None):
    """The verified local copy of entry["reference"] under cache (named .gguf: the publisher's `.gguf_file` suffix
    only keeps the two layouts apart on the Hub)."""
    ref = entry["reference"]
    name = re.sub(r"\.gguf_file$", ".gguf", os.path.basename(ref["file"]))
    dst = os.path.join(cache, ref["repo"].replace("/", "--"), ref["revision"], name)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        base = base or os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
        url = f"{base.rstrip('/')}/{ref['repo']}/resolve/{ref['revision']}/{ref['file']}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {})
        part = f"{dst}.part{os.getpid()}"
        try:
            with urllib.request.urlopen(req, timeout=600) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 16 << 20)
            os.replace(part, dst)
        finally:
            if os.path.exists(part):
                os.remove(part)
    if ref.get("size") is not None and os.path.getsize(dst) != ref["size"]:
        os.remove(dst)
        raise ValueError(f"{dst}: {os.path.getsize(dst) if os.path.exists(dst) else '?'} bytes, expected {ref['size']} (removed)")
    got = sha256_of(dst)
    if got != ref["sha256"]:
        os.remove(dst)
        raise ValueError(f"{dst}: sha256 {got} != pinned {ref['sha256']} (removed)")
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pkg")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", default=None, help="the llama.cpp ROCm release the gates run (bNNNN)")
    a = ap.parse_args(argv)
    entry = lookup(a.pkg)
    if entry is None:
        return 0
    why = check_tag(entry, a.tag)
    if why:
        print(f"reference_model: {why}", file=sys.stderr)
        return 1
    try:
        path = fetch(entry, a.cache)
    except Exception as x:  # noqa: BLE001 - one message for the job log
        print(f"reference_model: {x}", file=sys.stderr)
        return 1
    print(f"reference_model: {entry['checkpoint']} {entry['gguf']} -> {entry['reference']['file']} (sha256 ok)", file=sys.stderr)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
