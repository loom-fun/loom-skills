#!/usr/bin/env python3
"""Adds (or refreshes) one published model in the fleet manifest from its package directory (a Hugging Face snapshot of the
staging repo: llm_config.json, model.json, model.json.lock, tokenizer.json). Only small files are read; every GGUF the build
reads is dumped from the HF cache or by HTTP range reads at the revision the package's lock pins (gguf_stub.py).

usage: add_model.py SLUG PACKAGE_DIR [--staging-revision REV] [--hf-cache DIR] [--refresh-dumps]
Writes fleet/<slug>/{llm_config.json, tokenizer_min.json, gguf/<file>.json.gz} and fleet/manifest.json[slug].
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gguf_stub  # noqa: E402

HF = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
DROP = ("tensors", "generator", "draft", "resolved")   # outputs of a build (build.py rewrites them), not inputs


def origin_file(lock, origin):
    """'@key/path' -> (repo, revision, path)."""
    key, path = origin[1:].split("/", 1)
    o = lock["origins"][key]["requested"]
    return o["repository"], lock["origins"][key]["resolved"]["revision"], path, key


def source_of(repo, rev, path):
    p = f"{HF}/models--{repo.replace('/', '--')}/snapshots/{rev}/{path}"
    return p if os.path.exists(p) else f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


def entry(slug, pkg, staging_revision=None, refresh=False, root=None):
    root = root or os.path.dirname(HERE)
    cfg = json.load(open(f"{pkg}/llm_config.json"))
    mj = json.load(open(f"{pkg}/model.json"))
    lock = json.load(open(f"{pkg}/model.json.lock"))
    presets = mj["presets"]
    name = "chat" if "chat" in presets else next(iter(presets))
    ex = presets[name]["execution"]
    out = os.path.join(root, "fleet", slug)
    os.makedirs(f"{out}/gguf", exist_ok=True)
    files = {}
    for f in lock["files"]:
        if f["origin"].endswith("tokenizer.json") or not f["origin"].endswith(".gguf"):
            continue
        repo, rev, path, key = origin_file(lock, f["origin"])
        base = os.path.basename(path)
        dump = f"{slug}/gguf/{base}.json.gz"
        dp = os.path.join(root, "fleet", dump)
        if refresh or not os.path.exists(dp):
            d = gguf_stub.dump(source_of(repo, rev, path), os.path.join(root, "fleet", "blobs"))
            assert d["size"] == f["size"], (base, d["size"], f["size"])
            gguf_stub.save(d, dp)
        files[base] = {"origin": key, "repo": repo, "revision": rev, "path": path, "size": f["size"], "dump": dump}
    # tokenizer: the programs read only the added tokens' content and id (gen_llm.py stop_restore)
    tok = json.load(open(f"{pkg}/tokenizer.json", encoding="utf-8"))
    json.dump({"added_tokens": [{"content": t["content"], "id": t["id"]} for t in tok.get("added_tokens", [])]},
              open(f"{out}/tokenizer_min.json", "w"), separators=(",", ":"))
    json.dump({k: v for k, v in cfg.items() if k not in DROP}, open(f"{out}/llm_config.json", "w"), indent=1)
    build = {"preset": name, "context": ex.get("context_capacity"), "pool": ex.get("pool_capacity"),
             "prefill": ex.get("prefill_capacity"), "rows": ex.get("rows"), "max_tokens": ex.get("max_tokens"),
             "draft_depth": ex.get("draft_depth")}
    return {"staging": {"repo": f"loom-fun-staging/{slug}", "revision": staging_revision},
            "model_id": mj["model"]["id"], "kit": ((cfg.get("generator") or {}).get("kit") or {}).get("example"),
            "arch": cfg.get("arch"), "build": {k: v for k, v in build.items() if v is not None}, "files": files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("package")
    ap.add_argument("--staging-revision")
    ap.add_argument("--refresh-dumps", action="store_true")
    a = ap.parse_args()
    mp = os.path.join(HERE, "manifest.json")
    m = json.load(open(mp)) if os.path.exists(mp) else {"models": {}}
    m["models"][a.slug] = entry(a.slug, a.package, a.staging_revision, a.refresh_dumps)
    m["models"] = dict(sorted(m["models"].items()))
    json.dump(m, open(mp, "w"), indent=1)


if __name__ == "__main__":
    main()
