#!/usr/bin/env python3
"""Publish a finished package to a loom-fun repo: the files loom-fetch needs, with pinned origins, and the card.

  publish_package.py prepare PKG OUT --source ORG/REPO [--loom-fetch EXE]
      Run on the machine that holds the GGUFs (where the package was built). Copies PKG into OUT without weights
      (parameter/, *.gguf) and without build/test leftovers (model_target.json, model_d*.json), points every
      parameter file at its Hugging Face origin (checkpoint = --source; another repo, e.g. a DFlash drafter, gets
      its own origin named after the parameter) at the exact revision the package was validated on (read from the
      HF cache path the file links to), then writes model.json.lock with
      `loom-fetch --update-lock --all-variants` (the local snapshot dirs stand in for the origins: nothing is
      downloaded). OUT is what the repo holds apart from the card.
  publish_package.py upload OUT --repo loom-fun/NAME [--message MSG] [--card-only]
      Uploads OUT (with the card: README.md, info.json, REPORT.md made by make_info.py build + render_card.py)
      and deletes repo files OUT no longer has (an in-place update of an older package leaves nothing stale;
      .gitattributes stays). --card-only uploads just README.md, info.json and REPORT.md. Needs huggingface_hub
      and a write token (HF_TOKEN or ~/.cache/huggingface/token).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

SKIP = re.compile(r"^(parameter(/.*)?|.*\.gguf|model_target\.json|model_d\d+\.json|\.loom-fetch\.lock|\.cache(/.*)?)$")
CARD = ("README.md", "info.json", "REPORT.md")


def hf_origin(path):
    """(repo, revision, snapshot dir, file) of a file inside the Hugging Face cache, else None."""
    # Follow the link chain (package -> work dir -> snapshots/REV/file -> blobs/SHA) and stop at the snapshot entry:
    # realpath would land in blobs/, which names no file.
    p = os.path.abspath(path)
    for _ in range(40):
        m = re.search(r"/models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})/(.+)$", p)
        if m:
            return f"{m.group(1)}/{m.group(2)}", m.group(3), p[:m.start(4)].rstrip("/"), m.group(4)
        if not os.path.islink(p):
            return None
        p = os.path.normpath(os.path.join(os.path.dirname(p), os.readlink(p)))
    return None


def cmd_prepare(a):
    pkg, out = os.path.abspath(a.pkg), os.path.abspath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    for root, dirs, files in os.walk(pkg):
        rel = os.path.relpath(root, pkg)
        dirs[:] = [d for d in dirs if not SKIP.match(os.path.normpath(os.path.join(rel, d)))]
        for f in files:
            r = os.path.normpath(os.path.join(rel, f))
            if not SKIP.match(r) and not f.endswith(".pyc"):
                os.makedirs(os.path.dirname(os.path.join(out, r)), exist_ok=True)
                shutil.copy2(os.path.join(root, f), os.path.join(out, r))
    card = json.load(open(os.path.join(out, "model.json"), encoding="utf-8"))
    origins, roots = {}, {}
    for variant in card.get("variants", {}).values():
        for service in variant.get("services", {}).values():
            for pname, param in service.get("parameters", {}).items():
                for entry in param.get("files", []):
                    src = hf_origin(os.path.join(pkg, entry["path"]))
                    if src is None:
                        sys.exit(f"{entry['path']}: not a file of the Hugging Face cache; loom-fetch could not acquire it")
                    repo, rev, snap, fname = src
                    name = "checkpoint" if repo == a.source else ("drafter" if pname in ("draft", "drafter") else pname)
                    if name in origins and origins[name]["repository"] != repo:
                        name = f"{name}_{len(origins)}"
                    origins[name] = {"kind": "hf", "repository": repo, "revision": rev}
                    roots[name] = snap
                    entry["origin"] = f"@{name}/{fname}"
    if a.source not in [o["repository"] for o in origins.values()]:
        sys.exit(f"no parameter file comes from {a.source}")
    card["origins"] = origins
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=1)
        f.write("\n")
    # llm_config.json names the GGUFs by file name only (no local paths in a published file)
    cfgp = os.path.join(out, "llm_config.json")
    cfg = json.load(open(cfgp, encoding="utf-8"))
    def base(d, k):
        if isinstance(d, dict) and isinstance(d.get(k), str):
            d[k] = os.path.basename(d[k])
    base(cfg, "gguf"); base(cfg.get("vision"), "mmproj")
    for k in ("draft", "drafter", "assistant"):
        base(cfg.get(k), "gguf")
    json.dump(cfg, open(cfgp, "w", encoding="utf-8"), indent=1)
    cmd = [a.loom_fetch, os.path.join(out, "model.json"), "--update-lock", "--all-variants", "--no-input", "--quiet",
           *[f"--origin={n}={r}" for n, r in roots.items()]]
    subprocess.run(cmd, check=True)
    lock = json.load(open(os.path.join(out, "model.json.lock"), encoding="utf-8"))
    print(f"prepared {out}: origins {json.dumps(origins)}; lock: {len(lock.get('files', []))} files")


def cmd_upload(a):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    if a.card_only:
        missing = [f for f in CARD if not os.path.isfile(os.path.join(a.out, f))]
        if missing:
            sys.exit(f"card files missing in {a.out}: {missing}")
        api.upload_folder(repo_id=a.repo, folder_path=a.out, allow_patterns=list(CARD),
                          commit_message=a.message or "Card")
        print(f"card uploaded to https://huggingface.co/{a.repo}")
        return
    for f in ("model.json", "model.json.lock", "llm_config.json", *CARD):
        if not os.path.isfile(os.path.join(a.out, f)):
            sys.exit(f"{f} missing in {a.out} (prepare, then make_info.py build + render_card.py)")
    local = {os.path.relpath(os.path.join(r, f), a.out).replace(os.sep, "/") for r, _, fs in os.walk(a.out) for f in fs}
    stale = [f for f in api.list_repo_files(a.repo) if f not in local and f != ".gitattributes"]
    api.upload_folder(repo_id=a.repo, folder_path=a.out, delete_patterns=stale or None,
                      commit_message=a.message or "Loom package and card")
    print(f"uploaded https://huggingface.co/{a.repo} ({len(local)} files{'; removed ' + ', '.join(stale) if stale else ''})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("pkg"); p.add_argument("out")
    p.add_argument("--source", required=True, help="the GGUF repo the checkpoint comes from")
    p.add_argument("--loom-fetch", default="loom-fetch")
    u = sub.add_parser("upload")
    u.add_argument("out"); u.add_argument("--repo", required=True)
    u.add_argument("--message"); u.add_argument("--card-only", action="store_true")
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "upload": cmd_upload}[a.cmd](a)


if __name__ == "__main__":
    main()
