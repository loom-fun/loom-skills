#!/usr/bin/env python3
"""Publish a finished package to a Hugging Face org's repo (default org loom-fun): the files loom-fetch needs, with pinned origins, and the card.

  publish_package.py prepare PKG OUT --source ORG/REPO [--loom-fetch EXE]
      Run on the machine that holds the GGUFs (where the package was built). Copies PKG into OUT without weights
      (parameter/, *.gguf) and without build/test leftovers (model_target.json, model_d*.json), points every
      parameter file at its Hugging Face origin (checkpoint = --source; another repo, e.g. a DFlash drafter, gets
      its own origin named after the parameter) at the exact revision the package was validated on (read from the
      HF cache path the file links to), then writes model.json.lock with
      `loom-fetch --update-lock --all-variants` (the local snapshot dirs stand in for the origins: nothing is
      downloaded). OUT is what the repo holds apart from the card.
  publish_package.py upload OUT --repo NAME|ORG/NAME [--org ORG] [--dry-run] [--message MSG] [--card-only] [--parent-commit SHA]
      The target is ORG/NAME: ORG is --org (default loom-fun), NAME the last part of --repo, so the same repo names go
      to another org (the staging flow: --org loom-fun-staging). The card's info.json "repo" must be that target (its
      Quick start fetches from it; make_info.py build --org writes it). --dry-run checks everything and lists what would
      be uploaded and deleted, without a token or a network call.
      Uploads OUT (with the card: README.md, info.json, REPORT.md made by make_info.py build + render_card.py)
      and deletes repo files OUT no longer has (an in-place update of an older package leaves nothing stale;
      .gitattributes stays). --card-only uploads just README.md, info.json and REPORT.md, and model.json when it differs from the repo's only in the R9700 preset (`<first>-r9700`, written by make_info.py build --fit from the fit row's "r9700_execution"; anything else differing is refused). --parent-commit SHA: the
      repo commit the card was built from (the revision whose info.json you re-read, e.g. `hf download` of it, or
      the Hub API's model sha): the upload fails if the repo has moved since (another worker published), instead
      of overwriting that card; re-read info.json and rebuild. Needs huggingface_hub and a write token (HF_TOKEN or
      ~/.cache/huggingface/token).
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
    # A catalog package declares origins of its own (the checkpoint its tokenizer comes from, the quantized repo its
    # weights come from) that its files name as @origin/file: they stay, and a weight file keeps the origin name
    # its entry already uses. Only the revisions are pinned to the cache the files were validated from.
    origins, roots, param_repos = dict(card.get("origins", {})), {}, set()
    for variant in card.get("variants", {}).values():
        for service in variant.get("services", {}).values():
            for pname, param in service.get("parameters", {}).items():
                for entry in param.get("files", []):
                    src = hf_origin(os.path.join(pkg, entry["path"]))
                    if src is None:
                        sys.exit(f"{entry['path']}: not a file of the Hugging Face cache; loom-fetch could not acquire it")
                    repo, rev, snap, fname = src
                    declared = re.match(r"@([^/]+)/", entry.get("origin", ""))
                    declared = declared.group(1) if declared and origins.get(declared.group(1), {}).get("repository") == repo else None
                    name = declared or ("checkpoint" if repo == a.source else ("drafter" if pname in ("draft", "drafter") else pname))
                    if not declared and name in origins and origins[name]["repository"] != repo:
                        name = f"{name}_{len(origins)}"
                    origins[name] = {**origins.get(name, {}), "kind": "hf", "repository": repo, "revision": rev}
                    param_repos.add(repo)
                    roots[name] = snap
                    entry["origin"] = f"@{name}/{fname}"
    if a.source not in param_repos:
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
    for k in ("draft", "drafter", "assistant", "mtp"):
        base(cfg.get(k), "gguf")
    json.dump(cfg, open(cfgp, "w", encoding="utf-8"), indent=1)
    cmd = [a.loom_fetch, os.path.join(out, "model.json"), "--update-lock", "--all-variants", "--no-input", "--quiet",
           *[f"--origin={n}={r}" for n, r in roots.items()]]
    subprocess.run(cmd, check=True)
    lock = json.load(open(os.path.join(out, "model.json.lock"), encoding="utf-8"))
    print(f"prepared {out}: origins {json.dumps(origins)}; lock: {len(lock.get('files', []))} files")


STRAY = re.compile(r"^(ev_[\w.-]+\.json|\.loom-fetch\.lock)$")   # working files, never part of a card repo


DEFAULT_ORG = "loom-fun"


def target_repo(org, repo):
    """ORG/NAME: the org (default loom-fun) and the name part of --repo (NAME or any ORG/NAME)."""
    return f"{org or DEFAULT_ORG}/{repo.split('/')[-1]}"


def check_card_text(out, repo=None):
    """Refuse to publish a card whose free text is not template output or a short factual caveat, or whose repo is
    not the one it is published to (its Quick start fetches from info.json's repo)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import card_lib
    info = card_lib.load_json(os.path.join(out, "info.json"))
    card_lib.require_valid(info, os.path.join(out, "info.json"))
    if repo and info.get("repo") != repo:
        sys.exit(f"info.json is the card of {info.get('repo')}, not {repo}: rebuild it with make_info.py build --org "
                 f"{repo.split('/')[0]} (the Quick start fetches from the card's repo)")
    if info.get("report") != "generated":
        sys.exit("info.json: only a generated REPORT.md is published (an attached, hand-written report is narrative)")


def card_model(a, api):
    """["model.json"] when a card-only upload must carry it: the local model.json differs from the repo's only in the R9700
    preset (`<first>-r9700`, written by make_info.py build --fit from the fit row's r9700_execution: a serving setting of the
    card, not a program or weight change); [] when they are the same. Anything else differing is refused (a stale or other build)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import card_lib
    from huggingface_hub import hf_hub_download
    local = card_lib.load_json(os.path.join(a.out, "model.json"))
    remote = card_lib.load_json(hf_hub_download(a.repo, "model.json", revision=a.parent_commit))
    alt = card_lib.gpu_preset_name(local)
    strip = lambda m: {**m, "presets": {k: v for k, v in m["presets"].items() if k != alt}}
    if strip(local) != strip(remote):
        sys.exit(f"{a.out}/model.json differs from {a.repo}'s beyond the {alt} preset: --card-only uploads no other model.json change "
                 "(publish the whole package)")
    return [] if local == remote else ["model.json"]


def cmd_upload(a):
    a.repo = target_repo(a.org, a.repo)
    check_card_text(a.out, a.repo)
    local = {os.path.relpath(os.path.join(r, f), a.out).replace(os.sep, "/") for r, _, fs in os.walk(a.out) for f in fs}
    if a.dry_run:
        files = sorted(f for f in local if not a.card_only or f in CARD)
        print(f"dry run: would upload {len(files)} files to https://huggingface.co/{a.repo}"
              + (" (card only)" if a.card_only else "") + ": " + ", ".join(files[:12]) + (" ..." if len(files) > 12 else ""))
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    if a.card_only:
        files = list(CARD) + card_model(a, api)
        missing = [f for f in CARD if not os.path.isfile(os.path.join(a.out, f))]
        if missing:
            sys.exit(f"card files missing in {a.out}: {missing}")
        strays = [f for f in api.list_repo_files(a.repo) if STRAY.match(f)]
        api.upload_folder(repo_id=a.repo, folder_path=a.out, allow_patterns=files, delete_patterns=strays or None,
                          commit_message=a.message or "Card", parent_commit=a.parent_commit)
        print(f"card uploaded to https://huggingface.co/{a.repo}")
        return
    for f in ("model.json", "model.json.lock", "llm_config.json", *CARD):
        if not os.path.isfile(os.path.join(a.out, f)):
            sys.exit(f"{f} missing in {a.out} (prepare, then make_info.py build + render_card.py)")
    stale = [f for f in api.list_repo_files(a.repo) if f not in local and f != ".gitattributes"]
    api.upload_folder(repo_id=a.repo, folder_path=a.out, delete_patterns=stale or None,
                      commit_message=a.message or "Loom package and card", parent_commit=a.parent_commit)
    print(f"uploaded https://huggingface.co/{a.repo} ({len(local)} files{'; removed ' + ', '.join(stale) if stale else ''})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("pkg"); p.add_argument("out")
    p.add_argument("--source", required=True, help="the GGUF repo the checkpoint comes from")
    p.add_argument("--loom-fetch", default="loom-fetch")
    u = sub.add_parser("upload")
    u.add_argument("out"); u.add_argument("--repo", required=True, help="NAME or ORG/NAME (the name part is used)")
    u.add_argument("--org", default=DEFAULT_ORG, help=f"the Hugging Face org to publish to (default {DEFAULT_ORG}; staging: loom-fun-staging)")
    u.add_argument("--dry-run", action="store_true", help="check the card and list what would be uploaded; no token, no network")
    u.add_argument("--message"); u.add_argument("--card-only", action="store_true")
    u.add_argument("--parent-commit", help="fail unless the repo's main is still this commit (the one the card was "
                   "built from): concurrent publishes fail instead of overwriting each other")
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "upload": cmd_upload}[a.cmd](a)


if __name__ == "__main__":
    main()
