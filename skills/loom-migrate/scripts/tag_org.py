#!/usr/bin/env python3
"""Tag the head of every `*-loom` model repo of a Hugging Face org with the OUTGOING SDK version, before an SDK migration's staging cards
are promoted (the old cards stay addressable at .../resolve/<tag>/model.json).

usage: tag_org.py --org ORG --tag vX.Y.Z [--message MSG] [--only REPO ...] [--force] [--dry-run]

Per repo: `exists` (never moved; `not the head` when the tag points elsewhere), `created`, `would create`, `refused` (the head's llm_config.json
pins another SDK than the tag: tagging it would label a migrated card with the old version; --force overrides) or `FAILED` (a Hub error; the
other repos go on). Ends with a summary; exit 1 when any repo failed, was refused or has the tag on another commit. Needs huggingface_hub
and a write token (HF_TOKEN or ~/.cache/huggingface/token) unless --dry-run.
"""
import argparse
import json
import re
import sys
from collections import Counter


def head_on_sdk(pin, tag):
    """the head's sdk pin (None = pin-less = 0.1.0) belongs to the version the tag names (v0.1.0 -> 0.1.x)"""
    want = tag[1:].rsplit(".", 1)[0] + "."        # v0.1.0 -> "0.1."
    return (pin or "0.1.0").startswith(want)


def tag_repo(api, repo, tag, message, force, dry):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    head = api.model_info(repo).sha
    tags = {t.name for t in api.list_repo_refs(repo).tags}
    if tag in tags:
        at = api.model_info(repo, revision=tag).sha
        return ("exists", f"{tag} -> {at[:10]}") if at == head else ("not-head", f"{tag} -> {at[:10]}, head {head[:10]}")
    try:
        pin = (json.load(open(hf_hub_download(repo, "llm_config.json", revision=head))).get("sdk") or {}).get("hrx_loom")
    except EntryNotFoundError:
        pin = None
    if not head_on_sdk(pin, tag) and not force:
        return "refused", f"head {head[:10]} pins hrx-loom {pin}, not {tag} (--force to tag anyway)"
    if dry:
        return "would-create", f"{tag} at {head[:10]}"
    api.create_tag(repo, tag=tag, revision=head, tag_message=message or f"Cards built for Loom SDK {tag}")
    return "created", f"{tag} at {head[:10]}"


def main(argv=None, api=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--org", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--message", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="repo names (without the org) instead of every *-loom repo")
    ap.add_argument("--force", action="store_true", help="tag a head whose sdk pin is not the tag's version")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"v\d+\.\d+\.\d+\S*", a.tag):
        ap.error(f"--tag {a.tag}: use the version form vX.Y.Z (the card URL is .../resolve/vX.Y.Z/model.json)")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    names = a.only if a.only is not None else sorted(m.id.split("/", 1)[1] for m in api.list_models(author=a.org) if m.id.endswith("-loom"))
    count = Counter()
    for name in names:
        repo = f"{a.org}/{name}"
        try:
            status, text = tag_repo(api, repo, a.tag, a.message, a.force, a.dry_run)
        except Exception as e:  # one repo's Hub error must not hide the others
            status, text = "FAILED", str(e)
        count[status] += 1
        print(f"{repo}: {status} {text}")
    print("summary: " + ", ".join(f"{k} {v}" for k, v in sorted(count.items())) + f" ({len(names)} repos)")
    return 1 if count["FAILED"] or count["refused"] or count["not-head"] else 0


if __name__ == "__main__":
    sys.exit(main())
