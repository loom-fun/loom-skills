#!/usr/bin/env python3
"""Tag the production repos with the OUTGOING SDK version before an SDK migration's staging cards are promoted (the old cards stay
addressable at .../resolve/<tag>/model.json). Release-checklist section 5 step 1: run it before the merge, the release tag and the replay.

usage: tag_org.py --bundle DIR --manifest-sha256 SHA [--message MSG] [--dry-run]      (a release: the reviewed tag plan)
       tag_org.py --org ORG --tag vX.Y.Z [--message MSG] [--only REPO ...] [--force] [--dry-run]   (every head of an org)

Bundle mode tags exactly the reviewed plan of `replay_promotion.py prepare` (manifest.json `tags`, verified against the reviewed
SHA256): each repo at its reviewed parent, text and image packages alike, plus held repos the plan lists. Per repo: `exists` (the tag
is at the parent), `created` (then read back), `would-create`, `skip` (already on the release SDK, tag state unchanged), `not-head`
(the tag points elsewhere), `moved` (the head is no longer the reviewed parent: prepare again), `changed` or `FAILED`.
Org mode tags each head; `refused` when the head's llm_config.json pins another SDK than the tag (--force overrides).
Ends with a summary; exit 1 when any repo is not `exists`/`created`/`would-create`/`skip`. Needs huggingface_hub and a write
token (HF_TOKEN or ~/.cache/huggingface/token) unless --dry-run.
"""
import argparse
import hashlib
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


def bundle_plan(bundle, manifest_sha):
    """The reviewed outgoing tag plan of a prepared release bundle."""
    import os
    with open(os.path.join(bundle, "manifest.json"), "rb") as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != manifest_sha:
        raise SystemExit("reviewed manifest checksum mismatch")
    document = json.loads(data)
    if document.get("schema") != 2 or not isinstance(document.get("tags"), dict):
        raise SystemExit("not a release bundle manifest with a tag plan (prepare it again)")
    release = document.get("release") or {}
    if release.get("inventory") is None or release.get("card_check") != "passed":
        raise SystemExit("not an executable release bundle (partial --only or unchecked); prepare the full release again")
    plan = []
    for slug in sorted(document["tags"]):
        entry, release = document["tags"][slug], document.get("release") or {}
        ok = (isinstance(entry, dict) and isinstance(entry.get("repo"), str) and entry["repo"] == f"loom-fun/{slug}"
              and isinstance(entry.get("parent"), str) and re.fullmatch(r"[0-9a-f]{40}", entry["parent"])
              and entry.get("tag") == release.get("outgoing_tag")
              and entry.get("action") in ("create", "already-at-parent", "already-on-target-sdk"))
        if not ok:
            raise SystemExit(f"{slug}: invalid tag plan entry (prepare the release again)")
        plan.append(entry)
    return plan


def tag_planned(api, entry, message, dry):
    """One plan row: the tag at exactly the reviewed parent, never at a newer head."""
    repo, parent, tag = entry["repo"], entry["parent"], entry["tag"]
    tags = {t.name for t in api.list_repo_refs(repo).tags}
    current = api.model_info(repo, revision=tag).sha if tag in tags else None
    if entry["action"] == "already-on-target-sdk":
        if current == entry["existing"]:
            return "skip", f"on the release SDK; {tag} {current[:10] if current else 'absent'}"
        return "changed", f"{tag} is {current} since review, was {entry['existing']}"
    if current == parent:
        return "exists", f"{tag} -> {parent[:10]}"
    if current is not None:
        return "not-head", f"{tag} -> {current[:10]}, reviewed parent {parent[:10]}"
    head = api.model_info(repo).sha
    if head != parent:
        return "moved", f"head {head[:10]} is not the reviewed parent {parent[:10]}: prepare the release again"
    if dry:
        return "would-create", f"{tag} at {parent[:10]}"
    api.create_tag(repo, tag=tag, revision=parent, tag_message=message or f"Cards built for Loom SDK {tag}")
    readback = api.model_info(repo, revision=tag).sha
    if readback != parent:
        return "FAILED", f"{tag} reads back {readback}, not {parent}"
    return "created", f"{tag} at {parent[:10]}"


def main(argv=None, api=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", help="a prepared release bundle: tag exactly its reviewed plan")
    ap.add_argument("--manifest-sha256", help="the reviewed manifest SHA256 (with --bundle)")
    ap.add_argument("--org")
    ap.add_argument("--tag")
    ap.add_argument("--message", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="repo names (without the org) instead of every *-loom repo")
    ap.add_argument("--force", action="store_true", help="tag a head whose sdk pin is not the tag's version")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.bundle:
        if not a.manifest_sha256 or a.org or a.tag or a.only is not None or a.force:
            ap.error("--bundle takes --manifest-sha256 [--message] [--dry-run] only")
        plan = bundle_plan(a.bundle, a.manifest_sha256)
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        count = Counter()
        for entry in plan:
            try:
                status, text = tag_planned(api, entry, a.message, a.dry_run)
            except Exception as e:  # one repo's Hub error must not hide the others
                status, text = "FAILED", str(e)
            count[status] += 1
            print(f"{entry['repo']}: {status} {text}")
        print("summary: " + ", ".join(f"{k} {v}" for k, v in sorted(count.items())) + f" ({len(plan)} repos)")
        return 1 if set(count) - {"exists", "created", "would-create", "skip"} else 0
    if not a.org or not a.tag:
        ap.error("--org and --tag are required without --bundle")
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
