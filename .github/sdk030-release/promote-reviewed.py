"""Check or apply the reviewed package release and its outgoing Hub tags."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token

from replay_promotion import ENDPOINT, PRODUCTION, STAGING, digest, inspect_bundle, read_hub, replay, require

ROOT = Path(__file__).resolve().parent
REPAIRS = {
    "amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom",
    "unsloth-Inkling-2-Mini-loom",
}
OUT_OF_SCOPE = "unsloth-Inkling-2-Mini-loom-r9700"


def load(path):
    return json.loads(path.read_bytes())


def tag_commit(api, repo, name):
    return next((tag.target_commit for tag in read_hub(api.list_repo_refs, repo, repo_type="model").tags
                 if tag.name == name), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tag-plan-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path, default=Path("production-commits.jsonl"))
    args = parser.parse_args()
    bundle = ROOT / "release"
    tag_file = ROOT / "tag-plan.json"
    require(digest(tag_file) == args.tag_plan_sha256, "reviewed tag plan checksum differs")
    tags = load(tag_file)
    scope = load(ROOT / "inputs/eligible.json")
    held = load(ROOT / "inputs/held.json")
    rows = dict(inspect_bundle(bundle, args.manifest_sha256))
    repairs = rows.keys() & REPAIRS
    require(len(rows) == 58 + len(repairs) and rows.keys() == scope.keys() == tags.keys()
            and held.keys() == (REPAIRS - repairs) | {OUT_OF_SCOPE}
            and not rows.keys() & held.keys()
            and held[OUT_OF_SCOPE]["current_disposition"] == "out of scope"
            and all(held[slug]["current_disposition"] == "pending qualified repair" for slug in REPAIRS - repairs),
            "reviewed release scope differs")
    if args.apply:
        require(os.environ.get("PROMOTE_CONFIRM") == PRODUCTION and get_token(),
                "apply requires an authenticated operator and PROMOTE_CONFIRM=loom-fun")
    api = HfApi(endpoint=ENDPOINT, token=False)
    staging_heads = {model.id: model.sha for model in api.list_models(author=STAGING, full=True, token=False, limit=1000)}
    pending = []
    for slug, row in rows.items():
        tag = tags[slug]
        dst, src = f"{PRODUCTION}/{slug}", f"{STAGING}/{slug}"
        require((tag["repo"], tag["parent"], tag["tag"]) == (dst, row["target_revision"], "v0.2.0"),
                f"{slug}: outgoing tag identity differs")
        require(staging_heads.get(src) == row["staging_revision"],
                f"{slug}: staging head changed after review")
        current = tag_commit(api, dst, tag["tag"])
        if tag["action"] == "already-on-target-sdk":
            require(tag["outgoing_sdk"] == "0.3.0" and current == tag["existing"],
                    f"{slug}: 0.3.0 outgoing tag state changed after review")
        else:
            require(tag["outgoing_sdk"] == "0.2.0" and tag["action"] in ("create", "already-at-parent"),
                    f"{slug}: tag requires review")
            require(current == tag["parent"] or (current is None and tag["action"] == "create"),
                    f"{slug}: released tag conflicts with reviewed parent")
            if current is None:
                pending.append(tag)
        print(f"{dst}: {tag['tag']} {current or 'absent'} ({tag['action']})", flush=True)
    replay(api, bundle, args.manifest_sha256, False, args.receipt)
    if not args.apply:
        return
    for tag in pending:
        api.create_tag(tag["repo"], repo_type="model", revision=tag["parent"], tag=tag["tag"],
                       token=True, exist_ok=False)
        require(tag_commit(api, tag["repo"], tag["tag"]) == tag["parent"], f"{tag['repo']}: tag readback differs")
    replay(api, bundle, args.manifest_sha256, True, args.receipt)


if __name__ == "__main__":
    main()
