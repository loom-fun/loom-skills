"""Publish reviewed release entry points after the production model rollout."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download


ROOT = Path(__file__).resolve().parent
MESSAGE = "Update public Loom Quick starts for SDK 0.3.0"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gh(args, request=None):
    return json.loads(subprocess.check_output(["gh", "api", *args], input=request, text=True))


def live(item, revision=None):
    target, path = item["target"], item["path"]
    if item["kind"] == "space":
        head = revision or HfApi().space_info(target).sha
        data = Path(hf_hub_download(target, path, repo_type="space", revision=head)).read_bytes()
    else:
        head = revision or gh([f"repos/{target}/git/ref/heads/main"])["object"]["sha"]
        data = base64.b64decode(gh([f"repos/{target}/contents/{path}?ref={head}"])["content"])
    return head, sha(data)


def publish(item, parent, candidate):
    target, path = item["target"], item["path"]
    if item["kind"] == "space":
        token = get_token()
        if not token:
            raise SystemExit("Hugging Face authentication is required for Space writes")
        return HfApi(token=token).create_commit(
            target, repo_type="space", revision="main", parent_commit=parent, commit_message=MESSAGE,
            operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=candidate)],
        ).oid
    request = {"query": "mutation($input: CreateCommitOnBranchInput!) { createCommitOnBranch(input:$input) { commit { oid } } }",
               "variables": {"input": {
                   "branch": {"repositoryNameWithOwner": target, "branchName": "main"},
                   "expectedHeadOid": parent, "message": {"headline": MESSAGE},
                   "fileChanges": {"additions": [{"path": path, "contents": base64.b64encode(candidate).decode("ascii")}]},
               }}}
    result = gh(["graphql", "--input", "-"], json.dumps(request))
    return result["data"]["createCommitOnBranch"]["commit"]["oid"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = []
    for item in json.loads((ROOT / "manifest.json").read_text(encoding="utf-8-sig")):
        candidate = (ROOT / "reviewed" / item["candidate"]).read_bytes()
        if sha(candidate) != item["candidate_sha256"]:
            raise SystemExit(f"Reviewed bytes changed for {item['target']}")
        head, file_sha = live(item)
        if file_sha == item["candidate_sha256"]:
            print(f"VERIFIED {item['target']}@{head}")
            continue
        if file_sha != item["original_sha256"]:
            raise SystemExit(f"Unreviewed file change at {item['target']}@{head}")
        if item["kind"] == "space" and head != item["expected_parent"]:
            raise SystemExit(f"Unreviewed Space head at {item['target']}@{head}")
        plan.append((item, head, candidate))
        print(f"READY {item['target']}@{head} {item['path']}")
    if args.apply:
        for item, parent, candidate in plan:
            commit = publish(item, parent, candidate)
            _, file_sha = live(item, revision=commit)
            _, current_sha = live(item)
            if file_sha != item["candidate_sha256"] or current_sha != file_sha:
                raise SystemExit(f"Readback mismatch at {item['target']}@{commit}")
            print(f"VERIFIED {item['target']}@{commit} {file_sha}")


if __name__ == "__main__":
    main()
