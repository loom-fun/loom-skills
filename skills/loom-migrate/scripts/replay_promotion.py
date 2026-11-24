#!/usr/bin/env python3
"""Prepare immutable production package bytes and replay their reviewed manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path, PurePosixPath

STAGING = "loom-fun-staging"
PRODUCTION = "loom-fun"
ENDPOINT = "https://huggingface.co"
REVISION = r"[0-9a-f]{40}"
SHA256 = r"[0-9a-f]{64}"
SLUG = r"[A-Za-z0-9][A-Za-z0-9._-]*"
FETCH = re.compile(r"(?m)^([ \t]*loom-fetch[ \t]+)(?:([\x22\x27])([^\r\n]*?)\2|([^\s]+))")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    return (name == path.as_posix() and not path.is_absolute() and path.parts
            and all(part not in (".", "..", ".git") and "\\" not in part and ":" not in part
                    for part in path.parts))


def load(path):
    return json.loads(path.read_bytes())


def pinned_sha(value):
    return isinstance(value, str) and re.fullmatch(REVISION, value)


def records(pins):
    require(isinstance(pins, dict) and pins, "empty release")
    for slug, row in sorted(pins.items()):
        require(re.fullmatch(SLUG, slug) and ".." not in slug, f"invalid slug {slug}")
        require(all(pinned_sha(row.get(field)) for field in ("staging_revision", "target_revision")),
                f"{slug}: exact staging and target revisions required")
        retained = row.get("retain_target_files", {})
        require(isinstance(retained, dict) and all(safe_path(n) and re.fullmatch(SHA256, sha)
                for n, sha in retained.items()), f"{slug}: invalid retained target file map")
        yield slug, row


def read_hub(operation, repo, *args, **kwargs):
    for attempt in range(3):
        try:
            return operation(repo, *args, token=False, **kwargs)
        except Exception as error:
            response = getattr(error, "response", None)
            if response is None or response.status_code != 429 or attempt == 2:
                raise
            reset = re.search(r"(?:^|;)\s*t=(\d+)", response.headers.get("RateLimit", ""))
            delay = response.headers.get("Retry-After") or (reset[1] if reset else "300")
            seconds = int(delay) if delay.isdecimal() else 300
            print(f"Hub read rate limited; retrying {repo} in {seconds + 1} seconds", flush=True)
            time.sleep(seconds + 1)


def model_info(api, repo, **kwargs):
    return read_hub(api.model_info, repo, **kwargs)


def snapshot(api, repo, revision):
    from huggingface_hub import hf_hub_download

    info = model_info(api, repo, revision=revision, files_metadata=True)
    require(info.id == repo and info.sha == revision and info.siblings, f"{repo}: incorrect or empty revision {revision}")
    files = [file for file in info.siblings if file.rfilename != ".gitattributes"]
    require(files and all(safe_path(file.rfilename) for file in files), f"{repo}: empty package or unsafe path")

    def verify(file):
        name = file.rfilename
        local = Path(read_hub(hf_hub_download, repo, name, revision=revision, endpoint=ENDPOINT))
        root = local.parents[len(PurePosixPath(name).parts) - 1]
        size = local.stat().st_size
        sha = digest(local)
        if file.lfs:
            require(sha == file.lfs.sha256, f"{repo}: corrupt LFS blob {name}")
        else:
            git = hashlib.sha1(f"blob {size}\0".encode())
            with local.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    git.update(block)
            require(git.hexdigest() == file.blob_id, f"{repo}: corrupt Git blob {name}")
        require(file.size is None or size == file.size, f"{repo}: incorrect size {name}")
        return root, name, sha

    # All paths are already in model_info. Resolve them directly without listing the snapshot again.
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify, files))
    path = verified[0][0]
    require(all(root == path for root, _, _ in verified), f"{repo}: inconsistent snapshot cache paths")
    return path, {name: sha for _, name, sha in verified}


def require_head(api, repo, revision):
    live = model_info(api, repo, revision="main").sha
    require(live == revision, f"{repo}: main moved from {revision} to {live}")


def rewrite_readme(data, slug, model):
    src, dst = f"{STAGING}/{slug}", f"{PRODUCTION}/{slug}"
    converted = []

    def rewrite(match):
        prefix, quote, quoted, plain = match.groups()
        source = quoted if quote else plain
        if STAGING not in source:
            return match[0]
        modern = re.fullmatch(rf"huggingface:{re.escape(src)}:([A-Za-z0-9_.-]+)", source)
        legacy = source == f"https://huggingface.co/{src}/resolve/main/model.json"
        if legacy:
            args = match.string[match.end():].split("\n", 1)[0]
            preset = re.search(r"--preset(?:=| +)([A-Za-z0-9_.-]+)", args)
        else:
            preset = modern
        require(preset and preset[1] in model["presets"], f"{slug}: unexpected fetch selector {source}")
        converted.append(preset[1])
        quote = quote or ""
        return prefix + quote + source.replace(src, dst, 1) + quote

    text = FETCH.sub(rewrite, data.decode("utf-8"))
    require(converted, f"{slug}: no staging fetch example")
    return text.encode(), converted


def inspect_refs(name, data, slug):
    if STAGING.encode() not in data:
        return
    text = data.decode("utf-8")
    if name.endswith(".md") or name.endswith("qualification.json"):
        pinned = rf"{STAGING}/{SLUG}(?:@{REVISION}|/(?:resolve|commit|tree|blob)/{REVISION})(?![0-9a-f])"
        text = re.sub(pinned, "", text)
    if name == "quick-start-qualification.json" and STAGING in text:
        proof = json.loads(data)
        repo = f"{STAGING}/{slug}"
        require(proof.get("staging_repo") == repo and pinned_sha(proof.get("staging_parent")),
                f"{slug}: unpinned quick-start qualification")
        text, count = re.subn(rf'("staging_repo"\s*:\s*"){re.escape(repo)}(")', r"\1\2", text, count=1)
        require(count == 1, f"{slug}: invalid quick-start provenance")
    require(STAGING not in text, f"{slug}/{name}: found unhandled staging reference")


def check_image(path, component_hashes, slug, presets, old_revision):
    proof = load(path / "sdk-qualification.json")
    comparison = proof["comparison"]
    require(proof["sdk"]["hrx_loom"] == "0.3.0" and comparison["programs_match"]
            and comparison["pixels_match"] and comparison["max_abs_channel_diff"] == 0
            and comparison["repo"] == f"{PRODUCTION}/{slug}" and comparison["revision"] == old_revision
            and set(presets) == {comparison["preset"]}, f"{slug}: image qualification does not match release")
    programs = {name: sha for name, sha in component_hashes.items() if name.endswith(".loom")}
    for version in ("0.2.0", "0.3.0"):
        require(programs and comparison["runs"][version]["programs"] == programs,
                f"{slug}: image programs differ from {version} qualification")


def compose(slug, row, source, source_hashes, target, target_hashes, out):
    require({"README.md", "model.json", "model.json.lock"} <= source_hashes.keys(), f"{slug}: package missing")
    model = load(source / "model.json")
    readme, presets = rewrite_readme((source / "README.md").read_bytes(), slug, model)
    require(b'hrx-loom[hf]==0.3.0' in readme, f"{slug}: no exact release install pin")
    changed = {"README.md": readme}
    if "info.json" in source_hashes:
        text = (source / "info.json").read_bytes().decode("utf-8")
        require(json.loads(text).get("repo") == f"{STAGING}/{slug}", f"{slug}: incorrect info.json repo")
        src = re.escape(f"{STAGING}/{slug}")
        new_text, count = re.subn(rf'("repo"\s*:\s*"){src}(")', rf"\g<1>{PRODUCTION}/{slug}\g<2>", text, count=1)
        require(count == 1 and json.loads(new_text)["repo"] == f"{PRODUCTION}/{slug}", f"{slug}: cannot rewrite card metadata")
        changed["info.json"] = new_text.encode()
    else:
        require(all(service.get("kind") == "image" for service in model["services"].values()),
                f"{slug}: missing generated card metadata on non-image package")
        require("sdk-qualification.json" in source_hashes, f"{slug}: image SDK qualification missing")
        check_image(source, source_hashes, slug, presets, row["target_revision"])
    retained = {name: target_hashes[name] for name in sorted(target_hashes.keys() - source_hashes.keys())}
    require(retained == row.get("retain_target_files", {}),
            f"{slug}: target-only files require explicit hash-pinned retention: {retained}")
    output_names = source_hashes.keys() | retained.keys()
    for name in sorted(output_names):
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name in changed:
            dest.write_bytes(changed[name])
        else:
            shutil.copyfile((target if name in retained else source) / name, dest)
        inspect_refs(name, dest.read_bytes(), slug)
    outputs = {name: digest(out / name) for name in sorted(output_names)}
    return {**row, "source": source_hashes, "target": target_hashes, "output": outputs,
            "presets": presets, "operations": sorted(n for n, sha in outputs.items() if target_hashes.get(n) != sha)}


def prepare(api, pins, bundle):
    rows = list(records(load(pins)))
    bundle.mkdir(parents=True)
    release = {}
    for slug, row in rows:
        source_repo, target_repo = f"{STAGING}/{slug}", f"{PRODUCTION}/{slug}"
        require_head(api, target_repo, row["target_revision"])
        source, source_hashes = snapshot(api, source_repo, row["staging_revision"])
        target, target_hashes = snapshot(api, target_repo, row["target_revision"])
        release[slug] = compose(slug, row, source, source_hashes, target, target_hashes, bundle / "packages" / slug)
        print(f"Prepared {slug} ({len(source_hashes)} files)", flush=True)
    manifest = bundle / "manifest.json"
    manifest.write_bytes(canonical({"schema": 1, "packages": release}))
    print(f"Release manifest SHA256 {digest(manifest)}")


def inspect_bundle(bundle, manifest_sha):
    manifest = bundle / "manifest.json"
    require(re.fullmatch(SHA256, manifest_sha) and digest(manifest) == manifest_sha, "reviewed manifest checksum mismatch")
    document = load(manifest)
    require(document.get("schema") == 1, "unsupported release manifest schema")
    rows = list(records(document["packages"]))
    require({p.name for p in (bundle / "packages").iterdir()} == {s for s, _ in rows}, "bundle slug coverage differs")
    for slug, row in rows:
        root = bundle / "packages" / slug
        files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() or p.is_symlink()}
        require(files.keys() == row["output"].keys(), f"{slug}: bundle file coverage differs")
        require(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob("*")), f"{slug}: bundle contains symlinks")
        for name, path in files.items():
            require(safe_path(name) and digest(path) == row["output"][name], f"{slug}: incorrect bundle file {name}")
        expected_ops = sorted(n for n, sha in row["output"].items() if row["target"].get(n) != sha)
        require(expected_ops == row["operations"] and not row["target"].keys() - row["output"].keys(),
                f"{slug}: incorrect operation coverage")
    return rows


def checked_result(api, repo, revision, output):
    _, result = snapshot(api, repo, revision)
    require(result == output, f"{repo}@{revision}: readback differs from review")
    require_head(api, repo, revision)


def read_progress(path, manifest_sha, rows):
    if not path.exists():
        return {}
    require(not path.is_symlink(), "receipt must be a regular file")
    repo_to_row = {f"{PRODUCTION}/{slug}": row for slug, row in rows}
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        repo, commit = event["repo"], event["commit"]
        require(event["manifest_sha256"] == manifest_sha and pinned_sha(commit) and repo in repo_to_row,
                "receipt does not belong to the reviewed release")
        require(repo not in entries or entries[repo] == commit, f"{repo}: ambiguous production receipt")
        entries[repo] = commit
    return entries


def replay(api, bundle, manifest_sha, execute, receipt_path):
    rows = inspect_bundle(bundle, manifest_sha)
    if execute:
        require(receipt_path and os.environ.get("PROMOTE_CONFIRM") == PRODUCTION,
                "execution requires --receipt and explicit PROMOTE_CONFIRM=loom-fun")
    prior = read_progress(receipt_path, manifest_sha, rows) if receipt_path else {}
    # Preflight every reviewed parent before any upload. Each actual commit also uses CAS.
    for slug, row in rows:
        repo = f"{PRODUCTION}/{slug}"
        if repo in prior:
            checked_result(api, repo, prior[repo], row["output"])
        else:
            require_head(api, repo, row["target_revision"])
            _, content = snapshot(api, repo, row["target_revision"])
            require(content == row["target"], f"{repo}: target bytes differ from review")
    if not execute:
        for slug, row in rows:
            print(f"{PRODUCTION}/{slug}@{row['target_revision']}: {len(row['operations'])} change(s)", flush=True)
        return
    from huggingface_hub import CommitOperationAdd

    with receipt_path.open("a", encoding="utf-8", buffering=1) as log:
        for slug, row in rows:
            repo = f"{PRODUCTION}/{slug}"
            if repo in prior or not row["operations"]:
                print(f"{repo}: verified, no pending upload", flush=True)
                continue
            commit = api.create_commit(
                repo, repo_type="model", revision="main", parent_commit=row["target_revision"], token=True,
                commit_message=f"Promote reviewed package from staging {row['staging_revision']}",
                operations=[CommitOperationAdd(name, bundle / "packages" / slug / name) for name in row["operations"]],
            )
            require(pinned_sha(commit.oid), f"{repo}: missing upload commit; reconcile before retry")
            event = {"repo": repo, "commit": commit.oid, "manifest_sha256": manifest_sha}
            print(json.dumps({**event, "verified": False}), file=log)
            os.fsync(log.fileno())
            checked_result(api, repo, commit.oid, row["output"])
            print(json.dumps({**event, "verified": True}), file=log)
            print(f"{repo}@{commit.oid}: verified", flush=True)


def main():
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--pins", required=True, type=Path)
    prep.add_argument("--bundle", required=True, type=Path)
    run = commands.add_parser("replay")
    run.add_argument("--bundle", required=True, type=Path)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    api = HfApi(endpoint=ENDPOINT, token=False)
    if args.command == "prepare":
        prepare(api, args.pins, args.bundle)
    else:
        replay(api, args.bundle, args.manifest_sha256, args.execute, args.receipt)


if __name__ == "__main__":
    main()
