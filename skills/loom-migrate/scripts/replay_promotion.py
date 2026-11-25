#!/usr/bin/env python3
"""Prepare immutable production package bytes for an SDK release and replay the reviewed manifest.

The release scope is data, never code: `--eligible` maps each promoted slug to its exact `staging_revision` and
`target_revision` plus optional per-package exceptions (`retain_target_files`, `delete_target_files` + `rollback_tag`,
`image_readme_refresh`); `--held` maps each production repo that is NOT promoted to its `reason` (and optionally
`tag_revision`, the exact production head that still gets the outgoing SDK tag). Every production `*-loom` repo must be
in exactly one of the two files (a repo published after the snapshot fails the release instead of being left behind).

  prepare  --eligible E.json --held H.json --sdk X.Y.Z --outgoing-tag vA.B.C --bundle DIR
           [--loom-fetch PATH] [--only SLUG...]
  replay   --bundle DIR --manifest-sha256 SHA [--execute --receipt FILE]
  dry-run  the arguments of prepare: prepare + outgoing tag plan + read-only replay; anonymous Hub reads only

Order of the production writes (release-checklist.md section 5): outgoing tags (`tag_org.py --bundle`), then the loom-skills
merge and release tag, then `replay --execute`. Replay refuses to write before every planned outgoing tag resolves to its
reviewed parent. A crash between the Hub commit and the receipt write is recovered on retry when the live head is the
release's own commit (exact title and manifest marker, the reviewed parent, the exact byte inventory); anything else
stops with a reconcile error.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
import posixpath

STAGING = "loom-fun-staging"
PRODUCTION = "loom-fun"
ENDPOINT = "https://huggingface.co"
REVISION = r"[0-9a-f]{40}"
SHA256 = r"[0-9a-f]{64}"
SLUG = r"[A-Za-z0-9][A-Za-z0-9._-]*"
LOOM_REPO = re.compile(r".+-loom(?:-[A-Za-z0-9._-]+)?")
HUB_ATTRIBUTES = ".gitattributes"
CARD_SCRIPTS = Path(__file__).resolve().parents[2] / "loom-benchmark" / "scripts" / "card"
FETCH = re.compile(r"(?m)^([ \t]*loom-fetch[ \t]+)(?:([\x22\x27])([^\r\n]*?)\2|([^\s]+))")
SCHEMA = 2
PIN_FIELDS = ("staging_revision", "target_revision", "retain_target_files", "delete_target_files", "rollback_tag",
              "image_readme_refresh", "programs_sha256")
TAG_ACTIONS = ("create", "already-at-parent", "already-on-target-sdk")
MARKER = "Reviewed promotion manifest SHA256: "


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


def valid_slug(slug):
    return isinstance(slug, str) and re.fullmatch(SLUG, slug) and ".." not in slug


def commit_title(row):
    return f"Promote reviewed package from staging {row['staging_revision']}"


# ---- release inputs (data) -------------------------------------------------------------------------------------------

def release_meta(sdk, outgoing_tag, card_check="passed"):
    require(isinstance(sdk, str) and re.fullmatch(r"\d+\.\d+\.\d+", sdk), f"--sdk {sdk!r}: use X.Y.Z")
    require(isinstance(outgoing_tag, str) and re.fullmatch(r"v\d+\.\d+\.\d+", outgoing_tag),
            f"--outgoing-tag {outgoing_tag!r}: use vX.Y.Z")
    require(outgoing_tag[1:] != sdk, "the outgoing SDK tag must name the previous SDK, not the release SDK")
    require(card_check in ("passed", "skipped"), f"card check state {card_check!r}")
    return {"sdk": sdk, "outgoing_tag": outgoing_tag, "card_check": card_check,
            "staging_org": STAGING, "production_org": PRODUCTION}


def records(pins, allow_empty=False):
    """Validate the pin fields of every row (eligible input or manifest row) and yield (slug, row) sorted."""
    require(isinstance(pins, dict) and (pins or allow_empty), "empty release")
    for slug, row in sorted(pins.items()):
        require(valid_slug(slug), f"invalid slug {slug}")
        require(isinstance(row, dict), f"{slug}: row must be an object")
        require(all(pinned_sha(row.get(field)) for field in ("staging_revision", "target_revision")),
                f"{slug}: exact staging and target revisions required")
        for field in ("retain_target_files", "delete_target_files"):
            paths = row.get(field, {})
            require(isinstance(paths, dict) and all(n != HUB_ATTRIBUTES and safe_path(n) and re.fullmatch(SHA256, sha)
                    for n, sha in paths.items()), f"{slug}: invalid {field} file map")
        if row.get("delete_target_files"):
            require(re.fullmatch(r"v\d+\.\d+\.\d+", row.get("rollback_tag", "")),
                    f"{slug}: deleted files require a pinned outgoing rollback tag")
        require("programs_sha256" not in row or re.fullmatch(SHA256, row["programs_sha256"] or ""),
                f"{slug}: programs_sha256 must be a sha256")
        if "image_readme_refresh" in row:
            refresh = row["image_readme_refresh"]
            hashes = {"reference_readme_sha256", "target_readme_sha256", "staging_readme_sha256"}
            require(isinstance(refresh, dict)
                    and refresh.keys() == hashes | {"reference_revision"}
                    and pinned_sha(refresh["reference_revision"])
                    and refresh["reference_revision"] != row["target_revision"]
                    and all(isinstance(refresh[name], str) and re.fullmatch(SHA256, refresh[name]) for name in hashes),
                    f"{slug}: invalid image README refresh")
        yield slug, row


def pin_rows(eligible):
    """The eligible file's rows reduced to the fields the release acts on (the other fields are evidence, not scope)."""
    return {slug: {k: row[k] for k in PIN_FIELDS if k in row} for slug, row in records(eligible)}


def held_rows(held):
    require(isinstance(held, dict), "held file must map slug to an object")
    for slug, row in held.items():
        require(valid_slug(slug) and isinstance(row, dict), f"held: invalid row {slug}")
        require(isinstance(row.get("reason"), str) and row["reason"].strip(), f"held {slug}: a reason is required")
        require("tag_revision" not in row or pinned_sha(row["tag_revision"]),
                f"held {slug}: tag_revision must be an exact commit")
    return {slug: {k: row[k] for k in ("reason", "tag_revision", "current_disposition") if k in row}
            for slug, row in sorted(held.items())}


def production_inventory(api):
    models = read_hub(lambda token, **kw: list(api.list_models(token=token, **kw)), None, author=PRODUCTION, limit=10000)
    return sorted(name for name in (m.id.split("/", 1)[1] for m in models) if LOOM_REPO.fullmatch(name))


def check_scope(pins, held, inventory):
    require(not pins.keys() & held.keys(), f"slugs both eligible and held: {sorted(pins.keys() & held.keys())}")
    if inventory is not None:
        missing = sorted(set(inventory) - pins.keys() - held.keys())
        unknown = sorted((pins.keys() | held.keys()) - set(inventory))
        require(not missing and not unknown,
                f"release scope differs from the production inventory: neither eligible nor held {missing}; "
                f"not in production {unknown}")


# ---- Hub reads --------------------------------------------------------------------------------------------------------

def read_hub(operation, repo, *args, **kwargs):
    for attempt in range(3):
        try:
            if repo is None:
                return operation(*args, token=False, **kwargs)
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


def download(repo, name, revision):
    from huggingface_hub import hf_hub_download

    return Path(read_hub(hf_hub_download, repo, name, revision=revision, endpoint=ENDPOINT))


def snapshot(api, repo, revision):
    info = model_info(api, repo, revision=revision, files_metadata=True)
    require(info.id == repo and info.sha == revision and info.siblings, f"{repo}: incorrect or empty revision {revision}")
    files = info.siblings
    require(files and all(safe_path(file.rfilename) for file in files), f"{repo}: empty package or unsafe path")

    def verify(file):
        name = file.rfilename
        local = download(repo, name, revision)
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


def live_head(api, repo):
    return model_info(api, repo, revision="main").sha


def require_head(api, repo, revision):
    live = live_head(api, repo)
    require(live == revision, f"{repo}: main moved from {revision} to {live}")


def tag_commit(api, repo, tag):
    """The commit a tag resolves to, or None when the repo has no such tag."""
    refs = read_hub(api.list_repo_refs, repo, repo_type="model")
    if tag not in {ref.name for ref in refs.tags}:
        return None
    return model_info(api, repo, revision=tag).sha


# ---- outgoing tag plan ------------------------------------------------------------------------------------------------

def sdk_pin(directory, names):
    """(pin, evidence file, its sha256): llm_config.json's sdk pin (pin-less = 0.1.0), else the README's install pin."""
    if "llm_config.json" in names:
        data = (directory / "llm_config.json").read_bytes()
        pin = (json.loads(data).get("sdk") or {}).get("hrx_loom") or "0.1.0"
        return pin, "llm_config.json", hashlib.sha256(data).hexdigest()
    require("README.md" in names, f"{directory}: package without llm_config.json or README.md")
    data = (directory / "README.md").read_bytes()
    match = re.search(rb"hrx-loom\[hf\]==([0-9]+\.[0-9]+\.[0-9]+)", data)
    return (match[1].decode() if match else None), "README.md", hashlib.sha256(data).hexdigest()


def tag_entry(api, repo, parent, pin, meta):
    """The outgoing-tag action for one production parent; refuses anything that is not plainly old or plainly new."""
    pin_value, pin_path, pin_sha = pin
    tag, outgoing = meta["outgoing_tag"], meta["outgoing_tag"][1:]
    existing = tag_commit(api, repo, tag)
    if pin_value == meta["sdk"]:
        action = "already-on-target-sdk"
    else:
        require(pin_value == outgoing, f"{repo}@{parent}: pins hrx-loom {pin_value}, neither {outgoing} nor {meta['sdk']}")
        require(existing in (None, parent), f"{repo}: {tag} already points to {existing}, not the reviewed parent {parent}")
        action = "already-at-parent" if existing == parent else "create"
    return {"repo": repo, "parent": parent, "tag": tag, "action": action, "existing": existing,
            "outgoing_sdk": pin_value, "pin_path": pin_path, "pin_sha256": pin_sha}


def held_tag_entry(api, slug, row, meta):
    repo, revision = f"{PRODUCTION}/{slug}", row["tag_revision"]
    require_head(api, repo, revision)
    names = {s.rfilename for s in model_info(api, repo, revision=revision).siblings}
    pick = "llm_config.json" if "llm_config.json" in names else "README.md"
    directory = download(repo, pick, revision).parent
    return tag_entry(api, repo, revision, sdk_pin(directory, {pick}), meta)


# ---- production-form composition ---------------------------------------------------------------------------------------

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
    """The production form names no staging repo anywhere (owner decision on #1415: no staging provenance on cards)."""
    require(STAGING.encode() not in data, f"{slug}/{name}: staging reference in the production form")


def card_file_check(slug, directory):
    """The card's own file-set and text check (loom-benchmark scripts/card/card_files.py); fail closed without it."""
    import importlib.util

    path = CARD_SCRIPTS / "card_files.py"
    require(path.is_file(), f"{slug}: card file-set check {path} is missing; refusing to compose")
    spec = importlib.util.spec_from_file_location("card_files", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    problems = module.check_package(directory)
    require(not problems, f"{slug}: production form fails the card file-set check: {problems}")


def programs_sha256(directory):
    """The card's programs hash (mirrors loom-benchmark card_lib.programs_sha256; a test keeps them equal): sha256 over
    program/**/*.loom by path, each file without the generator's first-line comment. None without program files."""
    root = directory / "program"
    files = sorted((p.relative_to(root).as_posix(), p) for p in root.rglob("*.loom") if p.is_file()) if root.is_dir() else []
    h = hashlib.sha256()
    for name, path in files:
        data = path.read_bytes()
        if data.startswith(b"// Generated by"):
            data = data.split(b"\n", 1)[1] if b"\n" in data else b""
        h.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest() if files else None


def check_image_readme_refresh(slug, refresh, source, target, reference):
    require(reference is not None and {n: sha for n, sha in target.items() if n != "README.md"}
            == {n: sha for n, sha in reference.items() if n != "README.md"},
            f"{slug}: reviewed image production changed outside README.md")
    require((reference.get("README.md"), target.get("README.md"), source.get("README.md"))
            == (refresh["reference_readme_sha256"], refresh["target_readme_sha256"], refresh["staging_readme_sha256"]),
            f"{slug}: reviewed image README identities differ")


def check_deleted_refs(name, data, deleted, slug):
    if deleted and name.endswith((".json", ".json.lock", ".loom")):
        for match in re.finditer(rb"[A-Za-z0-9_./-]+\.loom", data):
            reference = match[0].decode()
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), reference))
            for old in deleted:
                require(not (reference == old or resolved == old or old.endswith("/" + reference.lstrip("./"))),
                        f"{slug}/{name}: references deleted target file {old}")


def compose(slug, row, source, source_hashes, target, target_hashes, out, meta, reference_hashes=None):
    require({"README.md", "model.json", "model.json.lock"} <= source_hashes.keys(), f"{slug}: package missing")
    model = load(source / "model.json")
    readme, presets = rewrite_readme((source / "README.md").read_bytes(), slug, model)
    require(f"hrx-loom[hf]=={meta['sdk']}".encode() in readme, f"{slug}: no exact release install pin")
    changed = {"README.md": readme}
    refresh = row.get("image_readme_refresh")
    if "info.json" in source_hashes:
        require(refresh is None, f"{slug}: image README refresh requires a bespoke image package")
        text = (source / "info.json").read_bytes().decode("utf-8")
        require(json.loads(text).get("repo") == f"{STAGING}/{slug}", f"{slug}: incorrect info.json repo")
        src = re.escape(f"{STAGING}/{slug}")
        new_text, count = re.subn(rf'("repo"\s*:\s*"){src}(")', rf"\g<1>{PRODUCTION}/{slug}\g<2>", text, count=1)
        require(count == 1 and json.loads(new_text)["repo"] == f"{PRODUCTION}/{slug}", f"{slug}: cannot rewrite card metadata")
        changed["info.json"] = new_text.encode()
    else:
        require(all(service.get("kind") == "image" for service in model["services"].values()),
                f"{slug}: missing generated card metadata on non-image package")
        require(row.get("programs_sha256"), f"{slug}: an image package needs its qualified programs_sha256 in the release input")
        if refresh is not None:
            check_image_readme_refresh(slug, refresh, source_hashes, target_hashes, reference_hashes)
            row = {**row, "image_reference_hashes": reference_hashes}
    extras = {name: target_hashes[name] for name in sorted(target_hashes.keys() - source_hashes.keys())
              if name != HUB_ATTRIBUTES}
    retained = row.get("retain_target_files", {})
    deleted = row.get("delete_target_files", {})
    require(not retained.keys() & deleted.keys() and extras == retained | deleted,
            f"{slug}: target-only files require explicit retain or delete pins: {extras}")
    output_names = (source_hashes.keys() | retained.keys()) - {HUB_ATTRIBUTES}
    if HUB_ATTRIBUTES in target_hashes:
        output_names.add(HUB_ATTRIBUTES)
    for name in sorted(output_names):
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name in changed:
            data = changed[name]
        else:
            data = ((target if name in retained or name == HUB_ATTRIBUTES else source) / name).read_bytes()
        dest.write_bytes(data)
        inspect_refs(name, data, slug)
        check_deleted_refs(name, data, deleted, slug)
    if row.get("programs_sha256"):
        found = programs_sha256(out)
        require(found == row["programs_sha256"], f"{slug}: programs {found} are not the qualified {row['programs_sha256']}")
    if meta["card_check"] == "passed":
        card_file_check(slug, out)
    outputs = {name: digest(out / name) for name in sorted(output_names)}
    return {**row, "source": source_hashes, "target": target_hashes, "output": outputs, "deletions": sorted(deleted),
            "presets": presets, "operations": sorted(n for n, sha in outputs.items() if target_hashes.get(n) != sha)}


def fetch_check(fetch, root, slug, row, scratch):
    """The SDK's own fetch dry run of every preset of the production-form package (no download, no Hub write)."""
    package = root / "packages" / slug
    results = []
    for preset in load(package / "model.json")["presets"]:
        dry = subprocess.run([str(fetch), str(package / "model.json"), "--preset", preset, "--output",
                              str(scratch / slug / preset), "--json", "--no-input", "--dry-run"],
                             capture_output=True, text=True, timeout=120, check=False)
        require(dry.returncode == 0, f"SDK fetch dry run failed for {slug}:{preset}: {dry.stderr[-1400:]}")
        plan = json.loads(dry.stdout)
        require(plan["preset"] == preset and plan["service"] and plan["files"] and plan["lock"] == "verify"
                and not plan["unverified_parameters"], f"unexpected SDK fetch plan for {slug}:{preset}")
        closure = json.dumps(plan["files"], sort_keys=True)
        require(not any(path.removeprefix("program/") in closure for path in row.get("deletions", [])),
                f"{slug}:{preset} still references an explicitly deleted path")
        results.append({"preset": preset, "advertised": preset in row["presets"], "service": plan["service"],
                        "files": len(plan["files"])})
    require(results, f"{slug}: no presets")
    require(not (scratch / slug).exists(), f"{slug}: SDK dry run wrote package files")
    return results


# ---- prepare ----------------------------------------------------------------------------------------------------------

def runbook(document, manifest_sha):
    """The release's RELEASE.md: release-checklist.md section 5 in order, with this bundle's identities."""
    meta, packages, held, tags = document["release"], document["packages"], document["held"], document["tags"]
    creates = sum(t["action"] == "create" for t in tags.values())
    lines = [
        f"# Promotion runbook: hrx-loom {meta['outgoing_tag'][1:]} -> {meta['sdk']}", "",
        f"Generated by `replay_promotion.py prepare`. Manifest SHA256 `{manifest_sha}`. Scope: {len(packages)} promoted "
        f"package(s), {len(held)} held repo(s); outgoing tag plan: {len(tags)} repo(s), {creates} tag(s) to create. "
        f"Card file-set check: {meta['card_check']}.", "",
        "Every write below needs the owner's explicit go for this exact manifest SHA256. Run the steps in this order "
        "(release-checklist.md section 5); a failed step stops the release. `DIR` is this bundle, `SHA` the reviewed "
        "manifest SHA256; the scripts are loom-skills `skills/loom-migrate/scripts/` at the reviewed tool commit.", "",
        "0. Read-only check: `replay_promotion.py replay --bundle DIR --manifest-sha256 SHA` and "
        "`tag_org.py --bundle DIR --manifest-sha256 SHA --dry-run`.",
        "1. Outgoing tags first, at the exact reviewed parents (text and image repos alike): "
        "`tag_org.py --bundle DIR --manifest-sha256 SHA`. Every row must read `exists`, `created` or `skip`.",
        "2. Merge the integration branch into loom-skills `main` (pinned to the reviewed head) and tag the merge commit "
        f"`sdk{meta['sdk']}-uN` (loom-fleet publish.md \"Release tags\").",
        "3. Replay: `PROMOTE_CONFIRM=loom-fun replay_promotion.py replay --bundle DIR --manifest-sha256 SHA --execute "
        "--receipt production-commits.jsonl`. It refuses before any write unless step 1 is complete. Keep the receipt; "
        "after an interruption rerun the same command with the same receipt (a commit whose receipt line was lost is "
        "recovered from its title, manifest marker, parent and bytes; anything else stops with a reconcile error).",
        "4. Merge the production HF org page change (a Hub pull request on the org's README Space).",
        "5. Merge the loom-dev website pull request; check the device pages.",
        "6. Merge the extension and weaver PRs; take the Lemonade PR out of draft; Gufo PR ready if its maintainers agree.",
        "7. Clean up: experiment PRs, merged branches, campaign dirs, final report.", "",
        "## Held (no package write; an outgoing tag only where a tag revision is listed)", ""]
    lines += [f"- `{slug}`: {row['reason']}" + (f" (outgoing tag at `{row['tag_revision']}`)" if row.get("tag_revision")
                                                else "") for slug, row in held.items()] or ["- none"]
    return "\n".join(lines) + "\n"


def prepare_one(api, slug, row, bundle, meta):
    source_repo, target_repo = f"{STAGING}/{slug}", f"{PRODUCTION}/{slug}"
    require_head(api, target_repo, row["target_revision"])
    source, source_hashes = snapshot(api, source_repo, row["staging_revision"])
    target, target_hashes = snapshot(api, target_repo, row["target_revision"])
    refresh = row.get("image_readme_refresh")
    reference_hashes = snapshot(api, target_repo, refresh["reference_revision"])[1] if refresh else None
    composed = compose(slug, row, source, source_hashes, target, target_hashes, bundle / "packages" / slug, meta,
                       reference_hashes)
    tag = tag_entry(api, target_repo, row["target_revision"], sdk_pin(target, target_hashes), meta)
    if row.get("rollback_tag"):
        require(row["rollback_tag"] == meta["outgoing_tag"] and tag["action"] != "already-on-target-sdk",
                f"{slug}: rollback tag must be the planned outgoing tag")
    return composed, tag


def prepare(api, eligible_path, held_path, bundle, meta, only=None, fetch=None, keep_going=False):
    """Compose every eligible package and the outgoing tag plan into BUNDLE; return (manifest sha256, failures).
    A release prepare stops at the first failure. keep_going (dry run) reports every failing package instead and leaves
    them out of a non-executable bundle."""
    pins = pin_rows(load(eligible_path))
    held = held_rows(load(held_path)) if held_path else {}
    if only:
        require(set(only) <= pins.keys(), f"--only names slugs that are not eligible: {sorted(set(only) - pins.keys())}")
        pins = {slug: pins[slug] for slug in sorted(only)}
        held, inventory = {}, None
        print(f"--only: partial bundle of {len(pins)} package(s); inventory coverage not checked; not executable", flush=True)
    else:
        inventory = production_inventory(api)
        check_scope(pins, held, inventory)
    bundle.mkdir(parents=True)
    release, tags, failures = {}, {}, {}
    for slug, row in pins.items():
        try:
            release[slug], tags[slug] = prepare_one(api, slug, row, bundle, meta)
            if fetch:
                release[slug]["fetch_check"] = fetch_check(fetch, bundle, slug, release[slug],
                                                           bundle.parent / f".{bundle.name}-fetch-scratch")
        except Exception as error:  # noqa: BLE001 - a dry run reports every package; a release stops at the first
            if not keep_going:
                raise
            failures[slug] = f"{type(error).__name__}: {error}"
            release.pop(slug, None)
            tags.pop(slug, None)
            shutil.rmtree(bundle / "packages" / slug, ignore_errors=True)
            print(f"FAIL {slug}: {failures[slug]}", flush=True)
            continue
        print(f"Prepared {slug} ({len(release[slug]['source'])} files, {len(release[slug]['operations'])} update(s), "
              f"{len(release[slug]['deletions'])} removal(s), {tags[slug]['tag']} {tags[slug]['action']})", flush=True)
    for slug, row in held.items():
        if row.get("tag_revision"):
            tags[slug] = held_tag_entry(api, slug, row, meta)
            print(f"Held {slug}: {tags[slug]['tag']} {tags[slug]['action']} at {row['tag_revision']}", flush=True)
    if failures:
        inventory = None  # an incomplete bundle can never be executed
        held = {slug: row for slug, row in held.items() if slug in tags}
    (bundle / "packages").mkdir(exist_ok=True)
    document = {"schema": SCHEMA, "release": {**meta, "inventory": inventory,
                                              "tool_sha256": digest(Path(__file__).resolve())},
                "packages": release, "held": held, "tags": tags}
    manifest = bundle / "manifest.json"
    manifest.write_bytes(canonical(document))
    sha = digest(manifest)
    (bundle / "RELEASE.md").write_text(runbook(document, sha), encoding="utf-8")
    print(f"Release manifest SHA256 {sha}" + (f" (dry run: {len(failures)} package(s) failed)" if failures else ""))
    return sha, failures


# ---- replay -----------------------------------------------------------------------------------------------------------

def inspect_bundle(bundle, manifest_sha):
    """Verify the bundle against the reviewed checksum; return (document, rows)."""
    manifest = bundle / "manifest.json"
    require(isinstance(manifest_sha, str) and re.fullmatch(SHA256, manifest_sha) and digest(manifest) == manifest_sha,
            "reviewed manifest checksum mismatch")
    document = load(manifest)
    require(document.get("schema") == SCHEMA, "unsupported release manifest schema (prepare the bundle again)")
    meta = document["release"]
    release_meta(meta["sdk"], meta["outgoing_tag"], meta["card_check"])
    require((meta["staging_org"], meta["production_org"]) == (STAGING, PRODUCTION), "bundle names other organizations")
    rows = list(records(document["packages"], allow_empty=meta["inventory"] is None))
    held = document["held"]
    held_rows(held)
    check_scope(document["packages"], held, meta["inventory"])
    tags = document["tags"]
    require(tags.keys() == document["packages"].keys() | {s for s, r in held.items() if r.get("tag_revision")},
            "tag plan coverage differs from the release scope")
    for slug, tag in tags.items():
        parent = document["packages"][slug]["target_revision"] if slug in document["packages"] else held[slug]["tag_revision"]
        require(tag["repo"] == f"{PRODUCTION}/{slug}" and tag["parent"] == parent and tag["tag"] == meta["outgoing_tag"]
                and tag["action"] in TAG_ACTIONS, f"{slug}: invalid tag plan entry")
    require({p.name for p in (bundle / "packages").iterdir()} == {s for s, _ in rows}, "bundle slug coverage differs")
    for slug, row in rows:
        if (refresh := row.get("image_readme_refresh")) is not None:
            check_image_readme_refresh(slug, refresh, row["source"], row["target"], row.get("image_reference_hashes"))
        if row.get("rollback_tag"):
            require(row["rollback_tag"] == meta["outgoing_tag"] and tags[slug]["action"] != "already-on-target-sdk",
                    f"{slug}: rollback tag must be the planned outgoing tag")
        root = bundle / "packages" / slug
        files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() or p.is_symlink()}
        require(files.keys() == row["output"].keys(), f"{slug}: bundle file coverage differs")
        require(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob("*")), f"{slug}: bundle contains symlinks")
        for name, path in files.items():
            require(safe_path(name) and digest(path) == row["output"][name], f"{slug}: incorrect bundle file {name}")
        expected_ops = sorted(n for n, sha in row["output"].items() if row["target"].get(n) != sha)
        deletions = {n: sha for n, sha in row["target"].items() if n not in row["output"]}
        require(expected_ops == row["operations"] and deletions == row.get("delete_target_files", {})
                and sorted(deletions) == row.get("deletions", [])
                and row["output"].get(HUB_ATTRIBUTES) == row["target"].get(HUB_ATTRIBUTES),
                f"{slug}: incorrect operation coverage")
    return document, rows


def checked_result(api, repo, revision, output):
    _, result = snapshot(api, repo, revision)
    require(result == output, f"{repo}@{revision}: readback differs from review")
    require_head(api, repo, revision)


def recover(api, repo, row, manifest_sha, head):
    """A live head that is this release's own commit for this repo (a lost receipt line): its title, manifest marker,
    parent and full byte inventory must all match. Anything else is a foreign change: stop for manual reconciliation."""
    reconcile = f"{repo}: main moved from the reviewed parent {row['target_revision']} to {head}"
    commits = read_hub(api.list_repo_commits, repo, revision=head)
    require(len(commits) >= 2 and commits[0].commit_id == head,
            f"{reconcile}; its history cannot be read: reconcile before retry")
    first = commits[0]
    require(first.title == commit_title(row) and f"{MARKER}{manifest_sha}" in f"{first.title}\n{first.message or ''}",
            f"{reconcile}, which is not this release's commit: reconcile before retry")
    require(commits[1].commit_id == row["target_revision"],
            f"{reconcile}, whose parent {commits[1].commit_id} is not the reviewed parent: reconcile before retry")
    checked_result(api, repo, head, row["output"])


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


def write_event(log, event):
    print(json.dumps(event, sort_keys=True), file=log, flush=True)
    os.fsync(log.fileno())


def replay(api, bundle, manifest_sha, execute, receipt_path):
    """Read-only by default: verify every bundle file, production parent, staging head and outgoing tag; return the plan.
    --execute then writes one parent-guarded commit per pending repo and records each in the receipt."""
    document, rows = inspect_bundle(bundle, manifest_sha)
    meta, tags, by_slug = document["release"], document["tags"], dict(rows)
    if execute:
        require(receipt_path and os.environ.get("PROMOTE_CONFIRM") == PRODUCTION,
                "execution requires --receipt and explicit PROMOTE_CONFIRM=loom-fun")
        require(meta["inventory"] is not None and meta["card_check"] == "passed",
                "a partial (--only) or unchecked (--skip-card-check) bundle is a dry run; it cannot be executed")
    prior = read_progress(receipt_path, manifest_sha, rows) if receipt_path else {}
    if meta["inventory"] is not None:
        live = production_inventory(api)
        require(live == meta["inventory"], "production inventory changed since review: "
                f"new {sorted(set(live) - set(meta['inventory']))}, gone {sorted(set(meta['inventory']) - set(live))}")
    recovered, untagged, plan = {}, [], []
    # Preflight every row before any upload. Each actual commit also uses the parent as compare-and-swap.
    for slug, row in rows:
        repo = f"{PRODUCTION}/{slug}"
        if repo in prior:
            checked_result(api, repo, prior[repo], row["output"])
            plan.append((slug, "done", prior[repo]))
            continue
        head = live_head(api, repo)
        if head != row["target_revision"]:
            recover(api, repo, row, manifest_sha, head)
            recovered[repo] = head
            plan.append((slug, "recovered", head))
            continue
        _, content = snapshot(api, repo, row["target_revision"])
        require(content == row["target"], f"{repo}: target bytes differ from review")
        staged = live_head(api, f"{STAGING}/{slug}")
        require(staged == row["staging_revision"],
                f"{STAGING}/{slug}: staging head moved from the reviewed {row['staging_revision']} to {staged}; review again")
        tag = tags[slug]
        if tag["action"] != "already-on-target-sdk" and tag_commit(api, repo, tag["tag"]) != tag["parent"]:
            untagged.append(repo)
        plan.append((slug, "pending", row["target_revision"]))
    for slug, state, revision in plan:
        row = by_slug[slug]
        shown = ", ".join(row["operations"][:6]) + (", ..." if len(row["operations"]) > 6 else "")
        print(f"{PRODUCTION}/{slug}@{revision}: {state}; {len(row['operations'])} update(s) [{shown}], "
              f"{len(row.get('deletions', []))} removal(s)"
              + (f"; outgoing {meta['outgoing_tag']} tag not yet at the parent" if f"{PRODUCTION}/{slug}" in untagged
                 else ""), flush=True)
    if not execute:
        return plan
    require(not untagged, f"outgoing {meta['outgoing_tag']} tags must resolve to the reviewed parents before any package "
            f"write (run tag_org.py --bundle first): {untagged}")
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    with receipt_path.open("a", encoding="utf-8", buffering=1) as log:
        for repo, commit in recovered.items():
            write_event(log, {"repo": repo, "commit": commit, "manifest_sha256": manifest_sha, "verified": True,
                              "recovered": True})
            print(f"{repo}@{commit}: recovered the missing receipt", flush=True)
        for slug, row in rows:
            repo = f"{PRODUCTION}/{slug}"
            deleted = row.get("deletions", [])
            if repo in prior or repo in recovered or not (row["operations"] or deleted):
                print(f"{repo}: verified, no pending upload", flush=True)
                continue
            commit = api.create_commit(
                repo, repo_type="model", revision="main", parent_commit=row["target_revision"], token=True,
                commit_message=commit_title(row), commit_description=f"{MARKER}{manifest_sha}",
                operations=[CommitOperationAdd(name, bundle / "packages" / slug / name) for name in row["operations"]]
                + [CommitOperationDelete(name, is_folder=False) for name in deleted],
            )
            require(pinned_sha(getattr(commit, "oid", None)),
                    f"{repo}: missing upload commit; rerun the same replay to recover, or reconcile")
            event = {"repo": repo, "commit": commit.oid, "manifest_sha256": manifest_sha}
            write_event(log, {**event, "verified": False})
            checked_result(api, repo, commit.oid, row["output"])
            write_event(log, {**event, "verified": True})
            print(f"{repo}@{commit.oid}: verified", flush=True)
    return plan


def main(argv=None, api=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "dry-run"):
        prep = commands.add_parser(name)
        prep.add_argument("--eligible", "--pins", dest="eligible", required=True, type=Path)
        prep.add_argument("--held", type=Path, help="slug -> {reason, [tag_revision]}; required unless --only")
        prep.add_argument("--sdk", required=True, help="the release SDK version X.Y.Z (the card install pin)")
        prep.add_argument("--outgoing-tag", required=True, help="the outgoing SDK card tag vX.Y.Z")
        prep.add_argument("--bundle", required=True, type=Path)
        prep.add_argument("--only", nargs="+", help="a partial, non-executable bundle of these eligible slugs")
        prep.add_argument("--skip-card-check", action="store_true",
                          help="dry-run only: compose without the card file-set check (the bundle is then not executable)")
        prep.add_argument("--loom-fetch", type=Path, help="run the SDK fetch dry run of every preset of the prepared form")
    run = commands.add_parser("replay")
    run.add_argument("--bundle", required=True, type=Path)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.command != "replay" and not args.only and not args.held:
        parser.error("--held is required for a full release bundle")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(endpoint=ENDPOINT, token=False)
    if args.command == "replay":
        replay(api, args.bundle, args.manifest_sha256, args.execute, args.receipt)
        return 0
    if args.skip_card_check and args.command != "dry-run":
        parser.error("--skip-card-check is for dry-run only")
    meta = release_meta(args.sdk, args.outgoing_tag, "skipped" if args.skip_card_check else "passed")
    dry = args.command == "dry-run"
    sha, failures = prepare(api, args.eligible, args.held, args.bundle, meta, args.only, args.loom_fetch, keep_going=dry)
    if dry:
        document, rows = inspect_bundle(args.bundle, sha)
        print("Outgoing tag plan:", flush=True)
        for _, tag in sorted(document["tags"].items()):
            print(f"  {tag['repo']}: {tag['tag']} {tag['action']} at {tag['parent']} (pins {tag['outgoing_sdk']})", flush=True)
        print("Replay plan (read-only):", flush=True)
        if rows:
            replay(api, args.bundle, sha, False, None)
        for slug, error in failures.items():
            print(f"FAILED {slug}: {error}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
