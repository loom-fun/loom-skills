"""Stage-lane, read-only promotion rehearsal for immutable release rows."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent
if not (ROOT / "replay_promotion.py").exists():
    sys.path.insert(0, str(ROOT.parents[1] / "skills/loom-migrate/scripts"))
from replay_promotion import canonical, digest, inspect_bundle, model_info, prepare, read_hub, replay, require

ENDPOINT = "https://huggingface.co"
TAG = "v0.2.0"
ATTR = ".gitattributes"


def load(path):
    return json.loads(path.read_bytes())


def save(path, data):
    path.write_bytes(canonical(data))


def fetch_content(repo, revision, path, token):
    try:
        with urlopen(f"{ENDPOINT}/{repo}/resolve/{revision}/{quote(path)}", timeout=90) as response:
            return response.read()
    except HTTPError as error:
        # Give the Hub retry helper the same status/header shape for a CDN response.
        error.status_code = error.code
        error.response = error
        raise


def blob_map(info):
    return {file.rfilename: {"kind": "lfs" if file.lfs else "git",
                             "oid": file.lfs.sha256 if file.lfs else file.blob_id, "size": file.size}
            for file in info.siblings}


def inspect(api, slug, row):
    src, dst = f"loom-fun-staging/{slug}", f"loom-fun/{slug}"
    source, target = row["staging_revision"], row["target_revision"]
    s = model_info(api, src, revision=source, files_metadata=True)
    d = model_info(api, dst, revision=target, files_metadata=True)
    require((s.id, s.sha, d.id, d.sha) == (src, source, dst, target), f"{slug}: immutable API resolution differs")
    sb, db = blob_map(s), blob_map(d)
    tags = {tag.name: tag.target_commit for tag in read_hub(api.list_repo_refs, dst, repo_type="model").tags}
    if "llm_config.json" in db:
        pin_path = "llm_config.json"
        pin_bytes = read_hub(fetch_content, dst, target, pin_path)
        sdk = (json.loads(pin_bytes).get("sdk") or {}).get("hrx_loom")
    else:
        pin_path = "README.md"
        pin_bytes = read_hub(fetch_content, dst, target, pin_path)
        match = re.search(r"hrx-loom\[hf\]==([0-9.]+)", pin_bytes.decode())
        sdk = match[1] if match else None
    require(sdk in ("0.2.0", "0.3.0"), f"{dst}: no reviewed outgoing SDK pin (found {sdk!r})")
    found = tags.get(TAG)
    action = "already-at-parent" if found == target else "collision" if found else "create"
    if sdk == "0.3.0":
        action = "already-on-target-sdk"
    tags_plan = {"repo": dst, "parent": target, "tag": TAG, "action": action, "existing": found,
                 "outgoing_sdk": sdk, "pin_path": pin_path, "pin_sha256": hashlib.sha256(pin_bytes).hexdigest()}
    return {"target_only": sorted(db.keys() - sb.keys() - {".gitattributes"}), "source": sb, "target": db,
            "tag": tags_plan}


def check_live(pins):
    api = HfApi(endpoint=ENDPOINT, token=False)
    for org, key in (("loom-fun-staging", "staging_revision"), ("loom-fun", "target_revision")):
        heads = {item.id: item.sha for item in api.list_models(author=org, full=True, token=False, limit=1000)}
        for slug, row in pins.items():
            require(heads.get(f"{org}/{slug}") == row[key], f"{org}/{slug}: live head differs from exact release pin")


def record_attributes(result, manifest):
    source = manifest["source"].get(ATTR)
    target = manifest["target"].get(ATTR)
    require(manifest["output"].get(ATTR) == target and ATTR not in manifest["operations"],
            f"{result['slug']}: release changes production Git attributes")
    result["attributes"] = {"source_sha256": source, "target_sha256": target, "output_sha256": target}


def restore_attributes(review, manifest, output, cache):
    slug = review["slug"]
    for kind, org, key in (("source", "loom-fun-staging", "staging_revision"), ("target", "loom-fun", "target_revision")):
        attributes = review[kind].get(ATTR)
        if not attributes or ATTR in manifest[kind]:
            continue
        require(attributes["kind"] == "git", f"{slug}: expected a Git blob for {ATTR}")
        oid = attributes["oid"]
        if oid not in cache:
            data = read_hub(fetch_content, f"{org}/{slug}", review["pins"][key], ATTR)
            git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            require(git_blob == oid and len(data) == attributes["size"], f"{slug}: Git attribute identity differs")
            cache[oid] = data
        data = cache[oid]
        value = hashlib.sha256(data).hexdigest()
        manifest[kind][ATTR] = value
        if kind == "target":
            (output / ATTR).write_bytes(data)
            manifest["output"][ATTR] = value
    review["output"] = manifest["output"]
    record_attributes(review, manifest)


def sdk_plan(fetch, root, slug, row, output):
    package = root / "packages" / slug
    commands = []
    deleted = row.get("delete_target_files", {})
    model = load(package / "model.json")
    advertised = set(row["presets"])
    for preset in model["presets"]:
        dry = subprocess.run([str(fetch), str(package / "model.json"), "--preset", preset, "--output",
                              str(output / slug / preset), "--json", "--no-input", "--dry-run"],
                             capture_output=True, text=True, timeout=120, check=False)
        require(dry.returncode == 0, f"SDK dry run failed for {slug}:{preset}: {dry.stderr[-1400:]}")
        plan = json.loads(dry.stdout)
        require(plan["preset"] == preset and plan["service"] and plan["files"] and plan["lock"] == "verify"
                and not plan["unverified_parameters"], f"unexpected SDK plan for {slug}:{preset}")
        closure = json.dumps(plan["files"], sort_keys=True)
        require(not any(path.removeprefix("program/") in closure for path in deleted),
                f"{slug}:{preset} still references an explicitly deleted path")
        commands.append({"preset": preset, "advertised": preset in advertised,
                         "service": plan["service"], "programs": plan["programs"], "files": len(plan["files"]),
                         "model_sha256": digest(package / "model.json"), "lock_sha256": digest(package / "model.json.lock"),
                         "deleted_paths_absent": sorted(deleted)})
    require(commands, f"{slug}: no listed presets")
    return commands


def review_one(task):
    slug, row, scratch, fetch = task
    api = HfApi(endpoint=ENDPOINT, token=False)
    output = scratch / slug
    output.mkdir()
    result = {"slug": slug, "pins": row}
    try:
        result.update(inspect(api, slug, row))
        require(result["tag"]["action"] != "collision", f"{slug}: {TAG} collision {result['tag']['existing']}")
        save(output / "pins.json", {slug: row})
        bundle = output / "bundle"
        prepare(api, output / "pins.json", bundle)
        replay(api, bundle, digest(bundle / "manifest.json"), False, None)
        manifest = load(bundle / "manifest.json")["packages"][slug]
        result["sdk_dry_run"] = sdk_plan(fetch, bundle, slug, manifest, output / "sdk-scratch")
        require(not (output / "sdk-scratch").exists(), f"{slug}: SDK dry run wrote package files")
        result["output"] = manifest["output"]
        record_attributes(result, manifest)
        result["change_paths"] = manifest["operations"]
        result["deleted_paths"] = sorted(result["target"].keys() - manifest["output"].keys() - {".gitattributes"})
        result["pass"] = True
    except (Exception, SystemExit) as exc:
        result["pass"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    print(f"{'PASS' if result['pass'] else 'FAIL'} {slug}: {result.get('error', 'composition and replay verified')}", flush=True)
    return result


def main():
    inputs, scratch, evidence, fetch = map(Path, sys.argv[1:5])
    pins = load(inputs / "pins.json")
    eligible, held = load(inputs / "eligible.json"), load(inputs / "held.json")
    require(pins.keys() == eligible.keys() and not pins.keys() & held.keys(), "release scope has overlapping or missing rows")
    check_live(pins)
    scratch.mkdir()
    package_dir = evidence / "release/packages"
    package_dir.mkdir(parents=True)
    reusable = {}
    prior_manifests = {}
    if len(sys.argv) == 6:
        prior = Path(sys.argv[5])
        prior_bundle = prior / "release"
        prior_manifests = dict(inspect_bundle(prior_bundle, (prior / "manifest.sha256").read_text().split()[0]))
        prior_run = load(inputs / "reuse.json")["run"]
        attribute_cache = {}
        for row in load(prior / "review.json"):
            slug = row["slug"]
            if not row["pass"] or row["pins"] != pins.get(slug):
                continue
            require(row["output"] == prior_manifests[slug]["output"], f"{slug}: reused row hashes differ")
            reusable[slug] = {**row, "reused_from": row.get("reused_from", prior_run),
                              "deleted_paths": row.get("deleted_paths", [])}
            shutil.copytree(prior_bundle / "packages" / slug, package_dir / slug)
            restore_attributes(reusable[slug], prior_manifests[slug], package_dir / slug, attribute_cache)
            print(f"REUSE {slug}: exact source and production heads and artifacts rechecked", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(review_one, ((slug, row, scratch, fetch) for slug, row in sorted(pins.items())
                                         if slug not in reusable)))
    manifests = {slug: prior_manifests[slug] for slug in reusable}
    for row in rows:
        if row["pass"]:
            slug = row["slug"]
            state = scratch / slug / "bundle"
            manifests.update(load(state / "manifest.json")["packages"])
            shutil.move(str(state / "packages" / slug), package_dir / slug)
    rows = sorted(rows + list(reusable.values()), key=lambda row: row["slug"])
    release = evidence / "release/manifest.json"
    save(release, {"schema": 1, "packages": manifests})
    save(evidence / "review.json", rows)
    save(evidence / "tag-plan.json", {row["slug"]: row["tag"] for row in rows if "tag" in row})
    if manifests:
        inspect_bundle(evidence / "release", digest(release))
    (evidence / "manifest.sha256").write_text(f"{digest(release)}  release/manifest.json\n")
    summary = ["# Read-only production-form promotion rehearsal", "",
               f"{len(manifests)}/{len(pins)} exact rows passed production composition, hash replay and SDK dry run.", "",
               "SDK dry runs resolved each reviewed production-form manifest from the local archive. The live production Hub revisions remained read-only.", "",
               "| Package | Staged | Production | Update paths | Deletes | Outgoing tag | Result |",
               "|---|---|---|---:|---:|---|---|"]
    for row in rows:
        result = "Pass" if row["pass"] else row["error"].replace("|", "\\|").replace("\n", " ")
        slug = row["slug"]
        stage = row["pins"]["staging_revision"]
        prod = row["pins"]["target_revision"]
        summary.append(f"| {slug} | [staged](https://huggingface.co/loom-fun-staging/{slug}/tree/{stage}) | "
                       f"[parent](https://huggingface.co/loom-fun/{slug}/tree/{prod}) | "
                       f"{len(row.get('change_paths', []))} | {len(row.get('deleted_paths', []))} | "
                       f"{row.get('tag', {}).get('action', 'unverified')} | {result} |")
    for slug, row in eligible.items():
        if row.get("qualification_note"):
            link = f"[Qualification]({row['qualification_url']})" if row.get("qualification_url") else (
                f"[Staging receipt]({row['staging_receipt_url']})")
            summary.extend(("", f"**{slug}** — {row['qualification_note']} {link}"))
    (evidence / "review.md").write_text("\n".join(summary) + "\n")
    require(len(manifests) == len(pins), "one or more packages require a targeted repair")


if __name__ == "__main__":
    main()
