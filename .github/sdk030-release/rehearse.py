"""Stage-lane, read-only promotion rehearsal for immutable release rows."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import quote
from urllib.request import urlopen

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills/loom-migrate/scripts"))
from replay_promotion import canonical, digest, prepare, replay, require

ENDPOINT = "https://huggingface.co"
TAG = "v0.2.0"


def load(path):
    return json.loads(path.read_bytes())


def save(path, data):
    path.write_bytes(canonical(data))


def contents(repo, revision, path):
    with urlopen(f"{ENDPOINT}/{repo}/resolve/{revision}/{quote(path)}", timeout=90) as response:
        return response.read()


def blob_map(info):
    return {file.rfilename: {"kind": "lfs" if file.lfs else "git",
                             "oid": file.lfs.sha256 if file.lfs else file.blob_id, "size": file.size}
            for file in info.siblings}


def inspect(api, slug, row):
    src, dst = f"loom-fun-staging/{slug}", f"loom-fun/{slug}"
    source, target = row["staging_revision"], row["target_revision"]
    s = api.model_info(src, revision=source, files_metadata=True, token=False)
    d = api.model_info(dst, revision=target, files_metadata=True, token=False)
    require((s.id, s.sha, d.id, d.sha) == (src, source, dst, target), f"{slug}: immutable API resolution differs")
    require(api.model_info(src, revision="main", token=False).sha == source, f"{src}: staging HEAD differs")
    require(api.model_info(dst, revision="main", token=False).sha == target, f"{dst}: production HEAD differs")
    sb, db = blob_map(s), blob_map(d)
    tags = {tag.name: tag.target_commit for tag in api.list_repo_refs(dst, repo_type="model", token=False).tags}
    if "llm_config.json" in db:
        pin_path = "llm_config.json"
        pin_bytes = contents(dst, target, pin_path)
        sdk = (json.loads(pin_bytes).get("sdk") or {}).get("hrx_loom")
    else:
        pin_path = "README.md"
        pin_bytes = contents(dst, target, pin_path)
        match = re.search(r"hrx-loom\[hf\]==([0-9.]+)", pin_bytes.decode())
        sdk = match[1] if match else None
    require(sdk in ("0.2.0", "0.3.0"), f"{dst}: no reviewed outgoing SDK pin (found {sdk!r})")
    found = tags.get(TAG)
    action = "already-at-parent" if found == target else "collision" if found else "create"
    if sdk == "0.3.0":
        action = "already-on-target-sdk"
    tags_plan = {"repo": dst, "parent": target, "tag": TAG, "action": action, "existing": found,
                 "outgoing_sdk": sdk, "pin_path": pin_path, "pin_sha256": hashlib.sha256(pin_bytes).hexdigest()}
    require(action != "collision", f"{dst}: {TAG} collision {found} vs reviewed {target}")
    return {"target_only": sorted(db.keys() - sb.keys() - {".gitattributes"}), "source": sb, "target": db,
            "tag": tags_plan}


def sdk_plan(fetch, root, slug, row, output):
    package = root / "packages" / slug
    commands = []
    for preset in dict.fromkeys(row["presets"]):
        dry = subprocess.run([str(fetch), str(package / "model.json"), "--preset", preset, "--output",
                              str(output / slug / preset), "--json", "--no-input", "--dry-run"],
                             capture_output=True, text=True, timeout=120, check=False)
        require(dry.returncode == 0, f"SDK dry run failed for {slug}:{preset}: {dry.stderr[-1400:]}")
        plan = json.loads(dry.stdout)
        require(plan["preset"] == preset and plan["service"] and plan["files"] and plan["lock"] == "verify"
                and not plan["unverified_parameters"], f"unexpected SDK plan for {slug}:{preset}")
        commands.append({"preset": preset, "service": plan["service"], "programs": plan["programs"], "files": len(plan["files"]),
                         "model_sha256": digest(package / "model.json"), "lock_sha256": digest(package / "model.json.lock")})
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
        require(not result["target_only"] or row.get("retain_target_files"),
                f"{slug}: target-only path(s) require an explicit retention review: {result['target_only']}")
        save(output / "pins.json", {slug: row})
        bundle = output / "bundle"
        prepare(api, output / "pins.json", bundle)
        replay(api, bundle, digest(bundle / "manifest.json"), False, None)
        manifest = load(bundle / "manifest.json")["packages"][slug]
        result["sdk_dry_run"] = sdk_plan(fetch, bundle, slug, manifest, output / "sdk-scratch")
        require(not (output / "sdk-scratch").exists(), f"{slug}: SDK dry run wrote package files")
        result["output"] = manifest["output"]
        result["change_paths"] = manifest["operations"]
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
    scratch.mkdir()
    package_dir = evidence / "release/packages"
    package_dir.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(review_one, ((slug, row, scratch, fetch) for slug, row in sorted(pins.items()))))
    manifests = {}
    for row in rows:
        if row["pass"]:
            slug = row["slug"]
            state = scratch / slug / "bundle"
            manifests.update(load(state / "manifest.json")["packages"])
            shutil.move(str(state / "packages" / slug), package_dir / slug)
    release = evidence / "release/manifest.json"
    save(release, {"schema": 1, "packages": manifests})
    save(evidence / "review.json", rows)
    save(evidence / "tag-plan.json", {row["slug"]: row["tag"] for row in rows if "tag" in row})
    (evidence / "manifest.sha256").write_text(f"{digest(release)}  release/manifest.json\n")
    summary = ["# Read-only production-form promotion rehearsal", "",
               f"{len(manifests)}/{len(pins)} exact rows passed production composition, hash replay and SDK dry run.", "",
               "| Package | Staged | Production | Files changed | Outgoing tag | Result |", "|---|---|---|---:|---|---|"]
    for row in rows:
        result = "Pass" if row["pass"] else row["error"].replace("|", "\\|").replace("\n", " ")
        summary.append(f"| {row['slug']} | {row['pins']['staging_revision']} | {row['pins']['target_revision']} | "
                       f"{len(row.get('change_paths', []))} | {row.get('tag', {}).get('action', 'unverified')} | {result} |")
    (evidence / "review.md").write_text("\n".join(summary) + "\n")
    require(len(manifests) == len(pins), "one or more packages require a targeted repair")


if __name__ == "__main__":
    main()
