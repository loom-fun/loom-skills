import hashlib
import json
from pathlib import Path
import subprocess
import sys

package, source, evidence, sdk = map(Path, sys.argv[1:])
card = json.loads((package / "model.json").read_text())
lock = json.loads((package / "model.json.lock").read_text())
local = {"model.json", "model.json.lock"}
remote = {}
lock_files = {entry["origin"]: entry for entry in lock["files"]}
revision_pinned_files = set()


def asset(value):
    entry = {"path": value} if isinstance(value, str) else value
    path = entry["path"]
    origin = entry.get("origin")
    if origin is None:
        assert (package / path).is_file(), path
        local.add(path)
    else:
        name = origin.split("/", 1)[0].removeprefix("@")
        assert name in card["origins"] and name in lock["origins"], origin
        revision = lock["origins"][name]["resolved"]["revision"]
        assert len(revision) == 40, (origin, revision)
        if origin in lock_files:
            assert len(lock_files[origin]["sha256"]) == 64 and lock_files[origin]["size"] > 0, origin
        else:
            assert Path(origin).name in {"LICENSE.md", "LICENSE.pdf"}, origin
            revision_pinned_files.add(origin)
        remote[path] = origin
    for child in entry.get("data", []):
        asset(child)


def variant_assets(value):
    if isinstance(value, dict):
        if "path" in value:
            asset(value)
        else:
            for child in value.values():
                variant_assets(child)
    elif isinstance(value, list):
        for child in value:
            variant_assets(child)


assert list(card["programs"]) == ["source"]
for key in ("inputs", "libraries"):
    for entry in card["programs"]["source"].get(key, []):
        asset(entry)
for variant in card["variants"].values():
    for service in variant["services"].values():
        assert service["programs"] == ["source"]
        variant_assets(service)
assert all(preset["programs"] == "source" for preset in card["presets"].values())
assert {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()} == local
assert (package / "model.json.lock").read_bytes() == (source / "model.json.lock").read_bytes()
plans = {}
for preset in card["presets"]:
    plan = subprocess.run([sdk / "loom-fetch", package / "model.json", f"--preset={preset}", "--dry-run", "--json",
                           "--no-input", f"--output={package.parent / f'plan-{package.name}-{preset}'}"],
                          capture_output=True, text=True)
    (evidence / f"plan-{preset}.log").write_text(plan.stderr)
    (evidence / f"plan-{preset}.json").write_text(plan.stdout)
    plan.check_returncode()
    files = json.loads(plan.stdout)["files"]
    planned = {item["path"] for item in files}
    declared = local | set(remote)
    assert planned <= declared, planned - declared
    assert planned & set(remote), preset
    plans[preset] = len(planned)
programs = sorted(path for path in local if path.endswith(".loom"))
parse_errors = {}
for path in programs:
    result = subprocess.run([sdk / "loom-opt", path, "--output=/dev/null"], cwd=package, capture_output=True, text=True)
    if result.returncode:
        parse_errors[path] = result.stderr
record = {"model": card["model"]["id"], "source_program_files": len(programs), "local_assets": len(local),
          "remote_destinations": len(remote), "revision_pinned_license_files": sorted(revision_pinned_files),
          "preset_dry_run_files": plans, "parse_errors": parse_errors,
          "staged_card_sha256": hashlib.sha256((package / "model.json").read_bytes()).hexdigest()}
(evidence / "result.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
assert not parse_errors, parse_errors
