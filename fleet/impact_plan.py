#!/usr/bin/env python3
"""Plan sound, parallel CPU impact builds; aggregate them without accepting missing models.

Build selection is based on changed input paths, never the PR author's scope claim. Only
edits confined to existing fleet model inputs can narrow the model set. Shared generators,
KB data, build tooling, manifest changes, and unknown paths conservatively rebuild all.
The historical ref's own validated manifest controls which selected additions are not
built there; every current model is still built at HEAD. Non-input changes still run
scope lint, including its independent serving-setting checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def catalog(repo, ref):
    """Read a ref's own fleet catalog; never infer absence from a failed build."""
    try:
        raw = git(repo, "show", f"{ref}:fleet/manifest.json")
        data = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise SystemExit(f"refusing impact plan: missing or corrupt fleet/manifest.json at {ref}: {e}")
    models = data.get("models") if isinstance(data, dict) else None
    if (not isinstance(models, dict) or
            any(not isinstance(slug, str) or not slug or not isinstance(model, dict) for slug, model in models.items())):
        raise SystemExit(f"refusing impact plan: invalid fleet/manifest.json catalog at {ref}")
    return data


def non_input(path):
    """Explicit non-build surfaces. Unknown inputs fall back to a complete fleet build."""
    if path.endswith(".md"):
        return True
    if path.startswith((".github/", "tests/", "docs/", ".claude-plugin/")):
        return True
    # The impact planner is never imported by build_fleet.py, gguf_stub.py, or the kit.
    # Keep this an exact exception: unknown fleet tools must still rebuild everything.
    if path in ("LICENSE", "THIRD_PARTY_NOTICES.md", ".gitignore", "install.sh", "scripts/install_skills.py", "fleet/impact_plan.py"):
        return True
    # These sibling workflows/tools are not loaded by the programs-only builder.
    return path.startswith(tuple(f"skills/{name}/" for name in (
        "loom", "loom-setup", "loom-benchmark", "loom-fleet", "loom-migrate",
        "loom-optimize", "loom-roofline", "loom-demo-ui")))


def select(paths, manifest):
    """(models, reasons): exact ownership only, with a global fallback on ambiguity."""
    models = manifest["models"]
    owners = {}
    for slug, model in models.items():
        for entry in model.get("files", {}).values():
            if entry.get("dump"):
                owners.setdefault("fleet/" + entry["dump"], set()).add(slug)
        for name in ("llm_config.json", "tokenizer_min.json"):
            owners.setdefault(f"fleet/{slug}/{name}", set()).add(slug)
    selected, reasons = set(), []
    for path in sorted(set(paths)):
        if path in owners:
            selected.update(owners[path])
            reasons.append(f"model input: {path}")
        elif non_input(path):
            continue
        else:
            return sorted(models), [f"shared or unknown build input: {path}"]
    return sorted(selected), reasons or ["no program build input changed"]


def make_plan(repo, base, head="HEAD", shards=4):
    base = git(repo, "merge-base", base, head)
    head = git(repo, "rev-parse", head)
    # --no-renames reports both the old and new path, so moving an input cannot evade selection.
    paths = git(repo, "diff", "--name-only", "--no-renames", base, head).splitlines()
    manifest = catalog(repo, head)
    base_manifest = catalog(repo, base)
    selected, reasons = select(paths, manifest)
    # Manifest edits must account for deletions/renames as explicit catalog changes.
    removed = sorted(set(base_manifest["models"]) - set(manifest["models"]))
    if "fleet/manifest.json" in paths:
        selected = sorted(set(selected) | set(manifest["models"]))
        reasons = ["fleet catalog changed; checking every HEAD model"]
    base_selected = sorted(set(selected) & set(base_manifest["models"]))
    head_selected = sorted(set(selected) & set(manifest["models"]))
    # The builder uses the HEAD checkout's fleet inputs/tooling for BOTH revisions.
    # Include that tree plus each revision's complete kit/KB trees in the cache key.
    inputs = git(repo, "rev-parse", f"{head}:fleet")
    include = []
    for side, ref, side_slugs in (("base", base, base_selected), ("head", head, head_selected)):
        trees = [git(repo, "rev-parse", f"{ref}:skills/{skill}") for skill in ("loom-port-llm", "loom-kb")]
        side_shards = max(1, min(shards, len(side_slugs)))
        buckets = [side_slugs[n::side_shards] for n in range(side_shards)] if side_slugs else []
        for n, slugs in enumerate(buckets):
            digest = hashlib.sha256(json.dumps([1, ref, inputs, trees, slugs], sort_keys=True).encode()).hexdigest()
            include.append({"side": side, "ref": ref, "shard": n, "slugs": ",".join(slugs), "key": digest})
    return {"base": base, "head": head, "changed": paths, "selected": selected,
            "base_selected": base_selected, "head_selected": head_selected,
            "base_absent_catalog": {s: "new model: absent from base fleet/manifest.json" for s in selected if s not in base_manifest["models"]},
            "removed_from_head": removed,
            "unchanged_inputs": sorted(set(manifest["models"]) - set(selected)), "reasons": reasons,
            "matrix": {"include": include}, "builds": len(include)}


def validate_pr_scope(repo, body_path):
    """Fail before planning builds when a PR scope declaration is malformed or unknown."""
    # scope_lint owns the grammar, selector matcher, facet list, and kernel-family registry.
    sys.path.insert(0, str(HERE))
    try:
        import scope_lint
    finally:
        sys.path.remove(str(HERE))
    errors = scope_lint.declaration_errors(repo, Path(body_path).read_text(encoding="utf-8"))
    if errors:
        raise SystemExit("invalid loom-scope declaration:\n" + "\n".join(f"- {error}" for error in errors))


def aggregate(plan, builds, output):
    # Import only for aggregation; the planner has no numpy/GGUF or SDK dependencies.
    sys.path.insert(0, str(HERE))
    import diff_fleet
    selected = plan["selected"]
    base_selected = plan.get("base_selected", selected)
    head_selected = plan.get("head_selected", selected)
    absent = plan.get("base_absent_catalog", {})
    if (not isinstance(selected, list) or len(set(selected)) != len(selected)
            or not isinstance(base_selected, list) or not isinstance(head_selected, list)
            or set(head_selected) != set(selected) or not set(base_selected) <= set(selected)
            or set(absent) != set(selected) - set(base_selected)):
        raise SystemExit("refusing invalid impact plan model accounting")
    base, head = Path(builds) / "base", Path(builds) / "head"
    base.mkdir(parents=True, exist_ok=True)
    head.mkdir(parents=True, exist_ok=True)
    if selected:
        # Keep per-shard failure/skip evidence. Only the validated catalog comparison
        # above may mark a BASE model absent; build-reported skips remain fatal.
        for side in ("base", "head"):
            merged = {"models": {}, "failures": {}, "skipped": dict(absent) if side == "base" else {},
                      "catalog_absent": dict(absent) if side == "base" else {}}
            expected = set(plan.get(f"{side}_selected", selected))
            for job in plan["matrix"]["include"]:
                if job["side"] != side:
                    continue
                summary = Path(builds) / side / f"summary-{job['shard']}.json"
                if not summary.is_file():
                    raise SystemExit(f"refusing incomplete impact shards: missing {summary}")
                data = json.loads(summary.read_text())
                if (data.get("ref") != job["ref"] or data.get("rdna4") != "1"
                        or set(data.get("models", {})) != (set(job["slugs"].split(",")) if job["slugs"] else set())):
                    raise SystemExit(f"refusing incomplete impact shards: wrong revision/model set in {summary}")
                if data.get("failures") or data.get("skipped"):
                    raise SystemExit(f"refusing failed impact shard: {summary}")
                for key in merged:
                    merged[key].update(data.get(key, {}))
            if set(merged["models"]) != expected:
                raise SystemExit(f"refusing incomplete impact shards: missing models on {side}")
            (Path(builds) / side).mkdir(parents=True, exist_ok=True)
            (Path(builds) / side / "summary.json").write_text(json.dumps(merged))
        problems = diff_fleet.check_complete(str(base), list(base_selected) + sorted(absent))
        problems += diff_fleet.check_complete(str(head), head_selected)
        if problems:
            raise SystemExit("refusing incomplete impact shards:\n" + "\n".join(problems))
        report = diff_fleet.report(str(base), str(head), set(selected), set(plan.get("removed_from_head", [])))
    elif plan.get("removed_from_head"):
        report = diff_fleet.report(str(base), str(head), set(), set(plan["removed_from_head"]))
    else:
        report = {"base": plan["base"], "head": plan["head"], "models": {}, "untouched": [], "by_family": {},
                  "summary": {"models": 0, "touched": 0, "untouched": 0, "gpus": [], "rows": [], "strata": []}}
    report["selection"] = {k: plan[k] for k in ("selected", "unchanged_inputs", "reasons")}
    removed = len(plan.get("removed_from_head", []))
    total = len(selected) + len(plan["unchanged_inputs"]) + removed
    base_count = len(plan.get("base_selected", selected))
    head_count = len(plan.get("head_selected", selected))
    additions = len(plan.get("base_absent_catalog", {}))
    coverage = (f"\nBuild coverage: {base_count}/{total} models built at BASE; "
                f"{head_count}/{total} built at HEAD; {additions} selected models are new and absent from the BASE catalog; "
                f"{len(plan['unchanged_inputs'])} have unchanged build inputs; {removed} removed models are identified from the HEAD catalog.\n" +
                "\n".join(f"- {reason}" for reason in plan["reasons"]) + "\n")
    markdown = diff_fleet.markdown(report) if selected or plan.get("removed_from_head") else "## Fleet program diff: no build inputs changed\n"
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "impact.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "impact.md").write_text(markdown + coverage)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--repo", default=str(HERE.parent))
    plan.add_argument("--base", required=True)
    plan.add_argument("--head", default="HEAD")
    plan.add_argument("--shards", type=int, default=4)
    plan.add_argument("--pr-body", help="optional PR body to validate before creating a build matrix")
    plan.add_argument("--out", required=True)
    plan.add_argument("--github-output")
    finish = sub.add_parser("aggregate")
    finish.add_argument("--plan", required=True)
    finish.add_argument("--builds", required=True)
    finish.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.pr_body:
            validate_pr_scope(args.repo, args.pr_body)
        result = make_plan(args.repo, args.base, args.head, args.shards)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        if args.github_output:
            with open(args.github_output, "a") as out:
                for key in ("base", "builds", "matrix"):
                    value = json.dumps(result[key], separators=(",", ":")) if key == "matrix" else result[key]
                    out.write(f"{key}={value}\n")
        print(f"Build {len(result['selected'])} models in {result['builds']} revision/shard jobs: {'; '.join(result['reasons'])}")
    else:
        aggregate(json.loads(Path(args.plan).read_text()), args.builds, args.out)


if __name__ == "__main__":
    main()
