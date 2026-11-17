#!/usr/bin/env python3
"""Plan sound, parallel CPU impact builds; aggregate them without accepting missing models.

Build selection is based on changed input paths, never the PR author's scope claim. Only
edits confined to existing fleet model inputs can narrow the model set. Shared generators,
KB data, build tooling, manifest changes, and unknown paths conservatively rebuild all.
Non-input changes still run scope lint, including its independent serving-setting checks.
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
    manifest = json.loads(git(repo, "show", f"{head}:fleet/manifest.json"))
    selected, reasons = select(paths, manifest)
    shards = max(1, min(shards, len(selected)))
    buckets = [selected[n::shards] for n in range(shards)] if selected else []
    # The builder uses the HEAD checkout's fleet inputs/tooling for BOTH revisions.
    # Include that tree plus each revision's complete kit/KB trees in the cache key.
    inputs = git(repo, "rev-parse", f"{head}:fleet")
    include = []
    for side, ref in (("base", base), ("head", head)):
        trees = [git(repo, "rev-parse", f"{ref}:skills/{skill}") for skill in ("loom-port-llm", "loom-kb")]
        for n, slugs in enumerate(buckets):
            digest = hashlib.sha256(json.dumps([1, ref, inputs, trees, slugs], sort_keys=True).encode()).hexdigest()
            include.append({"side": side, "ref": ref, "shard": n, "slugs": ",".join(slugs), "key": digest})
    return {"base": base, "head": head, "changed": paths, "selected": selected,
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
    base, head = Path(builds) / "base", Path(builds) / "head"
    if selected:
        # Keep per-shard failure/skip evidence. In particular, a newly introduced source
        # format may be absent from the base kit and must remain a reported new model.
        for side in ("base", "head"):
            merged = {"models": {}, "failures": {}, "skipped": {}}
            for job in plan["matrix"]["include"]:
                if job["side"] != side:
                    continue
                summary = Path(builds) / side / f"summary-{job['shard']}.json"
                if not summary.is_file():
                    raise SystemExit(f"refusing incomplete impact shards: missing {summary}")
                data = json.loads(summary.read_text())
                if (data.get("ref") != job["ref"] or data.get("rdna4") != "1"
                        or set(data.get("models", {})) != set(job["slugs"].split(","))):
                    raise SystemExit(f"refusing incomplete impact shards: wrong revision/model set in {summary}")
                if data.get("failures") or (side == "head" and data.get("skipped")):
                    raise SystemExit(f"refusing failed impact shard: {summary}")
                for key in merged:
                    merged[key].update(data.get(key, {}))
            if set(merged["models"]) != set(selected):
                raise SystemExit(f"refusing incomplete impact shards: missing models on {side}")
            (Path(builds) / side / "summary.json").write_text(json.dumps(merged))
        problems = [problem for d in (base, head) for problem in diff_fleet.check_complete(str(d), selected)]
        if problems:
            raise SystemExit("refusing incomplete impact shards:\n" + "\n".join(problems))
        report = diff_fleet.report(str(base), str(head), set(selected))
    else:
        report = {"base": plan["base"], "head": plan["head"], "models": {}, "untouched": [], "by_family": {},
                  "summary": {"models": 0, "touched": 0, "untouched": 0, "gpus": [], "rows": [], "strata": []}}
    report["selection"] = {k: plan[k] for k in ("selected", "unchanged_inputs", "reasons")}
    total = len(selected) + len(plan["unchanged_inputs"])
    coverage = (f"\nBuild coverage: {len(selected)}/{total} models built at both revisions; "
                f"{len(plan['unchanged_inputs'])} have unchanged build inputs.\n" +
                "\n".join(f"- {reason}" for reason in plan["reasons"]) + "\n")
    markdown = diff_fleet.markdown(report) if selected else "## Fleet program diff: no build inputs changed\n"
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
    plan.add_argument("--pr-body", required=True, help="PR body to validate before creating a build matrix")
    plan.add_argument("--out", required=True)
    plan.add_argument("--github-output")
    finish = sub.add_parser("aggregate")
    finish.add_argument("--plan", required=True)
    finish.add_argument("--builds", required=True)
    finish.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
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
