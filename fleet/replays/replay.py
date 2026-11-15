#!/usr/bin/env python3
"""Replay a past PR's intent through the guards: fleet impact, scope lint and sampler of each way of expressing it.

usage: replay.py NAME [--variants A,B] [--jobs N] [--work DIR] [--keep]      (NAME = a directory under fleet/replays/)

A replay directory holds
    meta.json        {"pr": "#357", "title": "...", "intent": "...", "variants": ["A", "B"], "base_from": "NAME:V" (chain), "commit": SHA (a
                     merged PR replayed file by file, see commit_replay), "base_args" / "head_args": extra build_fleet.py arguments}
    base.patch       (optional) the pre-PR state: a patch on origin/refactor that turns the feature off (kit default + registry legacy)
    <V>.patch        variant V on top of the base: A = as originally done (a changed default / the original edits),
                     B = as the architecture wants it (an opt-in scope rule at the measured scope, or a declared horizontal change)
    <V>.scope        the loom-scope block the author would declare for variant V
    <V>2.scope ...   more declared scopes for the same patch (patch: <V>.patch)
Output, next to the inputs: <V>.impact.md, <V>.impact.json (slimmed), <V>.lint.txt, <V>.plan.txt, <V>.summary.json
(touched models, leak list, lint verdict, files edited, sampler representatives).
Trees are built with fleet/build_fleet.py (CPU only) and cached by git tree id, so chained replays share builds.
The guards under test (diff_fleet.py, scope_lint.py) are this checkout's, not the replayed tree's.
Needs python with gguf (numpy). Stdlib otherwise.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET = os.path.dirname(HERE)
REPO = os.path.dirname(FLEET)
PY = sys.executable
BASE_REF = os.environ.get("REPLAY_BASE_REF", "origin/refactor")


def run(cmd, cwd=None, check=True, **kw):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)
    if check and p.returncode:
        sys.exit(f"FAILED {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p


def make_commit(work, label, patches, start=None):
    """origin/refactor + patches (in order) as a local scratch commit; returns (sha, tree). Scratch branches stay local."""
    wt = os.path.join(work, "wt-" + label)
    if os.path.isdir(wt):
        run(["git", "worktree", "remove", "--force", wt], cwd=REPO, check=False)
    run(["git", "worktree", "add", "--detach", wt, start or BASE_REF], cwd=REPO)
    for p in patches:
        run(["git", "apply", "--whitespace=nowarn", os.path.abspath(p)], cwd=wt)
    run(["git", "add", "-A"], cwd=wt)
    run(["git", "-c", "user.name=replay", "-c", "user.email=replay@example.invalid", "commit", "-q", "--allow-empty", "-m", "replay " + label], cwd=wt)
    sha = run(["git", "rev-parse", "HEAD"], cwd=wt).stdout.strip()
    tree = run(["git", "rev-parse", "HEAD^{tree}"], cwd=wt).stdout.strip()
    return wt, sha, tree


def build(tree_dir, tree_id, work, jobs, args=()):
    out = os.path.join(work, "builds", tree_id[:12] + ("".join("-" + x for x in args) if args else ""))
    sj = os.path.join(out, "summary.json")
    if os.path.isfile(sj) and not json.load(open(sj)).get("failures"):
        return out
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    r = run([PY, os.path.join(tree_dir, "fleet", "build_fleet.py"), "--tree", tree_dir, "--out", out, "--jobs", str(jobs), *args], check=False)
    if r.returncode:
        sys.exit("build failed:\n" + (r.stdout + r.stderr)[-3000:])
    fails = json.load(open(os.path.join(out, "summary.json"))).get("failures")
    if fails:
        sys.exit(f"build of {tree_id[:12]} failed for: {sorted(fails)}")
    return out


def commit_replay(name, meta, d, a):
    """A merged PR (meta "commit"): base = REF with the PR's files as they were before the commit, head = the same files after it (the
    PR's own diff, standalone, on top of REF). A = the PR as one change under A.scope; B = the same change split by what each file is
    (protected -> architecture, card tooling -> card, docs -> docs, scripts / CI -> tooling), each part linted under its own kind."""
    c = meta["commit"]
    sys.path.insert(0, FLEET)
    import scope_lint as SL
    own = json.load(open(os.path.join(FLEET, "ownership.json")))
    files = run(["git", "diff", "--name-only", c + "^", c], cwd=REPO).stdout.split()

    def state(wt, ref, label):
        """the files of the PR as they were at REF (deleted where REF has none), over a checkout of origin/refactor"""
        for f in files:
            if run(["git", "cat-file", "-e", f"{ref}:{f}"], cwd=REPO, check=False).returncode == 0:
                run(["git", "checkout", ref, "--", f], cwd=wt)
            else:
                run(["git", "rm", "-q", "-f", "--ignore-unmatch", f], cwd=wt)
        run(["git", "add", "-A"], cwd=wt)
        run(["git", "-c", "user.name=replay", "-c", "user.email=replay@example.invalid", "commit", "-q", "--allow-empty", "-m", label], cwd=wt)
        return run(["git", "rev-parse", "HEAD"], cwd=wt).stdout.strip()

    wt = os.path.join(a.work, "wt-" + name + "-base")
    if os.path.isdir(wt):
        run(["git", "worktree", "remove", "--force", wt], cwd=REPO, check=False)
    run(["git", "worktree", "add", "--detach", wt, BASE_REF], cwd=REPO)
    bsha = state(wt, c + "^", "base " + name)
    hwt = os.path.join(a.work, "wt-" + name + "-head")
    if os.path.isdir(hwt):
        run(["git", "worktree", "remove", "--force", hwt], cwd=REPO, check=False)
    run(["git", "worktree", "add", "--detach", hwt, bsha], cwd=REPO)
    head = state(hwt, c, "head " + name)
    dropped = []
    changed = run(["git", "diff", "--name-only", bsha, head], cwd=REPO).stdout.split()
    bt, ht = (run(["git", "rev-parse", x + "^{tree}"], cwd=REPO).stdout.strip() for x in (bsha, head))
    bout, hout = build(wt, bt, a.work, a.jobs), build(hwt, ht, a.work, a.jobs)
    os.makedirs(os.path.join(a.work, "json"), exist_ok=True)
    imp = os.path.join(a.work, "json", name + "-A.impact")
    run([PY, os.path.join(REPO, "fleet", "diff_fleet.py"), bout, hout, "--json", imp + ".json", "--md", os.path.join(d, "A.impact.md")])
    impact = json.load(open(imp + ".json"))
    touched = sorted((impact.get("models") or {}))
    res = {"commit": c, "files_changed": changed, "dropped": dropped, "impact_models": touched}

    def lint(label, block, repo_head):
        pb = os.path.join(a.work, f"{name}-{label}.body.md")
        open(pb, "w").write(f"```loom-scope\n{block}\n```\n")
        r = run([PY, os.path.join(REPO, "fleet", "scope_lint.py"), "--repo", repo_head[0], "--base", bsha, "--head", repo_head[1], "--pr-body", pb,
                 "--impact", imp + ".json", "--no-kb-lint"], cwd=repo_head[0], check=False)   # old copies of kb files would fail the structure lint
        return r.returncode, r.stdout + r.stderr

    sf = os.path.join(d, "A.scope")
    code, txt = lint("A", scope_block(sf), (hwt, head))
    open(os.path.join(d, "A.lint.txt"), "w").write(f"exit {code}\n{txt}")
    res["A"] = {"lint_exit": code, "files": len(changed)}
    # B: split by file category
    groups = {}
    protected = [p["glob"] for p in own["protected"]]
    for f in changed:
        if SL.globs_match(protected, f):
            k = "architecture"
        elif SL.globs_match(own["kinds"]["card"]["allow"], f):
            k = "card"
        elif SL.globs_match(own["kinds"]["docs"]["allow"], f):
            k = "docs"
        elif SL.globs_match(own["kinds"]["tooling"]["allow"], f) and not SL.globs_match(own["kinds"]["tooling"].get("deny", []), f):
            k = "tooling"
        else:
            k = "unclassified"
        groups.setdefault(k, []).append(f)
    out, parts = [], {}
    for k, fs in sorted(groups.items()):
        pw = os.path.join(a.work, f"wt-{name}-part-{k}")
        if os.path.isdir(pw):
            run(["git", "worktree", "remove", "--force", pw], cwd=REPO, check=False)
        run(["git", "worktree", "add", "--detach", pw, bsha], cwd=REPO)
        for f in fs:
            if run(["git", "cat-file", "-e", f"{head}:{f}"], cwd=REPO, check=False).returncode == 0:
                run(["git", "checkout", head, "--", f], cwd=pw)
            else:
                run(["git", "rm", "-q", "-f", "--ignore-unmatch", f], cwd=pw)
        run(["git", "add", "-A"], cwd=pw)
        run(["git", "-c", "user.name=replay", "-c", "user.email=replay@example.invalid", "commit", "-q", "--allow-empty", "-m", "part " + k], cwd=pw)
        psha = run(["git", "rev-parse", "HEAD"], cwd=pw).stdout.strip()
        kind = "tooling" if k == "unclassified" else k
        code, txt = lint("B-" + k, f"kind: {kind}", (pw, psha))
        parts[k] = {"files": len(fs), "lint_exit": code}
        out.append(f"=== part kind {kind}: {len(fs)} file(s), exit {code}\n" + "\n".join("  " + f for f in fs) + "\n" + txt)
        run(["git", "worktree", "remove", "--force", pw], cwd=REPO, check=False)
    open(os.path.join(d, "B.lint.txt"), "w").write("\n".join(out))
    res["B"] = {"parts": parts, "lint_exit": max((p["lint_exit"] for p in parts.values()), default=0)}
    json.dump(res, open(os.path.join(d, "summary.json"), "w"), indent=1)
    print(name, "A lint", res["A"]["lint_exit"], "B parts", {k: v["lint_exit"] for k, v in parts.items()}, "touched", len(touched), "dropped", len(dropped))


def scope_block(path):
    text = open(path, encoding="utf-8").read().strip()
    return text


def patch_files(path):
    return re.findall(r"^diff --git a/(\S+) b/", open(path, encoding="utf-8").read(), re.M)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--variants", default="")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--work", default="/home/jfowers/lt/refactor/eng/replay/work")
    ap.add_argument("--keep", action="store_true", help="keep the build outputs")
    a = ap.parse_args()
    d = os.path.join(HERE, a.name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    os.makedirs(a.work, exist_ok=True)
    if meta.get("commit"):
        return commit_replay(a.name, meta, d, a)
    base_patch = [os.path.join(d, "base.patch")] if os.path.isfile(os.path.join(d, "base.patch")) else []
    if meta.get("base_from"):   # chained: the base is another replay's variant (its base.patch chain + that variant's patch)
        base_patch = chain(meta["base_from"])
    bwt, bsha, btree = make_commit(a.work, a.name + "-base", base_patch)
    run(["git", "branch", "-f", f"replay-{a.name}-base", bsha], cwd=REPO)
    bout = build(bwt, btree, a.work, a.jobs, tuple(meta.get("base_args", ())))
    variants = [v for v in (a.variants.split(",") if a.variants else meta.get("variants", ["A", "B"])) if v]
    result = {}
    built = {}
    for v in variants:
        pv = os.path.join(d, v + ".patch")
        src = os.path.join(d, v[0] + ".patch") if not os.path.isfile(pv) else pv
        if not os.path.isfile(src):
            print(f"{a.name} {v}: no patch")
            continue
        key = os.path.basename(src)
        if key not in built:
            wt, sha, tree = make_commit(a.work, f"{a.name}-{key[:-6]}", [src], start=bsha)
            run(["git", "branch", "-f", f"replay-{a.name}-{key[:-6]}", sha], cwd=REPO)
            hout = build(wt, tree, a.work, a.jobs, tuple(meta.get("head_args", ())))
            os.makedirs(os.path.join(a.work, "json"), exist_ok=True)
            imp = os.path.join(a.work, "json", f"{a.name}-{key[:-6]}.impact")   # the JSON reports are big: kept out of the repo, the md stays
            run([PY, os.path.join(REPO, "fleet", "diff_fleet.py"), bout, hout, "--json", imp + ".json", "--md", os.path.join(d, key[:-6] + ".impact.md")])
            for fam in ("gfx11", "gfx12"):   # one GPU family's view (twins of the other family normalized away)
                run([PY, os.path.join(wt, "fleet", "replay_impact.py"), bout, hout, "--family", fam, "--out", f"{imp}.{fam}.json"])
            built[key] = (wt, sha, imp)
        wt, sha, imp = built[key]
        sf = os.path.join(d, v + ".scope")
        txt = ""
        if os.path.isfile(sf):
            body = f"```loom-scope\n{scope_block(sf)}\n```\n"
            pb = os.path.join(a.work, f"{a.name}-{v}.body.md")
            open(pb, "w").write(body)
            lint = run([PY, os.path.join(REPO, "fleet", "scope_lint.py"), "--repo", wt, "--base", bsha, "--head", sha, "--pr-body", pb,
                        "--impact", imp + ".json"], cwd=wt, check=False)
            txt = lint.stdout + lint.stderr
            open(os.path.join(d, v + ".lint.txt"), "w").write(f"exit {lint.returncode}\n" + txt)
        plan = run([PY, os.path.join(wt, "fleet", "sample_plan.py"), imp + ".json", "--configs", os.path.join(wt, "fleet")], cwd=wt, check=False)
        open(os.path.join(d, v + ".plan.txt"), "w").write(plan.stdout + plan.stderr)
        plan = run([PY, os.path.join(wt, "fleet", "sample_plan.py"), imp + ".json", "--configs", os.path.join(wt, "fleet"), "--granularity", "shape"],
                   cwd=wt, check=False)   # the coarser strata: kernel instantiation signature without the arch family
        open(os.path.join(d, v + ".plan.shape.txt"), "w").write(plan.stdout + plan.stderr)
        impact = json.load(open(imp + ".json"))
        result[v] = {"patch": os.path.basename(src), "files_edited": patch_files(src), "lint_exit": lint.returncode if txt else None,
                     "touched_by_family": {fam: sum(1 for x in json.load(open(f"{imp}.{fam}.json"))["models"].values() if x.get("kernels"))
                                           for fam in ("gfx11", "gfx12")},
                     "impact_summary": {k: impact["summary"][k] for k in ("models", "touched", "gpus", "rows")},
                     "touched_models": sorted(m for m, x in (impact.get("models") or {}).items() if x.get("changed") or x.get("kernels") or x.get("files")) if isinstance(impact.get("models"), dict) else None}
        json.dump(result[v], open(os.path.join(d, v + ".summary.json"), "w"), indent=1)
        print(a.name, v, "lint", result[v]["lint_exit"], "touched", len(result[v]["touched_models"] or []))
    if not a.keep:
        pass


def chain(spec):
    """spec 'name:variant' -> that replay's base.patch chain + the variant patch."""
    name, v = spec.split(":")
    d = os.path.join(HERE, name)
    m = json.load(open(os.path.join(d, "meta.json")))
    out = chain(m["base_from"]) if m.get("base_from") else ([os.path.join(d, "base.patch")] if os.path.isfile(os.path.join(d, "base.patch")) else [])
    return out + [os.path.join(d, v[0] + ".patch")]


if __name__ == "__main__":
    main()
