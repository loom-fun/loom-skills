#!/usr/bin/env python3
"""Promote loom-fun-staging cards to loom-fun: the staging repo's files, with the card rewritten for its new org.

  promote_staging.py [--work DIR] [--report REPORT.md] [--json REPORT.json] [--execute] SLUG... | --all

Per repo (the same name in both orgs):
  1. fetch the staging repo at its current commit (every file but .gitattributes: the package is small, its weights are
     fetched from their origins by loom-fetch);
  2. the publish-time org rewrite: info.json "repo" -> loom-fun/NAME, then render_card.py regenerates README.md and REPORT.md
     (Quick start URLs, links) from it, and render_card.py --check verifies them; no file may still name loom-fun-staging;
  3. compare with the loom-fun repo at its current commit: files added / changed / removed / unchanged (git blob ids, LFS sha256),
     and a README diff summary;
  4. publish_package.py upload --org loom-fun --dry-run (the card checks the real upload runs).
--execute uploads each repo for real (publish_package.py upload --org loom-fun --parent-commit <loom-fun commit compared>: the
upload fails if loom-fun moved since the dry run). Only on the user's explicit say-so; also requires PROMOTE_CONFIRM=loom-fun.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
import card_lib  # noqa: E402

SRC, DST = "loom-fun-staging", "loom-fun"


def git_blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def remote_files(api, repo):
    info = api.model_info(repo, files_metadata=True)
    out = {}
    for s in info.siblings:
        lfs = s.lfs if isinstance(s.lfs, dict) else (vars(s.lfs) if s.lfs else None)
        out[s.rfilename] = ("lfs", lfs["sha256"]) if lfs else ("git", s.blob_id)
    return info.sha, out


def promote(api, slug, work, execute):
    from huggingface_hub import snapshot_download
    src, dst = f"{SRC}/{slug}", f"{DST}/{slug}"
    r = {"slug": slug, "staging": src, "target": dst}
    ssha, sfiles = remote_files(api, src)
    try:
        dsha, dfiles = remote_files(api, dst)
    except Exception as e:  # noqa: BLE001
        dsha, dfiles = None, {}
        r["target_note"] = f"no {dst}: {type(e).__name__}"
    r["staging_commit"], r["target_commit"] = ssha, dsha
    out = os.path.join(work, slug, "out")
    shutil.rmtree(os.path.join(work, slug), ignore_errors=True)
    snapshot_download(src, revision=ssha, local_dir=out, ignore_patterns=[".gitattributes"])
    shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)
    # the org rewrite
    ip = os.path.join(out, "info.json")
    info = card_lib.load_json(ip)
    if info.get("repo") != src:
        raise SystemExit(f"{slug}: staging info.json repo is {info.get('repo')}, expected {src}")
    readme_old = open(os.path.join(out, "README.md"), encoding="utf-8").read()
    info["repo"] = dst
    card_lib.write_json(ip, info)
    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "render_card.py"), out], check=True, capture_output=True, text=True)
    chk = subprocess.run([py, os.path.join(HERE, "render_card.py"), "--check", out, os.path.join(out, "README.md")],
                         capture_output=True, text=True)
    r["render_check"] = "pass" if chk.returncode == 0 else f"FAIL: {(chk.stdout + chk.stderr).strip()[-300:]}"
    left = []
    for root, _, fs in os.walk(out):
        for f in fs:
            p = os.path.join(root, f)
            try:
                if SRC in open(p, encoding="utf-8").read():
                    left.append(os.path.relpath(p, out))
            except UnicodeDecodeError:
                pass
    r["staging_refs_left"] = left
    readme_new = open(os.path.join(out, "README.md"), encoding="utf-8").read()
    r["readme_vs_staging"] = sum(1 for d in difflib.ndiff(readme_old.splitlines(), readme_new.splitlines()) if d[:1] in "+-")
    # compare with the loom-fun repo
    local = {}
    for root, _, fs in os.walk(out):
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), out).replace(os.sep, "/")
            local[rel] = os.path.join(root, f)
    ch = {"added": [], "changed": [], "removed": [], "unchanged": []}
    for rel, p in sorted(local.items()):
        if rel not in dfiles:
            ch["added"].append(rel)
            continue
        kind, h = dfiles[rel]
        same = (sha256(p) if kind == "lfs" else git_blob(p)) == h
        ch["unchanged" if same else "changed"].append(rel)
    ch["removed"] = sorted(f for f in dfiles if f not in local and f != ".gitattributes")
    r["files"] = ch
    if dsha and "README.md" in dfiles:
        from huggingface_hub import hf_hub_download
        old = open(hf_hub_download(dst, "README.md", revision=dsha), encoding="utf-8").read()
        d = [x for x in difflib.unified_diff(old.splitlines(), readme_new.splitlines(), lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---")]
        heads = lambda t: [ln for ln in t.splitlines() if ln.startswith("#")]
        r["readme_diff"] = {"added_lines": sum(1 for x in d if x[0] == "+"), "removed_lines": sum(1 for x in d if x[0] == "-"),
                            "new_headings": [h for h in heads(readme_new) if h not in heads(old)],
                            "dropped_headings": [h for h in heads(old) if h not in heads(readme_new)]}
        r["schema"] = {"target": json.load(open(hf_hub_download(dst, "info.json", revision=dsha))).get("schema"), "new": info.get("schema")}
    dry = subprocess.run([py, os.path.join(HERE, "publish_package.py"), "upload", out, "--repo", slug, "--org", DST, "--dry-run"],
                         capture_output=True, text=True)
    r["dry_run"] = "pass" if dry.returncode == 0 else f"FAIL: {(dry.stdout + dry.stderr).strip()[-300:]}"
    r["ok"] = r["render_check"] == "pass" and not left and r["dry_run"] == "pass"
    if execute:
        if not r["ok"]:
            r["executed"] = "skipped: checks failed"
        else:
            cmd = [py, os.path.join(HERE, "publish_package.py"), "upload", out, "--repo", slug, "--org", DST,
                   "--message", f"Promote from {SRC}/{slug}@{ssha[:12]}"] + (["--parent-commit", dsha] if dsha else [])
            u = subprocess.run(cmd, capture_output=True, text=True)
            r["executed"] = "uploaded" if u.returncode == 0 else f"FAIL: {(u.stdout + u.stderr).strip()[-300:]}"
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every model of .github/nightly-models.json")
    ap.add_argument("--work", required=True, help="a scratch dir (on disk) for the rewritten repos")
    ap.add_argument("--report", help="write a Markdown dry-run report here")
    ap.add_argument("--json", help="write the per-repo results as JSON here")
    ap.add_argument("--execute", action="store_true", help="really upload to loom-fun (the user's call only)")
    a = ap.parse_args()
    if a.execute and os.environ.get("PROMOTE_CONFIRM") != DST:
        sys.exit("--execute publishes to loom-fun: set PROMOTE_CONFIRM=loom-fun (only on the user's say-so)")
    slugs = a.slugs
    if a.all:
        slugs = [m["repo"].split("/")[-1] for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]]
    from huggingface_hub import HfApi
    api = HfApi()
    res = []
    for s in slugs:
        try:
            res.append(promote(api, s, a.work, a.execute))
        except BaseException as e:  # noqa: BLE001
            res.append({"slug": s, "ok": False, "error": f"{type(e).__name__}: {e}"[:400]})
        x = res[-1]
        print(f"{'OK  ' if x.get('ok') else 'FAIL'} {s}: " + (x.get("error") or
              f"+{len(x['files']['added'])} ~{len(x['files']['changed'])} -{len(x['files']['removed'])} ={len(x['files']['unchanged'])}; "
              f"render {x['render_check']}; dry run {x['dry_run']}"), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if a.report:
        L = ["# Promotion dry run: loom-fun-staging -> loom-fun", "",
             f"{sum(1 for x in res if x.get('ok'))}/{len(res)} repos pass every check (org rewrite, render --check, no "
             "loom-fun-staging reference left, publish_package.py upload --org loom-fun --dry-run). Nothing was uploaded.", "",
             "| repo | staging commit | loom-fun commit | added | changed | removed | unchanged | README +/- | schema | checks |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for x in res:
            if x.get("error"):
                L.append(f"| {x['slug']} | | | | | | | | | FAIL: {x['error'][:120]} |")
                continue
            f, rd = x["files"], x.get("readme_diff") or {}
            L.append(f"| {x['slug']} | {x['staging_commit'][:8]} | {(x['target_commit'] or 'none')[:8]} | {len(f['added'])} | {len(f['changed'])} | "
                     f"{len(f['removed'])} | {len(f['unchanged'])} | +{rd.get('added_lines', '?')}/-{rd.get('removed_lines', '?')} | "
                     f"{(x.get('schema') or {}).get('target')} -> {(x.get('schema') or {}).get('new')} | {'pass' if x['ok'] else 'FAIL'} |")
        L += ["", "## Per repo", ""]
        for x in res:
            if x.get("error"):
                continue
            f, rd = x["files"], x.get("readme_diff") or {}
            L += [f"### {x['slug']}", f"- changed: {', '.join(f['changed']) or 'none'}", f"- added: {', '.join(f['added']) or 'none'}",
                  f"- removed: {', '.join(f['removed']) or 'none'}",
                  f"- README new headings: {'; '.join(rd.get('new_headings') or []) or 'none'}; dropped: {'; '.join(rd.get('dropped_headings') or []) or 'none'}",
                  f"- checks: render {x['render_check']}; staging refs left {x['staging_refs_left'] or 'none'}; dry run {x['dry_run']}", ""]
        open(a.report, "w").write("\n".join(L) + "\n")
    sys.exit(0 if all(x.get("ok") for x in res) else 1)


if __name__ == "__main__":
    main()
