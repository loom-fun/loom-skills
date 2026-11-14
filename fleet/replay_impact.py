#!/usr/bin/env python3
"""Impact JSON of two program trees, for a replay of past drift (stand-in for fleet/diff_fleet.py).

usage: replay_impact.py BASE_DIR HEAD_DIR [--family gfx11] [--out impact.json]

BASE_DIR and HEAD_DIR hold <slug>/program/*.loom. Programs are compared kernel by kernel in one GPU family's view
(loomtext.view: twin kernels of the other family, target declarations and template selection are normalized away), so a gfx12
twin added next to an unchanged gfx11 kernel is not a change on gfx11. Output = the impact shape sample_plan.py reads:
{"models": {slug: {"kernels": [{"name", "kind", "target"}]}}}. Models with no change are listed with an empty kernel list.
fleet/diff_fleet.py is the CI tool; this one exists to compare two trees that were not built by build_fleet.py (the loom-fun
package vs the loom-fun-staging package in the measure-5 replay).
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loomtext  # noqa: E402


def write_view_trees(src, dst, family="gfx11", slugs=None):
    """Write <dst>/<slug>/program/view.loom = the normalized one-GPU-family view of <src>/<slug>/program (blocks sorted by key, twins,
    target declarations and template selection resolved) and link llm_config.json: a tree fleet/diff_fleet.py can compare between
    two packages that were built by different generator versions (the loom-fun package has no gfx12 twins, the staging one has)."""
    for slug in sorted(slugs or os.listdir(src)):
        p = os.path.join(src, slug, "program")
        if not os.path.isdir(p):
            continue
        d = os.path.join(dst, slug, "program")
        os.makedirs(d, exist_ok=True)
        v = loomtext.view(p, family)
        with open(os.path.join(d, "view.loom"), "w") as fh:
            fh.write("\n".join(v[k] + "\n" for k in sorted(v)))
        cfg = os.path.join(src, slug, "llm_config.json")
        if os.path.exists(cfg):
            shutil.copyfile(cfg, os.path.join(dst, slug, "llm_config.json"))


def impact(base, head, family="gfx11"):
    models = {}
    for slug in sorted(set(os.listdir(base)) & set(os.listdir(head))):
        a, b = os.path.join(base, slug, "program"), os.path.join(head, slug, "program")
        if not (os.path.isdir(a) and os.path.isdir(b)):
            continue
        d = loomtext.diff_views(loomtext.view(a, family), loomtext.view(b, family))
        models[slug] = {"kernels": [{"name": k.split(":", 1)[1], "kind": k.split(":", 1)[0], "target": family} for k in d]}
    return {"schema": "loom-impact-replay/1", "base": base, "head": head, "family": family, "models": models}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--family", default="gfx11")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = impact(a.base, a.head, a.family)
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    touched = [k for k, v in res["models"].items() if v["kernels"]]
    print(f"{len(res['models'])} models compared, {len(touched)} changed on {a.family}")


if __name__ == "__main__":
    main()
