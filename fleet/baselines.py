#!/usr/bin/env python3
"""Rebuild fleet/baselines.json: the currently published Loom value per (model, GPU, row) with its run link (the ratchet store).

usage: baselines.py [--strix-dir DIR] [--r9700-dir DIR] [--out fleet/baselines.json]

Strix Halo: the loom-fun cards, DIR/<slug>/info.json (default /home/jfowers/lt/r9700/pkgs: copies of the published packages).
Radeon R9700: the loom-fun-staging cards, DIR/<slug>/info.json (download info.json only: `hf download loom-fun-staging/<slug> info.json`).
Both are read through card_lib (loom-card/2 and /3). A row's value is the card value (the median over sessions of each session's
median sample, methodology rule 5); its run is the DevLab run of the session that measured it.
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import card_lib  # noqa: E402

SOURCES = {"strix-halo": "/home/jfowers/lt/r9700/pkgs", "radeon-r9700": "/home/jfowers/lt/refactor/eng/sampler/staging"}


def card_rows(info_path, gpu):
    info = card_lib.load_info(info_path)
    view = card_lib.gpu_view(info, gpu)
    if not card_lib.has_results((info.get("devices") or {}).get(gpu)):
        return None, (info.get("devices") or {}).get(gpu, {}).get("state") or "no results"
    out = {}
    for row, (m, idx) in card_lib.card_metrics(view).items():
        if row == "spec_acceptance" or m.get("loom") is None:
            continue
        s = view["performance"]["sessions"][idx]
        src = (info.get("sources") or {}).get(s.get("source")) or {}
        out[row] = {"value": m["loom"], "date": s.get("date"), "run": src.get("url") or s.get("source")}
    return out, None


def build(dirs, only=None):
    """only: the slugs of the fleet (default: the slugs that have a card in the last directory, the R9700 / staging one)"""
    models = {}
    only = set(only) if only else {s for s in os.listdir(list(dirs.values())[-1]) if os.path.exists(os.path.join(list(dirs.values())[-1], s, "info.json"))}
    for gpu, d in dirs.items():
        for slug in sorted(only):
            p = os.path.join(d, slug, "info.json")
            if not os.path.exists(p):
                continue
            rows, why = card_rows(p, gpu)
            if rows is None:
                models.setdefault(slug, {})[gpu] = {"_state": why}
            else:
                models.setdefault(slug, {})[gpu] = rows
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--strix-dir", default=SOURCES["strix-halo"])
    ap.add_argument("--r9700-dir", default=SOURCES["radeon-r9700"])
    ap.add_argument("--out", default=os.path.join(ROOT, "fleet", "baselines.json"))
    a = ap.parse_args()
    models = build({"strix-halo": a.strix_dir, "radeon-r9700": a.r9700_dir})
    doc = {"schema": "loom-baselines/1", "generated": datetime.date.today().isoformat(),
           "sources": {"strix-halo": "loom-fun cards", "radeon-r9700": "loom-fun-staging cards"}, "models": models}
    json.dump(doc, open(a.out, "w"), indent=1, sort_keys=True)
    n = sum(1 for m in models.values() for g in m.values() if "_state" not in g)
    print(f"{len(models)} models, {n} (model, GPU) baselines -> {a.out}")


if __name__ == "__main__":
    main()
