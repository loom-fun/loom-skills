#!/usr/bin/env python3
"""The production inventory of an SDK migration: {org/repo: revision} of every PUBLIC package repo of the production org (read-only).

  sdk_inventory.py snapshot [--org loom-fun] -o FILE        list the org's public model repos and their head revisions (private repos are not
                                                            production: drafts and retired packages)
  sdk_inventory.py drift OLD.json NEW.json [--json OUT]     added / removed / moved repos between two inventories (exit 0; the report is the output)

The accepted inventory of a migration lives at skills/loom-migrate/versions/vOLD-to-vNEW-production.json (fleet/regen.py --programs production reads it;
packages held out of the release are listed in the sibling vOLD-to-vNEW-held.json). Re-snapshot it right before a sweep and record the drift.
"""
import argparse
import json
import sys


def snapshot(org):
    from huggingface_hub import HfApi
    return {m.id: m.sha for m in HfApi().list_models(author=org, limit=10_000, full=True) if not m.private}


def drift(old, new):
    return {"added": {k: new[k] for k in sorted(set(new) - set(old))},
            "removed": {k: old[k] for k in sorted(set(old) - set(new))},
            "moved": {k: {"from": old[k], "to": new[k]} for k in sorted(set(old) & set(new)) if old[k] != new[k]}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("snapshot")
    p.add_argument("--org", default="loom-fun")
    p.add_argument("-o", "--out", required=True)
    p = sub.add_parser("drift")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        inv = snapshot(a.org)
        with open(a.out, "w") as f:
            json.dump(dict(sorted(inv.items(), key=lambda kv: kv[0].lower())), f, indent=2)
            f.write("\n")
        print(f"{len(inv)} public repos of {a.org}: {a.out}")
        return 0
    d = drift(json.load(open(a.old)), json.load(open(a.new)))
    for k, v in d["added"].items():
        print(f"added   {k} @ {v[:10]}")
    for k, v in d["removed"].items():
        print(f"removed {k} (was {v[:10]})")
    for k, v in d["moved"].items():
        print(f"moved   {k}: {v['from'][:10]} -> {v['to'][:10]}")
    if a.json:
        json.dump(d, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
