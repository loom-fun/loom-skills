#!/usr/bin/env python3
"""Locked read/update of a campaign's status.json and log.md (several agents write them).

usage: status.py [--dir CAMPAIGN_DIR] set PATH.TO.KEY VALUE [PATH VALUE ...]   dotted paths; VALUE is JSON or a plain string
       status.py [--dir CAMPAIGN_DIR] get [PATH.TO.KEY]                        print the value (whole file without a path)
       status.py [--dir CAMPAIGN_DIR] log WHO TEXT...                          append 'YYYY-MM-DD HH:MM WHO: TEXT' to log.md

CAMPAIGN_DIR defaults to $CAMPAIGN_DIR, else the current directory. The schema is in reference/campaign.md (section "State files").
Only the tech lead writes status.json; engineers use `log` and their mailbox.
"""
import fcntl
import json
import os
import sys
import time


def campaign_dir(args):
    if args[:1] == ["--dir"]:
        return args[1], args[2:]
    return os.environ.get("CAMPAIGN_DIR", "."), args


def locked(path):
    fh = open(path + ".lock", "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def set_path(doc, dotted, value):
    keys = dotted.split(".")
    node = doc
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def parse(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main(argv):
    base, args = campaign_dir(argv)
    if not args or args[0] not in ("set", "get", "log"):
        print(__doc__)
        return 2
    cmd, rest = args[0], args[1:]
    if cmd == "log":
        if len(rest) < 2:
            print("log WHO TEXT...")
            return 2
        line = f"{time.strftime('%Y-%m-%d %H:%M')} {rest[0]}: {' '.join(rest[1:])}\n"
        path = os.path.join(base, "log.md")
        with locked(path):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
        return 0
    path = os.path.join(base, "status.json")
    with locked(path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if cmd == "get":
            node = doc
            for k in (rest[0].split(".") if rest else []):
                node = node[k]
            print(json.dumps(node, indent=1))
            return 0
        if len(rest) < 2 or len(rest) % 2:
            print("set needs PATH VALUE pairs")
            return 2
        for i in range(0, len(rest), 2):
            set_path(doc, rest[i], parse(rest[i + 1]))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
