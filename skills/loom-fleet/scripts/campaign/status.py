#!/usr/bin/env python3
"""Locked read/update of a campaign's status.json and log.md (several agents write them).

usage: status.py [--dir CAMPAIGN_DIR] set PATH.TO.KEY VALUE [PATH VALUE ...]   dotted paths; VALUE is JSON or a plain string
       status.py [--dir CAMPAIGN_DIR] get [PATH.TO.KEY]                        print the value (whole file without a path)
       status.py [--dir CAMPAIGN_DIR] --pointer set /PATH/TO/KEY VALUE [/PATH VALUE ...]
       status.py [--dir CAMPAIGN_DIR] --pointer get /PATH/TO/KEY
       status.py [--dir CAMPAIGN_DIR] log WHO TEXT...                          append 'YYYY-MM-DD HH:MM WHO: TEXT' to log.md

With --pointer, / separates object keys, dots are literal, and ~1 and ~0 encode / and ~ in a key.
For example: status.py --pointer get /models/unsloth-Qwen3.5-4B-loom/gfx1152/verdict
In Git Bash with Windows Python, use Windows-style campaign and script paths and set
MSYS2_ARG_CONV_EXCL='*' for the command to pass pointer paths through unchanged.

CAMPAIGN_DIR defaults to $CAMPAIGN_DIR, else the current directory. The schema is in reference/campaign.md (section "State files").
Only the tech lead writes status.json; engineers use `log` and their mailbox.
"""
import errno
import json
import os
import re
import sys
import time
from contextlib import contextmanager

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def campaign_dir(args):
    if args[:1] == ["--dir"]:
        return args[1], args[2:]
    return os.environ.get("CAMPAIGN_DIR", "."), args


@contextmanager
def locked(path):
    with open(path + ".lock", "a+b") as fh:
        if os.name != "nt":
            fcntl.flock(fh, fcntl.LOCK_EX)
            yield
            return

        # Lock byte zero even in existing empty lock files. Use a retry loop
        # because the CRT's blocking mode gives up after ten attempts.
        fh.seek(0)
        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as exc:
                if exc.errno != errno.EACCES:
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def path_keys(path, pointer):
    if not pointer:
        return path.split(".")
    if not path.startswith("/") or re.search(r"~(?![01])", path):
        raise ValueError(f"invalid pointer path {path!r}: start with / and use ~1 for / and ~0 for ~ in keys")
    return [key.replace("~1", "/").replace("~0", "~") for key in path[1:].split("/")]


def set_path(doc, keys, value):
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
    pointer = args[:1] == ["--pointer"]
    if pointer:
        args = args[1:]
    commands = ("set", "get") if pointer else ("set", "get", "log")
    if not args or args[0] not in commands:
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
    if cmd == "set" and (len(rest) < 2 or len(rest) % 2):
        print("set needs PATH VALUE pairs")
        return 2
    if cmd == "get" and pointer and len(rest) != 1:
        print("--pointer get needs one /PATH", file=sys.stderr)
        return 2
    path_args = rest[:1] if cmd == "get" else rest[::2]
    try:
        paths = [path_keys(path, pointer) for path in path_args]
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    path = os.path.join(base, "status.json")
    with locked(path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if cmd == "get":
            node = doc
            for k in (paths[0] if paths else []):
                node = node[k]
            print(json.dumps(node, indent=1))
            return 0
        for keys, value in zip(paths, rest[1::2]):
            set_path(doc, keys, parse(value))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
