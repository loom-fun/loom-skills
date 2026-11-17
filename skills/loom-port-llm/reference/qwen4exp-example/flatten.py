#!/usr/bin/env python3
"""Flatten the hrx-loom-library Flash-Next package into the kit's self-contained catalog/ directory.

usage: flatten.py LIBRARY_CHECKOUT OUT_DIR

LIBRARY_CHECKOUT is a hrx-loom-library tree (the release tag, with library_delta.patch applied: see README.md "Refreshing").
The package lives in model/Qwen/Qwen3.8-Flash-Next and pulls shared files from the repo-level program/, motif/ and kernel/
trees; the kit's catalog/ must stand alone, so package files keep their path below program/ and shared files go below
program/shared/. Only the source-form programs ("source", "vision-source") are kept: the bytecode "archive" programs
need the library's Bazel build.
"""
import json
import os
import shutil
import sys

PKG = "model/Qwen/Qwen3.8-Flash-Next"


def main(src, out):
    with open(os.path.join(src, PKG, "model.json"), encoding="utf-8") as f:
        m = json.load(f)
    copied = {}

    def dest(path):
        if path.startswith(PKG + "/program/"):
            return "program/" + path[len(PKG) + 9:]
        return "program/shared/" + path

    def cp(s):
        d = dest(s)
        os.makedirs(os.path.dirname(os.path.join(out, d)), exist_ok=True)
        shutil.copyfile(os.path.join(src, s), os.path.join(out, d))
        copied[d] = s
        return d

    def conv(x):
        if isinstance(x, str):
            return cp(x)
        y = dict(x)
        y["path"] = cp(x["path"])
        if "data" in x:
            y["data"] = [cp(d) for d in x["data"]]
        return y

    progs = {}
    for name in ("source", "vision-source"):
        p = m["programs"][name]
        progs[name] = {"directory": "program", "origin": "./", "inputs": [conv(i) for i in p["inputs"]],
                       "libraries": [conv(lib) for lib in p["libraries"]]}
    m["programs"] = progs
    for v in m["variants"].values():
        for s in v["services"].values():
            s["programs"] = [p for p in s["programs"] if p in progs]
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    print(len(copied), "files")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
