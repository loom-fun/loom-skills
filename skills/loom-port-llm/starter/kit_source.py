"""Where a package's programs came from: the loom-skills commit of the kit and the Loom SDK that built them.

build.py stamps generator_stamp() into llm_config.json["generator"]; the loom-fun cards show it.

  kit_source.py write KIT [--example NAME]      (apply_example.sh runs this) records in KIT/KIT_SOURCE the
      loom-skills commit the kit was copied from, whether that checkout had local changes under
      skills/loom-port-llm, and a hash of the kit's files, so a build can tell whether the kit was edited since.

generator_stamp():
  {"skills_commit": full sha or "unknown", "dirty": bool or null,
   "kit": {"example": NAME or null, "modified": bool} (a kit made by apply_example.sh),
   "hrx_loom": $LOOM_ROOT/loom-install.json (install_loom.sh), else {"hrx_loom": version, "fingerprint":
               first 16 hex of sha256(dist-info RECORD), "verified": false}, else "unknown"}
The commit comes from KIT/KIT_SOURCE, else from git when this file is inside a loom-skills checkout (the
starter used in place), else "unknown". Standard library only.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REL = os.path.join("skills", "loom-port-llm")


def kit_sha256(kit):
    """sha256 over the kit's files (relative path and content), without KIT_SOURCE and Python caches."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(kit):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f == "KIT_SOURCE" or f.endswith(".pyc"):
                continue
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, kit).replace(os.sep, "/").encode() + b"\0")
            with open(p, "rb") as fh:
                h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def git_commit(path):
    """(commit, dirty) of the loom-skills checkout containing path, or (None, None)."""
    def git(*args):
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        top = git("rev-parse", "--show-toplevel")
        if not os.path.isdir(os.path.join(top, SKILL_REL)):
            return None, None
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", os.path.join(top, SKILL_REL)))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def hrx_loom_identity():
    root = os.environ.get("LOOM_ROOT")
    venv = os.environ.get("VENV") or (os.path.join(root, ".venv") if root else None)
    if root and os.path.isfile(os.path.join(root, "loom-install.json")):
        with open(os.path.join(root, "loom-install.json"), encoding="utf-8") as f:
            return json.load(f)
    if venv:
        found = sorted(glob.glob(os.path.join(venv, "lib", "python*", "site-packages", "hrx_loom-*.dist-info"))
                       + glob.glob(os.path.join(venv, "Lib", "site-packages", "hrx_loom-*.dist-info")))
        if found:
            with open(os.path.join(found[0], "RECORD"), "rb") as f:
                fp = hashlib.sha256(f.read()).hexdigest()[:16]
            return {"hrx_loom": os.path.basename(found[0])[len("hrx_loom-"):-len(".dist-info")], "fingerprint": fp,
                    "verified": False}
    return "unknown"


def generator_stamp(kit=HERE):
    src = os.path.join(kit, "KIT_SOURCE")
    if os.path.isfile(src):
        with open(src, encoding="utf-8") as f:
            ks = json.load(f)
        stamp = {"skills_commit": ks.get("skills_commit") or "unknown", "dirty": ks.get("dirty"),
                 "kit": {"example": ks.get("example"), "modified": kit_sha256(kit) != ks.get("kit_sha256")}}
    else:
        commit, dirty = git_commit(kit)
        if not commit and os.environ.get("LOOM_SKILLS_COMMIT"):   # an exported snapshot (git archive) of that commit
            commit, dirty = os.environ["LOOM_SKILLS_COMMIT"], None
        stamp = {"skills_commit": commit or "unknown", "dirty": dirty}
    stamp["hrx_loom"] = hrx_loom_identity()
    return stamp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("kit")
    w.add_argument("--example")
    w.add_argument("--skills-dir", default=os.path.dirname(HERE), help="the loom-port-llm skill the kit came from")
    s = sub.add_parser("stamp")
    s.add_argument("kit", nargs="?", default=HERE)
    a = ap.parse_args(argv)
    if a.cmd == "stamp":
        print(json.dumps(generator_stamp(a.kit), indent=1, sort_keys=True))
        return
    commit, dirty = git_commit(a.skills_dir)
    ks = {"skills_commit": commit or "unknown", "dirty": dirty, "example": a.example, "kit_sha256": kit_sha256(a.kit)}
    with open(os.path.join(a.kit, "KIT_SOURCE"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(ks, indent=1, sort_keys=True) + "\n")
    print(f"KIT_SOURCE: loom-skills {ks['skills_commit'][:12]}{' (local changes)' if dirty else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()
