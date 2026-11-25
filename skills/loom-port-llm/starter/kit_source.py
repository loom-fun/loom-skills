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


def _load_sdk_pin(skills_dir=None):
    """The SDK the kit's programs need: the newest verified release of loom-setup/scripts/sdk_pin.py (the one source of truth), written to
    llm_config.json "sdk". The kit is copied out of the repository, so `write` records the pin in KIT_SOURCE and a copy reads it there."""
    import importlib.util
    here_sdk = os.path.join(skills_dir or os.path.join(HERE, ".."), "..", "loom-setup", "scripts", "sdk_pin.py")
    if os.path.isfile(here_sdk):
        spec = importlib.util.spec_from_file_location("sdk_pin", here_sdk)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.release_pin()
    try:
        with open(os.path.join(HERE, "KIT_SOURCE"), encoding="utf-8") as f:
            pin = json.load(f).get("sdk_pin")
    except (OSError, ValueError):
        pin = None
    if not pin:
        raise RuntimeError("kit_source: no SDK pin: run `kit_source.py write KIT --skills-dir <loom-port-llm skill>` from the loom-skills checkout")
    return pin


def __getattr__(name):   # SDK_PIN is resolved on first use, so `write` can run in a fresh copy that has no KIT_SOURCE yet
    if name == "SDK_PIN":
        globals()["SDK_PIN"] = pin = _load_sdk_pin()
        return pin
    raise AttributeError(name)


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
    try:
        sdk_pin = _load_sdk_pin(a.skills_dir)
    except RuntimeError:   # a skills tree without loom-setup: the kit has no pin, and a build that needs SDK_PIN says so
        sdk_pin = None
    kb = os.path.abspath(os.path.join(a.skills_dir, "..", "loom-kb"))   # the knowledge base beside the skill the kit came from
    ks = {"skills_commit": commit or "unknown", "dirty": dirty, "example": a.example, "kit_sha256": kit_sha256(a.kit),
          "kb_root": kb if os.path.isdir(kb) else None, **({"sdk_pin": sdk_pin} if sdk_pin else {})}
    with open(os.path.join(a.kit, "KIT_SOURCE"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(ks, indent=1, sort_keys=True) + "\n")
    print(f"KIT_SOURCE: loom-skills {ks['skills_commit'][:12]}{' (local changes)' if dirty else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()
