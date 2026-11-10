#!/usr/bin/env python3
"""Install the complete, shared Loom suite into native agent discovery roots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[1] / "skills"
MARKER = ".loom-skills-install.json"
OWNER = "loom-fun/loom-skills"


def rename(source, target):
    # Windows scanners can briefly hold newly copied files open.
    for attempt in range(6):
        try:
            source.rename(target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"Refusing symlink inside managed skill: {path}")
        if path.is_file() and path.name != MARKER:
            result[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def managed(target):
    if target.is_symlink() or target.is_junction():
        raise ValueError(f"Refusing to replace symlink: {target}")
    if not target.exists():
        return False
    marker = target / MARKER
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unmanaged skill; move it aside before installing: {target}") from exc
    if record.get("owner") != OWNER or record.get("files") != inventory(target):
        raise ValueError(f"Locally modified skill; preserve your changes before updating/removing: {target}")
    return True


def destinations(agent, home):
    shared = Path(os.environ.get("AGENTS_SKILLS_DIR", home / ".agents/skills"))
    roots = {
        "claude": Path(os.environ.get("CLAUDE_SKILLS_DIR", home / ".claude/skills")),
        "codex": shared,
        "pi": Path(os.environ.get("PI_SKILLS_DIR", shared)),
    }
    return list(dict.fromkeys(roots[a].expanduser().resolve() for a in (roots if agent == "all" else [agent])))


def install(roots, uninstall=False, source=SOURCE):
    source = source.resolve()
    skills = sorted(p for p in source.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())
    if not skills:
        raise ValueError(f"No skills found in {source}")
    targets = []
    for root in roots:
        root = root.resolve()
        if root == source or source in root.parents or root in source.parents:
            raise ValueError(f"Installation root overlaps source skills: {root}")
        for skill in skills:
            target = root / skill.name
            exists = managed(target)
            targets.append((skill, target, exists))
    # Preflight every destination before changing any installed skill.
    for skill, target, exists in targets:
        if uninstall:
            if exists:
                shutil.rmtree(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".loom-install-", dir=target.parent) as temp:
            stage = Path(temp) / "new"
            shutil.copytree(skill, stage, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (stage / MARKER).write_text(json.dumps({"owner": OWNER, "files": inventory(stage)}, indent=2) + "\n", encoding="utf-8")
            backup = Path(temp) / "old"
            if exists:
                rename(target, backup)
            try:
                rename(stage, target)
            except OSError:
                if exists:
                    rename(backup, target)
                raise
    return len(skills)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["all", "claude", "codex", "pi"], default="all")
    parser.add_argument("--dest", type=Path, help="Explicit skill directory instead of native discovery roots")
    parser.add_argument("--uninstall", action="store_true", help="Remove only unchanged copies owned by this installer")
    args = parser.parse_args()
    roots = [args.dest.expanduser().resolve()] if args.dest else destinations(args.agent, Path.home())
    try:
        count = install(roots, args.uninstall)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    action = "Removed" if args.uninstall else "Installed"
    print(f"{action} {count} skills in: " + ", ".join(map(str, roots)))
    if not args.uninstall:
        print("Start a fresh agent session. Claude: /loom; Codex: $loom; Pi: /skill:loom.")


if __name__ == "__main__":
    main()
