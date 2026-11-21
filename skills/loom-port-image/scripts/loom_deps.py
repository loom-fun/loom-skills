#!/usr/bin/env python3
"""BUILD dependency list per program file of a Loom model package: which program file or library motif defines each symbol it declares.

usage: loom_deps.py PROGRAM_DIR [--motif-root HRX_LOOM_LIBRARY/motif]

PROGRAM_DIR is a card's program/ directory. --motif-root defaults to ../../../../motif from it (the
hrx-loom-library layout model/<publisher>/<model>/program). For every *.loom file it prints the
`loom_library` deps (`:sibling` program files and `//motif/<dir>:<stem>` labels) its `*.decl` lines
need, then the motif set in first-use order. A declared symbol nothing defines is printed as MISSING:
a typo, or a motif that is not in this library revision. Standard library only; text match, not a parser.
"""
import argparse
import pathlib
import re
import sys

DEF = re.compile(r"^(?:kernel|command|template|func)\.def(?:<[^>]*>)?(?:\s+target\([^)]*\))?"
                 r"(?:\s+requires\s+\[[^\]]*\])?(?:\s+(?:public|device))*\s+(@[\w.]+)", re.M)
TDEF = re.compile(r"^template\.def<(@[\w.]+)>", re.M)
DECL = re.compile(r"^(?:kernel|command|template|func)\.decl\s+(@[\w.]+)", re.M)


def defined(text):
    return set(DEF.findall(text)) | set(TDEF.findall(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("program")
    ap.add_argument("--motif-root")
    args = ap.parse_args()
    program = pathlib.Path(args.program)
    motif_root = pathlib.Path(args.motif_root) if args.motif_root else program.resolve().parents[3] / "motif"
    if not motif_root.is_dir():
        sys.exit(f"motif root not found: {motif_root}")

    owners = {}
    for path in sorted(motif_root.rglob("*.loom")):
        if path.name.endswith("_test.loom") or "test" in path.relative_to(motif_root).parts:
            continue
        label = "//motif/" + path.parent.relative_to(motif_root).as_posix() + ":" + path.stem
        for symbol in defined(path.read_text(encoding="utf-8")):
            owners.setdefault(symbol, label)
    for path in sorted(program.glob("*.loom")):
        for symbol in defined(path.read_text(encoding="utf-8")):
            owners[symbol] = ":" + path.stem

    motifs, missing_any = [], False
    for path in sorted(program.glob("*.loom")):
        text = path.read_text(encoding="utf-8")
        mine = defined(text)
        deps, missing = [], []
        for symbol in DECL.findall(text):
            if symbol in mine:
                continue
            owner = owners.get(symbol)
            if owner is None:
                missing.append(symbol)
            elif owner != ":" + path.stem and owner not in deps:
                deps.append(owner)
                if owner.startswith("//") and owner not in motifs:
                    motifs.append(owner)
        missing_any |= bool(missing)
        print(path.stem, sorted(deps), ("MISSING " + " ".join(missing)) if missing else "")
    print("motifs", motifs)
    sys.exit(1 if missing_any else 0)


if __name__ == "__main__":
    main()
