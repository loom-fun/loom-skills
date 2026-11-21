#!/usr/bin/env python3
"""Turn an hrx-loom-library card into a self-contained package for a Hugging Face model repo.

usage: stage_hf.py CARD_DIR OUT_DIR [--library-root HRX_LOOM_LIBRARY]

CARD_DIR is model/<publisher>/<model> in an hrx-loom-library checkout (--library-root defaults to
three levels up). OUT_DIR receives:
  program/*.loom    the card's program sources, unchanged
  library/motif/... every motif the card references, copied from the library (paths unchanged)
  model.json        the card with motif entries {"path", "origin": "../.."} reduced to bare paths, the
                    "archive" program set (Bazel-built bytecode) dropped and every service using "source"
  model.json.lock   unchanged
Checkpoint origins are kept, so parameters, tokenizer and license are still fetched from the publisher's
repository and never re-uploaded. README.md is not written here
(reference/card-template.md); existing files in OUT_DIR other than these are left alone.
Check the result before upload: loom-fetch OUT_DIR/model.json into a clean directory and run image_check on it.
Standard library only.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def flatten(entries, motifs):
    out = []
    for e in entries:
        if isinstance(e, dict) and e.get("origin", "").startswith("../"):
            motifs.append(e["path"])
            out.append(e["path"])
        else:
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("card")
    ap.add_argument("out")
    ap.add_argument("--library-root")
    args = ap.parse_args()
    src, dst = Path(args.card).resolve(), Path(args.out)
    root = Path(args.library_root).resolve() if args.library_root else src.parents[2]
    if not (root / "motif").is_dir():
        sys.exit(f"no motif/ under {root}; pass --library-root")

    card = json.loads((src / "model.json").read_text(encoding="utf-8"))
    motifs = []
    source = card["programs"]["source"]
    source["libraries"] = flatten(source.get("libraries", []), motifs)
    for e in source.get("inputs", []):
        if isinstance(e, dict) and "data" in e:
            e["data"] = flatten(e["data"], motifs)
    card["programs"] = {"source": source}
    for variant in card["variants"].values():
        for service in variant["services"].values():
            service["programs"] = ["source"]

    (dst / "program").mkdir(parents=True, exist_ok=True)
    for f in sorted((src / "program").glob("*.loom")):
        shutil.copy2(f, dst / "program" / f.name)
    for m in motifs:
        rel = Path(m).relative_to("library/motif")
        (dst / m).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "motif" / rel, dst / m)
    shutil.copy2(src / "model.json.lock", dst / "model.json.lock")
    (dst / "model.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{dst}: {len(list((dst / 'program').glob('*.loom')))} programs, {len(motifs)} motifs")


if __name__ == "__main__":
    main()
