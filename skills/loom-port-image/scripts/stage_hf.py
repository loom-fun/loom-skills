#!/usr/bin/env python3
"""Turn an hrx-loom-library card into a self-contained package for a Hugging Face model repo.

usage: stage_hf.py CARD_DIR OUT_DIR [--library-root HRX_LOOM_LIBRARY]

CARD_DIR is model/<publisher>/<model> in an hrx-loom-library checkout (--library-root defaults to
three levels up). OUT_DIR receives:
  program/...       the card's referenced program sources and local input assets
  library/motif/... the referenced catalog motifs with their package paths preserved
  model.json        the card with catalog motif origins converted to local references and
                    services and presets limited to the source program set
  model.json.lock   unchanged
Named remote origins are preserved. OUT_DIR must be a new directory. Add README.md after staging.
Check the result before upload: loom-fetch OUT_DIR/model.json into a clean directory and run image_check.
Standard library only.
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath


def package_path(value):
    path = PurePosixPath(value)
    if any(part in ("", ".", "..") for part in value.split("/")) or "\\" in value or ":" in value:
        raise ValueError(f"unsafe package path {value!r}")
    return path


def add_file(files, destination, base, source):
    base = base.resolve()
    candidate = (base / source).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise ValueError(f"missing or out-of-tree package asset {base / source}")
    if destination in files and files[destination] != candidate:
        raise ValueError(f"conflicting package assets at {destination}")
    files[destination] = candidate


def prepare_package(src, root):
    card = json.loads((src / "model.json").read_text(encoding="utf-8"))
    files = {}
    known_origins = card.get("origins", {})

    def asset(value):
        record = value if isinstance(value, dict) else {"path": value}
        path = package_path(record["path"])
        if path.as_posix() in {"model.json", "model.json.lock"}:
            raise ValueError(f"reserved package asset path {path}")
        origin = record.get("origin")
        if origin and origin.startswith("../"):
            prefix = PurePosixPath("library/motif")
            if not path.is_relative_to(prefix):
                raise ValueError(f"unexpected catalog origin {origin!r} for {path}")
            relative = path.relative_to(prefix)
            expected = root / "motif" / relative
            if (root / "motif").is_symlink() or origin != Path(os.path.relpath(expected, src)).as_posix():
                raise ValueError(f"catalog origin {origin!r} does not match {path}")
            add_file(files, path, root / "motif", relative)
            record.pop("origin")
        elif origin and origin.startswith("@"):
            name, _, remote_path = origin[1:].partition("/")
            if name == "document":
                add_file(files, path, src, package_path(remote_path))
                record.pop("origin")
            elif name not in known_origins:
                raise ValueError(f"unknown package origin {name!r}")
            else:
                package_path(remote_path)
        else:
            add_file(files, path, src, package_path(origin) if origin else path)
            record.pop("origin", None)
        if "data" in record:
            record["data"] = [asset(entry) for entry in record["data"]]
        if set(record) == {"path"}:
            return record["path"]
        return record

    def variant_assets(value):
        if isinstance(value, dict):
            if "path" in value:
                prepared = asset(value)
                return prepared if isinstance(prepared, dict) else {"path": prepared}
            return {key: variant_assets(item) for key, item in value.items()}
        if isinstance(value, list):
            return [variant_assets(item) for item in value]
        return value

    source = card["programs"]["source"]
    if "directory" in source:
        package_path(source["directory"])
    for key in ("inputs", "libraries"):
        if key in source:
            source[key] = [asset(entry) for entry in source[key]]
    card["programs"] = {"source": source}
    for variant in card["variants"].values():
        variant["services"] = variant_assets(variant["services"])
        for service in variant["services"].values():
            if "source" not in service["programs"]:
                raise ValueError("variant service does not offer the source program set")
            service["programs"] = ["source"]
    for preset in card.get("presets", {}).values():
        preset["programs"] = "source"
    add_file(files, PurePosixPath("model.json.lock"), src, PurePosixPath("model.json.lock"))
    return card, files


def stage(src, dst, root):
    if dst.exists() or dst.is_symlink():
        raise ValueError(f"output already exists: {dst}; choose a new directory")
    card, files = prepare_package(src, root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-hf-", dir=dst.parent) as tmp:
        package = Path(tmp) / "package"
        package.mkdir()
        for path, source in sorted(files.items()):
            target = package / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (package / "model.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        package.rename(dst)
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("card")
    ap.add_argument("out")
    ap.add_argument("--library-root")
    args = ap.parse_args()
    src, dst = Path(args.card).resolve(), Path(args.out)
    root = Path(args.library_root).resolve() if args.library_root else src.parents[2]
    try:
        count = stage(src, dst, root)
    except (KeyError, OSError, TypeError, ValueError) as e:
        ap.exit(2, f"stage_hf: {e}\n")
    print(f"{dst}: {count} referenced files and model.json")


if __name__ == "__main__":
    main()
