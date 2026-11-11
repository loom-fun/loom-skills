"""Read hf CLI paths and stage checkpoint files across Windows/Linux filesystems."""
import argparse
from pathlib import Path
import os
import shutil
import sys
import tempfile


def from_log(log, filename):
    for line in reversed(Path(log).read_text(encoding="utf-8").splitlines()):
        candidate = line.strip()
        if candidate.startswith("path="):
            candidate = candidate[5:]
        if candidate.endswith(filename) and Path(candidate).is_file():
            return Path(os.path.abspath(candidate))   # not resolve(): the snapshots/REV/FILE path names the revision
    raise ValueError(f"No existing {filename} path in {log}")


def link_checkpoint(source, target):
    # Link the HF snapshot entry itself (snapshots/REV/FILE), not the blob it points to: publish_package.py reads the
    # repository and revision of a package's weights from that path.
    source = Path(os.path.abspath(source))
    if not source.is_file():
        raise FileNotFoundError(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.samefile(source):
        return
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=target.parent) as temp:
        stage = Path(temp) / "weights"
        try:
            stage.symlink_to(source)
        except OSError:
            try:
                os.link(source, stage)
            except OSError:
                shutil.copyfile(source, stage)  # Different volumes, or no link privileges.
        os.replace(stage, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("from-log")
    read.add_argument("log"); read.add_argument("filename")
    link = commands.add_parser("link")
    link.add_argument("source"); link.add_argument("target")
    args = parser.parse_args()
    if args.command == "from-log":
        sys.stdout.write(str(from_log(args.log, args.filename)))
    else:
        link_checkpoint(args.source, args.target)
