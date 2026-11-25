#!/usr/bin/env python3
"""What a production-form loom-fun card repository may hold: the file set and the text rules of the card format.

usage: card_files.py DIR [DIR ...]      prints each package's problems; exit 1 when any has one

check_package(directory) -> list[str] of problems (empty: the package passes). DIR is a composed production-form package
(the files that would be committed to the loom-fun repo). replay_promotion.py prepare and lint_card.py call it and fail
closed on any problem.

The file set is the one the published card format defines (render_card.py's Files section, the image cards'
skills/loom-port-image card template): the card (README.md, REPORT.md, model.json, model.json.lock, info.json,
llm_config.json), the tokenizer, the weights containers (*weights.loom), the generated programs (program/), the
generator configs (gen/) and the shared Loom library (library/), plus .gitattributes. Anything else (an SDK or Quick
start qualification record, a rehearsal leftover, a dev/ bench tree, a porting note) is not a card file.

Text rules, over the card files that are text (never program/ or library/, whose identifiers are code): no reference
to the staging org or to staging at all (a production card is composed from the staging one with every staging
reference removed), no rehearsal or qualification-record wording, and an info.json without the retired hand-assembled
release comparison blocks.
"""
import json
import os
import re
import sys

TOP_FILES = {".gitattributes", "README.md", "REPORT.md", "model.json", "model.json.lock", "info.json", "llm_config.json",
             "tokenizer.json"}
WEIGHTS = re.compile(r"^[a-z0-9_]*weights\.loom$")   # weights.loom, head_weights.loom, vision_weights.loom, draft_weights.loom
TREES = ("program/", "gen/", "library/")
TEXT = {"README.md", "REPORT.md", "model.json", "model.json.lock", "info.json", "llm_config.json"}   # tokenizer.json: model data
# Production-form text never mentions staging (the org, a staging package, a staging main), a rehearsal, or the qualification
# records the card format does not define.
FORBIDDEN = [(re.compile(r"loom-fun-staging", re.I), "a loom-fun-staging reference"),
             (re.compile(r"(?<!loom-fun-)\bstag(?:ing|ed)\s+(?:package|main|org|organi[sz]ation|card|repo(?:sitory)?|copy|branch|"
                         r"revision|head|preset|run|release)s?\b", re.I), "staging wording"),
             (re.compile(r"\brehears", re.I), "rehearsal wording"),
             (re.compile(r"(?:quick-start|sdk|sdk-capacity)-qualification|sdk-comparison", re.I), "a qualification-record reference")]


def card_file(rel):
    """Whether a repository path (POSIX, relative) is a file of the card format."""
    if rel in TOP_FILES or WEIGHTS.match(rel):
        return True
    return rel.startswith(TREES) and not rel.endswith("/")


def package_files(directory):
    """The repository paths of a package directory (POSIX, relative), without VCS metadata."""
    out = []
    # followlinks: lint_card.py --form staging composes the production form with each top-level entry a symlink into the card, and its trees
    # (program/, gen/, dev/ ...) must be listed and checked like real directories
    for root, dirs, files in os.walk(directory, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in (".git", ".cache", "__pycache__"))
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(root, f), directory).replace(os.sep, "/"))
    return out


PROSE_STAGING = re.compile(r"(?<!loom-fun-)\bstaging\b", re.I)   # the card's own prose never says staging at all

RELEASE_HISTORY = re.compile(r"\b(?:previous card|prior SDK|migration-only|new port in SDK|regressions?|"
                             r"SDK[- ]delta|ABBA|kit build (?:was|kept)|change from the (?:last|previous))\b", re.I)
INFO_PROSE = {"note", "notes", "explanation", "explanations", "footnote", "footnotes", "next_steps",
              "differs", "reason", "detail", "summary", "description", "title", "text"}


def info_history_problems(value, path="info.json", prose=False):
    """Check narrative fields at any depth, including lists and keyed collections of notes."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from info_history_problems(item, f"{path}.{key}", prose or key in INFO_PROSE)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from info_history_problems(item, f"{path}[{index}]", prose)
    elif prose and isinstance(value, str):
        match = RELEASE_HISTORY.search(value)
        if match:
            yield f"{path}: release history ({match.group()})"


def markdown_history_lines(text):
    """Yield prose with its source line number; fenced code and inline command records are evidence."""
    fence = ""
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = ""
            continue
        if marker:
            fence = marker[1]
            continue
        # The renderer labels its records "Commands". Preserve surrounding prose on the same line.
        line = re.sub(r"(?i)(\bcommands?:\s*)`([^`]+)`", r"\1", line)
        line = re.sub(r"`(?:[./\w-]+\.(?:sh|py)|loom-serve|loom-fetch|python3?|bash)\s+[^`]+`", "", line)
        yield number, line


def text_problems(rel, text):
    """The text rules of one card text file (README.md and REPORT.md, the card's prose, never say "staging"; the
    configs' recorded evidence strings may say "staged" for a kernel's pipeline, so only staging-flow phrases count)."""
    out = []
    pats = FORBIDDEN + ([(PROSE_STAGING, "staging wording")] if rel in ("README.md", "REPORT.md") else [])
    if rel in ("README.md", "REPORT.md"):
        hits = set()
        first = None
        for number, line in markdown_history_lines(text):
            for match in RELEASE_HISTORY.finditer(line):
                first = first or number
                hits.add(match.group())
        if hits:
            out.append(f"{rel}:{first}: release history ({', '.join(sorted(hits)[:3])})")
    for pat, what in pats:
        hits = sorted({m.group(0) for m in pat.finditer(text)})
        if hits:
            line = next(i for i, ln in enumerate(text.splitlines(), 1) if pat.search(ln))
            out.append(f"{rel}:{line}: {what} ({', '.join(hits[:3])})")
    if rel == "info.json":
        try:
            info = json.loads(text)
        except ValueError as e:
            return out + [f"info.json: not JSON ({e})"]
        out.extend(info_history_problems(info))
        for field in ("promotion_hold", "sdk_delta"):
            if field in info:
                out.append(f"info.json: {field} belongs in the release report")
    return out


def check_package(directory, files=None):
    """Problems of a composed production-form package directory (empty list: it passes). files: its repository paths
    (default: walk the directory; a caller with a remote listing passes it and has the text files on disk)."""
    files = package_files(directory) if files is None else list(files)
    extra, trees = [], {}
    for rel in files:
        if not card_file(rel):
            if "/" in rel:   # a whole foreign tree (dev/, bench/ ...) is one problem
                trees.setdefault(rel.split("/")[0], []).append(rel)
            else:
                extra.append(rel)
    problems = [f"{rel}: not a file of the card format" for rel in extra]
    problems += [f"{t}/ ({len(v)} files): not part of the card format" for t, v in sorted(trees.items())]
    if "README.md" not in files:
        problems.append("README.md: missing")
    if "model.json" not in files:
        problems.append("model.json: missing")
    for rel in files:
        if rel in TEXT and os.path.isfile(os.path.join(directory, rel)):
            with open(os.path.join(directory, rel), encoding="utf-8", errors="replace") as f:
                problems += text_problems(rel, f.read())
    return problems


def main(argv=None):
    dirs = (argv if argv is not None else sys.argv[1:])
    if not dirs or dirs[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if dirs else 2
    bad = 0
    for d in dirs:
        problems = check_package(d)
        for p in problems:
            print(f"{d}: {p}")
        bad |= bool(problems)
        if not problems:
            print(f"{d}: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
