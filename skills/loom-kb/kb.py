#!/usr/bin/env python3
"""loom-kb: the knowledge-base router and structure lint (stdlib only).

  kb.py route --config LLM_CONFIG.json | --gguf FILE | --arch A [--facets f1,f2]
              [--device strix-halo|radeon-r9700|rdna3|rdna4|all] [--quant q4_k_m,...] [--topic t1,t2]
  kb.py lint  [--root REPO]      structure + duplicate lint of every skills/**/*.md (exit 1 on errors)
  kb.py dupes [--root REPO]      duplicate-paragraph count only (works on any checkout, e.g. an old ref)
  kb.py list                     valid devices, families, facets, topics

`route` prints, one path per line and in reading order: device family README, device README, quant READMEs,
the arch chain root -> leaf, facet READMEs, topic files; then a '# resolved choices' footer when resolve.py exists.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parent

# Used when device/ has no device.json yet; the device registry (regs) takes over when present.
FALLBACK_DEVICES = {"strix-halo": "rdna3", "radeon-r9700": "rdna4"}
FALLBACK_FAMILIES = ("rdna3", "rdna4")

README_MAX = 300
TOC_ABOVE = 100
SKILL_MAX = 500
DUP_MIN_CHARS = 200
FIXTURE_PARTS = ("/scripts/card/test/",)  # golden card fixtures are generated text, not knowledge


def die(msg, code=2):
    print(f"kb.py: {msg}", file=sys.stderr)
    sys.exit(code)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------- nodes

def scope_files(root):
    return sorted(Path(root).glob("arch/**/scope.json")) + sorted(Path(root).glob("facet/*/scope.json"))


def load_scopes(root=HERE):
    """-> (families, facets). families: list of dicts with dir, topology, name; facets: dict tag -> dict."""
    families, facets = [], {}
    for path in scope_files(root):
        node = read_json(path)
        node["dir"] = path.parent
        rel = path.parent.relative_to(root).as_posix()
        node["rel"] = rel
        if rel.startswith("facet/"):
            facets[rel.split("/", 1)[1]] = node
        else:
            parts = rel.split("/")  # arch/llm/<topology>/<family>
            if len(parts) == 4 and parts[1] == "llm":
                node["topology"], node["name"] = parts[2], parts[3]
                families.append(node)
    return families, facets


def topics(root=HERE):
    return sorted(p.stem for p in (Path(root) / "topic").glob("*.md"))


def devices(root=HERE):
    """-> {device: family} from device/<family>/<device>/device.json, falling back to the built-ins."""
    found = {}
    for path in sorted(Path(root).glob("device/*/*/device.json")):
        found[path.parent.name] = path.parent.parent.name
    for name, fam in FALLBACK_DEVICES.items():
        found.setdefault(name, fam)
    return found


def families_of_devices(root=HERE):
    fams = set(FALLBACK_FAMILIES)
    fams.update(p.parent.name for p in Path(root).glob("device/*/family.json"))
    fams.update(devices(root).values())
    return sorted(fams)


# ---------------------------------------------------------------- cfg and matching

def read_gguf_meta(path):
    """Minimal stdlib GGUF header reader -> flat dict of metadata (arrays are skipped, not stored)."""
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            die(f"{path}: not a GGUF file")
        _version, _tensors, count = struct.unpack("<IQQ", f.read(20))

        def string():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def value(t, keep=True):
            if t in sizes:
                fmt = "<" + sizes[t]
                (v,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
                return v
            if t == 8:
                return string()
            if t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if et in sizes:
                    f.seek(struct.calcsize("<" + sizes[et]) * n, 1)
                else:
                    for _ in range(n):
                        value(et, False)
                return None
            die(f"{path}: unsupported GGUF value type {t}")

        meta = {}
        for _ in range(count):
            key = string()
            (t,) = struct.unpack("<I", f.read(4))
            v = value(t)
            if v is not None:
                meta[key] = v
        return meta


def cfg_from_gguf(path):
    meta = read_gguf_meta(path)
    arch = meta.get("general.architecture")
    if not arch:
        die(f"{path}: no general.architecture")
    cfg = {"arch": arch}
    heads, kv = meta.get(f"{arch}.attention.head_count"), meta.get(f"{arch}.attention.head_count_kv")
    if isinstance(kv, int):
        cfg["kv_heads"] = kv
    if isinstance(heads, int):
        cfg["heads"] = heads
    if meta.get(f"{arch}.attention.sliding_window"):
        cfg["sliding_window"] = meta[f"{arch}.attention.sliding_window"]
    if meta.get(f"{arch}.expert_count"):
        cfg["moe"] = {"experts": meta[f"{arch}.expert_count"]}
    return cfg


def load_cfg(path):
    p = Path(path)
    if p.is_dir():
        p = p / "llm_config.json"
    if not p.is_file():
        die(f"config not found: {path}")
    return read_json(p)


def truthy(v):
    return bool(v) and v != "none" and not (isinstance(v, dict) and v.get("kind") == "none")


def matches(rule, cfg):
    """A scope.json 'match' dict against an llm_config-like dict. All given keys must hold.
    arch: list; has/lacks: cfg keys; kv_heads: [lo, hi]; sliding_window: [lo, hi]; head_dim: [lo, hi];
    recurrent_kv: true -> kv_layers < layers and no shortconv_layers (gated-delta-net hybrids)."""
    if not rule:
        return False
    if "arch" in rule and cfg.get("arch") not in rule["arch"]:
        return False
    for key in rule.get("has", []):
        if not truthy(cfg.get(key)):
            return False
    for key in rule.get("lacks", []):
        if truthy(cfg.get(key)):
            return False
    for key in ("kv_heads", "sliding_window", "head_dim", "layers"):
        if key in rule:
            lo, hi = rule[key]
            v = cfg.get(key)
            if v is None or (lo is not None and v < lo) or (hi is not None and v > hi):
                return False
    if rule.get("recurrent_kv"):
        if not (cfg.get("kv_layers") and cfg.get("layers") and cfg["kv_layers"] < cfg["layers"]
                and not cfg.get("shortconv_layers")):
            return False
    return True


# ---------------------------------------------------------------- route

def family_for(families, arch):
    hits = [f for f in families if arch in f["match"].get("arch", [])]
    return hits[0] if len(hits) == 1 else None


def route(args, root=HERE):
    families, facet_nodes = load_scopes(root)
    valid_arch = sorted({a for f in families for a in f["match"].get("arch", [])})
    cfg = None
    if args.config:
        cfg = load_cfg(args.config)
    elif args.gguf:
        cfg = cfg_from_gguf(args.gguf)
    elif args.arch:
        cfg = {"arch": args.arch}
    else:
        die("give --config, --gguf or --arch")
    arch = cfg.get("arch")
    fam = family_for(families, arch)
    if fam is None:
        die(f"unknown arch {arch!r}; valid: {', '.join(valid_arch)}")

    facets = []
    for tag, node in facet_nodes.items():
        if matches(node["match"], cfg):
            facets.append(tag)
    for tag in fam.get("facets", []):
        if tag not in facets:
            facets.append(tag)
    for tag in [t for t in (args.facets or "").split(",") if t]:
        if tag not in facet_nodes:
            die(f"unknown facet {tag!r}; valid: {', '.join(sorted(facet_nodes))}")
        if tag not in facets:
            facets.append(tag)
    facets = [t for t in sorted(facet_nodes) if t in facets]

    devs = devices(root)
    sel = (args.device or "").strip()
    chosen_devices = []
    if sel:
        valid = sorted(devs) + families_of_devices(root) + ["all"]
        for item in sel.split(","):
            if item not in valid:
                die(f"unknown device {item!r}; valid: {', '.join(sorted(set(valid)))}")
        chosen = sel.split(",")
        if "all" in chosen:
            chosen_devices = sorted(devs)
            fam_list = families_of_devices(root)
        else:
            chosen_devices = [d for d in chosen if d in devs]
            fam_list = sorted({devs[d] for d in chosen_devices} | {d for d in chosen if d not in devs})
    else:
        fam_list = []

    wanted_topics = [t for t in (args.topic or "").split(",") if t]
    valid_topics = topics(root)
    for t in wanted_topics:
        if t not in valid_topics:
            die(f"unknown topic {t!r}; valid: {', '.join(valid_topics)}")

    quants = [q for q in (args.quant or "").split(",") if q]
    if not quants and isinstance(cfg.get("formats"), list):
        quants = [str(q).lower() for q in cfg["formats"]]

    files = []
    for f in fam_list:
        files.append(Path("device") / f / "README.md")
    for d in chosen_devices:
        files.append(Path("device") / devs[d] / d / "README.md")
    for q in quants:
        files.append(Path("quant") / q / "README.md")
    files += [Path("arch/llm/README.md"), Path("arch/llm") / fam["topology"] / "README.md",
              Path(fam["rel"]) / "README.md"]
    for tag in facets:
        files.append(Path("facet") / tag / "README.md")
        for extra in facet_nodes[tag].get("also", []):
            files.append(Path(extra))
    for t in wanted_topics:
        files.append(Path("topic") / f"{t}.md")

    seen = set()
    for rel in files:
        if rel in seen:
            continue
        seen.add(rel)
        if (Path(root) / rel).is_file():
            print((rel if args.relative else Path(root) / rel).as_posix())
    print(f"# arch {arch} -> {fam['rel']}; facets: {','.join(facets) or '-'}; "
          f"devices: {','.join(chosen_devices or fam_list) or '-'}; topics: {','.join(wanted_topics) or '-'}")
    footer = resolve_footer(cfg, chosen_devices or fam_list, root)
    print("# resolved choices")
    print(footer)
    return 0


def resolve_footer(cfg, devs, root):
    if not (Path(root) / "resolve.py").is_file():
        return "# (resolve.py not installed: no registered choices)"
    sys.path.insert(0, str(root))
    try:
        import resolve  # noqa: PLC0415  (lazy: owned by the registry)
        out = resolve.resolve(cfg, devs)
        return out if isinstance(out, str) else json.dumps(out, indent=1, sort_keys=True, default=str)
    except Exception as exc:  # the router must not break on a registry error
        return f"# resolve.py failed: {type(exc).__name__}: {exc}"
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------- lint

def md_files(repo):
    skills = Path(repo) / "skills"
    out = []
    for p in sorted(skills.rglob("*.md")):
        s = p.as_posix()
        if any(part in s for part in FIXTURE_PARTS) or "/node_modules/" in s:
            continue
        out.append(p)
    return out


def strip_fences(text):
    """-> list of (lineno, line) outside fenced code blocks."""
    out, fenced = [], False
    for i, line in enumerate(text.splitlines(), 1):
        if re.match(r"\s*(```|~~~)", line):
            fenced = not fenced
            continue
        if not fenced:
            out.append((i, line))
    return out


def paragraphs(path):
    """-> list of (lineno, normalized text) for paragraphs outside fences (blank-line separated)."""
    res, cur, start = [], [], None
    for i, line in strip_fences(path.read_text(encoding="utf-8", errors="replace")):
        if line.strip():
            if not cur:
                start = i
            cur.append(line.strip())
        elif cur:
            res.append((start, " ".join(cur)))
            cur = []
    if cur:
        res.append((start, " ".join(cur)))
    return [(n, re.sub(r"\s+", " ", t.lower())) for n, t in res]


def dup_groups(repo, allow=()):
    seen = {}
    for p in md_files(repo):
        for n, t in paragraphs(p):
            if len(t) < DUP_MIN_CHARS or t.startswith("|") or t.startswith("#"):
                continue
            key = hashlib.sha1(t.encode()).hexdigest()[:12]
            seen.setdefault(key, []).append((p, n))
    return {k: v for k, v in seen.items() if len(v) > 1 and k not in allow}


SENT_MIN_CHARS = 100


def dup_sentences(repo, min_chars=None):
    """Sentences (>= SENT_MIN_CHARS, normalized) that occur in two or more different files: repeated facts."""
    seen = {}
    for p in md_files(repo):
        text = " ".join(t for _, t in paragraphs(p))
        for s in re.split(r"(?<=[.;!?])\s+", text):
            if len(s) >= (min_chars or SENT_MIN_CHARS) and not s.startswith("|"):
                key = hashlib.sha1(s.encode()).hexdigest()[:12]
                seen.setdefault(key, set()).add(Path(p).relative_to(repo).as_posix())
    return {k: v for k, v in seen.items() if len(v) > 1}


def load_allow(repo):
    path = Path(repo) / "skills/loom-kb/lint_allow.json"
    if path.is_file():
        return set(read_json(path).get("duplicates", {}))
    return set()


def frontmatter(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    meta = {}
    key = None
    for line in m.group(1).splitlines():
        km = re.match(r"([A-Za-z_-]+):\s*(.*)", line)
        if km:
            key = km.group(1)
            meta[key] = km.group(2).strip().strip(">|").strip()
        elif key and line.strip():
            meta[key] += " " + line.strip()
    return meta


LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def links(path):
    for n, line in strip_fences(path.read_text(encoding="utf-8", errors="replace")):
        line = re.sub(r"`[^`]*`", "", line)  # code spans are not links
        for m in LINK.finditer(line):
            yield n, m.group(1)


def fleet_arches(repo):
    manifest = Path(repo) / "fleet/manifest.json"
    arches = {}
    if manifest.is_file():
        data = read_json(manifest)
        models = data.get("models", data) if isinstance(data, dict) else data
        items = models.items() if isinstance(models, dict) else [(m.get("slug"), m) for m in models]
        for slug, m in items:
            cfgp = Path(repo) / "fleet" / str(slug) / "llm_config.json"
            if cfgp.is_file():
                arches.setdefault(read_json(cfgp).get("arch"), []).append(slug)
        return arches, "fleet/manifest.json"
    pkgs = os.environ.get("LOOM_KB_FLEET_PKGS", "/home/jfowers/lt/r9700/pkgs")
    for p in sorted(glob.glob(os.path.join(pkgs, "*", "llm_config.json"))):
        arches.setdefault(read_json(p).get("arch"), []).append(Path(p).parent.name)
    return arches, pkgs if arches else None


def lint(repo, allow=None):
    repo = Path(repo)
    kb = repo / "skills/loom-kb"
    errors, notes = [], []
    rel = lambda p: Path(p).relative_to(repo).as_posix()  # noqa: E731

    # SKILL.md limits
    for skill in sorted((repo / "skills").glob("*/SKILL.md")):
        n = len(skill.read_text(encoding="utf-8").splitlines())
        if n > SKILL_MAX:
            errors.append(f"{rel(skill)}: {n} lines (> {SKILL_MAX})")
        meta = frontmatter(skill)
        if meta is None:
            errors.append(f"{rel(skill)}: missing frontmatter")
            continue
        name, desc = meta.get("name", ""), meta.get("description", "")
        if not re.fullmatch(r"[a-z0-9-]{1,64}", name):
            errors.append(f"{rel(skill)}: name {name!r} must be 1-64 chars of [a-z0-9-]")
        if name != skill.parent.name:
            errors.append(f"{rel(skill)}: name {name!r} != directory {skill.parent.name!r}")
        if not desc or len(desc) > 1024:
            errors.append(f"{rel(skill)}: description length {len(desc)} (must be 1..1024)")

    # node files: size, TOC, no node-to-node links
    nodes = [p for p in sorted(kb.rglob("*.md")) if p.name != "SKILL.md"] if kb.is_dir() else []
    nodeset = {p.resolve() for p in nodes}
    for p in nodes:
        text = p.read_text(encoding="utf-8")
        n = len(text.splitlines())
        if n > README_MAX:
            errors.append(f"{rel(p)}: {n} lines (> {README_MAX}); split it by scope")
        if n > TOC_ABOVE and not re.search(r"^#{1,3}\s*(table of )?contents\b", text, re.I | re.M):
            errors.append(f"{rel(p)}: {n} lines (> {TOC_ABOVE}) and no '## Contents' table of contents")
        for ln, target in links(p):
            t = target.split("#")[0]
            if t and not re.match(r"[a-z]+:", t) and (p.parent / t).resolve() in nodeset and (p.parent / t).resolve() != p.resolve():
                errors.append(f"{rel(p)}:{ln}: links to node {t} (one level only: the router `kb.py route` is the TOC)")

    # every relative link resolves
    for p in md_files(repo):
        for ln, target in links(p):
            t = target.split("#")[0]
            if not t or re.match(r"[a-z][a-z0-9+.-]*:", t) or t.startswith("#") or "${" in t or t.startswith("<"):
                continue
            if not (p.parent / t).exists():
                errors.append(f"{rel(p)}:{ln}: broken link {target}")

    # duplicate paragraphs
    allow = load_allow(repo) if allow is None else allow
    for key, locs in sorted(dup_groups(repo, allow).items()):
        where = "; ".join(f"{rel(p)}:{n}" for p, n in locs)
        errors.append(f"duplicate paragraph sha1:{key} (>= {DUP_MIN_CHARS} chars) at {where} (move it up to one node, "
                      f"or allowlist the key in skills/loom-kb/lint_allow.json)")

    # scope nodes
    if kb.is_dir():
        families, facets = load_scopes(kb)
        owner_of = {}
        for node in list(families) + list(facets.values()):
            d, r = node["dir"], node["rel"]
            for key in ("scope", "match", "owner", "choices"):
                if key not in node:
                    errors.append(f"skills/loom-kb/{r}/scope.json: missing key {key!r}")
            if node.get("scope") != r:
                errors.append(f"skills/loom-kb/{r}/scope.json: scope {node.get('scope')!r} != its path {r!r}")
            if not (d / "README.md").is_file():
                errors.append(f"skills/loom-kb/{r}: scope.json without README.md")
            for tag in node.get("facets", []):
                if tag not in facets:
                    errors.append(f"skills/loom-kb/{r}/scope.json: unknown facet {tag!r}")
            for extra in node.get("also", []):
                if not (kb / extra).is_file():
                    errors.append(f"skills/loom-kb/{r}/scope.json: 'also' file {extra} does not exist")
            for a in node.get("match", {}).get("arch", []) if r.startswith("arch/") else []:
                owner_of.setdefault(a, []).append(r)
        for a, rs in sorted(owner_of.items()):
            if len(rs) > 1:
                errors.append(f"arch {a!r} is claimed by {len(rs)} family nodes: {', '.join(rs)}")
        arches, source = fleet_arches(repo)
        if source is None:
            notes.append("fleet arch check skipped: no fleet/manifest.json and no LOOM_KB_FLEET_PKGS packages")
        else:
            for a, slugs in sorted(arches.items(), key=lambda kv: str(kv[0])):
                got = owner_of.get(a, [])
                if len(got) != 1:
                    errors.append(f"fleet arch {a!r} ({len(slugs)} models, e.g. {slugs[0]}) maps to {len(got)} family nodes "
                                  f"(need exactly 1); create skills/loom-kb/arch/llm/<topology>/<family>/")
    return errors, notes


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="kb.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("route", help="print the node files to read")
    r.add_argument("--config", help="llm_config.json (or a package dir containing it)")
    r.add_argument("--gguf", help="a GGUF file (arch and shape read from its header)")
    r.add_argument("--arch", help="an arch name (cfg arch or GGUF general.architecture)")
    r.add_argument("--facets", help="extra facets, comma separated")
    r.add_argument("--device", help="strix-halo | radeon-r9700 | rdna3 | rdna4 | all (comma separated)")
    r.add_argument("--quant", help="quant formats, comma separated (default: from the config)")
    r.add_argument("--topic", help="topics, comma separated (see `kb.py list`)")
    r.add_argument("--relative", action="store_true", help="print paths relative to the skill directory")
    lt = sub.add_parser("lint", help="structure + duplicate lint")
    lt.add_argument("--root", default=str(SKILLS.parent), help="repo root to lint (default: this repo)")
    dp = sub.add_parser("dupes", help="duplicate paragraph count (any checkout)")
    dp.add_argument("--root", default=str(SKILLS.parent))
    dp.add_argument("--no-allow", action="store_true", help="ignore lint_allow.json (raw count)")
    dp.add_argument("--sentences", action="store_true", help="also count repeated sentences across files")
    dp.add_argument("--min-sentence", type=int, default=SENT_MIN_CHARS, help="sentence length threshold (default 100)")
    sub.add_parser("list", help="valid devices, families, facets, topics")
    args = ap.parse_args(argv)

    if args.cmd == "route":
        return route(args)
    if args.cmd == "list":
        families, facets = load_scopes()
        print("devices:", ", ".join(sorted(devices())), "| families:", ", ".join(families_of_devices()))
        print("facets:", ", ".join(sorted(facets)))
        print("topics:", ", ".join(topics()))
        for f in sorted(families, key=lambda f: f["rel"]):
            print(f"{f['rel']}: {','.join(f['match'].get('arch', []))}")
        return 0
    if args.cmd == "dupes":
        groups = dup_groups(args.root, set() if args.no_allow else load_allow(args.root))
        for key, locs in sorted(groups.items()):
            print(f"sha1:{key} x{len(locs)}: " + "; ".join(f"{Path(p).relative_to(args.root)}:{n}" for p, n in locs))
        print(f"{len(groups)} duplicated paragraphs (>= {DUP_MIN_CHARS} chars)")
        if args.sentences:
            sent = dup_sentences(args.root, args.min_sentence)
            for key, locs in sorted(sent.items()):
                print(f"sentence sha1:{key} x{len(locs)}: " + "; ".join(sorted(locs)))
            print(f"{len(sent)} sentences (>= {args.min_sentence} chars) repeated in 2+ files")
        return 0
    errors, notes = lint(args.root)
    for n in notes:
        print(f"note: {n}")
    for e in errors:
        print(f"error: {e}")
    print(f"kb lint: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
