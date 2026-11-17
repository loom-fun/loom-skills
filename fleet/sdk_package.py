"""The v0.2 package of one model for loom-fun-staging (regen.py package): the published staging repo with its programs replaced by the canonical build.

Pure file work, no network: `assemble(snapshot, canon, out)` turns a snapshot of the staging repo (small files, never weights) and the programs-only
build of the base (build_fleet.py --with-gen) into the package directory publish_package.py uploads. What is replaced: program/**, the package-root
*.loom (weights policies), gen/**, llm_config.json, model.json. What stays: tokenizer.json, model.json.lock, README.md / info.json / REPORT.md (the old
card until `ingest` replaces it), everything else of the snapshot.

model.json is the canonical build's, with what publish_package.py prepare writes carried over from the published one (the `origins` table and each
parameter file's `origin`). It is checked against the published model.json first: only what the migration explains may differ (see model_changes).
"""
import copy
import json
import os
import shutil

PROGRAM_FILES = ("program", "gen")          # directories replaced as a whole
CARD_ROOT = ("README.md", "info.json", "REPORT.md")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _files(m):
    """every parameter file entry of a model.json: (variant, service, parameter, index, entry)"""
    for vn, v in (m.get("variants") or {}).items():
        for sn, s in (v.get("services") or {}).items():
            for pn, p in (s.get("parameters") or {}).items():
                for i, e in enumerate(p.get("files") or []):
                    yield vn, sn, pn, i, e


def normalize(m, alt_presets=()):
    """model.json without what a package rebuild does not decide: execution.pool_capacity (v0.2 rejects it), the origins (publish_package.py
    prepare), each parameter file's origin, and the per-GPU presets make_info.py build --fit writes (alt_presets)."""
    m = copy.deepcopy(m)
    m.pop("origins", None)
    for _, _, _, _, e in _files(m):
        e.pop("origin", None)
    pre = m.get("presets") or {}
    for k in list(pre):
        if k in alt_presets:
            del pre[k]
    for p in pre.values():
        (p.get("execution") or {}).pop("pool_capacity", None)
    return m


def model_changes(pub, new, alt_presets=()):
    """[(kind, path)] of what differs between the published model.json and the canonical one after normalize(); kind is added (a field the
    canonical build has and the published card lacks: a v0.2 field), removed or changed."""
    out = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a:
                    out.append(("added", f"{path}.{k}" if path else k))
                elif k not in b:
                    out.append(("removed", f"{path}.{k}" if path else k))
                else:
                    walk(a[k], b[k], f"{path}.{k}" if path else k)
        elif isinstance(a, list) and isinstance(b, list):
            if a == b:
                return
            if all(x in b for x in a):   # list entries the canonical build adds (e.g. llm_config.json among the package files)
                out.append(("added", f"{path}[+{len(b) - len(a)}]"))
            else:
                out.append(("changed", path))
        elif a != b:
            out.append(("changed", path))
    walk(normalize(pub, alt_presets), normalize(new, alt_presets), "")
    return out


def carry_over(pub, new):
    """the canonical model.json with the published origins table and parameter file origins; ValueError for a parameter file the published
    card does not know (no origin to fetch it from: the package needs publish_package.py prepare)"""
    m = copy.deepcopy(new)
    if "origins" in pub:
        m["origins"] = copy.deepcopy(pub["origins"])
    old = {(vn, sn, pn, e.get("path")): e for vn, sn, pn, _, e in _files(pub)}
    for vn, sn, pn, _, e in _files(m):
        k = (vn, sn, pn, e.get("path"))
        if k not in old or "origin" not in old[k]:
            raise ValueError(f"parameter file {e.get('path')} ({vn}/{sn}/{pn}) has no origin in the published model.json")
        e["origin"] = old[k]["origin"]
    return m


def assemble(snap, canon, out, alt_presets=(), prefill_changes=None):
    """Snapshot + canonical build -> OUT. A measured prefill transition may name exact old/new values per preset;
    all other removed/changed fields remain errors. The caller validates measured execution before using that option."""
    pub, new = load(os.path.join(snap, "model.json")), load(os.path.join(canon, "model.json"))
    changes = model_changes(pub, new, alt_presets)
    approved = set()
    for preset, (old, new_value) in (prefill_changes or {}).items():
        before = pub.get("presets", {}).get(preset, {}).get("execution", {}).get("prefill_capacity")
        after = new.get("presets", {}).get(preset, {}).get("execution", {}).get("prefill_capacity")
        path = f"presets.{preset}.execution.prefill_capacity"
        if (type(old) is not int or type(new_value) is not int or min(old, new_value) < 1
                or old == new_value or (before, after) != (old, new_value)
                or ("changed", path) not in changes):
            raise ValueError(f"prefill transition {preset}: expected {old}->{new_value}, found {before}->{after}")
        approved.add(("changed", path))
    bad = [c for c in changes if c[0] != "added" and c not in approved]
    if bad:
        raise ValueError("model.json differs from the published one beyond the migration: " + ", ".join(f"{k} {p}" for k, p in bad))
    model = carry_over(pub, new)
    for k in alt_presets:   # the per-GPU presets the card step wrote (make_info.py build --fit / device files) stay: they are the card's, not the build's
        if k in (pub.get("presets") or {}) and k not in (model.get("presets") or {}):
            model.setdefault("presets", {})[k] = copy.deepcopy(pub["presets"][k])
            (model["presets"][k].get("execution") or {}).pop("pool_capacity", None)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(snap, out, ignore=shutil.ignore_patterns(".cache"))
    for d in PROGRAM_FILES:
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)
        if os.path.isdir(os.path.join(canon, d)):
            shutil.copytree(os.path.join(canon, d), os.path.join(out, d))
    for f in os.listdir(out):   # package-root *.loom: the weights policies
        if f.endswith(".loom"):
            os.remove(os.path.join(out, f))
    for f in os.listdir(canon):
        if f.endswith(".loom"):
            shutil.copy2(os.path.join(canon, f), os.path.join(out, f))
    shutil.copy2(os.path.join(canon, "llm_config.json"), os.path.join(out, "llm_config.json"))
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
        f.write("\n")
    return changes


def same_files(a, b, skip=("README.md", "info.json", "REPORT.md", ".gitattributes")):
    """True when the two package dirs hold the same files with the same bytes, card files aside: an equal programs hash is not enough (a package
    re-pinned to another SDK changes llm_config.json only)"""
    def files(d):
        out = {}
        for root, dirs, fs in os.walk(d):
            dirs[:] = [x for x in dirs if x != ".cache"]
            for f in fs:
                rel = os.path.relpath(os.path.join(root, f), d).replace(os.sep, "/")
                if rel not in skip:
                    out[rel] = os.path.join(root, f)
        return out
    fa, fb = files(a), files(b)
    if set(fa) != set(fb):
        return False
    return all(open(fa[k], "rb").read() == open(fb[k], "rb").read() for k in fa)
