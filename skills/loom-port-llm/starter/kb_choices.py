"""Resolves a model's generator choices from the knowledge base (skills/loom-kb: choices.py, resolve.py, scope.json nodes) and
puts them into the environment the generators read.

build.py calls apply() once per build, before gen_llm.py starts; make_kernel_tests.py calls it too, so the tests exercise the
variants the model launches. With no scope.json setting anything for the model nothing is exported and every program is as
the generators' own defaults make it.

  apply(cfg, draft_kind, environ=os.environ, kit=None, devices=None)
      resolves, exports into ENVIRON every choice whose environment spelling differs from the model's default, and returns the
      block build.py records in the package's llm_config.json["resolved"]:
        {"scopes", "choices": {key: {"value", "from", "evidence"}}, "devices", "facets", "env": {NAME: value}, "dirty": [keys]}
      A registered variable already set by the caller wins over the resolution and, when it differs from what the resolution
      would have produced, is listed in "dirty": that build is not reproducible from the knowledge base, and publish_package.py
      refuses it unless --allow-dirty.

Where the knowledge base is: $LOOM_KB_ROOT, else kb_root of the kit's KIT_SOURCE (kit_source.py records it when a kit is made),
else ../../loom-kb next to this file (the starter used in place, or the installed skills). Standard library only.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = ("0", "off", "no", "false")


def find_kb_root(kit=HERE):
    cands = [os.environ.get("LOOM_KB_ROOT")]
    src = os.path.join(kit, "KIT_SOURCE")
    if os.path.isfile(src):
        try:
            cands.append(json.load(open(src, encoding="utf-8")).get("kb_root"))
        except (OSError, ValueError):
            pass
    cands.append(os.path.join(HERE, "..", "..", "loom-kb"))
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "resolve.py")):
            return os.path.abspath(c)
    return None


def load_resolver(kb_root):
    spec = importlib.util.spec_from_file_location("loom_kb_resolve", os.path.join(kb_root, "resolve.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def kit_example(kit=HERE):
    """The reference kit example (gemma4, lfm2moe, ...) a kit was made from (KIT_SOURCE), else None."""
    try:
        return json.load(open(os.path.join(kit, "KIT_SOURCE"), encoding="utf-8")).get("example")
    except (OSError, ValueError):
        return None


def apply(cfg, draft_kind, environ=None, kit=None, devices=None, kit_dir=HERE):
    environ = os.environ if environ is None else environ
    root = find_kb_root(kit_dir)
    if root is None:
        sys.exit("loom-kb not found: set LOOM_KB_ROOT to skills/loom-kb of the loom-skills checkout (the kit was copied away from it)")
    rs = load_resolver(root)
    ch = rs.choices_module(root)
    kit = kit if kit is not None else kit_example(kit_dir)
    if devices is None:
        devices = ("rdna3",) if environ.get("LOOM_RDNA4", "1").strip().lower() in OFF else ("rdna3", "rdna4")
    try:
        res = rs.resolve_full(cfg, devices, root, draft_kind, kit)
    except rs.ResolveError as e:
        sys.exit(f"choices: {e}")
    dirty = []
    for env, c in ch.BY_ENV.items():
        if env not in environ or (c.kits is not None and kit not in c.kits):
            continue
        want = c.to_env(res["values"][c.key])
        if environ[env] != (want if want is not None else ""):
            dirty.append(c.key)
    for env, val in res["env"].items():
        if env in environ:
            continue   # the caller's value wins (and is listed in dirty when it differs)
        if val is None:
            environ.pop(env, None)
        else:
            environ[env] = val
    block = rs.resolved_block(res)
    block["dirty"] = sorted(dirty)
    return block


def apply_for_tests(cfg, environ=None):
    """make_kernel_tests.py: the environment the kernels under test were generated with. A built package's llm_config.json carries
    it (resolved.env); a cfg.json before the build is resolved with the draft of cfg["draft"] or LLM_DRAFT (none: not drafted)."""
    environ = os.environ if environ is None else environ
    rec = cfg.get("resolved")
    if rec is not None:
        for env, val in (rec.get("env") or {}).items():
            if env not in environ:
                if val is None:
                    continue
                environ[env] = val
        return rec
    draft = cfg.get("draft") or json.loads(environ.get("LLM_DRAFT") or "null") or {}
    return apply(cfg, draft.get("kind") or None, environ)


def exempt_block(reason):
    """The resolved block of a build that runs no generator (catalog packages)."""
    return {"exempt": reason, "scopes": [], "choices": {}, "devices": [], "facets": [], "env": {}, "dirty": []}
