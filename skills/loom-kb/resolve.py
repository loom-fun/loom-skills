#!/usr/bin/env python3
"""Choice resolver: which value of every registered choice (choices.py) applies to one model on one set of devices.

  resolve(cfg, devices=("rdna3", "rdna4"), kb_root=None, draft=AUTO, kit=None) -> {
      "scopes":  [scope path of every scope.json node that matches, in precedence order],
      "choices": {key: {"value": v, "from": scope path | "model", "evidence": "..."}},   (only keys a scope or the model set)
      "devices": [device families of the build], "facets": [...], "env": {ENV: "value"}}   (env: what differs from the default)

Stdlib only. `python resolve.py --config llm_config.json [--device strix-halo ...] [--all] [--lint]` prints the JSON.

Scope nodes. Every `scope.json` under the KB root (skills/loom-kb/**) is a selector node:
    {"scope": "arch/llm/moe/qwen35moe",                 (the node's directory under the KB root)
     "match": {"arch": ["qwen35moe"], "facets": ["drafted"], "kv_heads": [1, 2], "head_dim": 256, "quant": ["q4_k"],
               "devices": ["rdna4"]},                    (all keys must hold; every key optional)
     "owner": "...",
     "choices": [{"when": {"facets": ["drafted"], "devices": ["rdna4"]},   (optional, narrows the node's match)
                  "set": {"decode.attn.kv_once.d256": false}, "evidence": "run/PR link and numbers"}]}
  match / when keys:  arch (GGUF arch, one or a list)  facets (all of)  not_facets (none of)  has / lacks (config keys that are / are
                      not set)  kv_heads, head_dim, heads, layers, sliding_window (a number, or [lo, hi] inclusive, null: open)
                      recurrent_kv (gated-delta-net hybrids)  quant (any tensor format of the model is one of)  devices (any of
                      the build's device families or device slugs). kb.py's matcher reads the same keys.
  An arch node may list `facets` its family implies (gemma3: sliding-window); a facet node (facet/<tag>) is on for a model its
  `match` holds for; built-in facets come from the config (derive_facets): drafted, draft-<kind>, mtp, vision, hybrid-gdn, moe,
  sliding-window, kv-le2, sinks, gated-attn, shortconv. `has: ["draft"]` sees the draft the build uses.

Precedence, later wins: registry default < device family (device/<fam>) < device (device/<fam>/<dev>) < quant (quant/<fmt>)
< arch root->leaf (arch/...) < facet (facet/<tag>) < model (cfg["choices"]). Two matching nodes at the same level (same
depth for arch) setting one key to different values for one model is an error (ResolveError).

Devices: a choice that cannot differ between the two targets of one dual-target build (not per_target) may only be set by
a rule whose devices cover every device family the choice affects; otherwise ResolveError.

Exports (`env`): each resolved choice that a scope or the model set, whose environment spelling differs from the model's
default (the registry default, the kit's, or the cfg-derived one), is exported as its environment variable for the
generators (gen_llm.py and its subprocesses read env; null: the variable is unset). With no scope.json setting anything the dict is empty and every
program is as before.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO = object()
LEVELS = {"device_family": 1, "device": 2, "quant": 3, "arch": 4, "facet": 5, "model": 6}
RANGE_KEYS = ("kv_heads", "head_dim", "heads", "layers", "sliding_window")
MATCH_KEYS = {"arch", "facets", "not_facets", "has", "lacks", "recurrent_kv", "quant", "devices", *RANGE_KEYS}
NODE_KEYS = ("scope", "match", "owner", "choices", "doc", "facets", "also")
FALLBACK_FAMILY = {"strix-halo": "rdna3", "gfx1151": "rdna3", "gfx1100": "rdna3", "radeon-r9700": "rdna4", "gfx1201": "rdna4"}


class ResolveError(Exception):
    pass


def _load_choices(kb_root):
    path = os.path.join(kb_root, "choices.py")
    if not os.path.isfile(path):
        path = os.path.join(HERE, "choices.py")
    spec = importlib.util.spec_from_file_location("loom_kb_choices_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CACHE = {}


def choices_module(kb_root=None):
    kb_root = os.path.abspath(kb_root or HERE)
    if kb_root not in _CACHE:
        _CACHE[kb_root] = _load_choices(kb_root)
    return _CACHE[kb_root]


# --- facets and matching ------------------------------------------------------------------------------------------------
def auto_draft_kind(cfg):
    """The speculative-decoding kind build.py picks with --draft auto for CFG (mirror of build.py): the optimizer's measured
    choice drafter.kind, else the GGUF's MTP block, else a DFlash drafter, else a Gemma-4 assistant, else none."""
    d = cfg.get("drafter") or {}
    return d.get("kind") or ("mtp" if cfg.get("mtp") else ("dflash" if d.get("gguf") else
                                                          ("assist" if (cfg.get("assistant") or {}).get("gguf") else None))) or None


def draft_kind(cfg, draft=AUTO):
    """DRAFT: AUTO (the package's stamped draft block if cfg has one, else auto_draft_kind), a kind string, a draft dict, or None."""
    if draft is AUTO:
        if "draft" in cfg:   # a built package's llm_config.json: the draft that was built
            draft = cfg["draft"]
        else:
            return auto_draft_kind(cfg)
    if isinstance(draft, dict):
        return draft.get("kind") or None
    return draft if draft and draft != "none" else None


def derive_facets(cfg, kind):
    f = set()
    if kind:
        f |= {"drafted", f"draft-{kind}"}
    if cfg.get("mtp"):
        f.add("mtp")
    if cfg.get("vision"):
        f.add("vision")
    if cfg.get("recurrent_layers"):
        f.add("hybrid-gdn")
    if cfg.get("moe"):
        f.add("moe")
    if (cfg.get("sliding_window") or 0) >= 1:
        f.add("sliding-window")
    if cfg.get("kv_heads") is not None and cfg["kv_heads"] <= 2:
        f.add("kv-le2")
    if cfg.get("attn_sinks"):
        f.add("sinks")
    if cfg.get("attn_gate"):
        f.add("gated-attn")
    if cfg.get("shortconv_layers"):
        f.add("shortconv")
    return sorted(f)


def quant_formats(cfg):
    return sorted({str(t.get("type", "")).lower() for t in (cfg.get("tensors") or {}).values() if isinstance(t, dict)} - {""})


class Env:
    """What a match is evaluated against."""

    def __init__(self, cfg, facets, devices, family_of):
        self.cfg, self.facets, self.devices, self.family_of = cfg, set(facets), devices, family_of
        self.formats = set(quant_formats(cfg))
        self.names = set(devices) | {family_of(d) for d in devices}
        self.families = {family_of(d) for d in devices}


def _num(spec, v):
    if v is None:
        return False
    if isinstance(spec, list):
        return len(spec) == 2 and (spec[0] is None or v >= spec[0]) and (spec[1] is None or v <= spec[1])
    return v == spec


def truthy(v):
    return bool(v) and v != "none" and not (isinstance(v, dict) and v.get("kind") == "none")


def match(m, env, where="match"):
    for k in m:
        if k not in MATCH_KEYS:
            raise ResolveError(f"{where}: unknown match key {k!r} (valid: {sorted(MATCH_KEYS)})")
    cfg = env.cfg
    if "arch" in m:
        a = m["arch"] if isinstance(m["arch"], list) else [m["arch"]]
        if cfg.get("arch") not in a:
            return False
    if not set(m.get("facets", [])) <= env.facets or env.facets & set(m.get("not_facets", [])):
        return False
    for k in RANGE_KEYS:
        if k in m and not _num(m[k], cfg.get(k)):
            return False
    if not all(truthy(cfg.get(k)) for k in m.get("has", [])) or any(truthy(cfg.get(k)) for k in m.get("lacks", [])):
        return False
    if m.get("recurrent_kv") and not (cfg.get("kv_layers") and cfg.get("layers") and cfg["kv_layers"] < cfg["layers"]
                                      and not cfg.get("shortconv_layers")):
        return False
    if "quant" in m and not (env.formats & {q.lower() for q in (m["quant"] if isinstance(m["quant"], list) else [m["quant"]])}):
        return False
    if "devices" in m and not (env.names & set(m["devices"])):
        return False
    return True


# --- scope nodes ---------------------------------------------------------------------------------------------------------
def level_of(scope):
    """(level, depth) of a scope path; ResolveError for a path outside the selector axes."""
    p = scope.split("/")
    if p[0] == "device" and len(p) == 2:
        return LEVELS["device_family"], 0
    if p[0] == "device" and len(p) == 3:
        return LEVELS["device"], 0
    if p[0] == "quant" and len(p) == 2:
        return LEVELS["quant"], 0
    if p[0] == "arch" and len(p) >= 2:
        return LEVELS["arch"], len(p)
    if p[0] == "facet" and len(p) == 2:
        return LEVELS["facet"], 0
    raise ResolveError(f"scope {scope!r}: must be device/<family>[/<device>], quant/<fmt>, arch/<...> or facet/<tag>")


def load_scopes(kb_root=None):
    """[(scope path, node dict)] for every scope.json under the KB root (sorted)."""
    kb_root = os.path.abspath(kb_root or HERE)
    out = []
    for dp, dn, fn in os.walk(kb_root):
        dn[:] = sorted(d for d in dn if d not in ("__pycache__",) and not d.startswith("."))
        if "scope.json" in fn:
            rel = os.path.relpath(dp, kb_root).replace(os.sep, "/")
            try:
                with open(os.path.join(dp, "scope.json"), encoding="utf-8") as f:
                    node = json.load(f)
            except ValueError as e:
                raise ResolveError(f"{rel}/scope.json: not JSON ({e})")
            if node.get("scope") != rel:
                raise ResolveError(f"{rel}/scope.json: \"scope\" must be {rel!r}, is {node.get('scope')!r}")
            level_of(rel)
            out.append((rel, node))
    return sorted(out)


def device_index(kb_root=None):
    """{device slug: family} from device/<family>/<device>/ directories, plus the built-in names."""
    kb_root = os.path.abspath(kb_root or HERE)
    idx = dict(FALLBACK_FAMILY)
    base = os.path.join(kb_root, "device")
    if os.path.isdir(base):
        for fam in sorted(os.listdir(base)):
            if os.path.isdir(os.path.join(base, fam)):
                idx[fam] = fam
                for dev in sorted(os.listdir(os.path.join(base, fam))):
                    if os.path.isdir(os.path.join(base, fam, dev)):
                        idx[dev] = fam
    idx.setdefault("rdna3", "rdna3")
    idx.setdefault("rdna4", "rdna4")
    return idx


def validate_node(scope, node, ch):
    """Schema errors of one scope node (list of strings)."""
    errs = []
    for k in node:
        if k not in NODE_KEYS:
            errs.append(f"{scope}: unknown key {k!r}")
    try:
        match(node.get("match") or {}, Env({}, [], ["rdna3"], lambda d: d), f"{scope} match")
    except ResolveError as e:
        errs.append(str(e))
    if not isinstance(node.get("choices", []), list):
        return errs + [f"{scope}: choices must be a list"]
    for i, ent in enumerate(node.get("choices", [])):
        w = f"{scope} choices[{i}]"
        if not isinstance(ent, dict) or not isinstance(ent.get("set"), dict) or not ent["set"]:
            errs.append(f"{w}: needs a non-empty \"set\" object")
            continue
        if not str(ent.get("evidence", "")).strip():
            errs.append(f"{w}: needs \"evidence\" (the run, PR or measurement that justifies it)")
        try:
            match(ent.get("when") or {}, Env({}, [], ["rdna3"], lambda d: d), f"{w} when")
        except ResolveError as e:
            errs.append(str(e))
        for k, v in ent["set"].items():
            c = ch.CHOICES.get(k)
            if c is None:
                errs.append(f"{w}: unknown choice {k!r}")
                continue
            try:
                c.check(v)
            except ValueError as e:
                errs.append(f"{w}: {e}")
    return errs


def lint(kb_root=None, cfgs=None, devices_list=(("rdna3", "rdna4"),)):
    """Errors of every scope.json (schema, known keys, types, evidence) and, for each cfg of CFGS, the resolution conflicts."""
    kb_root = os.path.abspath(kb_root or HERE)
    ch = choices_module(kb_root)
    errs = []
    try:
        nodes = load_scopes(kb_root)
    except ResolveError as e:
        return [str(e)]
    for scope, node in nodes:
        errs += validate_node(scope, node, ch)
    if errs:
        return errs
    for name, cfg in (cfgs or {}).items():
        for devs in devices_list:
            try:
                resolve(cfg, devs, kb_root)
            except ResolveError as e:
                errs.append(f"{name} {list(devs)}: {e}")
    return errs


# --- resolution ----------------------------------------------------------------------------------------------------------
def _check_devices(c, when_devices, build_families, family_of, where):
    """A rule restricted to some devices must be expressible in one dual-target build."""
    if not when_devices:
        return
    covered = {family_of(d) for d in when_devices}
    affected = set(c.devices) if c.devices else set(build_families)
    uncovered = (affected & set(build_families)) - covered
    if uncovered and not c.per_target:
        raise ResolveError(f"{where}: {c.key} cannot differ per device (it changes programs shared by {sorted(uncovered)}); "
                           f"set it for the whole build or make it a twin (per_target)")


def resolve(cfg, devices=("rdna3", "rdna4"), kb_root=None, draft=AUTO, kit=None):
    """The resolution of CFG (see the module docstring); resolve_full() adds the value of every registered choice."""
    r = resolve_full(cfg, devices, kb_root, draft, kit)
    r.pop("values")
    return r


def resolve_full(cfg, devices=("rdna3", "rdna4"), kb_root=None, draft=AUTO, kit=None):
    kb_root = os.path.abspath(kb_root or HERE)
    ch = choices_module(kb_root)
    idx = device_index(kb_root)
    family_of = lambda d: idx.get(d, d)
    devices = list(devices) or ["rdna3", "rdna4"]
    kind = draft_kind(cfg, draft)
    cfg = dict(cfg, draft={"kind": kind} if kind else None)   # what `has: ["draft"]` of a facet node sees: the draft this build uses
    nodes = load_scopes(kb_root)
    facets = set(derive_facets(cfg, kind))
    for _ in range(4):   # facet nodes switch on by their own match; arch nodes imply facets (gemma3: sliding-window)
        before = len(facets)
        e = Env(cfg, facets, devices, family_of)
        for scope, node in nodes:
            lvl = level_of(scope)[0]
            if lvl == LEVELS["arch"] and node.get("facets") and match(node.get("match") or {}, e):
                facets |= set(node["facets"])
            elif lvl == LEVELS["facet"] and node.get("match") and match(node["match"], e):
                facets.add(scope.split("/")[1])
        if len(facets) == before:
            break
    facets = sorted(facets)
    env = Env(cfg, facets, devices, family_of)
    matched, contrib = [], {}
    for scope, node in nodes:
        errs = validate_node(scope, node, ch)
        if errs:
            raise ResolveError("; ".join(errs))
        lvl, depth = level_of(scope)
        parts = scope.split("/")
        m = dict(node.get("match") or {})
        if lvl in (LEVELS["device_family"], LEVELS["device"]):   # the directory is the selector
            if not (env.names & {parts[-1]}):
                continue
        if not match(m, env, f"{scope} match"):
            continue
        matched.append(((lvl, depth), scope))
        for i, ent in enumerate(node.get("choices", [])):
            when = ent.get("when") or {}
            if not match(when, env, f"{scope} when"):
                continue
            for k, v in ent["set"].items():
                c = ch.CHOICES[k]
                where = f"{scope} choices[{i}]"
                node_devs = [parts[-1]] if lvl in (LEVELS["device_family"], LEVELS["device"]) else []
                rule_devs = list(when.get("devices", [])) or node_devs or list(m.get("devices", []))
                _check_devices(c, rule_devs, env.families, family_of, where)
                contrib.setdefault(k, []).append(((lvl, depth), scope, c.check(v), ent.get("evidence", "")))
    for k, v in (cfg.get("choices") or {}).items():
        c = ch.CHOICES.get(k)
        if c is None:
            raise ResolveError(f"model choices: unknown choice {k!r}")
        try:
            contrib.setdefault(k, []).append(((LEVELS["model"], 0), "model", c.check(v), "llm_config.json choices"))
        except ValueError as e:
            raise ResolveError(f"model choices: {e}")
    chosen = {}
    for k, items in contrib.items():
        top = max(i[0] for i in items)
        at_top = [i for i in items if i[0] == top]
        if len({repr(i[2]) for i in at_top}) > 1:
            raise ResolveError(f"{k}: conflicting values at one level for this model: " +
                               ", ".join(f"{i[1]}={i[2]!r}" for i in at_top))
        chosen[k] = {"value": at_top[0][2], "from": at_top[0][1], "evidence": at_top[0][3]}
    # effective values in registry order (cfg-driven defaults read the values resolved before them), then exports
    ctx = {"draft_kind": kind, "families": sorted(env.families)}
    values, exports = {}, {}
    for k, c in ch.CHOICES.items():
        dflt = c.default(cfg, values, ctx, kit)
        v = chosen[k]["value"] if k in chosen else dflt
        values[k] = v
        if k in chosen and (c.kits is None or kit in c.kits):
            ev, dv = c.to_env(v), c.to_env(dflt)
            if ev != dv:
                exports[c.env] = ev
    return {"scopes": [s for _, s in sorted(matched)], "choices": {k: chosen[k] for k in ch.CHOICES if k in chosen},
            "devices": sorted(env.families), "facets": facets, "env": dict(sorted(exports.items())), "values": values}


def effective(cfg, devices=("rdna3", "rdna4"), kb_root=None, draft=AUTO, kit=None):
    """resolve_full() with every registered choice's value (defaults included) under "all"."""
    r = resolve_full(cfg, devices, kb_root, draft, kit)
    ch = choices_module(kb_root)
    r["all"] = {k: {"value": r["values"][k], "from": r["choices"][k]["from"] if k in r["choices"] else "default",
                    "env": ch.CHOICES[k].env} for k in ch.CHOICES}
    return r


def resolved_block(result):
    """The part of a resolution a package records (llm_config.json["resolved"])."""
    return {"scopes": result["scopes"], "choices": result["choices"], "devices": result["devices"], "facets": result["facets"],
            "env": result["env"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="an llm_config.json (a source cfg or a built package's)")
    ap.add_argument("--device", action="append", help="device family or slug of the build (default: rdna3 and rdna4)")
    ap.add_argument("--kb-root", default=HERE)
    ap.add_argument("--kit", help="reference kit example the build uses (gemma4, lfm2moe, ...)")
    ap.add_argument("--draft", default=None, help="draft kind (none for ordinary decoding); default: as built / auto")
    ap.add_argument("--all", action="store_true", help="every registered choice with its value")
    ap.add_argument("--lint", action="store_true", help="validate every scope.json (and resolve CONFIG if given)")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config, encoding="utf-8")) if a.config else None
    if a.lint:
        errs = lint(a.kb_root, {a.config: cfg} if cfg else None)
        for e in errs:
            print("ERROR", e)
        print(f"{len(errs)} error(s)")
        return 1 if errs else 0
    if cfg is None:
        ap.error("--config is required")
    draft = AUTO if a.draft is None else (None if a.draft == "none" else a.draft)
    fn = effective if a.all else resolve
    try:
        r = fn(cfg, tuple(a.device or ("rdna3", "rdna4")), a.kb_root, draft, a.kit)
    except ResolveError as e:
        print(f"resolve: {e}", file=sys.stderr)
        return 2
    r.pop("values", None)
    json.dump(r, sys.stdout, indent=1, sort_keys=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
