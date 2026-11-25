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
  match / when keys:  slug / model_id (the package slug / the checkpoint id: one or a list; first-class model nodes, below)
                      hidden, ffn, vocab, params_b (shape: a number or [lo, hi]; params_b = billions of parameters estimated from the
                      shape, a bucket not a count)
                      arch (GGUF arch, one or a list)  facets (all of)  not_facets (none of)  has / lacks (config keys that are / are
                      not set)  kv_heads, head_dim, heads, layers, sliding_window (a number, or [lo, hi] inclusive, null: open)
                      recurrent_kv (gated-delta-net hybrids)  quant (any tensor format of the model is one of)  devices (any of
                      the build's device families or device slugs). kb.py's matcher reads the same keys.
  An arch node may list `facets` its family implies (gemma3: sliding-window); a facet node (facet/<tag>) is on for a model its
  `match` holds for; built-in facets come from the config (derive_facets): drafted, draft-<kind>, mtp, vision, hybrid-gdn, moe,
  sliding-window, kv-le2, sinks, gated-attn, shortconv. `has: ["draft"]` sees the draft the build uses.

Model nodes: `model/<slug>/scope.json` (+ README.md) with match {"slug": ["<slug>"]} (or "model_id") is the scope of ONE package; its
rules read like any other node's (`when` narrows by device, facets, shape). The slug comes from the caller (resolve(..., slug=),
build.py --slug, make_info.py's repo, fleet/<slug>) or cfg["slug"]; model_id comes from the config.

Execution choices (choices.py exec.*): serving settings of one device's preset in model.json, not generator inputs. resolve() leaves
them out (a build's "resolved" block and programs are what they were); device_execution(cfg, device) resolves them for ONE device.

Precedence, later wins: registry default < device family (device/<fam>) < device (device/<fam>/<dev>) < quant (quant/<fmt>)
< arch root->leaf (arch/...) < facet (facet/<tag>) < model node (model/<slug>) < package override (cfg["choices"]). Two matching nodes at the same level (same
depth for arch) setting one key to different values for one model is an error (ResolveError).

Per-target choices: a choice declared per_target=True may be set by a rule whose `when`/`match` devices name one family
(a twin K / K_rdna4 in one dual-target build). Where the families end up with different values the resolved value is
{"rdna3": false, "rdna4": true} (default value for a family no rule reaches); the base variable carries the build's first
family (rdna3: the gfx11 kernel) and `<ENV>__RDNA4` the twin's value, which the generator of the twin applies
(flash2_rdna4.twin_env). A rule that names no device sets every family, as before.

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
LEVELS = {"device_family": 1, "device": 2, "quant": 3, "arch": 4, "facet": 5, "model_node": 6, "model": 7}   # "model": cfg["choices"]
RANGE_KEYS = ("kv_heads", "head_dim", "heads", "layers", "sliding_window", "hidden", "ffn", "vocab")
LIST_KEYS = ("slug", "model_id")   # identity: one value or a list, any of
MATCH_KEYS = {"arch", "facets", "not_facets", "has", "lacks", "recurrent_kv", "quant", "devices", "params_b", *LIST_KEYS, *RANGE_KEYS}
NODE_KEYS = ("scope", "match", "owner", "choices", "doc", "facets", "also")
FALLBACK_FAMILY = {"strix-halo": "rdna3", "gfx1151": "rdna3", "gfx1100": "rdna3", "radeon-r9700": "rdna4", "gfx1201": "rdna4"}


class ResolveError(Exception):
    pass


def _is_exec(c):
    """An execution choice (exec.*): a registry without the attribute (an older choices.py) has none."""
    return bool(getattr(c, "execution", False))


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

    def __init__(self, cfg, facets, devices, family_of, slug=None):
        self.cfg, self.facets, self.devices, self.family_of = cfg, set(facets), devices, family_of
        self.slug = slug or cfg.get("slug")
        self.formats = set(quant_formats(cfg))
        self.names = set(devices) | {family_of(d) for d in devices}
        self.families = {family_of(d) for d in devices}


def _num(spec, v):
    if v is None:
        return False
    if isinstance(spec, list):
        return len(spec) == 2 and (spec[0] is None or v >= spec[0]) and (spec[1] is None or v <= spec[1])
    return v == spec


def params_b(cfg):
    """Billions of parameters estimated from the shape (attention + FFN per layer, experts counted, embeddings): a bucket for selecting a size
    class, good to ~20 %; None when the config lacks the shape."""
    try:
        h, L, f, v = cfg["hidden"], cfg["layers"], cfg["ffn"], cfg["vocab"]
        hd, nh, kv = cfg["head_dim"], cfg["heads"], cfg["kv_heads"]
    except KeyError:
        return None
    moe = cfg.get("moe")
    experts = (moe.get("experts") or 1) if isinstance(moe, dict) else 1
    per = h * hd * (2 * nh + 2 * kv) + 3 * h * f * experts
    return (L * per + (1 if cfg.get("tied_embeddings") else 2) * v * h) / 1e9


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
    if "params_b" in m and not _num(m["params_b"], params_b(cfg)):
        return False
    for k in LIST_KEYS:
        if k in m:
            have = env.slug if k == "slug" else cfg.get(k)
            if have is None or have not in (m[k] if isinstance(m[k], list) else [m[k]]):
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
    if p[0] == "model" and len(p) == 2:
        return LEVELS["model_node"], 0
    raise ResolveError(f"scope {scope!r}: must be device/<family>[/<device>], quant/<fmt>, arch/<...>, facet/<tag> or model/<slug>")


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
    if scope.startswith("model/") and not ({"slug", "model_id"} & set(node.get("match") or {})):
        errs.append(f"{scope}: a model node must match an identity (match \"slug\" or \"model_id\")")
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


def resolve(cfg, devices=("rdna3", "rdna4"), kb_root=None, draft=AUTO, kit=None, slug=None):
    """The resolution of CFG (see the module docstring); resolve_full() adds the value of every registered choice."""
    r = resolve_full(cfg, devices, kb_root, draft, kit, slug)
    r.pop("values")
    return r


def device_execution(cfg, device, kb_root=None, draft=AUTO, slug=None):
    """{key: {"value", "from", "evidence"}} of the execution choices (exec.*) that a scope sets for CFG on ONE device (a slug or a family);
    empty when none does."""
    r = resolve_full(cfg, (device,), kb_root, draft, None, slug, execution=True)
    ch = choices_module(kb_root)
    return {k: v for k, v in r["choices"].items() if _is_exec(ch.CHOICES[k])}


def resolve_full(cfg, devices=("rdna3", "rdna4"), kb_root=None, draft=AUTO, kit=None, slug=None, execution=False):
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
        e = Env(cfg, facets, devices, family_of, slug)
        for scope, node in nodes:
            lvl = level_of(scope)[0]
            if lvl == LEVELS["arch"] and node.get("facets") and match(node.get("match") or {}, e):
                facets |= set(node["facets"])
            elif lvl == LEVELS["facet"] and node.get("match") and match(node["match"], e):
                facets.add(scope.split("/")[1])
        if len(facets) == before:
            break
    facets = sorted(facets)
    env = Env(cfg, facets, devices, family_of, slug)
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
                if _is_exec(c) and not execution:
                    continue   # a serving setting of a device's preset: device_execution()
                where = f"{scope} choices[{i}]"
                node_devs = [parts[-1]] if lvl in (LEVELS["device_family"], LEVELS["device"]) else []
                rule_devs = list(when.get("devices", [])) or node_devs or list(m.get("devices", []))
                if not _is_exec(c):
                    _check_devices(c, rule_devs, env.families, family_of, where)
                fams = {family_of(d) for d in rule_devs} if (rule_devs and c.per_target) else None
                if c.family_program:
                    fams = set(c.devices) if fams is None else fams.intersection(c.devices)
                    if not fams.intersection(env.families):
                        continue
                contrib.setdefault(k, []).append(((lvl, depth), scope, c.check(v), ent.get("evidence", ""), fams))
    for k, v in (cfg.get("choices") or {}).items():
        c = ch.CHOICES.get(k)
        if c is None:
            raise ResolveError(f"model choices: unknown choice {k!r}")
        if _is_exec(c) and not execution:
            continue
        try:
            fams = set(c.devices) if c.family_program else None
            if fams and not fams.intersection(env.families):
                continue
            contrib.setdefault(k, []).append(((LEVELS["model"], 0), "model", c.check(v), "llm_config.json choices", fams))
        except ValueError as e:
            raise ResolveError(f"model choices: {e}")
    def pick(k, items):
        if not items:
            return None
        top = max(i[0] for i in items)
        at_top = [i for i in items if i[0] == top]
        if len({repr(i[2]) for i in at_top}) > 1:
            raise ResolveError(f"{k}: conflicting values at one level for this model: " +
                               ", ".join(f"{i[1]}={i[2]!r}" for i in at_top))
        return {"value": at_top[0][2], "from": at_top[0][1], "evidence": at_top[0][3]}

    chosen, by_family = {}, {}   # by_family: {key: {family: value}} for a per_target choice whose value differs between families
    for k, items in contrib.items():
        if ch.CHOICES[k].per_target and any(i[4] is not None for i in items):
            per = {f: pick(k, [i for i in items if i[4] is None or f in i[4]]) for f in sorted(env.families)}
            if len({repr(p and p["value"]) for p in per.values()}) > 1:
                dflt = ch.CHOICES[k].default(cfg, {}, {"draft_kind": kind, "families": sorted(env.families)}, kit)
                vals = {f: (p["value"] if p else dflt) for f, p in per.items()}
                if len({repr(v) for v in vals.values()}) > 1:
                    by_family[k] = vals
                    chosen[k] = {"value": vals, "from": ", ".join(sorted({p["from"] for p in per.values() if p})),
                                 "evidence": " | ".join(dict.fromkeys(p["evidence"] for p in per.values() if p))}
                    continue
        chosen[k] = pick(k, items)
    # effective values in registry order (cfg-driven defaults read the values resolved before them), then exports
    ctx = {"draft_kind": kind, "families": sorted(env.families)}
    values, exports = {}, {}
    base = "rdna3" if "rdna3" in env.families else sorted(env.families)[0]
    for k, c in ch.CHOICES.items():
        dflt = c.default(cfg, values, ctx, kit)
        if k in by_family:   # the gfx11 kernel (the build's first family) carries the base variable, a twin its own <ENV>__<FAMILY>
            v = by_family[k][base]
        else:
            v = chosen[k]["value"] if k in chosen else dflt
        values[k] = v
        if k in chosen and not _is_exec(c) and (c.kits is None or kit in c.kits):
            ev, dv = c.to_env(v), c.to_env(dflt)
            if ev != dv:
                exports[c.env] = ev
            for f, fv in sorted(by_family.get(k, {}).items()):
                if f != base and c.to_env(fv) != ev:
                    exports[f"{c.env}__{f.upper()}"] = c.to_env(fv) if c.to_env(fv) is not None else ""
    return {"scopes": [s for _, s in sorted(matched)], "choices": {k: chosen[k] for k in ch.CHOICES if k in chosen},
            "devices": sorted(env.families), "facets": facets, "env": dict(sorted(exports.items())), "values": values}


def effective(cfg, devices=("rdna3", "rdna4"), kb_root=None, draft=AUTO, kit=None, slug=None):
    """resolve_full() with every registered choice's value (defaults included) under "all"."""
    r = resolve_full(cfg, devices, kb_root, draft, kit, slug)
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
    ap.add_argument("--slug", help="the package slug (matched by model nodes: model/<slug>)")
    ap.add_argument("--execution", action="store_true", help="print the execution choices (exec.*) of the one --device instead")
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
        if a.execution:
            if not a.device or len(a.device) != 1:
                ap.error("--execution needs exactly one --device")
            json.dump(device_execution(cfg, a.device[0], a.kb_root, draft, a.slug), sys.stdout, indent=1)
            print()
            return 0
        r = fn(cfg, tuple(a.device or ("rdna3", "rdna4")), a.kb_root, draft, a.kit, a.slug)
    except ResolveError as e:
        print(f"resolve: {e}", file=sys.stderr)
        return 2
    r.pop("values", None)
    json.dump(r, sys.stdout, indent=1, sort_keys=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
