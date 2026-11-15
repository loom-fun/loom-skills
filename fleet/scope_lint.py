#!/usr/bin/env python3
"""Scope lint: does a PR stay inside the mandate its `loom-scope` block declares? (DECISIONS.md section 6)

usage: scope_lint.py [--base REF] [--head REF] [--pr-body FILE | --scope-block TEXT] [--impact IMPACT.json]
                     [--json OUT.json] [--md OUT.md] [--local] [--no-impact] [--no-kb-lint] [--repo DIR]

The PR body carries a fenced block (the first one wins):

    ```loom-scope
    kind: targeted | horizontal | architecture | tooling | docs | card
    scopes: arch/llm/moe/qwen35moe+drafted, device/rdna4     (selector nodes under skills/loom-kb, `+facet` narrows)
    kernels: attention_decode                                (horizontal: kernel families, fleet/ownership.json)
    leak: <reason>                                           (only with horizontal/architecture)
    ```

Checks (exit 1 and one precise message per violation):
  1 files     every changed file is allowed for the kind (fleet/ownership.json "kinds"; targeted: its scope nodes' directories and
              tests; horizontal: its kernel families' files, its nodes, tests)
  2 leak      models whose programs / model.json differ (IMPACT.json = fleet/diff_fleet.py --json) that the declared scopes do not
              match; fails unless `leak:` is given with horizontal/architecture; every leak is listed (leak rate = leaked / touched).
              docs / card / tooling PRs must have an empty impact. Horizontal: every changed block belongs to a declared family.
              A `device/...` scope also limits the GPU families the changed blocks may target.
  3 protected a protected core file changed without kind architecture (a horizontal PR may change a `families_ok` file that a
              declared family lists)
  2b exec     the serving settings of a device's preset (exec.* choices: prefill chunk, context/pool caps ...) resolved for every fleet
              model and device at base and head: each (model, device) that changes must be matched by the declared selector scopes and
              lie inside the declared device scopes (a model node `model/<slug>` + `device/rdna4` is the targeted scope of a per-device
              setting of one package); the changes are listed. No program changes, so the program leak check cannot see them.
  4 registry  choices.py: a removed key or a changed legacy / env / type / derive, or any change that is not the addition of a new
              CHOICE / NOT / helper, needs architecture
  5 kb lint   `kb.py lint` (structure) on the head tree

--local: BASE = merge-base of HEAD with origin/refactor (else origin/r9700, else origin/main); HEAD = the working tree (HEAD plus uncommitted changes and new files); when
a change can affect programs and no --impact is given, builds the fleet at both (CPU, ~2 min) with fleet/build_fleet.py.
Stdlib only (--local building the fleet needs gguf numpy jinja2 pyyaml: checked up front, fleet/preflight.py). Imports skills/loom-kb/resolve.py when the tree has it (the matcher of the router and the build), else a built-in
copy of its documented match semantics.
"""
import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("targeted", "horizontal", "architecture", "tooling", "docs", "card")
GPU_FAMILY = {"gfx11": "rdna3", "gfx12": "rdna4"}
FAMILY_GPU = {v: k for k, v in GPU_FAMILY.items()}
DEVICE_FAMILY = {"strix-halo": "rdna3", "gfx1151": "rdna3", "gfx1100": "rdna3", "radeon-r9700": "rdna4", "gfx1201": "rdna4"}
MAX_ROWS = 15
KNOWN_FACETS = {"drafted", "mtp", "vision", "hybrid-gdn", "moe", "sliding", "kv-le2", "sinks", "gated-attn", "shortconv"}


# --- globs ----------------------------------------------------------------------------------------------------------------
def glob_re(g):
    out, i = "", 0
    while i < len(g):
        if g.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif g.startswith("**", i):
            out += ".*"
            i += 2
        elif g[i] == "*":
            out += "[^/]*"
            i += 1
        elif g[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(g[i])
            i += 1
    return re.compile("^" + out + "$")


def globs_match(globs, path):
    return any(glob_re(g).match(path) for g in globs)


# --- git ------------------------------------------------------------------------------------------------------------------
def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def changed_files(repo, base, head):
    """[(status, path)] of base..head (merge-base diff; renames reported as delete + add)."""
    mb = git(repo, "merge-base", base, head, check=False).strip() or base
    out = []
    for line in git(repo, "diff", "--name-status", "--no-renames", mb, head).splitlines():
        st, _, p = line.partition("\t")
        out.append((st[0], p))
    return mb, out


def show(repo, ref, path):
    r = subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


# --- the loom-scope block --------------------------------------------------------------------------------------------------
def parse_block(text):
    """(dict, [errors]) of the first ```loom-scope block of TEXT (or of TEXT itself when it has no fence)."""
    m = re.search(r"```[ \t]*loom-scope[ \t]*\r?\n(.*?)```", text, re.S)
    if not m and "```" in text:
        return None, ["no ```loom-scope block in the PR body"]
    body = m.group(1) if m else text
    d, errs = {}, []
    for raw in body.splitlines():
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("leak:") else raw.rstrip()
        if not line.strip():
            continue
        k, sep, v = line.partition(":")
        k = k.strip().lower()
        if not sep or k not in ("kind", "scopes", "kernels", "leak"):
            errs.append(f"loom-scope: cannot read line {raw.strip()!r} (keys: kind, scopes, kernels, leak)")
            continue
        d[k] = v.strip()
    kind = d.get("kind", "")
    if kind not in KINDS:
        errs.append(f"loom-scope: kind must be one of {', '.join(KINDS)} (got {kind!r})")
    out = {"kind": kind, "scopes": [s.strip() for s in d.get("scopes", "").split(",") if s.strip() and s.strip() != "-"],
           "kernels": [s.strip() for s in d.get("kernels", "").split(",") if s.strip() and s.strip() != "-"],
           "leak": d.get("leak", "").strip()}
    return out, errs


# --- matching models against scopes -------------------------------------------------------------------------------------------
def load_module(path, name):
    sys.dont_write_bytecode = True   # never leave __pycache__ in the tree under lint
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Matcher:
    """Selector nodes of a KB tree and the match of a model's config against them (resolve.py when present)."""

    def __init__(self, tree):
        self.kb = os.path.join(tree, "skills", "loom-kb")
        rp = os.path.join(self.kb, "resolve.py")
        if not os.path.isfile(rp):
            raise SystemExit(f"scope_lint: {rp} missing (resolve.py is the one matcher)")
        sys.path.insert(0, self.kb)
        try:
            self.r = load_module(rp, "loom_kb_resolve_for_lint")
        finally:
            sys.path.remove(self.kb)
        self.nodes = {}
        for dp, dn, fn in os.walk(self.kb):
            dn[:] = sorted(d for d in dn if not d.startswith((".", "__")))
            if "scope.json" in fn:
                rel = os.path.relpath(dp, self.kb).replace(os.sep, "/")
                try:
                    with open(os.path.join(dp, "scope.json"), encoding="utf-8") as fh:
                        self.nodes[rel] = json.load(fh)
                except ValueError:
                    self.nodes[rel] = None
        self.devdirs = {}
        base = os.path.join(self.kb, "device")
        if os.path.isdir(base):
            for fam in os.listdir(base):
                if os.path.isdir(os.path.join(base, fam)):
                    self.devdirs[fam] = fam
                    for dev in os.listdir(os.path.join(base, fam)):
                        if os.path.isdir(os.path.join(base, fam, dev)):
                            self.devdirs[dev] = fam

    def family_of(self, dev):
        return self.devdirs.get(dev) or DEVICE_FAMILY.get(dev) or dev

    def facets(self, cfg, draft):
        kind = self.r.draft_kind(cfg, self.r.AUTO) if draft is None else (None if draft == "none" else draft)
        return self.r.derive_facets(cfg, kind)

    def node_matches(self, scope, cfg, facets, devices=("rdna3", "rdna4"), slug=None):
        """True / False; raises ValueError when the node is unknown or invalid."""
        node = self.nodes.get(scope, "missing")
        if node == "missing":
            raise ValueError(f"scope {scope!r}: no scope.json under skills/loom-kb/{scope}")
        if node is None:
            raise ValueError(f"scope {scope!r}: scope.json is not JSON")
        env = self.r.Env(cfg, facets, list(devices), self.family_of, slug)
        try:
            return bool(self.r.match(node.get("match") or {}, env, scope))
        except self.r.ResolveError as e:
            raise ValueError(str(e))


def parse_scope(entry):
    """'arch/llm/moe/qwen35moe+drafted+kv-le2' -> ('arch/llm/moe/qwen35moe', ['drafted', 'kv-le2'])."""
    node, *facets = entry.split("+")
    return node.strip().strip("/"), [f.strip() for f in facets if f.strip()]


def scope_dir(entry):
    node, _ = parse_scope(entry)
    return f"skills/loom-kb/{node}"


# --- choices.py ---------------------------------------------------------------------------------------------------------------
def exec_choices(src):
    """{key: signature} of a choices.py source (exec'd in a scratch namespace)."""
    ns = {"__name__": "choices_under_lint"}
    exec(compile(src, "choices.py", "exec"), ns)   # noqa: S102  (the PR's registry: stdlib `re` only, by its own contract)
    out = {}
    for k, c in ns["CHOICES"].items():
        d = getattr(c, "derive", None)
        out[k] = {"legacy": repr(c.legacy), "env": c.env, "type": c.type, "enc": repr(c.enc), "kit_legacy": repr(sorted(c.kit_legacy.items())),
                  "derive": (d.__code__.co_code.hex() + repr(d.__code__.co_consts)) if d else None, "kits": repr(c.kits)}
    return out


def machinery(src):
    """Top-level statements that are not registrations: {name or dump: ast dump}; CHOICE/NOT calls are registrations."""
    tree = ast.parse(src)
    out, anon = {}, {}
    for n in tree.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id in ("CHOICE", "NOT"):
            continue
        name = getattr(n, "name", None)
        if name is None and isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
        dump = ast.dump(n)
        if name:
            out[name] = dump
        else:
            anon[dump] = anon.get(dump, 0) + 1
    return out, anon


def choices_diff(base_src, head_src):
    """(problems, added keys, changed/removed keys) of the registry between two sources of choices.py."""
    problems, added = [], []
    try:
        b, h = exec_choices(base_src), exec_choices(head_src)
    except Exception as e:   # noqa: BLE001
        return [f"choices.py cannot be loaded: {type(e).__name__}: {e}"], [], []
    removed = sorted(set(b) - set(h))
    changed = sorted(k for k in set(b) & set(h) if b[k] != h[k])
    added = sorted(set(h) - set(b))
    for k in removed:
        problems.append(f"registry key {k!r} was removed")
    for k in changed:
        what = ", ".join(f for f in b[k] if b[k][f] != h[k][f])
        problems.append(f"registry key {k!r} changed ({what}); legacy/env/type/derive define today's behaviour")
    bm, ba = machinery(base_src)
    hm, ha = machinery(head_src)
    for n in sorted(set(bm) - set(hm)):
        problems.append(f"choices.py: {n!r} was removed")
    for n in sorted(n for n in set(bm) & set(hm) if bm[n] != hm[n]):
        problems.append(f"choices.py: {n!r} (registry machinery or shared helper) changed")
    if any(ha.get(d, 0) < c for d, c in ba.items()):
        problems.append("choices.py: a top-level statement of the registry machinery changed or was removed")
    return problems, added, changed + removed


# --- the lint -----------------------------------------------------------------------------------------------------------------
class Result:
    def __init__(self):
        self.errors, self.notes = [], []
        self.info = {}

    def err(self, rule, msg):
        self.errors.append({"rule": rule, "message": msg})


def load_cfg(tree, slug):
    p = os.path.join(tree, "fleet", slug, "llm_config.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def exec_changes(repo, mb, tree, matcher):
    """[{model, device, before, after}]: the exec.* settings (choices.py `execution` choices) each fleet model resolves to on each registry
    device, base (the merge-base's skills/loom-kb) against the tree's. Empty when the base has no execution choices."""
    fleet = os.path.join(tree, "fleet")
    if not os.path.isdir(fleet) or not hasattr(matcher.r, "device_execution"):
        return []
    work = tempfile.mkdtemp(prefix="scope_lint_base_")
    try:
        tar = subprocess.run(["git", "-C", repo, "archive", mb, "skills/loom-kb"], capture_output=True)
        if tar.returncode:
            return []
        subprocess.run(["tar", "-x", "-C", work], input=tar.stdout, check=True)
        bkb = os.path.join(work, "skills", "loom-kb")
        if not os.path.isfile(os.path.join(bkb, "resolve.py")):
            return []
        sys.path.insert(0, bkb)
        try:
            br = load_module(os.path.join(bkb, "resolve.py"), "loom_kb_resolve_base_for_lint")
        finally:
            sys.path.remove(bkb)
        if not hasattr(br, "device_execution"):
            br = None   # a base without execution choices: everything it resolves is empty
        devices = sorted({d for d, f in matcher.devdirs.items() if d != f})   # device slugs (device/<family>/<device>)
        out = []
        for slug in sorted(os.listdir(fleet)):
            cfg = load_cfg(tree, slug)
            if cfg is None:
                continue
            for dev in devices:
                try:
                    after = {k: v["value"] for k, v in matcher.r.device_execution(cfg, dev, matcher.kb, slug=slug).items()}
                    before = {k: v["value"] for k, v in br.device_execution(cfg, dev, bkb, slug=slug).items()} if br else {}
                except Exception as e:   # noqa: BLE001  (an invalid scope.json is reported by the kb lint and the resolver tests)
                    out.append({"model": slug, "device": dev, "before": {}, "after": {}, "error": f"{type(e).__name__}: {e}"})
                    continue
                if before != after:
                    out.append({"model": slug, "device": dev, "before": before, "after": after})
        return out
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def lint(repo, base, head, block, errs_in, impact=None, kb_lint=True, own=None, tree=None):
    """Run the checks; returns a Result. OWN: the ownership data (default: the one next to this file in REPO, else beside the script)."""
    res = Result()
    tree = tree or repo
    for e in errs_in:
        res.err("scope-block", e)
    if block is None or block["kind"] not in KINDS:
        return res
    kind = block["kind"]
    own = own or json.loads(Path(HERE, "ownership.json").read_text())
    mb, files = changed_files(repo, base, head)
    paths = [p for _, p in files]
    res.info.update(kind=kind, base=mb, scopes=block["scopes"], kernels=block["kernels"], files=paths)
    # declared scopes and kernels exist
    matcher = Matcher(tree)
    for s in block["scopes"]:
        node, facets = parse_scope(s)
        if not os.path.isdir(os.path.join(matcher.kb, node)):
            res.err("scope-block", f"scope {s!r}: skills/loom-kb/{node} does not exist")
        for f in facets:
            if f not in KNOWN_FACETS and not f.startswith("draft-") and not os.path.isdir(os.path.join(matcher.kb, "facet", f)):
                res.err("scope-block", f"scope {s!r}: unknown facet {f!r}")
    for k in block["kernels"]:
        if k not in own["families"]:
            res.err("scope-block", f"kernels: unknown family {k!r} (known: {', '.join(sorted(own['families']))})")
    if kind == "targeted" and not block["scopes"]:
        res.err("scope-block", "kind targeted needs scopes: (the selector nodes the change is for)")
    if kind == "horizontal" and not block["kernels"]:
        res.err("scope-block", "kind horizontal needs kernels: (the kernel families it may touch)")
    if kind in ("targeted", "docs", "card", "tooling") and block["kernels"]:
        res.err("scope-block", f"kernels: is only for horizontal PRs (kind is {kind})")
    if block["leak"] and kind not in ("horizontal", "architecture"):
        res.err("scope-block", f"leak: is only allowed with kind horizontal or architecture (kind is {kind}); split the PR instead")
    # 1 + 3: files and protected files
    prot = own["protected"]
    choices_path = own["choices_file"]
    family_files = [g for k in block["kernels"] if k in own["families"] for g in own["families"][k]["files"]]
    node_dirs = [scope_dir(s) + "/**" for s in block["scopes"]]
    allow = list(own["kinds"][kind]["allow"])
    if kind in ("targeted", "horizontal"):
        allow += node_dirs
    if kind == "horizontal":
        allow += family_files
    choices_touched = choices_path in paths
    for st, p in files:
        if p == choices_path:
            continue
        pr = next((x for x in prot if glob_re(x["glob"]).match(p)), None)
        if pr and kind != "architecture":
            if kind == "horizontal" and pr.get("families_ok") and globs_match(family_files, p):
                pass
            else:
                res.err("protected", f"{p} is a protected core file ({pr['why']}); changing it needs kind: architecture")
                continue
        if not globs_match(allow, p) or globs_match(own["kinds"][kind].get("deny", []), p):
            hint = {"targeted": "a targeted PR may change only its scope nodes' directories and tests",
                    "horizontal": "a horizontal PR may change only its kernel families' files, its scope nodes and tests",
                    "docs": "a docs PR may change only *.md files", "card": "a card PR may change only card tooling and its tests",
                    "tooling": "a tooling PR may not change generators, kb data or cards"}[kind]
            res.err("files", f"{p}: not allowed for kind {kind} ({hint})")
    # 4: registry
    if choices_touched:
        base_src, head_src = show(repo, mb, choices_path), show(repo, head, choices_path)
        if base_src is None or head_src is None:
            if kind != "architecture":
                res.err("registry", f"{choices_path} added or removed: needs kind: architecture")
        else:
            probs, added, _ = choices_diff(base_src, head_src)
            res.info["choices_added"] = added
            if probs and kind != "architecture":
                for pr_ in probs:
                    res.err("registry", pr_ + "; needs kind: architecture")
            elif kind in ("docs", "card", "tooling"):
                res.err("files", f"{choices_path}: not allowed for kind {kind}")
            if kind == "architecture" and probs:
                res.notes.append("registry changed (architecture): " + "; ".join(probs))
    # 2: leak and kernel families
    prog_inputs = [p for p in paths if globs_match(own.get("program_inputs", []), p)]
    res.info["program_inputs_changed"] = prog_inputs
    leak = {"touched": 0, "leaked": [], "rate": 0.0}
    if impact is None:
        if not prog_inputs:
            res.info["impact"] = {"touched": 0, "models": [], "note": "no program input changed"}
        elif kind != "architecture":
            res.err("leak", "this change can affect programs (" + ", ".join(prog_inputs[:3]) + (", ..." if len(prog_inputs) > 3 else "") +
                    ") but no --impact report was given; run fleet/build_fleet.py + diff_fleet.py (or --local)")
    else:
        models = {s: m for s, m in impact.get("models", {}).items()}
        leak["touched"] = len(models)
        res.info["impact"] = {"touched": len(models), "models": sorted(models)}
        if kind in ("docs", "card", "tooling") and models:
            res.err("leak", f"kind {kind} must not change any program, but {len(models)} model(s) differ: " + ", ".join(sorted(models)[:6]))
        if kind in ("targeted", "horizontal") or (kind == "architecture" and block["scopes"]):
            ent = [parse_scope(s) for s in block["scopes"]]
            devs = [n.split("/") for n, _ in ent if n.startswith("device/")]
            dev_fams = {matcher.family_of(d[-1]) for d in devs}
            selectors = [(n, f) for n, f in ent if not n.startswith("device/")]
            rules_ok = {r for k in block["kernels"] if k in own["families"] for r in own["families"][k]["rules"]}
            for slug, m in sorted(models.items()):
                if "error" in m:
                    res.err("leak", f"{slug}: {m['error']}")
                    continue
                why = []
                cfg = load_cfg(tree, slug)
                if selectors:
                    if cfg is None:
                        why.append("no fleet config to match the scopes against")
                    else:
                        facets = matcher.facets(cfg, (m.get("stratum") or {}).get("draft"))
                        ok = False
                        for node, extra in selectors:
                            try:
                                hit = matcher.node_matches(node, cfg, facets, slug=slug) and set(extra) <= set(facets)
                            except ValueError as e:
                                res.err("scope-block", str(e))
                                hit = False
                            ok = ok or hit
                        if not ok:
                            why.append("not matched by " + ", ".join(block["scopes"]))
                if dev_fams:
                    wrong = sorted(g for g in m.get("gpus", []) if GPU_FAMILY.get(g, g) not in dev_fams)
                    if wrong:
                        why.append(f"changes {'/'.join(wrong)} programs; the declared device scope is {'/'.join(sorted(dev_fams))}")
                if kind == "horizontal":
                    stray = sorted({b["rule"] for b in m.get("blocks", []) if b.get("rule") not in rules_ok and
                                    not (b.get("rule", "").startswith(("template", "program")) and block["kernels"])})
                    if stray:
                        why.append("changes blocks outside the declared kernel families (rules " + ", ".join(stray) + ")")
                if why:
                    leak["leaked"].append({"model": slug, "why": "; ".join(why)})
            leak["rate"] = round(len(leak["leaked"]) / len(models), 3) if models else 0.0
            if leak["leaked"]:
                if block["leak"] and kind in ("horizontal", "architecture"):
                    res.notes.append(f"leak declared ({block['leak']}): {len(leak['leaked'])} of {len(models)} model(s) outside the scope")
                else:
                    for x in leak["leaked"]:
                        res.err("leak", f"{x['model']}: {x['why']}; narrow the change (or declare `leak:` with horizontal/architecture)")
        elif kind == "architecture" and models:
            res.notes.append(f"architecture PR: {len(models)} model(s) change")
    res.info["leak"] = leak
    # 2b: serving settings of the devices' presets (exec.*): they change no program, so the fleet diff cannot show them
    if any(p.startswith("skills/loom-kb/") and (p.endswith("scope.json") or p == choices_path or p.endswith("resolve.py")) for p in paths):
        ex = exec_changes(repo, mb, tree, matcher)
        res.info["exec_changed"] = ex
        if ex:
            ent = [parse_scope(s) for s in block["scopes"]]
            dev_names = {n.split("/")[-1] for n, _ in ent if n.startswith("device/")}
            dev_fams = {matcher.family_of(d) for d in dev_names}
            selectors = [(n, f) for n, f in ent if not n.startswith("device/")]
            scoped = kind in ("targeted", "horizontal") or (kind == "architecture" and block["scopes"])
            for x in ex:
                if "error" in x:
                    res.err("exec", f"{x['model']} on {x['device']}: {x['error']}")
                    continue
                why = []
                if scoped:
                    cfg = load_cfg(tree, x["model"])
                    if selectors:
                        facets = matcher.facets(cfg, None)
                        def hit(n, f):
                            try:
                                return matcher.node_matches(n, cfg, facets, slug=x["model"]) and set(f) <= set(facets)
                            except ValueError:   # an unknown node: reported by the scope-block rule
                                return False
                        if not any(hit(n, f) for n, f in selectors):
                            why.append("not matched by " + ", ".join(block["scopes"]))
                    if dev_names and not ({x["device"], matcher.family_of(x["device"])} & (dev_names | dev_fams)):
                        why.append(f"the declared device scope is {'/'.join(sorted(dev_names))}")
                    if why and not (block["leak"] and kind in ("horizontal", "architecture")):
                        res.err("exec", f"{x['model']} on {x['device']}: serving settings change ({x['before']} -> {x['after']}) but " + "; ".join(why))
                elif kind in ("docs", "card", "tooling"):
                    res.err("exec", f"kind {kind} must not change serving settings ({x['model']} on {x['device']})")
            res.notes.append(f"serving settings (exec.*) change for {len({x['model'] for x in ex})} model(s) on {len({x['device'] for x in ex})} device(s)")
    # 5: kb lint
    if kb_lint:
        kb = os.path.join(tree, "skills", "loom-kb", "kb.py")
        if os.path.isfile(kb):
            r = subprocess.run([sys.executable, kb, "lint"], capture_output=True, text=True, cwd=tree)
            out = (r.stdout + r.stderr).strip()
            res.info["kb_lint"] = out.splitlines()[-1] if out else ""
            if r.returncode:
                for line in out.splitlines():
                    if line.strip() and not line.startswith("kb lint:"):
                        res.err("kb-lint", line.strip())
                if not any(e["rule"] == "kb-lint" for e in res.errors):
                    res.err("kb-lint", "kb.py lint failed")
    return res


def markdown(res):
    i = res.info
    L = [f"### Scope lint: {'FAIL' if res.errors else 'pass'}"]
    if i.get("kind"):
        L.append(f"kind `{i['kind']}`; scopes {', '.join(f'`{s}`' for s in i.get('scopes', [])) or '-'}; "
                 f"kernels {', '.join(f'`{s}`' for s in i.get('kernels', [])) or '-'}; {len(i.get('files', []))} file(s) changed.")
    for x in (i.get("exec_changed") or [])[:MAX_ROWS]:
        if "error" not in x:
            L.append(f"- serving settings: `{x['model']}` on `{x['device']}`: {x['before'] or '-'} -> {x['after'] or '-'}")
    lk = i.get("leak") or {}
    if "impact" in i:
        L.append(f"Program impact: {lk.get('touched', 0)} model(s) touched{' (no program input changed)' if i['impact'].get('note') else ''}; leaked outside the declared scope: {len(lk.get('leaked', []))} "
                 f"(leak rate {lk.get('rate', 0):.0%}).")
    if res.errors:
        L += ["", "| rule | problem |", "|---|---|"] + [f"| {e['rule']} | {e['message'].replace('|', '/')} |" for e in res.errors[:MAX_ROWS]]
        if len(res.errors) > MAX_ROWS:
            L.append(f"| ... | {len(res.errors) - MAX_ROWS} more (all in the JSON report) |")
    for n in res.notes:
        L.append(f"- note: {n}")
    for x in lk.get("leaked", [])[:MAX_ROWS]:
        L.append(f"- leaked: `{x['model']}`: {x['why']}")
    if i.get("choices_added"):
        L.append("- registry additions: " + ", ".join(f"`{k}`" for k in i["choices_added"]))
    if i.get("kb_lint"):
        L.append(f"- kb lint: {i['kb_lint']}")
    return "\n".join(L) + "\n"


def local_base(repo):
    for ref in ("origin/refactor", "origin/r9700", "origin/main"):
        if subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "-q", ref], capture_output=True).returncode == 0:
            return ref
    raise SystemExit("--local: none of origin/refactor, origin/r9700, origin/main exists; pass --base")


def local_head(repo):
    """HEAD with the uncommitted working-tree changes (tracked edits and new files that are not ignored) as one dangling commit,
    so --local lints what would be committed; HEAD itself when the tree is clean."""
    if not git(repo, "status", "--porcelain").strip():
        return "HEAD"
    idx = tempfile.mktemp(prefix="scope_lint_index_")
    env = dict(os.environ, GIT_INDEX_FILE=idx, GIT_AUTHOR_NAME="scope-lint", GIT_AUTHOR_EMAIL="x@x", GIT_COMMITTER_NAME="scope-lint", GIT_COMMITTER_EMAIL="x@x")
    try:
        for cmd in (["read-tree", "HEAD"], ["add", "-A"]):
            r = subprocess.run(["git", "-C", repo, *cmd], capture_output=True, text=True, env=env)
            if r.returncode:
                raise SystemExit(f"--local: git {cmd[0]}: {r.stderr.strip()}")
        tree = subprocess.run(["git", "-C", repo, "write-tree"], capture_output=True, text=True, env=env, check=True).stdout.strip()
        sha = subprocess.run(["git", "-C", repo, "commit-tree", tree, "-p", "HEAD", "-m", "scope-lint working tree"], capture_output=True, text=True, env=env, check=True).stdout.strip()
    finally:
        if os.path.exists(idx):
            os.remove(idx)
    print(f"--local: the working tree has uncommitted changes; linting them as {sha[:10]} on top of HEAD", file=sys.stderr)
    return sha


def build_impact(repo, base, out_dir):
    """Fleet impact of REPO's working tree against BASE (needs a python with gguf): returns the impact dict."""
    mb = git(repo, "merge-base", base, "HEAD").strip()
    bdir, hdir = os.path.join(out_dir, "base"), os.path.join(out_dir, "head")
    bf = os.path.join(repo, "fleet", "build_fleet.py")
    sys.path.insert(0, os.path.join(repo, "fleet"))
    import preflight   # noqa: E402  (the tools about to run need the same packages)
    preflight.require()
    for cmd in ([sys.executable, bf, "--ref", mb, "--out", bdir], [sys.executable, bf, "--tree", repo, "--out", hdir]):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        if r.returncode:
            raise SystemExit("build_fleet failed:\n" + (r.stdout + r.stderr)[-1500:])
    ij = os.path.join(out_dir, "impact.json")
    subprocess.run([sys.executable, os.path.join(repo, "fleet", "diff_fleet.py"), bdir, hdir, "--json", ij, "--md", os.path.join(out_dir, "impact.md")],
                   check=True, capture_output=True)
    return json.loads(Path(ij).read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--pr-body", help="file with the PR description (its ```loom-scope block) or just the block")
    ap.add_argument("--scope-block")
    ap.add_argument("--impact")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--no-impact", action="store_true", help="--local: do not build the fleet to get the impact")
    ap.add_argument("--no-kb-lint", action="store_true")
    ap.add_argument("--repo", default=os.path.dirname(HERE))
    ap.add_argument("--work", help="--local: directory for the fleet builds (default: a temp dir, removed)")
    a = ap.parse_args(argv)
    repo = os.path.abspath(a.repo)
    base = a.base or (local_base(repo) if a.local else None)
    head = a.head or (local_head(repo) if a.local else "HEAD")
    if not base:
        ap.error("--base is required (or --local)")
    text = a.scope_block if a.scope_block is not None else (Path(a.pr_body).read_text(encoding="utf-8") if a.pr_body else "")
    block, errs = parse_block(text) if text.strip() else (None, ["no loom-scope block given (--pr-body FILE or --scope-block TEXT)"])
    impact = json.loads(Path(a.impact).read_text()) if a.impact else None
    if impact is None and a.local and not a.no_impact and block and block["kind"] != "architecture":
        _, files = changed_files(repo, base, head)
        own = json.loads((Path(repo) / "fleet" / "ownership.json").read_text())
        if any(globs_match(own.get("program_inputs", []), p) for _, p in files):
            work = a.work or tempfile.mkdtemp(prefix="scope_lint_")
            print(f"building the fleet at {base} and the working tree in {work} ...", file=sys.stderr)
            impact = build_impact(repo, base, work)
    own = json.loads((Path(repo) / "fleet" / "ownership.json").read_text()) if (Path(repo) / "fleet" / "ownership.json").is_file() else None
    res = lint(repo, base, head, block, errs, impact, kb_lint=not a.no_kb_lint, own=own, tree=repo)
    md = markdown(res)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"ok": not res.errors, "errors": res.errors, "notes": res.notes, **res.info}, fh, indent=1)
    print(md)
    return 1 if res.errors else 0


if __name__ == "__main__":
    sys.exit(main())
