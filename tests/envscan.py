"""Finds every environment variable name the generators touch (read, default, write): AST scan of the .py files, text scan of the
kit patches. Used by test_choices.py (a knob must be registered in skills/loom-kb/choices.py) and runnable as a script."""
import ast, os, re, collections

NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
HELPERS = {"on", "gib"}
PATCH_RE = re.compile(r"""(?:environ|env\w*)(?:\.get|\.setdefault|\.pop)?\(\s*["']([A-Z][A-Z0-9_]*)["'](?:\s*,\s*([^)\n]*))?""")


def _envish(node):
    n = node
    if isinstance(n, ast.Attribute):
        return "env" in n.attr.lower()
    if isinstance(n, ast.Name):
        return "env" in n.id.lower()
    return False


def _lit(n):
    try:
        return ast.literal_eval(n)
    except Exception:
        return "<expr>"


def scan_py(path, text=None):
    """[(name, lineno, kind, default)]: kind in get/set/sub/in/write."""
    if text is None:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    src = text
    t = ast.parse(src)
    out = []
    for n in ast.walk(t):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in ("get", "setdefault", "pop", "getenv") and (_envish(f.value) or f.attr == "getenv") \
                    and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str) and NAME.match(n.args[0].value):
                d = _lit(n.args[1]) if len(n.args) > 1 else None
                out.append((n.args[0].value, n.lineno, "set" if f.attr == "setdefault" else "get", d))
            elif isinstance(f, ast.Name) and f.id in HELPERS and n.args and isinstance(n.args[0], ast.Constant) \
                    and isinstance(n.args[0].value, str) and NAME.match(n.args[0].value):
                out.append((n.args[0].value, n.lineno, "get", "<helper>"))
            elif isinstance(f, ast.Name) and f.id == "dict" and n.args and _envish(n.args[0]):
                out += [(k.arg, n.lineno, "write", _lit(k.value)) for k in n.keywords if k.arg and NAME.match(k.arg)]
            elif isinstance(f, ast.Attribute) and f.attr == "update" and _envish(f.value):
                out += [(k.arg, n.lineno, "write", _lit(k.value)) for k in n.keywords if k.arg and NAME.match(k.arg)]
        elif isinstance(n, ast.Subscript) and _envish(n.value) and isinstance(n.slice, ast.Constant) \
                and isinstance(n.slice.value, str) and NAME.match(n.slice.value):
            out.append((n.slice.value, n.lineno, "write" if isinstance(n.ctx, ast.Store) else "sub", None))
        elif isinstance(n, ast.Compare) and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str) and NAME.match(n.left.value) \
                and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops) and any(_envish(c) for c in n.comparators):
            out.append((n.left.value, n.lineno, "in", None))
        elif isinstance(n, (ast.GeneratorExp, ast.ListComp)) and isinstance(n.elt, ast.Call) and n.elt.args \
                and isinstance(n.elt.func, ast.Attribute) and _envish(n.elt.func.value) and isinstance(n.elt.args[0], ast.Name):
            for g in n.generators:
                if isinstance(g.iter, (ast.Tuple, ast.List)):
                    out += [(e.value, n.lineno, "get", None) for e in g.iter.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str) and NAME.match(e.value)]
    return out


def scan_patch(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines, 1):
        if line.startswith("+") and not line.startswith("+++"):
            for m in PATCH_RE.finditer(line):
                d = m.group(2)
                out.append((m.group(1), i, "get", d.strip() if d else None))
            for m in re.finditer(r"""env\w*\[\s*["']([A-Z][A-Z0-9_]*)["']\s*\]""", line):
                out.append((m.group(1), i, "sub", None))
            for m in re.finditer(r"""\b(FLASH2_[A-Z0-9_]+|KTEST_[A-Z0-9_]+)=""", line):
                out.append((m.group(1), i, "write", None))
    return out


def scan(roots, patches=()):
    res = collections.defaultdict(list)
    for root in roots:
        for dp, dn, fs in os.walk(root):
            dn[:] = [d for d in dn if d != "__pycache__"]
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.join(dp, f)
                    for name, ln, kind, d in scan_py(p):
                        res[name].append((p, ln, kind, d))
    for p in patches:
        for name, ln, kind, d in scan_patch(p):
            res[name].append((p, ln, kind, d))
    return res

if __name__ == "__main__":
    import sys, glob
    r = sys.argv[1]
    roots = [f"{r}/skills/loom-port-llm/starter", f"{r}/skills/loom-optimize/reference", f"{r}/skills/loom-port-llm/reference"]
    patches = glob.glob(f"{r}/skills/loom-port-llm/reference/*/starter.patch")
    res = scan(roots, patches)
    for k in sorted(res):
        print(k, len(res[k]), "|", "; ".join(f"{os.path.basename(p)}:{l}:{kd}={d}" for p, l, kd, d in res[k][:3]))
