"""Kernel-test cases for generated variants of tested kernels, made from the tested kernel's own case (make_kernel_tests.py):

- K_lv (gen_llm.live_row_skip): the per-row kernel that skips padding rows. The clone launches K_lv where the case launched K,
  with every row live (an info buffer whose word 2 is 1 is appended when K_lv takes one K does not), so it must produce K's
  expected outputs.
- Specialized copies: a kernel hrx-loom's loom-test cannot run as a case root is copied into the test file as `kt_<name>`, changed
  only where a case root differs from the program's launch, and the copy is tested (ktest_select counts kt_K as a case of K):
  X_listed (gen_llm.listed) reads its workgroup count from a buffer (indirect dispatch): the copy takes the count as a scalar and
  runs over the list of every token where a case launched the direct analogue (g16_*_listed <- g_*: the 16-split prefill
  fallback attention against the decode attention chain); a kernel that proves a vector bound only for the program's constant
  width (SwiGLU g_silu_mul) gets that constant.

clones(cases, lib, untested, save, ks) -> [case text] (K_lv); listed_cases(...) -> [helper kernel and case texts]."""
import re

import numpy as np

LAUNCH = re.compile(r"^(?P<ind>\s*)kernel\.launch @(?P<k>[\w.]+)\[(?P<grid>[^\]]*)\]\((?P<args>[^)]*)\) : \[(?P<gt>[^\]]*)\]\((?P<types>.*)\)$", re.M)


def _params(lib, name):
    """The launch parameter names of kernel NAME in LIB, or None."""
    a = lib.find(f'export("{name}")')
    if a < 0:
        return None
    m = re.search(r"^\} launch\(([^)]*)\) \{", lib[a:], re.M)
    return [p.split(":")[0].strip() for p in m.group(1).split(",")] if m else None


def _rename_case(case, sfx):
    return re.sub(r"check\.case public @(\w+) \{", lambda m: f"check.case public @{m.group(1)}{sfx} {{", case, count=1)


def _literal(case, var):
    m = re.search(r"^\s*" + re.escape(var) + r" = check\.literal value\((\d+)\) : index$", case, re.M)
    return int(m.group(1)) if m else None


def _types(s):
    return [t.strip() for t in re.split(r",\s*(?![^<]*>)", s)]


def lv_clone(case, bases, lib, save, ks):
    """CASE with every launch of a kernel in BASES on its _lv variant (all rows live), or None."""
    extras = {}
    for base in bases:
        lvp, bp = _params(lib, base + "_lv"), _params(lib, base)
        if lvp is None or bp is None or lvp[:len(bp)] != bp or len(lvp) - len(bp) > 1:
            return None
        extras[base] = len(lvp) > len(bp)
    decls = {}

    def sub(m):
        if m["k"] not in bases:
            return m.group(0)
        args, types = m["args"], m["types"]
        if extras[m["k"]]:
            n = _literal(case, args.split(",")[0].strip())
            if n is None:
                raise ValueError("no row count")
            if n not in decls:
                info = np.zeros((n, 4), np.int32)
                info[:, 2] = 1
                save(f"lvinfo_{n}", info)
                decls[n] = f'  %lvinfo{n} = check.file.read.npy path("lvinfo_{n}{ks}.npy") : tensor<{n}x4xi32>\n'
            args += f", %lvinfo{n}"
            types += f", tensor<{n}x4xi32>"
        return f"{m['ind']}kernel.launch @{m['k']}_lv[{m['grid']}]({args}) : [{m['gt']}]({types})"
    try:
        out = LAUNCH.sub(sub, case)
    except ValueError:
        return None
    return _rename_case(out, "_lv").replace("{\n", "{\n" + "".join(decls.values()), 1)


def clones(cases, lib, untested, save, ks=""):
    """K_lv clones; a case that launches kt_K (a specialized copy) is cloned onto kt_K_lv when that copy is in LIB."""
    out = []
    todo = {k[:-3] for k in untested if k.endswith("_lv")}
    todo |= {"kt_" + k for k in todo}
    for c in cases:
        if not todo:
            break
        bases = {k for k in set(re.findall(r"kernel\.launch @([\w.]+)\[", c)) if k in todo}
        if bases:
            cl = lv_clone(c, bases, lib, save, ks)
            if cl:
                out.append(cl)
                todo -= bases | {b[3:] if b.startswith("kt_") else "kt_" + b for b in bases}
    return out


def kernel_text(lib, name):
    m = re.search(r'(?ms)^kernel\.def[^\n]*export\("' + re.escape(name) + r'"\).*?^\} launch\(.*?^  kernel\.return\n\}\n', lib)
    return m.group(0) if m else None


def _as_copy(text, name):
    return text.replace(f'export("{name}") @{name}(', f'export("kt_{name}") @kt_{name}(', 1)


def listed_copy(lib, name):
    t = kernel_text(lib, name)
    if not t:
        return None
    t2 = re.sub(r"\(%count_buffer: buffer\) \{\n  %origin_c = .*?\n  %count = index\.cast %bounded : i32 to index\n", "(%count: index) {\n",
                t, count=1, flags=re.S)
    return _as_copy(t2, name) if t2 != t else None


def width_copy(lib, name, n):
    t = kernel_text(lib, name)
    if not t:
        return None
    t2 = re.sub(r"%n = index\.assume %n_arg \[[^\]]*\] : index", f"%n = index.constant {n} : index", t, count=1)
    return _as_copy(t2, name) if t2 != t else None


def listed_cases(cases, lib, untested, save, ks=""):
    """kt_ copies of the untested X_listed kernels and one case that launches them where the first case launching every direct
    analogue did, over the list of every token (the same outputs)."""
    listed = sorted(k for k in untested if k.endswith("_listed"))
    if not listed:
        return []
    pairs = {re.sub(r"^g\d+_", "g_", k[:-len("_listed")]): k for k in listed}
    helpers = [listed_copy(lib, k) for k in listed]
    if not all(helpers):
        return []
    for c in cases:
        if not set(pairs) <= set(re.findall(r"kernel\.launch @([\w.]+)\[", c)):
            continue
        decl = []

        def sub(m):
            if m["k"] not in pairs:
                return m.group(0)
            args = [a.strip() for a in m["args"].split(",")]
            types = _types(m["types"])
            n = _literal(c, args[0])
            if n is None:
                raise ValueError("no token count")
            if not decl:
                save(f"klist_{n}", np.arange(n, dtype=np.int32))
                decl.append(f'  %klist = check.file.read.npy path("klist_{n}{ks}.npy") : tensor<{n}xi32>\n')
            args.insert(1, "%klist")
            types.insert(1, f"tensor<{n}xi32>")
            return f"{m['ind']}kernel.launch @kt_{pairs[m['k']]}[{m['grid']}]({', '.join(args)}) : [{m['gt']}]({', '.join(types)})"
        try:
            out = LAUNCH.sub(sub, c)
        except ValueError:
            continue
        if decl:
            return helpers + [_rename_case(out, "_listed").replace("{\n", "{\n" + decl[0], 1)]
    return []
