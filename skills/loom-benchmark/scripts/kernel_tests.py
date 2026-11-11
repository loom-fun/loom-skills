#!/usr/bin/env python3
"""Runs a package's loom_test targets (test/BUILD.bazel) with loom-test, without Bazel.

usage: kernel_tests.py PKG_DIR_OR_REPO TEST_DIR [--loom-test BIN] [--device amdgpu] [--only a,b] [-o out.json]

PKG_DIR_OR_REPO  the repository root (the directory holding kernel/, motif/, program/, model/ with their BUILD.bazel files),
                 or any directory below it (the root is found by walking up, or in a `pkg/` child)
TEST_DIR         the test directory (BUILD.bazel + *.loom), absolute or relative to the root

How a target is translated (the Bazel rule's semantics, without Bazel):
  * BUILD.bazel files are evaluated as (near-)Python with stub rules; every loom_library / loom_motif_library /
    loom_kernel_library records name, srcs and deps, loom_test records srcs, deps, configs, variants, execution_profiles.
  * each library is compiled to a relocatable .loombc (`loom-link --allow-unresolved`, its srcs merged), cached per label.
  * the target's own srcs (labels like //pkg:file.loom and filegroups are resolved under the root) are merged the same way.
  * `loom-test test.loombc --library=<dep libs, dependencies first> --config=k=v ...`; `variants` run as target:variant.
Profiles: VM_REFERENCE_PROFILE targets run anywhere (no --device). AMD_GPU_HARDWARE_PROFILE targets need --device=amdgpu;
without --device loom-test still loads and plans the module (links libraries, binds configs) but the GPU scenarios are skipped
by the runner; such targets are reported "skipped", never "pass". A target also is "skipped" when no case/sample executed.
Statuses: pass | fail | skipped (+reason). A target that did not execute its checks is never "pass".
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, time

class Prof(str):
    def __getattr__(self, n): return Noop()

class Noop:
    def __call__(self, *a, **k): return Noop()
    def __getattr__(self, n): return Noop()
    def __iter__(self): return iter(())
    def __getitem__(self, k): return Noop()
    def __add__(self, o): return list(o) if isinstance(o, (list, tuple)) else Noop()
    __radd__ = __add__

def find_root(p):
    p = os.path.abspath(p)
    def ok(d): return all(os.path.isdir(os.path.join(d, x)) for x in ("motif", "program")) or \
                      (os.path.isdir(os.path.join(d, "model")) and os.path.isdir(os.path.join(d, "kernel")))
    for d in (p, os.path.join(p, "pkg")):
        if ok(d): return d
    d = p
    while d != "/":
        if ok(d): return d
        d = os.path.dirname(d)
    sys.exit(f"cannot find the repository root (motif/, program/, ...) from {p}")

class Build:
    """Evaluates BUILD.bazel files lazily, one per package."""
    def __init__(self, root): self.root, self.pk = root, {}
    def package(self, pkg):
        if pkg in self.pk: return self.pk[pkg]
        rules = {"lib": {}, "test": {}, "fg": {}}
        self.pk[pkg] = rules
        path = os.path.join(self.root, pkg, "BUILD.bazel")
        if not os.path.exists(path): path = os.path.join(self.root, pkg, "BUILD")
        if not os.path.exists(path): return rules
        text = open(path, encoding="utf-8").read()
        # drop load(...) statements (balanced parens, starting at line start)
        out, i = [], 0
        for m in re.finditer(r"^load\(", text, re.M):
            if m.start() < i: continue
            depth, j = 0, m.end() - 1
            while True:
                depth += {"(": 1, ")": -1}.get(text[j], 0); j += 1
                if depth == 0: break
            out.append(text[i:m.start()]); i = j
        out.append(text[i:]); text = "".join(out)
        def lib(name, srcs=(), deps=(), **kw): rules["lib"][name] = {"srcs": list(srcs), "deps": list(deps), "kw": kw}
        def test(name, srcs=(), deps=(), configs=None, variants=None, execution_profiles=(), **kw):
            rules["test"][name] = {"srcs": list(srcs), "deps": list(deps), "configs": dict(configs or {}),
                                   "variants": variants, "profiles": [str(p) for p in execution_profiles], "tags": kw.get("tags", [])}
        def fg(name, srcs=(), **kw): rules["fg"][name] = list(srcs)
        g = {"loom_library": lib, "loom_motif_library": lib, "loom_kernel_library": lib, "loom_test": test,
             "filegroup": fg, "package_name": lambda: pkg, "glob": lambda *a, **k: [], "select": lambda d, **k: [],
             "AMD_GPU_HARDWARE_PROFILE": Prof("AMD_GPU_HARDWARE_PROFILE"), "VM_REFERENCE_PROFILE": Prof("VM_REFERENCE_PROFILE"),
             "True": True, "False": False}
        g["native"] = type("N", (), {"filegroup": staticmethod(fg), "package_name": staticmethod(lambda: pkg)})()
        for _ in range(200):
            try:
                exec(compile(text, path, "exec"), g); break
            except NameError as e:
                m = re.search(r"name '(\w+)'", str(e))
                if not m: raise
                g[m.group(1)] = Noop()
        return rules

    def label(self, lab, cur):
        """-> (package, name) for ':x', '//p:x', '//p'."""
        if lab.startswith("//"):
            p, _, n = lab[2:].partition(":")
            return p, n or p.rsplit("/", 1)[-1]
        return cur, lab.lstrip(":")

    def files(self, srcs, cur, seen=None):
        """Expands srcs (files, file labels, filegroups, libraries' srcs) into [(path, label)]; unresolved -> (None, why)."""
        out = []
        for s in srcs:
            if s.startswith(("@", "$")): out.append((None, f"external label {s}")); continue
            pkg, name = self.label(s, cur) if (s.startswith(("//", ":"))) else (cur, s)
            r = self.package(pkg)
            if name in r["fg"] and "/" not in name and not name.endswith(".loom"):
                out += self.files(r["fg"][name], pkg)
            elif name in r["lib"] and not name.endswith(".loom"):
                out += self.files(r["lib"][name]["srcs"], pkg)
            else:
                f = os.path.join(self.root, pkg, name)
                out.append((f, s) if os.path.isfile(f) else (None, f"missing file {pkg}/{name}"))
        return out

def sh(cmd, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr), time.time() - t0
    except subprocess.TimeoutExpired as e:
        return 124, "timeout after %ss" % timeout, time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg"); ap.add_argument("tests")
    ap.add_argument("--loom-test", default="loom-test"); ap.add_argument("--loom-link", default="")
    ap.add_argument("--device", default="", help="e.g. amdgpu; enables AMD_GPU_HARDWARE_PROFILE targets")
    ap.add_argument("--only", default=""); ap.add_argument("-o", default="kernel_tests.json")
    ap.add_argument("--timeout", type=int, default=900); ap.add_argument("--work-dir", default="")
    a = ap.parse_args()
    root = find_root(a.pkg)
    tdir = a.tests if os.path.isabs(a.tests) and os.path.isdir(a.tests) else (a.tests if os.path.isdir(a.tests) else os.path.join(root, a.tests))
    tdir = os.path.abspath(tdir)
    tpkg = os.path.relpath(tdir, root)
    link = a.loom_link or (os.path.join(os.path.dirname(os.path.abspath(a.loom_test)), "loom-link") if os.sep in a.loom_test else "loom-link")
    if os.sep not in link and not any(os.access(os.path.join(d, link), os.X_OK) for d in os.environ.get("PATH", "").split(os.pathsep)):
        sys.exit("loom-link not found next to loom-test or on PATH (use --loom-link)")
    work = a.work_dir or tempfile.mkdtemp(prefix="kt_")
    os.makedirs(work, exist_ok=True)
    B = Build(root)
    tests = B.package(tpkg)["test"]
    only = set(filter(None, a.only.split(",")))
    libcache = {}   # label -> {"module": path|None (aggregator), "trans": [module paths, deps first], "err": str|None}

    def dep_modules(deps, cur):
        """Direct dependency modules (aggregators are transparent) and the ordered transitive closure behind them."""
        direct, trans, err = [], [], None
        for d in deps:
            if d.startswith("@"): continue
            pkg, name = B.label(d, cur)
            info = compile_lib(pkg, name)
            if info["err"]: return [], [], info["err"]
            for m in info["trans"]:
                if m not in trans: trans.append(m)
            if info["module"]:
                if info["module"] not in direct: direct.append(info["module"])
            else:
                for m in info["direct"]:
                    if m not in direct: direct.append(m)
        trans = [m for m in trans if m not in direct]
        return direct, trans, err

    def merge(srcs, direct, trans, out):
        cmd = [link, "--mode=merge", "--to=bc", f"--output={out}"] + srcs + \
              [f"--library={m}" for m in direct] + [f"--transitive-library={m}" for m in trans]
        return sh(cmd, a.timeout)

    def compile_lib(pkg, name):
        key = f"//{pkg}:{name}"
        if key in libcache: return libcache[key]
        info = libcache[key] = {"module": None, "direct": [], "trans": [], "err": None}
        lib = B.package(pkg)["lib"].get(name)
        if not lib: info["err"] = f"{key}: no such loom_library"; return info
        files = B.files(lib["srcs"], pkg)
        bad = [w for f, w in files if f is None] + [f for f, _ in files if f and not f.endswith(".loom")]
        if lib["kw"].get("input_format") or lib["kw"].get("inputopts"): bad.append("non-.loom input_format")
        if bad: info["err"] = f"{key}: cannot translate srcs: {bad}"; return info
        direct, trans, err = dep_modules(lib["deps"], pkg)
        if err: info["err"] = err; return info
        info["direct"] = direct
        info["trans"] = direct + trans   # closure behind this library (without itself)
        if files:
            out = os.path.join(work, hashlib.sha1(key.encode()).hexdigest()[:10] + "_" + name + ".loombc")
            rc, txt, _ = merge([f for f, _ in files], direct, trans, out)
            if rc: info["err"] = f"{key}: loom-link failed: {txt[-400:]}"; return info
            info["module"] = out
        return info

    results = []
    for name, t in tests.items():
        runs = [(name, {})] if not t["variants"] else \
               [(f"{name}:{v}", (spec or {}).get("configs", {})) for v, spec in t["variants"].items()]
        for rname, extra in runs:
            if only and name not in only and rname not in only: continue
            gpu = "AMD_GPU_HARDWARE_PROFILE" in t["profiles"]
            r = {"name": rname, "profile": "amdgpu" if gpu else "vm_reference"}
            results.append(r)
            def done(status, reason=None, tail=None):
                r["status"] = status
                if reason: r["reason"] = reason
                if tail: r["tail"] = tail[-800:]
                print(f"{status.upper():8s} {rname}" + (f"  ({reason[:140]})" if reason else ""), flush=True)
            files = B.files(t["srcs"], tpkg)
            bad = [w for f, w in files if f is None]
            if bad: done("skipped", "cannot translate srcs: " + "; ".join(bad)); continue
            direct, trans, err = dep_modules(t["deps"], tpkg)
            if err: done("skipped", "cannot translate deps: " + err); continue
            srcs = [f for f, _ in files if f.endswith(".loom")]
            if not srcs: done("skipped", "no .loom sources"); continue
            root_mod = os.path.join(work, rname.replace(":", "_") + "_root.loombc")
            rc, txt, _ = merge(srcs, direct, trans, root_mod)      # the loom_test's own relocatable module
            if rc: done("fail", "test sources do not link/parse", txt); continue
            mod = os.path.join(work, rname.replace(":", "_") + "_test.loombc")
            rc, txt, _ = sh([link, "--mode=link", "--include-input-tests", "--to=bc", f"--output={mod}", root_mod] +
                            [f"--library={m}" for m in direct] + [f"--transitive-library={m}" for m in trans], a.timeout)
            if rc: done("fail", "test module does not link against its libraries", txt); continue
            cfg = dict(t["configs"]); cfg.update(extra)
            base = [a.loom_test, mod] + [f"--config={k}={v}" for k, v in cfg.items()]
            cmd = base + ([f"--device={a.device}"] if gpu and a.device else [])
            rc, txt, secs = sh(cmd, a.timeout)
            r["seconds"] = round(secs, 1); r["command"] = " ".join(cmd)
            try: d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1]) if rc == 0 or "{" in txt else {}
            except ValueError: d = {}
            g = lambda k: d.get(k, 0)
            if d:
                r["executed"] = g("sample_count") + g("trial_count")
                r["failed"] = g("failed_sample_count") + g("failed_trial_count")
                r["skipped_records"] = g("skipped_case_count") + g("skipped_scenario_count")
            if rc == 124: done("fail", txt, txt)
            elif gpu and not a.device and ("requires a semantic function subject" in txt or "requires exactly one --device" in txt):
                done("skipped", "AMD GPU profile: needs a GPU device to execute (module loaded and linked ok; run with --device amdgpu)")
            elif rc != 0 or r.get("failed") or g("planning_issue_count") or g("xfail_mismatch_count"):
                done("fail", "loom-test exit %d, %s failed samples/trials" % (rc, r.get("failed", "?")), txt)
                print("    " + txt[-300:].replace("\n", "\n    "), flush=True)
            elif not d or not r["executed"]:
                why = "; ".join(sorted({"%s: %s" % (x.get("code"), x.get("display_message", "")) for x in d.get("skipped_scenarios", []) + d.get("skipped_cases", [])})) if d else "no result record"
                done("skipped", ("ran without a GPU device (%s); module loads and plans (compile check), nothing executed" % why) if gpu and not a.device
                     else "nothing executed: " + why)
            elif r["skipped_records"]:
                done("skipped", "%d executed ok but %d case/scenario records skipped" % (r["executed"], r["skipped_records"]))
            else:
                done("pass", tail=txt[-300:])
    counts = {k: sum(1 for r in results if r["status"] == k) for k in ("pass", "fail", "skipped")}
    by = {}
    for r in results:
        by.setdefault((r["profile"], r["status"]), 0); by[(r["profile"], r["status"])] += 1
    json.dump({"root": root, "test_dir": tpkg, "device": a.device or None, "counts": counts,
               "by_profile": {f"{p}/{s}": n for (p, s), n in sorted(by.items())}, "results": results},
              open(a.o, "w"), indent=1)
    print(f"{counts['pass']} pass, {counts['fail']} fail, {counts['skipped']} skipped of {len(results)}  ->  {a.o}")
    print(f"{counts['pass']}/{len(results)} passed")   # make_info.py gates reads this last line; a skipped target is not a pass
    sys.exit(1 if counts["fail"] else 0)

main()
