#!/usr/bin/env python3
"""Parse sweep + link sweep of a fleet (or one package) under any hrx-loom SDK, CPU only. Version-agnostic.

usage: sweep.py --sdk VENV FLEET_DIR [--slugs A,B] [--jobs N] [--json OUT] [--no-link]

FLEET_DIR is a directory of packages (each holds model.json; build_fleet.py --out DIR makes one) or one package. For every package:
  parse  loom-opt FILE for every program file model.json names (inputs and libraries): is the source valid in this SDK?
  link   loom-link --mode=link of each variant/service program with every public function as a root and the card config bound as the
         runtime binds it (what fleet/link_check.py does): a link rule broken by the new SDK fails only at load on a GPU otherwise.
         A package whose files do not parse is not linked (reported as link-skipped).
--sdk VENV is a venv (or its bin directory) holding loom-opt and loom-link. The output has a histogram of each package's FIRST error
class (parse, else link): the class names the breaking change (e.g. PARSE/006 unknown op = a renamed op), the count says how wide.
Exit 1 if any package has an error. Standard library only.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys

ERR = re.compile(r"(?:^|\n)(?:(?P<where>[^\s:][^\n]*?):(?P<line>\d+):(?P<col>\d+): )?error \[(?P<code>[A-Z_]+/\d+)\]: (?P<msg>[^\n]*)")
ROOTS = re.compile(r"^(?:func|command)\.def public @([\w.]+)", re.M)


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def write(p, text):
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def find_tool(sdk, name):
    for d in (sdk, os.path.join(sdk, "bin"), os.path.join(sdk, "Scripts")):
        for ext in ("", ".exe"):
            p = os.path.join(d, name + ext)
            if os.path.isfile(p):
                return os.path.abspath(p)
    sys.exit(f"{name} not found under {sdk}")


def sdk_version(sdk):
    for d in (sdk, os.path.join(sdk, "bin"), os.path.join(sdk, "Scripts")):
        for py in ("python", "python.exe", "python3"):
            p = os.path.join(d, py)
            if os.path.isfile(p):
                r = subprocess.run([p, "-c", "import importlib.metadata as m;print(m.version('hrx-loom'))"], capture_output=True, text=True)
                if r.returncode == 0:
                    return r.stdout.strip()
    return None


def classify(text):
    """First error of a tool's output as {class, msg}; the class is code + message with quoted names and numbers folded."""
    m = ERR.search(text)
    if not m:
        return {"class": "UNKNOWN", "msg": text.strip()[-300:]}
    msg = re.sub(r"'[^']*'", "'X'", m["msg"])
    msg = re.sub(r"\d+", "N", msg)
    return {"class": f"{m['code']} {msg}", "code": m["code"], "msg": m["msg"], "at": f"{m['where']}:{m['line']}" if m["where"] else None}


def program_files(model):
    out = []
    for prog in model.get("programs", {}).values():
        for i in prog.get("inputs", []) + prog.get("libraries", []):
            for p in ([i["path"], *i.get("data", [])] if isinstance(i, dict) else [i]):   # data inputs too: weight policies (head_weights.loom) sit at the package root
                p = p["path"] if isinstance(p, dict) else p
                if p.endswith(".loom") and p not in out:
                    out.append(p)
    return out


def parse_pkg(opt, pkg, files):
    errs = []
    for f in files:
        r = subprocess.run([opt, f, "--output=" + os.devnull], cwd=pkg, capture_output=True, text=True)
        if r.returncode:
            c = classify(r.stderr + r.stdout)
            c["file"] = f
            errs.append(c)
    return errs


def link_pkg(link, pkg, model):
    errs = []
    for vname, var in model.get("variants", {}).items():
        for sname, svc in var.get("services", {}).items():
            cfg = svc.get("config", {})
            for pname in svc.get("programs", []):
                prog = model["programs"][pname]
                inputs = [i["path"] if isinstance(i, dict) else i for i in prog.get("inputs", [])]
                libs = [entry["path"] if isinstance(entry, dict) else entry
                        for entry in prog.get("libraries", [])]
                texts = "\n".join(read(os.path.join(pkg, p)) for p in inputs + libs if p.endswith(".loom"))
                roots = sorted(set(ROOTS.findall(texts)))
                base = [f"--config={k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items()]
                runs = [("card config", base)]
                if "llm.logits" in cfg:
                    runs.append(("llm.logits on", [b for b in base if not b.startswith("--config=llm.logits=")] + ["--config=llm.logits=true"]))
                for what, extra in runs:
                    cmd = [link, *inputs, *[f"--library={p}" for p in libs], "--mode=link", *[f"--root=@{r}" for r in roots], *extra, "--to=text"]
                    r = subprocess.run(cmd, cwd=pkg, capture_output=True, text=True)
                    if r.returncode:
                        c = classify(r.stderr + r.stdout)
                        c["where"] = f"{vname}/{sname}/{pname} ({what})"
                        errs.append(c)
    return errs


def check(opt, link, pkg, do_link):
    try:
        model = json.loads(read(os.path.join(pkg, "model.json")))
    except (OSError, ValueError) as e:
        return {"parse": [{"class": "MODEL_JSON", "msg": str(e), "file": "model.json"}], "link": [], "link_skipped": True}
    files = program_files(model)
    perr = parse_pkg(opt, pkg, files)
    res = {"files": len(files), "parse": perr, "link": [], "link_skipped": False}
    if do_link:
        if perr:
            res["link_skipped"] = True
        else:
            res["link"] = link_pkg(link, pkg, model)
    return res


def packages(root, slugs):
    if os.path.isfile(os.path.join(root, "model.json")):
        return [(os.path.basename(os.path.abspath(root)), root)]
    names = slugs or sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "model.json")))
    return [(n, os.path.join(root, n)) for n in names]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fleet")
    ap.add_argument("--sdk", required=True, help="venv (or bin dir) with loom-opt and loom-link")
    ap.add_argument("--slugs")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("--no-link", action="store_true", help="parse sweep only")
    a = ap.parse_args(argv)
    opt, link = find_tool(a.sdk, "loom-opt"), find_tool(a.sdk, "loom-link")
    pkgs = packages(a.fleet, a.slugs.split(",") if a.slugs else None)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        results = dict(zip([n for n, _ in pkgs], ex.map(lambda p: check(opt, link, p[1], not a.no_link), pkgs)))
    first, every = collections.Counter(), collections.Counter()
    for r in results.values():
        errs = r["parse"] + r["link"]
        if errs:
            first[errs[0]["class"]] += 1
        for e in errs:
            every[e["class"]] += 1
    bad_parse = sorted(s for s, r in results.items() if r["parse"])
    bad_link = sorted(s for s, r in results.items() if r["link"])
    out = {
        "sdk": {"path": a.sdk, "version": sdk_version(a.sdk)},
        "totals": {"packages": len(results), "parse_failing": len(bad_parse), "link_failing": len(bad_link),
                   "link_skipped": sum(r["link_skipped"] for r in results.values()),
                   "parse_error_files": sum(len(r["parse"]) for r in results.values())},
        "first_error_class": dict(first.most_common()),
        "all_error_class": dict(every.most_common()),
        "packages": results,
    }
    t = out["totals"]
    print(f"sdk {out['sdk']['version'] or a.sdk}: {t['packages']} package(s), {t['parse_failing']} fail to parse ({t['parse_error_files']} files), "
          f"{t['link_failing']} fail to link, {t['link_skipped']} link-skipped")
    for c, n in first.most_common():
        print(f"  {n:4d} package(s) first fail with: {c}")
    if a.json:
        write(a.json, json.dumps(out, indent=1))
    return 1 if (bad_parse or bad_link) else 0


if __name__ == "__main__":
    sys.exit(main())
