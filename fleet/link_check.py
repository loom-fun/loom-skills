#!/usr/bin/env python3
"""Link check of fleet builds, CPU only: the runtime's link of every package's programs, offline (loom-link).

usage: link_check.py FLEET_DIR [--slugs A,B] [--jobs N] [--json OUT]       (FLEET_DIR = build_fleet.py --out DIR)

The runtime links model.json's program (the `inputs` files jointly, the `libraries` beside them) with the variant's card config. A config
key defined in two of those files, or any other link rule broken by a change of the programs, fails only at load time on a GPU
(`LINK/MATERIALIZE: exact link selection missed reachable source symbol`): no per-file tool (loom-opt, loom-compile) sees it. This tool
runs `loom-link --mode=link` with every public function of the program as a root, the card config bound as the runtime binds it, and
(for packages that declare llm.logits) once more with the capability on. Exit 1 on any failure. Needs loom-link (hrx-loom 0.1.0; set LOOM_LINK).
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys

ROOTS = re.compile(r"^(?:func|command)\.def public @(\w+)", re.M)


def tool():
    for c in (os.environ.get("LOOM_LINK"), shutil.which("loom-link"), "/home/jfowers/lt/loom010/.venv/bin/loom-link"):
        if c and os.path.exists(c):
            return c
    sys.exit("loom-link not found (set LOOM_LINK)")


def link(exe, pkg, prog, extra):
    inputs = [i["path"] if isinstance(i, dict) else i for i in prog["inputs"]]
    texts = "".join(open(os.path.join(pkg, p), encoding="utf-8").read() for p in inputs + prog.get("libraries", []))
    roots = sorted(set(ROOTS.findall(texts)))
    cmd = [exe, *inputs, *[f"--library={p}" for p in prog.get("libraries", [])], "--mode=link", *[f"--root=@{r}" for r in roots], *extra, "--to=text"]
    r = subprocess.run(cmd, cwd=pkg, capture_output=True, text=True)
    return r.returncode, (r.stderr or r.stdout)[-400:].strip()


def check(exe, pkg):
    model = json.load(open(os.path.join(pkg, "model.json")))
    problems, skipped = [], set()
    for vname, var in model["variants"].items():
        for sname, svc in var["services"].items():
            cfg = svc.get("config", {})
            for pname in svc["programs"]:
                prog = model["programs"][pname]
                base = [f"--config={k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items()]
                runs = [("card config", base)]
                if "llm.logits" in cfg:
                    runs.append(("llm.logits on", [b for b in base if not b.startswith("--config=llm.logits=")] + ["--config=llm.logits=true"]))
                for what, extra in runs:
                    rc, msg = link(exe, pkg, prog, extra)
                    if rc and "error [PARSE/" in msg:   # syntax the installed loom-link does not know (a newer port): not a link verdict
                        skipped.add(f"{vname}/{sname}/{pname}")
                    elif rc:
                        problems.append(f"{vname}/{sname}/{pname} ({what}): {msg}")
    return problems, sorted(skipped)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fleet")
    ap.add_argument("--slugs")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()
    exe = tool()
    slugs = a.slugs.split(",") if a.slugs else sorted(d for d in os.listdir(a.fleet) if os.path.isfile(os.path.join(a.fleet, d, "model.json")))
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = dict(zip(slugs, ex.map(lambda s: check(exe, os.path.join(a.fleet, s)), slugs)))
    bad = {s: p[0] for s, p in res.items() if p[0]}
    skipped = {s: p[1] for s, p in res.items() if p[1]}
    for s, p in sorted(bad.items()):
        print(f"FAIL {s}: " + " | ".join(p))
    for s in sorted(skipped):
        print(f"SKIP {s}: the installed loom-link cannot parse its programs ({', '.join(skipped[s])})")
    print(f"link check: {len(slugs) - len(bad) - len(skipped)} of {len(slugs)} package(s) link, {len(skipped)} skipped, {len(bad)} fail")
    if a.json:
        json.dump({"checked": slugs, "failed": bad, "skipped": skipped}, open(a.json, "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
