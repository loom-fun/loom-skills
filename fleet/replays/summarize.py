#!/usr/bin/env python3
"""Summary tables of the replays under fleet/replays/ (reads every replay's meta.json, *.summary.json, *.lint.txt, *.plan.txt).

usage: summarize.py [--md OUT.md] [--json OUT.json]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET = os.path.dirname(HERE)
sys.path.insert(0, FLEET)
import scope_lint as SL  # noqa: E402

ORDER = ["r357", "r451", "d7", "r573", "r656", "r480", "r692", "r694", "r686"]
OWN = json.load(open(os.path.join(FLEET, "ownership.json")))
PROTECTED = [p["glob"] for p in OWN["protected"]]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def lint_info(path):
    if not os.path.isfile(path):
        return None
    t = open(path).read()
    rules = []
    for m in re.finditer(r"^\| ([a-z -]+?) \| ", t, re.M):
        r = m.group(1)
        if r not in ("rule",) and r not in rules:
            rules.append(r)
    leak = re.search(r"leaked outside the declared scope: (\d+)", t)
    return {"exit": int(re.match(r"exit (\d+)", t).group(1)), "rules": rules, "leaked": int(leak.group(1)) if leak else 0}


def plan_info(path):
    if not os.path.isfile(path):
        return None
    reps = {"strix-halo": set(), "radeon-r9700": set()}
    strata = set()
    moe_drafted = False
    for line in open(path):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 5 and c[0] in reps:
            strata.add((c[0], c[1]))
            for m in re.finditer(r"(\S+-loom)", c[4]):
                reps[c[0]].add(m.group(1))
            if "arch=qwen35moe" in c[1] and "draft=none" not in c[1]:
                moe_drafted = True
    allr = sorted(set().union(*reps.values()))
    t = re.search(r"T1 canary ~([\d.]+) runner-h vs full A/B of every touched model ~([\d.]+)", open(path).read())
    return {"reps": allr, "n": len(allr), "strix": len(reps["strix-halo"]), "r9700": len(reps["radeon-r9700"]),
            "strata": len(strata), "qwen35moe_drafted": moe_drafted, "canary_h": float(t.group(1)) if t else None,
            "full_h": float(t.group(2)) if t else None}


def verdict(l):
    if l is None:
        return "-"
    return "pass" if l["exit"] == 0 else "FAIL (" + ", ".join(l["rules"]) + ")"


def collect():
    out = {}
    for name in ORDER:
        d = os.path.join(HERE, name)
        if not os.path.isfile(os.path.join(d, "meta.json")):
            continue
        meta = load(os.path.join(d, "meta.json"))
        e = {"meta": meta, "variants": {}}
        if meta.get("commit"):
            s = load(os.path.join(d, "summary.json")) or {}
            e["commit"] = s
            for v in ("A", "B"):
                lint = lint_info(os.path.join(d, v + ".lint.txt")) if v == "A" else {"exit": s.get("B", {}).get("lint_exit", 1), "rules": [], "leaked": 0}
                e["variants"][v] = {"lint": lint, "touched": len(s.get("impact_models", [])), "files": s.get("files_changed", [])}
        else:
            for f in sorted(os.listdir(d)):
                m = re.match(r"(.+)\.lint\.txt$", f)
                if not m:
                    continue
                v = m.group(1)
                s = load(os.path.join(d, v + ".summary.json")) or {}
                e["variants"][v] = {"lint": lint_info(os.path.join(d, f)), "plan": plan_info(os.path.join(d, v + ".plan.txt")),
                                    "plan_shape": plan_info(os.path.join(d, v + ".plan.shape.txt")),
                                    "touched": (s.get("impact_summary") or {}).get("touched"), "by_family": s.get("touched_by_family"),
                                    "files": s.get("files_edited") or [], "touched_models": s.get("touched_models")}
        out[name] = e
    return out


def row(name, e):
    m, V = e["meta"], e["variants"]
    if m.get("commit"):
        a, b = V.get("A", {}), V.get("B", {})
        parts = e["commit"].get("B", {}).get("parts", {})
        bl = ", ".join(f"{k} {'pass' if p['lint_exit'] == 0 else 'FAIL'}" for k, p in parts.items())
        return (f"| {m['pr']} | {a['touched']} / 0 / {verdict(a['lint'])} | {b['touched']} / {len(parts)} PRs: {bl} | none (no program touched) | "
                f"yes: nothing to sample |")
    a = V.get("A")
    b = V.get("B")
    ex = [v for v in V if v not in ("A", "B")]

    def reps(x):
        p, q = x.get("plan"), x.get("plan_shape")
        return (f"{p['n']} models ({p['strata']} strata)" + (f", {q['n']} by shape" if q else "")) if p else "-"
    leak = a["lint"]["leaked"] if a["lint"] else "-"
    vA = verdict(a["lint"]) + "".join(f"; {v}: {verdict(V[v]['lint'])}" for v in ex)
    narrow = "yes" if b["lint"] and b["lint"]["exit"] == 0 and b["lint"]["leaked"] == 0 else "no"
    return (f"| {m['pr']} | {a['touched']} / {leak} / {vA} | {b['touched']} / {verdict(b['lint'])} | A {reps(a)}; B {reps(b)} | "
            f"A: no, B: {narrow} |")


def tables(d):
    L = ["| PR | original: touched / leak / lint | scoped: touched / lint | sampler representatives | narrow? |", "|---|---|---|---|---|"]
    for n, e in d.items():
        L.append(row(n, e))
    L += ["", "| PR | files edited, original (protected) | files edited, scoped (protected) |", "|---|---|---|"]
    for n, e in d.items():
        if e["meta"].get("commit"):
            parts = e["commit"].get("B", {}).get("parts", {})
            fa = e["variants"]["A"]["files"]
            prot = sum(1 for f in fa if SL.globs_match(PROTECTED, f))
            L.append(f"| {e['meta']['pr']} | {len(fa)} ({prot}) | {len(fa)} in {len(parts)} PRs by kind ({prot} in the architecture PR) |")
            continue
        V = e["variants"]
        fa, fb = V["A"]["files"], V["B"]["files"]
        pa = sum(1 for f in fa if SL.globs_match(PROTECTED, f))
        pb = sum(1 for f in fb if SL.globs_match(PROTECTED, f))
        L.append(f"| {e['meta']['pr']} | {len(fa)} ({pa}) | {len(fb)} ({pb}) |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = collect()
    t = tables(d)
    if a.json:
        json.dump(d, open(a.json, "w"), indent=1)
    if a.md:
        open(a.md, "w").write(t)
    print(t)
