#!/usr/bin/env python3
"""route_load.py: how many instruction lines a skill run loads, per fleet model (stdlib only).

For each of the fleet configs x {port, optimize} x {strix-halo, radeon-r9700}:
  new  = lines of the files `kb.py route` prints for the task's topics + the workflow SKILL.md + loom-kb SKILL.md
  old  = the pre-refactor set at --base (default PRE_REFACTOR: r9700 before the refactor merge #850): port-llm SKILL.md + loom-ir-cookbook + porting-notes
         (+ vision.md for vision models); optimize SKILL.md + optimization-catalog
Tasks: port = topics porting,kernels-ir; optimize-prefill / optimize-decode = the one metric being optimized, optimize-both
= prefill,decode together (a run that also benchmarks or rooflines adds those SKILL.md files on both sides equally, so they
are left out). The "all topics" column adds the on-demand topics a run reads only when it hits that need (program ABI, IR
debugging, cfg review, starter reference; optimize-loop mechanics, roofline, concurrency, long-context): the worst case.

    python3 fleet/route_load.py [--tree .] [--base REF] [--md OUT.md] [--json OUT.json]
"""
import argparse
import contextlib
import io
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASKS = {"port": "porting,kernels-ir", "optimize-prefill": "prefill", "optimize-decode": "decode",
         "optimize-both": "prefill,decode"}
EXTRA = {"port": "program-abi,ir-debug,cfg-review,starter", "optimize-prefill": "optimize-loop,roofline,concurrency,long-context",
         "optimize-decode": "optimize-loop,roofline,concurrency,long-context", "optimize-both": "optimize-loop,roofline,concurrency,long-context"}
DEVICES = ("strix-halo", "radeon-r9700")
PRE_REFACTOR = "10956463e1409abd8bbe22d1e37c4b57a7d44a48"   # r9700 before the refactor merge (#850): the last tree with the pre-refactor instruction set
OLD = {
    "port": ["skills/loom-port-llm/SKILL.md", "skills/loom-port-llm/reference/loom-ir-cookbook.md",
             "skills/loom-port-llm/reference/porting-notes.md"],
    "optimize": ["skills/loom-optimize/SKILL.md", "skills/loom-optimize/reference/optimization-catalog.md"],
}
NEW_SKILL_KEY = lambda task: "port" if task == "port" else "optimize"
OLD_VISION = "skills/loom-port-llm/reference/vision.md"
NEW_SKILL = {"port": "skills/loom-port-llm/SKILL.md", "optimize": "skills/loom-optimize/SKILL.md"}
KB_SKILL = "skills/loom-kb/SKILL.md"


def nlines(path):
    return len(Path(path).read_text(encoding="utf-8").splitlines())


def old_lines(ref, path):
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"], capture_output=True, text=True, check=True)
    return len(out.stdout.splitlines())


def routed(kb, tree, cfg_path, facets, device, topics):
    ns = type("A", (), dict(config=str(cfg_path), gguf=None, arch=None, facets=facets, device=device,
                            quant=None, topic=topics, relative=False))()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        kb.route(ns, root=tree / "skills/loom-kb")
    return [Path(l) for l in buf.getvalue().splitlines() if l and not l.startswith("#")]


def stats(vals):
    return min(vals), statistics.median(vals), max(vals)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=str(REPO))
    ap.add_argument("--base", default=PRE_REFACTOR)
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    tree = Path(a.tree).resolve()
    sys.path.insert(0, str(tree / "skills/loom-kb"))
    import kb
    manifest = json.loads((REPO / "fleet/manifest.json").read_text())["models"]
    rows = []
    for slug, entry in sorted(manifest.items()):
        cfg_path = REPO / "fleet" / slug / "llm_config.json"
        cfg = json.loads(cfg_path.read_text())
        drafted = any(f["origin"] == "drafter" for f in entry["files"].values())
        vision = bool(cfg.get("vision")) or any(f["origin"] == "vision" for f in entry["files"].values())
        for task, topics in TASKS.items():
            old = sum(old_lines(a.base, p) for p in OLD[NEW_SKILL_KEY(task)])
            if vision and task == "port":
                old += old_lines(a.base, OLD_VISION)
            for dev in DEVICES:
                files = routed(kb, tree, cfg_path, "drafted" if drafted else None, dev, topics)
                new = sum(nlines(f) for f in files) + nlines(tree / NEW_SKILL[NEW_SKILL_KEY(task)]) + nlines(tree / KB_SKILL)
                allf = routed(kb, tree, cfg_path, "drafted" if drafted else None, dev, topics + "," + EXTRA[task])
                full = sum(nlines(f) for f in allf) + nlines(tree / NEW_SKILL[NEW_SKILL_KEY(task)]) + nlines(tree / KB_SKILL)
                rows.append(dict(model=slug, task=task, device=dev, old=old, new=new, full=full, files=len(files) + 2))
    md = ["| task | device | set | min | median | max | median vs pre-refactor | median, all topics |", "|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for dev in DEVICES:
            sel = [r for r in rows if r["task"] == task and r["device"] == dev]
            o, n = stats([r["old"] for r in sel]), stats([r["new"] for r in sel])
            f = statistics.median([r["full"] for r in sel])
            md.append(f"| {task} | {dev} | pre-refactor | {o[0]} | {o[1]:g} | {o[2]} | 100 % | {o[1]:g} |")
            md.append(f"| {task} | {dev} | routed | {n[0]} | {n[1]:g} | {n[2]} | {100 * n[1] / o[1]:.0f} % | {f:g} ({100 * f / o[1]:.0f} %) |")
    text = "\n".join(md) + f"\n\n{len(rows) // (len(TASKS) * len(DEVICES))} fleet configs, base {a.base}.\n"
    print(text)
    if a.md:
        Path(a.md).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
