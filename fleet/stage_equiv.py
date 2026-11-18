#!/usr/bin/env python3
"""Stage equivalence of two fleet builds, offline: do the default-path command programs (stages) of every package compile to the
same code at HEAD as at BASE?

usage: stage_equiv.py BASE_DIR HEAD_DIR [--json OUT.json] [--md OUT.md] [--jobs N] [--slugs A,B] [--compile] [--allow-added SYM,...]
       (BASE_DIR / HEAD_DIR: build_fleet.py outputs)

Command programs have no offline compile product (loom-compile: "command-program roots require loomc_cmd_program_product_build"), so the
proof runs loom-opt's folding pipeline (canonicalize, cse, dce, symbol-dce) over each package's program/llm.loom with the stage configs
bound (two epoch shapes: a decode shape and a prefill shape; the drafting configs too), renumbers the SSA names, splits the result into
top-level symbols and compares them by name. A package PASSES when every command program (stage) and every kernel/function/template of
BASE exists in HEAD with identical text after folding, and what HEAD adds or changes is only the allowed symbols (default: the logits
readback's kernel g_logits_export and its config symbols, which no default stage reaches). `llm.logits` is left at its default (off): this is
the property that lets BASE's card evidence carry over to HEAD. Both targets: a dual-target build (the fleet default) holds the gfx11
and gfx12 twins in the one text, so the comparison covers both; --compile also compiles every added kernel for gfx1151 and gfx1201.
The json holds, per package and shape, the sha256 of every stage's folded text, base and head.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loomtext  # noqa: E402

ALLOWED_ADDED = ("g_logits_export", "llm.logits", "llm.logit_slots")
SHAPES = {"decode": ("1", "1"), "prefill": ("256", "4")}   # llm.tokens, llm.spans
START = re.compile(r"^(command\.program\.def|command\.def|kernel\.def|func\.def|func\.decl|template\.def|template\.decl|global\.rodata\.def|config\.(?:def|decl)|target\.decl|amdgpu\.target)\b")
SYMBOL = re.compile(r'@([\w.$]+)')
PASSES = ["--pass=canonicalize", "--pass=cse", "--pass=dce", "--pass=symbol-dce"]
EXTRA = ["--config=llm.width=4", "--config=llm.gated=0", "--config=llm.epoch_spans=4", "--config=llm.epoch_tokens=64"]
TARGETS = ("amdgpu:gfx1151", "amdgpu:gfx1201")


def tool(name):
    for c in (os.environ.get(name.upper().replace("-", "_")), shutil.which(name)):
        if c and os.path.exists(c):
            return c
    sys.exit(f"{name} not found (set {name.upper().replace('-', '_')})")


def fold(path, shape):
    tokens, spans = SHAPES[shape]
    cfg = [f"--config=llm.context=4096", f"--config=llm.pool=4096", f"--config=llm.tokens={tokens}", f"--config=llm.spans={spans}"] + EXTRA
    # v0.1 programs read as v0.2 (the loom-opt of the SDK under test only parses its own spelling)
    with open(path) as fh:
        text = loomtext.upgrade_v01(fh.read())
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "llm.loom")
        with open(tmp, "w") as fh:
            fh.write(text)
        r = subprocess.run([tool("loom-opt"), tmp] + PASSES + cfg, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"loom-opt {path} ({shape}): {r.stderr[-300:]}")
    return re.sub(r"\$\d+", "$N", re.sub(r"%\d+", "%N", r.stdout))


def split(text):
    """{kind symbol: text} of the top-level symbols (unnamed ones are keyed by their ordinal)."""
    blocks, cur, key, n = {}, [], None, 0

    def close():   # comments and blank lines between symbols belong to no symbol
        while cur and (not cur[-1].strip() or cur[-1].lstrip().startswith("//")):
            cur.pop()
        return "\n".join(cur)
    for line in text.splitlines():
        if START.match(line):
            if cur and key:
                blocks[key] = close()
            m = re.search(r'@([\w.$]+)\s*\(', line) or SYMBOL.search(line)
            n += 1
            key = (START.match(line).group(1).replace("command.program.", "command.") + " " + (m.group(1) if m else f"#{n}"))
            cur = [line]
        elif cur:
            cur.append(line)
    if cur and key:
        blocks[key] = close()
    return blocks


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def compile_kernel(path, symbol, target):
    out = path + f".{symbol}.{target.split(':')[1]}.hsaco"
    r = subprocess.run([tool("loom-compile"), path, f"--root=@{symbol}", f"--target={target}", "--format=amdgpu-hsaco", f"--output={out}"],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    if os.path.exists(out):
        os.remove(out)
    return ok


def check(slug, base, head, allow, do_compile):
    res = {"slug": slug, "shapes": {}, "status": "PASS", "problems": []}
    bp, hp = os.path.join(base, slug, "program/llm.loom"), os.path.join(head, slug, "program/llm.loom")
    if not (os.path.exists(bp) and os.path.exists(hp)):
        return dict(res, status="SKIP", problems=["no program/llm.loom in base or head"])
    added_all = set()
    for shape in SHAPES:
        try:
            b, h = split(fold(bp, shape)), split(fold(hp, shape))
        except RuntimeError as e:
            return dict(res, status="ERROR", problems=[str(e)])
        stages = {k: {"base": sha(v), "head": sha(h[k]) if k in h else None} for k, v in b.items() if k.startswith("command.def")}
        removed = sorted(set(b) - set(h))
        changed = sorted(k for k in b if k in h and b[k] != h[k])
        added = sorted(set(h) - set(b))
        added_all.update(added)
        res["shapes"][shape] = {"stages": stages, "symbols_base": len(b), "removed": removed, "changed": changed, "added": added}
        # an allowed symbol (the logits readback, which no default stage reaches) may also CHANGE, e.g. a fix inside g_logits_export
        changed_ok = [k for k in changed if any(a in k for a in allow)]
        added_all.update(changed_ok)
        for what, items in (("removed", removed), ("changed", [k for k in changed if k not in changed_ok])):
            if items:
                res["problems"].append(f"{shape}: {what} {items[:5]}")
        extra = [k for k in added if not any(a in k for a in allow)]
        if extra:
            res["problems"].append(f"{shape}: added outside the allowed set {extra[:5]}")
    if do_compile and "kernel.def g_logits_export" in added_all:
        res["compiled"] = {t: compile_kernel(hp, "g_logits_export", t) for t in TARGETS}
        for t, ok in res["compiled"].items():
            if not ok:
                res["problems"].append(f"g_logits_export does not compile for {t}")
    res["status"] = "PASS" if not res["problems"] else "FAIL"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base"), ap.add_argument("head")
    ap.add_argument("--json"), ap.add_argument("--md")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--slugs")
    ap.add_argument("--compile", action="store_true", help="compile the added kernels for gfx1151 and gfx1201")
    ap.add_argument("--allow-added", default=",".join(ALLOWED_ADDED))
    a = ap.parse_args()
    allow = tuple(x for x in a.allow_added.split(",") if x)
    slugs = a.slugs.split(",") if a.slugs else sorted(d for d in os.listdir(a.base) if os.path.isdir(os.path.join(a.base, d)))
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda s: check(s, a.base, a.head, allow, a.compile), slugs))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = {"base": a.base, "head": a.head, "allowed_added": list(allow), "counts": counts, "packages": results}
    if a.json:
        json.dump(summary, open(a.json, "w"), indent=1)
    lines = [f"## Stage equivalence: {counts}", "",
             "| package | status | stages (decode/prefill) | added symbols | problems |", "|---|---|---|---|---|"]
    for r in results:
        sh = r["shapes"].get("decode", {})
        n = len(sh.get("stages", {}))
        same = sum(1 for v in sh.get("stages", {}).values() if v["base"] == v["head"])
        lines.append(f"| {r['slug']} | {r['status']} | {same}/{n} identical | {len(sh.get('added', []))} | {'; '.join(r['problems'])[:160]} |")
    text = "\n".join(lines) + "\n"
    if a.md:
        open(a.md, "w").write(text)
    print(text if not a.md else "\n".join(lines[:1]))
    return 0 if set(counts) <= {"PASS", "SKIP"} else 1


if __name__ == "__main__":
    sys.exit(main())
