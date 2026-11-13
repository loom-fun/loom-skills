#!/usr/bin/env python3
"""Offline compile sweep: build each package with THIS checkout's kit, then compile every kernel of every program file
for gfx1201 and gfx1151 (loom-compile --root=<kernel>, no GPU) and tabulate the failures by kernel family.

usage: python sweep_compile.py [--pkgs DIR] [--work DIR] [--out sweep.json] [--slugs A,B,...] [--targets gfx1201,gfx1151]
                               [--keep] [--jobs N]

--pkgs  directory of published packages (<slug>/llm_config.json, tokenizer.json, model.json); default
        ~/lt/r9700/pkgs. Every package whose slug is in --slugs (default: every package with an llm_config.json, minus
        SKIP) is rebuilt: inspect_gguf.py on the GGUF from the Hugging Face cache, the package's own cfg fields
        overlaid (chat block, drafter, vision, ...), then build.py (example kits through reference/apply_example.sh).
--work  scratch directory on a real disk (never /tmp: it is RAM). One package at a time; its build is deleted unless
        --keep.
--out   sweep.json: {"summary", "families", "packages": {slug: {"build", "kernels": {file:kernel: {gfx1201, gfx1151}}}}}

A kernel's result is "ok" or "CODE op / constraint" (or the error line). Failures are tabulated by kernel family and by
where they occur: one target only, or both. A failure on both targets is not a port bug when the kernel needs its
launch's constants to prove its bounds or strides (a standalone-root artifact: g_silu_mul(_lv) SUBRANGE/010, the
LFM2.5 g_moe_* fragment.store dynamic_stride); those compile in the model. Failures are the gfx1201-only rows.
Command programs cannot be compiled offline (loom-compile has no command-program product); kernels only.
"""
import argparse
import collections
import concurrent.futures as cf
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STARTER = os.path.join(HERE, "starter")
LOOMC = os.environ.get("LOOM_COMPILE", os.path.expanduser("~/lt/loom010/.venv/bin/loom-compile"))
HF = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
SKIP = {"bartowski-Ornith-1.5-122B-loom", "unsloth-Qwen3.8-Flash-Next-loom", "Qwen3-0.6B"}   # not on the R9700 (do not fit / Strix only)
# kernel -> (code) that fails as a standalone root on every target but compiles in its program (the bound proof needs
# the launch's constant n; see geglu_mul in starter/gen_llm.py): g_silu_mul(_lv) SUBRANGE/010.
KNOWN_ROOT_ARTIFACTS = {"g_silu_mul": "SUBRANGE/010", "g_silu_mul_lv": "SUBRANGE/010"}


def family(kernel):
    """g_gemm_q4_k_store_m128_q8 -> g_gemm_q4_k_store_m128_q8 with sizes/shape suffixes folded: k32768 -> kN, t16 -> tN."""
    k = re.sub(r"_k\d+", "_kN", kernel)
    k = re.sub(r"_t\d+", "_tN", k)
    k = re.sub(r"_h\d+", "_hN", k)
    k = re.sub(r"_d\d+", "_dN", k)
    k = re.sub(r"\d+x\d+", "NxN", k)
    return k


_INDEX = {}


def _add(p):
    """Index p by basename; of several files with one name (header stubs, truncated downloads) keep the largest."""
    n, rp = os.path.basename(p), os.path.realpath(p)
    if n not in _INDEX or os.path.getsize(rp) > os.path.getsize(_INDEX[n]):
        _INDEX[n] = rp


def find_in_cache(name):
    """GGUF `name` from the Hugging Face cache, else from any file named so under GGUF_DIRS (default ~/lt: the
    campaigns keep copies of GGUFs there when a snapshot was evicted). Dangling links are skipped."""
    if not _INDEX:
        for p in glob.glob(f"{HF}/models--*/snapshots/*/*.gguf"):
            if os.path.exists(p):
                _add(p)
        for root in os.environ.get("GGUF_DIRS", os.path.expanduser("~/lt")).split(":"):
            for dp, _, fn in os.walk(root):
                if "/eng/" in dp:
                    continue
                for f in fn:
                    p = os.path.join(dp, f)
                    if f.endswith(".gguf") and os.path.exists(p):
                        _add(p)
    return _INDEX.get(name)


def link_gguf(name, work):
    """Link GGUF `name` (and, for a split GGUF, every sibling in its snapshot) into work."""
    p = find_in_cache(name)
    if not p:
        return False
    d = os.path.dirname(p)
    stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", name)
    for f in os.listdir(d):
        if f == name or (stem != name and f.startswith(stem)):
            dst = os.path.join(work, f)
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(os.path.join(d, f)), dst)
    dst = os.path.join(work, name)
    if not os.path.lexists(dst):
        os.symlink(p, dst)
    return True


def sh(cmd, cwd, env=None, timeout=3600):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout + p.stderr)


def build(slug, pkgdir, work):
    """Rebuild slug's package into work/pkg with this checkout's kit. Returns (ok, message, program_dir)."""
    pcfg = json.load(open(f"{pkgdir}/llm_config.json"))
    os.makedirs(work, exist_ok=True)
    names = [pcfg["gguf"]]
    for k in ("drafter", "assistant"):
        if (pcfg.get(k) or {}).get("gguf"):
            names.append(pcfg[k]["gguf"])
    if (pcfg.get("vision") or {}).get("mmproj"):
        names.append(pcfg["vision"]["mmproj"])
    for n in names:
        if not link_gguf(os.path.basename(n), work):
            return False, f"GGUF not in the HF cache: {n}", None
    # kit: the starter, plus the example the package was generated with
    kit = os.path.join(work, "kit")
    ex = (pcfg.get("generator", {}).get("kit") or {}).get("example")
    if ex:
        rc, out = sh(["bash", f"{HERE}/reference/apply_example.sh", ex, kit], work)
        if rc:
            return False, "apply_example: " + out[-300:], None
    else:
        shutil.copytree(STARTER, kit)
    py = sys.executable
    fresh = os.path.join(work, "fresh.json")
    rc, out = sh([py, f"{kit}/inspect_gguf.py", os.path.basename(pcfg["gguf"]), fresh], work)
    if rc:
        return False, "inspect_gguf: " + out.strip().splitlines()[-1][-250:], None
    cfg = json.load(open(fresh))
    cfg.update({k: v for k, v in pcfg.items() if k not in ("tensors", "generator")})
    json.dump(cfg, open(os.path.join(work, "cfg.json"), "w"), indent=1)
    env = dict(os.environ, LOOM_RDNA4=os.environ.get("LOOM_RDNA4", "1"))
    rc, out = sh([py, f"{kit}/build.py", "cfg.json", "pkg", "--tokenizer", f"{pkgdir}/tokenizer.json"], work, env)
    if rc:
        return False, "build.py: " + out.strip().splitlines()[-1][-250:], None
    return True, "", os.path.join(work, "pkg", "program")


def compile_one(job):
    f, k, target = job
    p = subprocess.run([LOOMC, f, f"--target=amdgpu:{target}", f"--root={k}", "--format=amdgpu-hsaco", "--output=/dev/null"],
                       capture_output=True, text=True, timeout=900)
    if p.returncode == 0:
        return "ok"
    m = re.search(r"error \[([A-Z0-9/]+)\]: [^\n]*?rejected '([^']+)' in '[^']*': source-to-low constraint '([^']+)'", p.stderr)
    if m:   # the op the target rejects and the constraint it breaks
        return f"{m.group(1)} {m.group(2)} / {m.group(3)}"
    m = re.search(r"error \[([A-Z0-9/]+)\]: ([^\n]*)", p.stderr)
    if not m:
        return "ERR " + p.stderr[-160:].replace("\n", " ")
    code, why = m.groups()
    why = re.sub(r"'[^']*'", "'..'", why)
    why = re.sub(r"%\w+", "%v", why)
    why = re.sub(r"\b(?:final resources admit|but) [^,;]*", "", why)
    why = re.sub(r"\d+", "N", why)[:140]
    return f"{code} {why}"


def sweep_program(prog, targets, jobs):
    files = sorted(glob.glob(prog + "/**/*.loom", recursive=True))
    work = []
    for f in files:
        for tag, k in sorted(set(re.findall(r'kernel\.def (?:target\(@(\w+)\) )?(?:\w+ )?export\("([^"]+)"\)', open(f).read()))):
            for t in targets:
                # dual-target packages: a kernel tagged @rdna3 is a gfx11 kernel, one tagged @rdna4 a gfx12 kernel
                if (tag == "rdna3" and t.startswith("gfx12")) or (tag == "rdna4" and t.startswith("gfx11")):
                    continue
                work.append((f, k, t))
    with cf.ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(compile_one, work))
    out = {}
    for (f, k, t), r in zip(work, res):
        out.setdefault(f"{os.path.basename(f)}:{k}", {})[t] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkgs", default=os.path.expanduser("~/lt/r9700/pkgs"))
    ap.add_argument("--work", default=os.path.expanduser("~/lt/r9700stg/eng/misc/sweep_work"))
    ap.add_argument("--out", default=os.path.expanduser("~/lt/r9700stg/eng/misc/sweep.json"))
    ap.add_argument("--slugs", default="")
    ap.add_argument("--targets", default="gfx1201,gfx1151")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.work.startswith("/tmp"):
        sys.exit("--work must not be under /tmp (RAM)")
    targets = a.targets.split(",")
    slugs = a.slugs.split(",") if a.slugs else sorted(
        os.path.basename(os.path.dirname(p)) for p in glob.glob(a.pkgs + "/*/llm_config.json"))
    slugs = [s for s in slugs if s not in SKIP or a.slugs]
    results, t0 = {}, time.time()
    for s in slugs:
        w = os.path.join(a.work, s)
        shutil.rmtree(w, ignore_errors=True)
        ok, msg, prog = build(s, f"{a.pkgs}/{s}", w)
        if not ok:
            results[s] = {"build": "FAIL " + msg, "kernels": {}}
            print(f"{s}: BUILD FAIL {msg[:200]}", flush=True)
        else:
            ks = sweep_program(prog, targets, a.jobs)
            results[s] = {"build": "ok", "kernels": ks}
            bad = sum(1 for v in ks.values() if any(r != "ok" for r in v.values()))
            print(f"{s}: {len(ks)} kernels, {bad} failing", flush=True)
        if not a.keep:
            shutil.rmtree(w, ignore_errors=True)
    # families: (family, target, reason) -> [slug:kernel]; a kernel failing with the same reason on gfx1151 as well is
    # target-neutral (a standalone-root artifact, or a defect Strix has too) and goes to its own section
    fam = collections.defaultdict(list)
    counted = collections.Counter()
    for s, r in results.items():
        if r["build"] != "ok":
            fam[("<build>", "-", "-", r["build"][:110])].append(s)
            counted["build"] += 1
            continue
        for kk, v in r["kernels"].items():
            name = kk.split(":", 1)[1]
            for t, res in v.items():
                if res == "ok":
                    continue
                both = all(v.get(o, "ok") == res for o in targets)
                scope = "both" if both and len(targets) > 1 else t
                if both and t != targets[0]:
                    continue
                known = name in KNOWN_ROOT_ARTIFACTS and res.startswith(KNOWN_ROOT_ARTIFACTS[name])
                fam[(family(name), scope, "known-root-artifact" if known else "", res)].append(f"{s}:{name}")
                counted[("root_artifact" if known else scope)] += 1
    table = []
    for (f, t, note, res), items in sorted(fam.items(), key=lambda x: (x[0][1] == "both", -len(x[1]), x[0])):
        table.append(dict(family=f, target=t, note=note, reason=res, count=len(items), examples=items[:3]))
    summary = {"packages": len(results), "build_failures": counted["build"],
               **{f"{t}_only_failures": counted[t] for t in targets}, "both_targets_failures": counted["both"],
               "known_root_artifacts": counted["root_artifact"], "seconds": round(time.time() - t0)}
    json.dump({"summary": summary, "families": table, "packages": results}, open(a.out, "w"), indent=1)
    print(json.dumps(summary))
    print(f"{'count':>5}  {'fails on':8} family | reason   (fails on: one target only, or both)")
    for r in table[:80]:
        print(f"{r['count']:>5}  {r['target']:8} {r['family']}  |  {r['reason']} {r['note']}")
    if len(table) > 80:
        print(f"... {len(table) - 80} more rows in {a.out}")


if __name__ == "__main__":
    main()
