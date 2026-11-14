#!/usr/bin/env python3
"""Regenerate .github/devlab/r9700_fit.json: what each package needs on the Radeon AI PRO R9700, from the BUILT package.

usage: r9700_fit.py [--pkgs DIR (published packages, default ~/lt/r9700/pkgs)] [--work DIR] [--fit OLD.json] [-o NEW.json] [SLUG...]

For every model of .github/nightly-models.json the package is rebuilt with this checkout's kit (skills/loom-port-llm/
sweep_compile.build, GGUF files taken from the repos the package's model.json names) and build.py --sizing-json records
what loom-serve keeps resident: weights (every GGUF part, drafter and mmproj included), the largest command program's
scratch (build.py's workspace minus its 3 GiB slack), the image producer's scratch (0.65 x the projector), the recurrent state of the resident rows, and the KV bytes per position (full pages on every layer, sliding
windows included: that is what the pool allocates). Row fields (same as before):
  w     weights + workspace + recurrent state (the bytes that do not shrink with the context)
  kv    KV bytes at ctx (the native context, at most the kernels' 262144)       kv32  KV bytes at 32768
  need  w + kv + card_lib.SERVE_HEADROOM       verdict  fits (native ctx) | fits (reduced ctx) | does not fit
  mem   the card's memory_bytes     n_exp  experts (kept from the previous table)     kvpp  KV bytes per position
GGUFs may be header-only stubs (sparse files of the right size): only headers are read.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-port-llm"))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import card_lib  # noqa: E402
import sweep_compile as sc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("slugs", nargs="*")
ap.add_argument("--pkgs", default=os.path.expanduser("~/lt/r9700/pkgs"))
ap.add_argument("--work", default=os.path.expanduser("~/lt/r9700stg/eng/fit_work"))
ap.add_argument("--fit", default=os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"))
ap.add_argument("-o", "--out", default=os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"))
a = ap.parse_args()
if a.work.startswith("/tmp"):
    sys.exit("--work must not be under /tmp (RAM)")
POOL = card_lib.GPUS["radeon-r9700"]["memory_bytes"]
old = {r["slug"]: r for r in json.load(open(a.fit))}
models = {m["repo"].split("/")[-1]: m for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]}
kernel_ctx = 262144
VISION_SCRATCH = 0.65
rows, changes = [], []
for slug in a.slugs or sorted(models):
    o = old.get(slug, {})
    w = os.path.join(a.work, slug)
    shutil.rmtree(w, ignore_errors=True)
    sizing = os.path.join(w, "sizing.json")
    os.makedirs(w, exist_ok=True)
    ok, msg, _ = sc.build(slug, f"{a.pkgs}/{slug}", w, ("--sizing-json", sizing))
    if not ok or not os.path.exists(sizing):
        print(f"{slug}: no sizing ({msg or 'catalog kit'}): previous row kept", file=sys.stderr)
        if o:
            rows.append(o)
        shutil.rmtree(w, ignore_errors=True)
        continue
    z = json.load(open(sizing))
    shutil.rmtree(w, ignore_errors=True)
    # the command programs' real scratch (build.py's workspace without its 3 GiB pad), the recurrent state, and the image
    # producer's scratch (VISION_SCRATCH x the projector: Glimmer-30B served 48k tokens but failed images at 64k, 34 GB)
    fixed = int(z["weights_bytes"] + z["scratch_bytes"] + z["recurrent_bytes"] + VISION_SCRATCH * z["mmproj_bytes"])
    ctx = min(z["native_context"], kernel_ctx)
    kvpp = z["kv_bytes_per_position"]
    kv = kvpp * ctx
    need = fixed + kv + card_lib.SERVE_HEADROOM
    if fixed + card_lib.SERVE_HEADROOM >= POOL:
        verdict = "does not fit"
    else:
        verdict = "fits (native ctx)" if need <= POOL else "fits (reduced ctx)"
    row = {"slug": slug, "w": fixed, "kv": kv, "kv32": kvpp * 32768, "kvpp": kvpp, "ctx": ctx, "mem": models[slug].get("memory_bytes") or o.get("mem"),
           "need": float(need), "verdict": verdict, "n_exp": o.get("n_exp", 0)}
    rows.append(row)
    reduced = lambda r, v: card_lib.served_context(r["w"], r["kv"] / r["ctx"], POOL) if r.get("kv") and v.startswith("fits (red") else None
    sc_old = reduced(o, o.get("verdict", "")) if o else None
    sc_new = reduced(row, verdict)
    print(f"{slug}: w {fixed / 1e9:.1f} GB kv/pos {kvpp / 1024:.1f} KiB need {need / 1e9:.1f} GB {verdict} (was {o.get('verdict')}; served ctx {sc_old} -> {sc_new})", flush=True)
    if o.get("verdict") != verdict or sc_old != sc_new:
        changes.append((slug, o.get("verdict"), sc_old, verdict, sc_new))
if not a.slugs:   # the models this run did not rebuild (not in nightly-models) keep their rows
    seen = {r["slug"] for r in rows}
    rows += [o for s, o in old.items() if s not in seen]
rows.sort(key=lambda r: r["slug"])
json.dump(rows, open(a.out, "w"), indent=1)
print("\nverdict or served context changed:")
for c in changes:
    print("  %s: %s (served %s) -> %s (served %s)" % c)
