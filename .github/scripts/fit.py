#!/usr/bin/env python3
"""Regenerate a device's fit table (.github/devlab/r9700_fit.json for the R9700): what each package needs on the device, from the BUILT package.

usage: fit.py --device DEVICE [--pkgs DIR (published packages, default ~/lt/r9700/pkgs)] [--work DIR] [--fit OLD.json] [-o NEW.json] [SLUG...]
       fit.py --device DEVICE --approx [--from TABLE.json] [--json]   verdicts of the fleet on DEVICE from another device's table (no builds)
       r9700_fit.py [...]                                              fit.py --device radeon-r9700 (the original entry point)

The table path is device.json "fit_table" (r9700: .github/devlab/r9700_fit.json), else .github/devlab/<device>_fit.json; the pool is the
device's memory_bytes; a device without one (Strix Halo) takes the package's own memory and has no table.

For every model of .github/nightly-models.json the package is rebuilt with this checkout's kit (skills/loom-port-llm/
sweep_compile.build, GGUF files taken from the repos the package's model.json names) and build.py --sizing-json records
what loom-serve keeps resident: weights (every GGUF part, drafter and mmproj included), the largest command program's
scratch (build.py's workspace minus its 3 GiB slack), the image producer's scratch (0.65 x the projector), the recurrent state of the resident rows, and the KV bytes per position (full pages on every layer, sliding
windows included: that is what the pool allocates). Row fields (same as before):
  w     weights + workspace + recurrent state (the bytes that do not shrink with the context)
  kv    KV bytes at ctx (the native context, at most the kernels' 262144)       kv32  KV bytes at 32768
  need  w + kv + card_lib.SERVE_HEADROOM       verdict  fits (native ctx) | fits (reduced ctx) | does not fit
  mem   the card's memory_bytes     n_exp  experts (kept from the previous table)     kvpp  KV bytes per position
The table holds data only. The serving settings a package needs on the device beyond its default preset's (the program scratch at the default
prefill chunk does not fit beside the weights and the drafter, say; a measured limit on the served context) are scope rules, not table
columns: exec.prefill_capacity, exec.context_capacity, exec.pool_capacity, exec.rows, exec.draft_depth, exec.context_cap set on a model node
(skills/loom-kb/model/<slug>/scope.json, `when: {"devices": ["radeon-r9700"]}`) or a family node. The package then carries a second preset
`<first><preset suffix>` (`<first>-r9700`) in model.json (make_info.py build --fit writes it, the card's Quick start for the device fetches it,
device_jobs.py measures it); this script sizes the package at that preset's prefill_capacity and rows. card_lib.fit_rows() puts the rules back
on the row as "ctx_cap" and "<runner key>_execution" for the readers. A row of an older table that still has those keys keeps them while no rule
sets one.
A new device's table starts without these blocks (n_exp is seeded from another device's table: --seed).
GGUFs may be header-only stubs (sparse files of the right size): only headers are read.
--approx: no build; the verdicts of the fleet on DEVICE from the sizing columns of another table (--from, default the R9700's): `w` can be a little
larger than the default preset needs (the table was sized at the R9700's prefill chunk). Good enough to plan a first wave.
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

registry = card_lib.registry
KERNEL_CTX = 262144
VISION_SCRATCH = 0.65


def verdicts(device, rows, headroom=card_lib.SERVE_HEADROOM):
    """--approx: the fleet's verdicts on DEVICE from the sizing columns (w, kvpp or kv/ctx, ctx) of a table sized for another device."""
    pool = registry.device(device).get("memory_bytes")
    out = []
    for r in rows:
        fixed = r["w"]
        kvpp = r.get("kvpp") or (r["kv"] / r["ctx"])   # some rows carry only kv at ctx
        kv = kvpp * r["ctx"]
        if pool is None:
            v, served = "fits (native ctx)", r["ctx"]
        elif fixed + headroom >= pool:
            v, served = "does not fit", 0
        elif fixed + kv + headroom <= pool:
            v, served = "fits (native ctx)", r["ctx"]
        else:
            v, served = "fits (reduced ctx)", int((pool - fixed - headroom) // kvpp)
        out.append({"slug": r["slug"], "w": fixed, "need": fixed + kv + headroom, "ctx": r["ctx"], "served_context": served, "verdict": v})
    return out


def approx(a):
    src = a.src or os.path.join(ROOT, registry.fit_table("radeon-r9700"))
    res = verdicts(a.device, json.load(open(src)))
    if a.json:
        print(json.dumps(res, indent=1))
        return
    for r in res:
        print(f"{r['slug']:55s} w {r['w'] / 1e9:5.1f} GB need {r['need'] / 1e9:6.1f} GB  {r['verdict']}" +
              (f" (served ctx {r['served_context']})" if r["verdict"].startswith("fits (red") else ""))
    n = {}
    for r in res:
        n[r["verdict"]] = n.get(r["verdict"], 0) + 1
    print(a.device, "memory", registry.device(a.device).get("memory_bytes"), n)


def build_table(a):
    import sweep_compile as sc
    GPU = a.device
    POOL = card_lib.GPUS[GPU]["memory_bytes"]
    XKEY = registry.fit_execution_key(GPU)   # "r9700_execution": the fit row's block of the device preset's execution
    table = os.path.join(ROOT, registry.fit_table(GPU))
    fit_in, out_path = a.fit or table, a.out or table
    if a.work.startswith("/tmp"):
        sys.exit("--work must not be under /tmp (RAM)")
    old = {r["slug"]: r for r in json.load(open(fit_in))} if os.path.exists(fit_in) else {}
    seed = {}   # a device's first table: n_exp of the models comes from another device's table
    if not old:
        other = a.seed or next((os.path.join(ROOT, registry.fit_table(g)) for g in registry.devices() if g != GPU
                                and os.path.exists(os.path.join(ROOT, registry.fit_table(g)))), None)
        seed = {r["slug"]: {"n_exp": r.get("n_exp", 0)} for r in json.load(open(other))} if other else {}
        old = dict(seed)
    models = {m["repo"].split("/")[-1]: m for m in json.load(open(os.path.join(ROOT, ".github", "nightly-models.json")))["models"]}
    rows, changes = [], []
    for slug in a.slugs or sorted(models):
        o = old.get(slug, {})
        w = os.path.join(a.work, slug)
        shutil.rmtree(w, ignore_errors=True)
        sizing = os.path.join(w, "sizing.json")
        os.makedirs(w, exist_ok=True)
        # a package with the device's execution block is served from its `<first><suffix>` preset: size the program scratch at that preset's prefill chunk and rows
        rule = card_lib.scope_execution(slug, GPU)   # exec.* of the scope rules (model node, family ...)
        rx = {k: rule[k] for k in card_lib.EXEC_KEYS if k in rule} or o.get(XKEY) or {}
        ok, msg, _ = sc.build(slug, f"{a.pkgs}/{slug}", w, ("--sizing-json", sizing,
                              *(("--prefill", str(rx["prefill_capacity"])) if rx.get("prefill_capacity") else ()),
                              *(("--rows", str(rx["rows"])) if rx.get("rows") else ())))
        if not ok or not os.path.exists(sizing):
            print(f"{slug}: no sizing ({msg or 'catalog kit'}): previous row kept", file=sys.stderr)
            if o and o.get("w") is not None:
                rows.append(o)
            shutil.rmtree(w, ignore_errors=True)
            continue
        z = json.load(open(sizing))
        shutil.rmtree(w, ignore_errors=True)
        # the command programs' real scratch (build.py's workspace without its 3 GiB pad), the recurrent state, and the image
        # producer's scratch (VISION_SCRATCH x the projector: Glimmer-30B served 48k tokens but failed images at 64k, 34 GB)
        fixed = int(z["weights_bytes"] + z["scratch_bytes"] + z["recurrent_bytes"] + VISION_SCRATCH * z["mmproj_bytes"])
        ctx = min(z["native_context"], KERNEL_CTX)
        kvpp = z["kv_bytes_per_position"]
        kv = kvpp * ctx
        need = fixed + kv + card_lib.SERVE_HEADROOM
        if fixed + card_lib.SERVE_HEADROOM >= POOL:
            verdict = "does not fit"
        else:
            verdict = "fits (native ctx)" if need <= POOL else "fits (reduced ctx)"
        row = {"slug": slug, "w": fixed, "kv": kv, "kv32": kvpp * 32768, "kvpp": kvpp, "ctx": ctx, "mem": models[slug].get("memory_bytes") or o.get("mem"),
               "need": float(need), "verdict": verdict, "n_exp": o.get("n_exp", 0), **({"ctx_cap": o["ctx_cap"]} if o.get("ctx_cap") and "context_cap" not in rule else {}),
               **({XKEY: rx} if rx and not any(k in rule for k in card_lib.EXEC_KEYS) else {})}
        rows.append(row)
        reduced = lambda r, v: card_lib.served_context(r["w"], r["kv"] / r["ctx"], POOL) if r.get("kv") and v.startswith("fits (red") else None
        sc_old = reduced(o, o.get("verdict", "")) if o.get("w") is not None else None
        sc_new = reduced(row, verdict)
        print(f"{slug}: w {fixed / 1e9:.1f} GB kv/pos {kvpp / 1024:.1f} KiB need {need / 1e9:.1f} GB {verdict} (was {o.get('verdict')}; served ctx {sc_old} -> {sc_new})", flush=True)
        if o.get("verdict") != verdict or sc_old != sc_new:
            changes.append((slug, o.get("verdict"), sc_old, verdict, sc_new))
    seen = {r["slug"] for r in rows}   # the models this run did not rebuild (not named, or not in nightly-models) keep their rows
    rows += [o for s, o in old.items() if s not in seen and s not in seed]
    rows.sort(key=lambda r: r["slug"])
    json.dump(rows, open(out_path, "w"), indent=1)
    print("\nverdict or served context changed:")
    for c in changes:
        print("  %s: %s (served %s) -> %s (served %s)" % c)
    if not registry.device(GPU).get("fit_table"):
        print(f"note: {out_path} is {GPU}'s default table path; set device.json \"fit_table\" to pin it", file=sys.stderr)


def main(argv=None, device=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slugs", nargs="*")
    if device is None:
        ap.add_argument("--device", required=True, choices=list(registry.devices()))
    else:
        ap.set_defaults(device=device)
    ap.add_argument("--pkgs", default=os.path.expanduser("~/lt/r9700/pkgs"))
    ap.add_argument("--work", default=os.path.expanduser("~/lt/r9700stg/eng/fit_work"))
    ap.add_argument("--fit", default=None, help="the previous table (default: the device's table)")
    ap.add_argument("-o", "--out", default=None, help="the new table (default: the device's table)")
    ap.add_argument("--seed", default=None, help="a device's first table: take n_exp from this table (default: another device's)")
    ap.add_argument("--approx", action="store_true", help="no builds: verdicts from another device's sizing columns (--from)")
    ap.add_argument("--from", dest="src", default=None, help="--approx: the table to read (default: the R9700's)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not a.approx and card_lib.GPUS[a.device]["memory_bytes"] is None:
        sys.exit(f"{a.device}: no memory_bytes (the package's own memory is used): no fit table")
    (approx if a.approx else build_table)(a)


if __name__ == "__main__":
    main()
