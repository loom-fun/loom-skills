"""Builds a loom-serve package from a hand-written catalog package instead of generating the programs.

Some architectures are not in the starter (qwen4exp: hyper-connections, per-layer-embedding hash table, sparse indexed
attention) but have a package in ROCm/hrx-loom-library (model/<org>/<name>: model.json, program/, tests). A kit
that carries such a package in KIT/catalog/ (or CATALOG=DIR) turns the pipeline around: build.py hands over to
build(), which copies the package, keeps one variant + one preset, points its parameter files at the port's GGUFs
and writes llm_config.json, so everything downstream (serve, gates, perf, card, publish, nightly) is unchanged.

KIT/catalog/ holds the package files as the catalog has them (model.json with @origin files, program/, ...) and
catalog.json (schema below; every key optional except memory and source):
  {"source": {"repo": "unsloth/X-GGUF", "revision": "<sha>", "file": "UD/X-00001-of-00004.gguf",
              "extra": ["mtp-X-Q8_0.gguf"]},      # port.sh fetches these when called with no FILE (and `extra` always)
   "variant": "ud-q4-k-xl", "preset": "chat",      # the target-only variant and preset to serve
   "mtp_variant": "ud-q4-k-xl-mtp", "mtp_preset": "mtp",   # with the MTP head; its depth is the preset's draft_depth
   "context": 262144,                              # execution.context_capacity (default: the GGUF's native context)
   "prefill_capacity": 4096,                       # execution.prefill_capacity (default: the catalog preset's)
   "memory": {"resident_weights_bytes": N,         # what the package keeps resident (its own number: the PLE table
              "mtp_weights_bytes": N,              #   is read on demand, not resident)
              "kv_bytes_per_position": N, "mtp_kv_bytes_per_position": N, "recurrent_bytes_per_row": N,
              "workspace_gib": 4,
              "over_cap": true,                    # the resident set is above the default 60 %-of-RAM budget: allowed
              "note": "why"},                      #   for this model only (memcheck.sh then wants MemAvailable headroom)
   "sdk": {...}}                                   # the hrx-loom wheel the programs need (loom-setup/scripts/sdk_pin.py)
Kernel unit tests are the catalog's own (Bazel, in its CI); there is none to generate here.
"""
import copy
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from formats import memory_cap, split_parts  # noqa: E402

KEEP_OUT = ("catalog.json", "BUILD.bazel", "test", "__pycache__", "model.json.lock")


def find(kit=HERE):
    """The catalog package directory of this kit (CATALOG=DIR, else KIT/catalog), or None."""
    d = os.environ.get("CATALOG") or os.path.join(kit, "catalog")
    return os.path.abspath(d) if os.path.isfile(os.path.join(d, "catalog.json")) else None


def meta(cdir):
    with open(os.path.join(cdir, "catalog.json"), encoding="utf-8") as f:
        return json.load(f)


def parameter_files(variant):
    """[(service, parameter name, entry)] of a variant's weight files."""
    return [(s, p, e) for s, svc in variant.get("services", {}).items() for p, par in svc.get("parameters", {}).items()
            for e in par.get("files", [])]


def build(a, cfg, cdir, tokenizer):
    """Writes the package a.package from the catalog package in cdir; a: build.py's parsed arguments."""
    cat, mem = meta(cdir), meta(cdir)["memory"]
    if not os.path.isfile(os.path.join(cdir, "model.json")):
        sys.exit(f"{cdir}/model.json missing: copy the catalog package (model.json, program/, ...) next to catalog.json")
    src = json.load(open(os.path.join(cdir, "model.json"), encoding="utf-8"))
    gguf_dir = os.path.dirname(os.path.abspath(cfg["gguf"]))
    pkg = os.path.abspath(a.package)

    # Speculative decoding: the catalog's MTP variant (depth = its preset's draft_depth). --draft none/SPEC=off: target only.
    tuned = cfg.get("drafter") or {}
    kind = a.draft if a.draft != "auto" else (tuned.get("kind") if tuned.get("kind") in ("mtp", "none") else
                                              ("mtp" if cat.get("mtp_variant") else "none"))
    if kind not in ("mtp", "none"):
        sys.exit(f"--draft {kind}: a catalog package drafts with its MTP head only (or none)")
    if kind == "mtp" and not cat.get("mtp_variant"):
        sys.exit("--draft mtp: this catalog package has no MTP variant")
    variant, pname = (cat["mtp_variant"], cat.get("mtp_preset", "mtp")) if kind == "mtp" else (cat["variant"], cat.get("preset", "chat"))
    depth = 0
    if kind == "mtp":
        depth = a.draft_depth or (tuned.get("depth") if tuned.get("kind") == "mtp" else None) or src["presets"][pname]["execution"].get("draft_depth") or 3

    def card_for(variant, pname, depth):
        """The catalog's model.json cut down to one variant and preset (the shape every other package has)."""
        card = copy.deepcopy(src)
        card["variants"] = {variant: card["variants"][variant]}
        preset = card["presets"][pname]
        card["presets"] = {pname: preset}
        preset["variant"] = variant
        ex = preset.setdefault("execution", {})
        if "draft_depth" in ex or depth:
            ex["draft_depth"] = depth
        card["services"] = {s: v for s, v in card["services"].items() if s in card["variants"][variant]["services"]}
        for _, _, entry in parameter_files(card["variants"][variant]):   # the port's files, under their own names
            name = os.path.basename(entry["origin"].split("@", 1)[-1]) if entry.get("origin") else os.path.basename(entry["path"])
            entry["path"] = f"parameter/{name}"
        return card

    card = card_for(variant, pname, depth)
    files = parameter_files(card["variants"][variant])
    missing = [e["path"] for _, _, e in files if not os.path.exists(os.path.join(gguf_dir, os.path.basename(e["path"])))]
    if missing:
        sys.exit(f"catalog package: {missing} not next to {cfg['gguf']} (port.sh downloads source.file's parts and source.extra)")

    # Memory: the package's own resident set + KV pool + recurrent rows + workspace, against the default cap.
    ex = card["presets"][pname]["execution"]
    rows = a.rows
    native = cfg.get("context_length") or 32768
    context = a.context or cat.get("context") or native
    resident = mem["resident_weights_bytes"] + (mem.get("mtp_weights_bytes", 0) if kind == "mtp" else 0)
    kv_pos = mem["kv_bytes_per_position"] + (mem.get("mtp_kv_bytes_per_position", 0) if kind == "mtp" else 0)
    fixed = resident + int(mem.get("workspace_gib", 4) * 2**30) + 2 * rows * mem.get("recurrent_bytes_per_row", 0)
    want_pool = a.pool or context + max(65536, context // 4)
    need = fixed + want_pool * kv_pos
    cap, what, _ = memory_cap()
    over_cap = bool(mem.get("over_cap"))
    if a.memory_gib == "auto":
        budget = need if (over_cap or not cap) else min(need, cap)
        a.memory_gib = round(budget / 2**30, 1)
        print(f"memory budget {a.memory_gib} GiB (resident {resident / 2**30:.1f} GiB + a {want_pool}-position pool; "
              + (f"explicit per-model budget above the default cap {cap / 2**30:.0f} GiB ({what}): {mem.get('note', 'catalog.json memory.over_cap')})"
                 if over_cap and cap and need > cap else f"cap {what})"))
    budget = int(float(a.memory_gib) * 2**30)
    if resident > budget:
        sys.exit(f"the {a.memory_gib} GiB budget does not hold the package's resident set ({resident / 2**30:.1f} GiB)")
    if (context + 1024) * kv_pos > budget - fixed:
        fit = max(4096, ((budget - fixed) // kv_pos - 1024) // 1024 * 1024)
        print(f"WARNING: context {context} needs {context * kv_pos / 2**30:.1f} GiB of KV; {a.memory_gib} GiB budget fits {fit}")
        context = fit
    pool = max(min(want_pool, (budget - fixed) // kv_pos) // 1024 * 1024, -(-context // 1024) * 1024)
    ex["context_capacity"], ex["rows"] = context, rows
    if "memory" in card["presets"][pname]:   # the catalog's figure counts what its own package keeps resident
        card["presets"][pname]["memory"] = {"weights_bytes": resident, "workload": mem.get("note") or "resident parameter bank"}
    if a.prefill:
        ex["prefill_capacity"] = a.prefill
    elif cat.get("prefill_capacity"):   # catalog.json's own chunk size (a DevLab-sized staging budget)
        ex["prefill_capacity"] = cat["prefill_capacity"]
    elif cfg.get("prefill_capacity"):
        ex["prefill_capacity"] = cfg["prefill_capacity"]
    if a.pool or "pool_capacity" in ex:
        ex["pool_capacity"] = pool
    ex["max_tokens"] = a.max_tokens
    # the first (only) preset names the service's sampling default too
    opts = card["presets"][pname].setdefault("options", {}).setdefault("sampling", {})
    opts.update(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)

    # The package directory: the catalog's files, the weights as links, the tokenizer as the catalog names it.
    shutil.rmtree(pkg, ignore_errors=True)
    shutil.copytree(cdir, pkg, ignore=shutil.ignore_patterns(*KEEP_OUT), symlinks=True)
    os.makedirs(os.path.join(pkg, "parameter"), exist_ok=True)
    for _, _, entry in files:
        link = os.path.join(pkg, entry["path"])
        target = os.path.join(gguf_dir, os.path.basename(entry["path"]))
        try:
            os.symlink(target, link)
        except OSError:
            try:
                os.link(os.path.realpath(target), link)
            except OSError as e:
                sys.exit(f"cannot link {target} into {link} ({e}): enable Windows Developer Mode, or put the package on the GGUF's drive")
    shutil.copyfile(tokenizer, os.path.join(pkg, "tokenizer.json"))   # as upstream has it: its origin's hash is in the lock
    json.dump(card, open(os.path.join(pkg, "model.json"), "w"), indent=1)
    if kind == "mtp":   # the draft-off twin of the same package (port.sh's spec check, tune.py)
        tpreset = cat.get("preset", "chat")
        twin = card_for(cat["variant"], tpreset, 0)
        twin["presets"][tpreset]["execution"].update(context_capacity=context, rows=rows, max_tokens=a.max_tokens)
        json.dump(twin, open(os.path.join(pkg, "model_target.json"), "w"), indent=1)

    model_id = a.model_id or cfg.get("name", "llm").replace(" ", "-")
    bench = {k: v for k, v in cfg.items() if k != "tensors"}
    mtp_file = next((os.path.basename(e["path"]) for _, p, e in files if p == "draft"), None)
    if mtp_file:
        bench["mtp"] = {"gguf": os.path.join(gguf_dir, mtp_file), "layers": (cfg.get("mtp") or {}).get("layers", [])}
    bench.update(model_id=model_id, draft={"kind": "mtp", "depth": depth} if kind == "mtp" else None,
                 catalog={"variant": variant, "preset": pname, "draft_kinds": ["mtp"] if cat.get("mtp_variant") else [],
                          "memory": {k: v for k, v in mem.items()}},
                 sdk=cat.get("sdk"))
    from kit_source import generator_stamp  # noqa: E402
    bench["generator"] = generator_stamp()
    json.dump({k: v for k, v in bench.items() if v is not None}, open(os.path.join(pkg, "llm_config.json"), "w"), indent=1)

    print("package:", os.path.join(pkg, "model.json"), f"(model id {model_id!r}; catalog variant {variant}, preset {pname})")
    print("speculative decoding:", f"mtp, depth {depth}" if kind == "mtp" else "off")
    print(f"context {context} tokens (native {native}), KV pool {pool} positions = {pool * kv_pos / 2**30:.1f} GiB, "
          f"resident weights {resident / 2**30:.1f} GiB, {rows} rows")
    print(f"serve with --memory_bytes={int(float(a.memory_gib) * 2**30) // 10**9 + 1}000000000"
          " (too small shows up as 503 'retained state leaves insufficient completion capacity')")


if __name__ == "__main__":
    # catalog.py source DIR: repo, revision, file, tokenizer repo and extra files of catalog.json, one |-separated line (port.sh)
    if len(sys.argv) == 3 and sys.argv[1] == "source":
        c = meta(sys.argv[2]); sc = c.get("source", {})
        print("|".join([sc.get("repo", ""), sc.get("revision", ""), sc.get("file", ""), c.get("tokenizer", ""), " ".join(sc.get("extra", []))]))
    else:
        sys.exit("usage: catalog.py source CATALOG_DIR")
