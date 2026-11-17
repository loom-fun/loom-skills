#!/usr/bin/env python3
"""The committed reference cache of the gates (vLLM for safetensors cards, llama.cpp for GGUF cards): record once, reuse in every job.

A gates job (gates.sh LEVEL=dev ...) against vLLM costs about 72 % of a typical experiment job, almost all of it vLLM loading and answering
the same golden / agent / vision / compare requests for the unchanged checkpoint. The answers do not depend on the Loom build or the runner,
only on the checkpoint revision, the vLLM version and the request (bench_common.llama_completion keys them), so they are stored once:

  .github/devlab/reference_cache/<org>__<repo>__<ENGINE>.tgz    ENGINE = vllm-<VERSION> (the checkpoint's repo) or llama-<BUILD> (the card's repo)
  holds llama_cache/<xx>/<key>.json ... + reference.json (a vLLM reference's statement for the card)

devlab_perf.sh `gates` restores the file of the card's checkpoint (or card) and the job's engine version before gates.sh runs, so a run whose requests
are all cached never starts vLLM ("references: N cached, 0 recorded"); a request not in the file (a new prompt set, a revision bump) is
recorded by that job as before, and its gates_V/llama_cache.tgz is merged back with `merge`. A key hit is exact: nothing stale is served.

The store holds two recordings of the golden / vision requests where the engine is not run-to-run deterministic (vLLM: ~28 % of the keys differ between
jobs): the first, and a second under another salt (bench_common.ALT_SALT_TEXT / ALT_SALT_VISION) taken by a different job (REFERENCE_ALT=record in gates.sh).
compare.py and vision_golden.py read the second as their --alt / re-recording rule; `merge` adds keys only, so neither recording replaces the other.

  reference_cache.py restore --repo ORG/REPO --engine ENGINE --into BENCH_DIR [--store DIR]   -> prints the number of entries restored (0: none)
  reference_cache.py merge   --repo ORG/REPO --engine ENGINE  JOB_llama_cache.tgz ...          -> adds the entries (and reference.json) to the store file
  reference_cache.py count   TGZ_OR_DIR                                                         -> entries
  reference_cache.py path    --repo ORG/REPO --engine ENGINE                                     -> the store file's path
"""
import argparse
import io
import json
import os
import sys
import tarfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
STORE = os.path.join(ROOT, ".github", "devlab", "reference_cache")


def store_file(repo, engine, store=STORE):
    return os.path.join(store, f"{repo.replace('/', '__')}__{engine}.tgz")


def _entries(tgz):
    """{member name: bytes} of a cache tgz: llama_cache/<xx>/<key>.json and reference.json only (nothing else is trusted into a bench dir)."""
    out = {}
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            n = m.name
            ok = m.isfile() and (n == "reference.json" or (n.startswith("llama_cache/") and n.endswith(".json") and n.count("/") == 2 and ".." not in n))
            if ok:
                out[n] = t.extractfile(m).read()
    return out


def restore(repo, engine, into, store=STORE):
    p = store_file(repo, engine, store)
    if not os.path.exists(p):
        return 0
    ents = _entries(p)
    for n, data in ents.items():
        d = os.path.join(into, n)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if n == "reference.json" or not os.path.exists(d):
            with open(d, "wb") as f:
                f.write(data)
    return sum(1 for n in ents if n != "reference.json")


def merge(repo, engine, tgzs, store=STORE):
    """Union of the store file and the job tgzs; an entry is content-keyed, so the same key is the same answer (the stored one stays)."""
    p = store_file(repo, engine, store)
    ents = _entries(p) if os.path.exists(p) else {}
    before = len(ents)
    for t in tgzs:
        for n, data in _entries(t).items():
            if n == "reference.json":
                ents[n] = data   # the newest statement of the engine / revision / quantization
            else:
                ents.setdefault(n, data)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with tarfile.open(p, "w:gz") as t:
        for n in sorted(ents):
            ti = tarfile.TarInfo(n)
            ti.size, ti.mtime, ti.mode = len(ents[n]), 0, 0o644   # deterministic member headers: a merge of known entries leaves the content unchanged
            t.addfile(ti, io.BytesIO(ents[n]))
    return before, len(ents)


def count(path):
    if os.path.isdir(path):
        return sum(f.endswith(".json") for _, _, fs in os.walk(os.path.join(path, "llama_cache")) for f in fs)
    return sum(1 for n in _entries(path) if n != "reference.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("restore", "merge", "path"):
        s = sub.add_parser(c)
        s.add_argument("--repo", required=True)
        s.add_argument("--engine", required=True, help="vllm-VERSION or llama-BUILD")
        s.add_argument("--store", default=STORE)
        if c == "restore":
            s.add_argument("--into", required=True)
        if c == "merge":
            s.add_argument("tgz", nargs="+")
    sub.add_parser("count").add_argument("path")
    a = ap.parse_args(argv)
    if a.cmd == "restore":
        print(restore(a.repo, a.engine, a.into, a.store))
    elif a.cmd == "merge":
        b, n = merge(a.repo, a.engine, a.tgz, a.store)
        print(f"{store_file(a.repo, a.engine, a.store)}: {b} -> {n} entries")
    elif a.cmd == "count":
        print(count(a.path))
    else:
        print(store_file(a.repo, a.engine, a.store))


if __name__ == "__main__":
    main()
