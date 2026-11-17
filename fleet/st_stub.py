#!/usr/bin/env python3
"""Safetensors header dumps (stdlib only): the safetensors counterpart of gguf_stub.py.

The programs a build generates depend on a checkpoint's tensor table (names, dtypes, shapes), its config.json, generation_config.json,
tokenizer_config.json, the chat template and the tokenizer's added tokens, never on the weights. A dump keeps exactly that
(inspect_safetensors.Source.dump(): the header without data offsets, the small JSON files, the template) in one gzip JSON, so
fleet/build_fleet.py builds the model without weights and without the network.

usage: st_stub.py dump SOURCE OUT.json.gz [--revision REV]     SOURCE: a local checkpoint directory or an HF repo id (HTTP range reads)
"""
import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "loom-port-llm", "starter"))
import inspect_safetensors  # noqa: E402


def dump(source, revision="main"):
    return inspect_safetensors.load_source(source, revision).dump()


def save(d, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(d, f, separators=(",", ":"), sort_keys=True)


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["dump"])
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--revision", default="main")
    a = ap.parse_args()
    d = dump(a.source, a.revision)
    save(d, a.out)
    print(f"{a.out}: {len(d['header'])} tensors, {sum(d['files'].values())} bytes, {os.path.getsize(a.out)} bytes gz")


if __name__ == "__main__":
    main()
