#!/usr/bin/env python3
"""Ablation package: the cost of a kernel, measured end to end (hardware counters are unavailable).

usage: ablate.py PKG OUT_PKG REGEX [--skip] [--name NAME]
  Copies PKG to OUT_PKG (the GGUF link and tokenizer are shared), then in program/*.loom DOUBLES
  (default) or removes (--skip) every `kernel.launch` line whose text matches REGEX, and gives the
  copy its own model id (<id>-abl-<NAME>) so both serve from ONE loom-serve:
    serve.sh ab PORT amdgpu MEM PKG/model.json OUT_PKG/model.json
    ab_perf.py http://127.0.0.1:PORT <id> <id>-abl-<NAME> --metrics decode,prefill_2k
  Doubling: (B time - A time) per token = that kernel's cost. Skipping gives wrong output (fine for
  timing only; never ship it). Prints how many launches matched per file.
  Caveat: a doubled launch re-reads its weights from the 32 MB MALL when one layer's tensor fits
  there, so doubling UNDERestimates bandwidth-bound matvecs; use --skip for those (compute- or
  launch-bound kernels: either works).
  --skip can leave a weight unused (e.g. decode-only expert matvecs), and registration then fails
  ('stage parameter counts differ'): double those instead.
  Skipping the output head ends generation at once (decode then reports absurd tok/s): double it.
  Examples: REGEX '@attn_decode' (decode attention), '@mv_.*ffn_down' (one matvec), '@rms' (norms).
"""
import argparse
import json
import os
import re
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("pkg")
ap.add_argument("out")
ap.add_argument("regex")
ap.add_argument("--skip", action="store_true")
ap.add_argument("--name")
a = ap.parse_args()
rx = re.compile(a.regex)
name = a.name or re.sub(r"[^A-Za-z0-9]+", "-", a.regex).strip("-")[:24] or "x"
if os.path.exists(a.out):
    shutil.rmtree(a.out)
shutil.copytree(a.pkg, a.out, symlinks=True)
total = 0
for f in sorted(os.listdir(os.path.join(a.out, "program"))):
    if not f.endswith(".loom"):
        continue
    p = os.path.join(a.out, "program", f)
    out, n = [], 0
    for line in open(p):
        hit = "kernel.launch" in line and rx.search(line)
        if hit:
            n += 1
            if a.skip:
                continue
            out.append(line)
        out.append(line)
    if n:
        open(p, "w").write("".join(out))
        print(f"{f}: {'skipped' if a.skip else 'doubled'} {n} launches")
    total += n
if not total:
    raise SystemExit(f"no kernel.launch line matches {a.regex!r}")
card = json.load(open(os.path.join(a.out, "model.json")))
mid = card["model"]["id"] + f"-abl-{name}"
card["model"]["id"] = mid
json.dump(card, open(os.path.join(a.out, "model.json"), "w"), indent=1)
cfgp = os.path.join(a.out, "llm_config.json")
if os.path.exists(cfgp):
    cfg = json.load(open(cfgp))
    cfg["model_id"] = mid
    json.dump(cfg, open(cfgp, "w"), indent=1)
print(f"model id {mid}")
