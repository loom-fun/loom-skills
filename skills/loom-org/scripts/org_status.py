#!/usr/bin/env python3
"""Status of every model in a Loom HF org, and the work each target model still needs.

usage: org_status.py [--org loom-fun] [--targets ranked.json] [--cache DIR] [-o status.json]

For each model repo in the org, reads its model.json, model.json.lock, llm_config.json and info.json and reports:
  category  A = card made by the current skill: schema loom-card/2 and "report": "generated"
            B = card exists but came from an older skill (e.g. "report": "attached", imported by import_card.py)
                or an older schema; never use a B repo as a reference
  missing   gates and metrics the card should have (card_lib.gate_applies / metric_applies) but has no evidence for
  failing   gates whose latest evidence is not a pass
  quant     the checkpoint's quant (card_lib.quant_of); anything but plain Q4_K_M is flagged
With --targets (rank_models.py output), matches repos to target models by the checkpoint's base model and assigns
each target one action:
  create  no repo for the model                         -> loop step 1 (port + optimize, publish)
  update  only a category B repo, or one not built from plain Q4_K_M -> loop step 1, from the best A reference
  fix     category A with a failing gate                -> loop step 4 (fix the skill, regenerate)
  sweep   category A, gates or metrics missing          -> loop step 3 (DevLab sweep)
  done    category A, every applicable gate passes and every applicable metric is measured
Repos that duplicate a target's better repo, or match no target, are listed under "extra" (retire or keep: ask the
user; never delete a repo on your own).
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "loom-benchmark", "scripts", "card"))
from card_lib import (GATE_NAMES, METRIC_KEYS, SCHEMA_ID, gate_applies, load_package, metric_applies,  # noqa: E402
                      package_facts, quant_of)

from hub import HF, get_json  # noqa: E402
from hub import fetch as fetch_bytes  # noqa: E402
FILES = ("model.json", "model.json.lock", "llm_config.json", "info.json")
VARIANT = re.compile(r"(-gguf|-bf16|-fp16|-qat.*|-mobile.*|-unquantized)$", re.I)
ORDER = {"fix": 0, "update": 1, "create": 2, "sweep": 3, "done": 4}




def group_key(model_id):
    org, name = model_id.split("/", 1)
    return f"{org.lower()}/{VARIANT.sub('', name).lower()}"


def fetch(repo, cache):
    d = os.path.join(cache, repo.replace("/", "__"))
    os.makedirs(d, exist_ok=True)
    for f in FILES:
        p = os.path.join(d, f)
        b = fetch_bytes(f"{HF}/{repo}/resolve/main/{f}")
        if b is None:
            if os.path.exists(p):
                os.remove(p)
        else:
            with open(p, "wb") as o:
                o.write(b)
    return d


def assess(repo, d):
    out = {"repo": repo, "category": "B", "missing": [], "failing": [], "notes": []}
    if not os.path.exists(os.path.join(d, "model.json")):
        out["notes"].append("no model.json: not a loom package")
        return out
    facts = package_facts(load_package(d))
    ck = facts["checkpoint"]
    out.update(checkpoint=ck["repo"], file=ck["file"], quant=quant_of(ck["file"]),
               generator=(facts.get("generator") or {}).get("commit"))
    info_p = os.path.join(d, "info.json")
    if not os.path.exists(info_p):
        out["notes"].append("no info.json: no card")
        return out
    info = json.load(open(info_p))
    if info.get("schema") == SCHEMA_ID and info.get("report") == "generated":
        out["category"] = "A"
    out["report"] = info.get("report")
    gates = info.get("gates") or {}
    for g in GATE_NAMES:
        if not gate_applies(g, facts["kind"], facts["features"]):
            continue
        ev = gates.get(g)
        if not ev:
            out["missing"].append(f"gate:{g}")
        elif ev.get("status") != "pass":
            out["failing"].append(g)
    measured = set()
    for s in (info.get("performance") or {}).get("sessions", []):
        measured |= {k for k, m in s.get("metrics", {}).items() if (m or {}).get("loom") is not None}
    out["missing"] += [f"metric:{k}" for k in METRIC_KEYS
                       if metric_applies(k, facts["kind"], facts["features"]) and k not in measured]
    if out["quant"] != "Q4_K_M" or re.search(r"UD-", ck["file"]):
        out["notes"].append(f"checkpoint is {ck['file']}, not plain Q4_K_M")
    return out


def base_model(checkpoint):
    info = get_json(f"{HF}/api/models/{checkpoint}?expand[]=cardData") or {}
    bm = (info.get("cardData") or {}).get("base_model")
    if isinstance(bm, list):
        bm = bm[0] if len(bm) == 1 else None
    return bm if isinstance(bm, str) else checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", default="loom-fun")
    ap.add_argument("--targets", help="rank_models.py output")
    ap.add_argument("--cache", help="directory for the downloaded package files (default: a temp dir)")
    ap.add_argument("-o", "--output")
    a = ap.parse_args()
    cache = a.cache or tempfile.mkdtemp(prefix="org_status_")
    repos = [m["id"] for m in get_json(f"{HF}/api/models?author={a.org}&limit=1000") or []]

    def one(repo):
        r = assess(repo, fetch(repo, cache))
        r["base"] = base_model(r["checkpoint"]) if r.get("checkpoint") else None
        return r

    with cf.ThreadPoolExecutor(8) as ex:
        status = sorted(ex.map(one, repos), key=lambda r: r["repo"])
    for r in status:
        print(f"{r['category']}  {r['repo'][:58]:58} {(r.get('file') or '?')[:40]:40} "
              f"failing={','.join(r['failing']) or '-'} missing={len(r['missing'])}"
              + (f"  [{'; '.join(r['notes'])}]" if r["notes"] else ""))
    result = {"org": a.org, "repos": status}

    if a.targets:
        targets = json.load(open(a.targets))["models"]
        by_base = {}
        for r in status:
            if r.get("base"):
                by_base.setdefault(group_key(r["base"]), []).append(r)
        work, used = [], set()
        for t in targets:
            have = by_base.get(group_key(t["model"]), [])
            best = sorted(have, key=lambda r: (r["category"] != "A", len(r["failing"]), len(r["missing"]), bool(r["notes"])))
            primary = best[0] if best else None
            used |= {r["repo"] for r in have}
            if primary is None:
                action = "create"
            elif primary["category"] != "A" or primary.get("quant") != "Q4_K_M" or "UD-" in (primary.get("file") or ""):
                action = "update"   # older skill, or not the target's plain Q4_K_M
            elif primary["failing"]:
                action = "fix"
            elif primary["missing"]:
                action = "sweep"
            else:
                action = "done"
            work.append({"rank": t.get("rank"), "model": t["model"], "arch": t.get("arch"), "q4_repo": t.get("q4_repo"),
                         "action": action, "repo": primary["repo"] if primary else None,
                         "duplicates": [r["repo"] for r in best[1:]],
                         "missing": primary["missing"] if primary else [],
                         "failing": primary["failing"] if primary else [],
                         "notes": primary["notes"] if primary else []})
        extra = [r["repo"] for r in status if r["repo"] not in used]
        counts = {k: sum(w["action"] == k for w in work) for k in ORDER}
        print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items()))
        for w in sorted(work, key=lambda w: (ORDER[w["action"]], w["rank"] or 0)):
            print(f"{w['action']:6} #{w['rank']:<3} {w['model'][:50]:50} {w['repo'] or ''}"
                  + (f"  dup: {', '.join(w['duplicates'])}" if w["duplicates"] else ""))
        if extra:
            print("\nnot a target (keep or retire: ask the user): " + ", ".join(extra))
        result.update(work=work, extra=extra, counts=counts)

    if a.output:
        with open(a.output, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
