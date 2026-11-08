#!/usr/bin/env python3
"""Status of every model in a Loom HF org, and the work each target model still needs.

usage: org_status.py [--org loom-fun] [--targets ranked.json] [--cache DIR] [-o status.json]

For each model repo in the org, reads its model.json, model.json.lock, llm_config.json, info.json and program/*.loom
(downloaded again only when the Hub's git blob id changed) and reports:
  category  A = card made by the current skill: schema loom-card/2 and "report": "generated"
            B = card exists but came from an older skill (e.g. "report": "attached", imported by import_card.py)
                or an older schema; never use a B repo as a reference
  missing   gates and metrics the card should have (card_lib.gate_applies / metric_applies) but has no evidence for,
            the roofline, and devlab-perf (perf.devlab_perf: local values without a DevLab value on the repo's programs)
  failing   gates whose latest evidence is not a pass
  perf      the performance guardrails (card_lib.perf_flags), on the local (headline) values: far = metrics whose Loom
            value is below FAR_FRACTION x the efficiency Loom ports have demonstrated (roofline); stale = metrics whose
            value was measured on programs other than the repo's (card_lib.stale_metrics: the session's
            programs_sha256 vs the published program/*.loom); devlab = metrics only DevLab measured (no local value);
            devlab_perf = see missing. The local / DevLab ratio is never a flag.
  generator the loom-skills commit that generated the programs: llm_config.json "generator".skills_commit (build.py's
            stamp), else info.json's recovered versions.package.skills_commit; generator_source says which
  paused    true when info.json "status" is "paused" (work stopped; the card ends with its next steps); the
            action is assigned as for any other repo
  quant     the checkpoint's quant (card_lib.quant_of); anything but plain Q4_K_M is flagged
With --targets (rank_models.py output, {"models": [...]}, or a plain list of such entries or of model ids; a
missing rank is the entry's position, 1-based), matches repos to target models by the checkpoint's base model and assigns
each target one action:
  create  no repo for the model                         -> loop step 1 (port + optimize, publish)
  update  only a category B repo, or one not built from plain Q4_K_M -> loop step 1, from the best A reference
  fix     category A with a failing gate                -> loop step 4 (fix the skill, regenerate)
  remeasure  category A, a card value that a fresh local session (Loom and llama.cpp, quiet GPU, the repo's programs)
             must replace before any performance judgement; "remeasure" lists why (card_lib.remeasure_reasons):
             stale (measured on other programs, perf.stale) or devlab-primary (only DevLab measured it, perf.devlab)
  investigate  category A, far from roofline (perf.far) -> loop step 5 (local GPU worker, loom-optimize)
  sweep   category A, gates, metrics, the roofline or the DevLab perf session missing -> loop step 3 (gates,
          devlab-perf: DevLab; metrics: local)
  done    category A, every applicable gate passes, every applicable metric is measured locally and on DevLab, the
          roofline is computed and no flag is raised
Repos that duplicate a target's better repo, or match no target, are listed under "extra" (retire or keep: ask the
user; never delete a repo on your own).
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "loom-benchmark", "scripts", "card"))
from card_lib import (GATE_NAMES, METRIC_KEYS, SCHEMA_ID, gate_applies, generator_commit,  # noqa: E402
                      load_package, metric_applies, package_facts, perf_flags, programs_sha256, quant_of,
                      remeasure_reasons)

from hub import HF, get_json  # noqa: E402
from hub import fetch as fetch_bytes  # noqa: E402
FILES = ("model.json", "model.json.lock", "llm_config.json", "info.json")
VARIANT = re.compile(r"(-gguf|-bf16|-fp16|-qat.*|-mobile.*|-unquantized)$", re.I)
ORDER = {"fix": 0, "remeasure": 1, "investigate": 2, "update": 3, "create": 4, "sweep": 5, "done": 6}




def group_key(model_id):
    org, name = model_id.split("/", 1)
    return f"{org.lower()}/{VARIANT.sub('', name).lower()}"


def git_blob_id(path):
    """The git blob id of a local file (the Hub tree API's "oid" of a non-LFS file)."""
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_programs(repo, d):
    """The repo's program/*.loom into d/program (for card_lib.programs_sha256): only files whose blob id changed are
    downloaded; files the repo no longer has are removed."""
    tree = get_json(f"{HF}/api/models/{repo}/tree/main/program") or []
    want = {os.path.basename(t["path"]): t.get("oid") for t in tree
            if t.get("type") == "file" and t["path"].endswith(".loom")}
    pd = os.path.join(d, "program")
    os.makedirs(pd, exist_ok=True)
    for n in os.listdir(pd):
        if n not in want:
            os.remove(os.path.join(pd, n))
    for n, oid in want.items():
        p = os.path.join(pd, n)
        if oid and os.path.exists(p) and git_blob_id(p) == oid:
            continue
        b = fetch_bytes(f"{HF}/{repo}/resolve/main/program/{n}")
        if b is not None:
            with open(p, "wb") as o:
                o.write(b)


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
    if os.path.exists(os.path.join(d, "info.json")):   # a card: its sessions name the programs they measured
        fetch_programs(repo, d)
    return d


def assess(repo, d):
    out = {"repo": repo, "category": "B", "missing": [], "failing": [], "notes": []}
    if not os.path.exists(os.path.join(d, "model.json")):
        out["notes"].append("no model.json: not a loom package")
        return out
    facts = package_facts(load_package(d))
    ck = facts["checkpoint"]
    out.update(checkpoint=ck["repo"], file=ck["file"], quant=quant_of(ck["file"]))
    info_p = os.path.join(d, "info.json")
    info = json.load(open(info_p)) if os.path.exists(info_p) else {}
    out["generator"], out["generator_source"] = generator_commit(facts.get("generator"), info.get("versions"))
    if not info:
        out["notes"].append("no info.json: no card")
        return out
    if info.get("schema") == SCHEMA_ID and info.get("report") == "generated":
        out["category"] = "A"
    out["report"] = info.get("report")
    if info.get("status") == "paused":
        out["paused"] = True
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
    if out["category"] == "A":
        if not info.get("roofline"):
            out["missing"].append("roofline")
        out["programs_sha256"] = programs_sha256(d)
        out["perf"] = perf_flags(info, facts["kind"], facts["features"], programs=out["programs_sha256"] or "")
        if out["perf"]["devlab_perf"]:
            out["missing"].append("devlab-perf")
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

    def flagged(r):
        return bool((r.get("perf") or {}).get("far"))

    def remeasure(r):
        return remeasure_reasons(r.get("perf"))

    def flag_text(p):
        out = [f"stale {k} (measured programs {v[:8]})" for k, v in (p.get("stale") or {}).items()]
        if p.get("devlab"):
            out.append("devlab-primary " + ",".join(sorted(p["devlab"])) + " (no local session)")
        out += [f"far {k} {v[0]:.0%} of roofline (demonstrated {v[1]:.0%})" for k, v in p["far"].items()]
        return "; ".join(out)

    with cf.ThreadPoolExecutor(8) as ex:
        status = sorted(ex.map(one, repos), key=lambda r: r["repo"])
    for r in status:
        print(f"{r['category']}  {r['repo'][:58]:58} {(r.get('file') or '?')[:40]:40} "
              f"failing={','.join(r['failing']) or '-'} missing={len(r['missing'])}"
              + (f"  [{'; '.join(r['notes'])}]" if r["notes"] else "")
              + (f"  perf: {flag_text(r['perf'])}" if flagged(r) or remeasure(r) else ""))
    result = {"org": a.org, "repos": status}

    if a.targets:
        targets = json.load(open(a.targets))
        targets = targets["models"] if isinstance(targets, dict) else targets
        targets = [dict({"model": t} if isinstance(t, str) else t) for t in targets]
        for i, t in enumerate(targets, 1):
            t.setdefault("rank", i)   # a plain list: rank by order
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
            elif remeasure(primary):
                action = "remeasure"   # other programs or no local session: a fresh local session first
            elif flagged(primary):
                action = "investigate"   # far from roofline: back to a local GPU worker
            elif primary["missing"]:
                action = "sweep"
            else:
                action = "done"
            work.append({"rank": t.get("rank"), "model": t["model"], "arch": t.get("arch"), "q4_repo": t.get("q4_repo"),
                         "action": action, "repo": primary["repo"] if primary else None,
                         "paused": bool(primary and primary.get("paused")),
                         "duplicates": [r["repo"] for r in best[1:]],
                         "missing": primary["missing"] if primary else [],
                         "failing": primary["failing"] if primary else [],
                         "notes": primary["notes"] if primary else [],
                         "perf": primary.get("perf") if primary else None,
                         "remeasure": remeasure(primary) if primary else []})
        extra = [r["repo"] for r in status if r["repo"] not in used]
        counts = {k: sum(w["action"] == k for w in work) for k in ORDER}
        print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items()))
        for w in sorted(work, key=lambda w: (ORDER[w["action"]], w["rank"] or 0)):
            print(f"{w['action']:11} #{w['rank']:<3} {w['model'][:50]:50} {w['repo'] or ''}"
                  + (f"  dup: {', '.join(w['duplicates'])}" if w["duplicates"] else "")
                  + (f"  [{flag_text(w['perf'])}]" if w["action"] in ("investigate", "remeasure") else ""))
        if extra:
            print("\nnot a target (keep or retire: ask the user): " + ", ".join(extra))
        result.update(work=work, extra=extra, counts=counts)

    if a.output:
        with open(a.output, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
