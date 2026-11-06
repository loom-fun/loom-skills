#!/usr/bin/env python3
"""One-time importer: a card published before loom-card/1 -> a make_info.py evidence file.

usage: import_card.py REPO_DIR [--run-date YYYY-MM-DD] -o evidence.json
  REPO_DIR holds the card's current README.md and info.json (downloaded from Hugging Face, never the weights).

It maps what the old ad-hoc formats recorded and nothing else; a value they did not record stays unknown
(rendered as not measured / no evidence):
  - README.md front matter (license, tags) and its quick-start command (--output, --memory_bytes);
  - info.json "metrics" in any of the three older layouts ({"loom", "llama_cpp"} objects, flat *_llamacpp keys,
    prefill_1000-style names) -> performance, with the llama.cpp build and the run they came from;
  - the gate verdicts the old formats stored (golden_gate / correctness / consistency fields, the factory
    REPORT.md's own PASS lines, or a "GATES PASS level=... {...}" summary) -> gates;
  - a factory REPORT.md stating greedy-only decoding -> sampling: not supported.
Factory runs (info.json "run") are loom-fun/loom-skills Actions runs; their date comes from `gh run view`
unless --run-date is given.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_lib import EVIDENCE_ID, METRIC_KEYS, build_tag, load_json, write_json  # noqa: E402

# old metric name -> loom-card/1 metric
ALIASES = {
    "decode": ["decode_tps"], "decode_2k": ["decode_tps_2k"], "decode_32k": ["decode_tps_32k"],
    "prefill_1k": ["prefill_1k", "prefill_1k_tps", "prefill_1000"], "prefill_2k": ["prefill_2k", "prefill_2k_tps", "prefill_2000"],
    "prefill_8k": ["prefill_8k", "prefill_8k_tps", "prefill_8000"], "prefill_16k": ["prefill_16k", "prefill_16k_tps", "prefill_16000"],
    "prefill_32k": ["prefill_32k", "prefill_32k_tps", "prefill_32000"],
    "streams4": ["c4_steady_tps", "c4_steady", "c4_steady_tps_median"],
}


def first(v):
    return v[0] if isinstance(v, list) and v else v


def metric_pair(m, names):
    for n in names:
        if n not in m:
            continue
        v = m[n]
        if isinstance(v, dict):
            return v.get("loom"), v.get("llama_cpp", v.get("llamacpp"))
        return first(v), first(m.get(n + "_llamacpp"))
    return None, None


def front_matter(readme):
    m = re.match(r"---\n(.*?)\n---\n", readme, re.S)
    fm = m.group(1) if m else ""
    lic = re.search(r"^license:\s*(\S+)\s*$", fm, re.M)
    tags = re.search(r"^tags:\s*\[(.*?)\]\s*$", fm, re.M)
    tag_list = [t.strip() for t in tags.group(1).split(",")] if tags else re.findall(r"^- (\S+)\s*$", fm, re.M)
    return (lic.group(1) if lic else None), tag_list


def run_date(run_url, given):
    if given:
        return given
    rid = run_url.rstrip("/").split("/")[-1]
    out = subprocess.run(["gh", "run", "view", rid, "-R", "loom-fun/loom-skills", "--json", "createdAt", "-q", ".createdAt"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out[:10]


def verdict(text):
    m = re.search(r"\b(PASS|FAIL|passed|failed)\b", text or "")
    return None if not m else ("pass" if m.group(1) in ("PASS", "passed") else "fail")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_dir")
    ap.add_argument("--run-date")
    ap.add_argument("-o", required=True)
    a = ap.parse_args(argv)
    readme = open(os.path.join(a.repo_dir, "README.md"), encoding="utf-8").read()
    info = load_json(os.path.join(a.repo_dir, "info.json"))
    if info.get("schema"):
        sys.exit(f"{a.repo_dir}/info.json is already {info['schema']}; nothing to import")
    m = info.get("metrics") or {}
    report = info.get("report") or ""

    lic, tags = front_matter(readme)
    out_dir = re.search(r"--output=(\S+)", readme)
    mem = re.search(r"--memory_bytes=(\d+)", readme)
    ev = {"evidence": EVIDENCE_ID,
          "card": {"license": lic, "tags": tags, "output": out_dir.group(1) if out_dir else None,
                   "memory_bytes": int(mem.group(1)) if mem else None}}

    # where the old numbers came from: a loom-factory run, or (current cards) a local measurement
    run = info.get("run")
    if run:
        src = {"kind": "factory", "run": run.rstrip("/").split("/")[-1], "url": run}
        date = run_date(run, a.run_date)
    else:
        src = {"kind": "local", "run": None, "url": None}
        date = (re.search(r"\b(20\d\d-\d\d-\d\d)\b", json.dumps(m)) or [None, None])[1] or a.run_date
    build_text = " ".join(str(x) for x in (m.get("llama_cpp_build"), m.get("llamacpp_build"), m.get("measured"),
                                         info.get("measurement")) if x)
    build_text = build_text or (re.search(r"llama\.cpp[^\n]*?\bb1\d{3}\b", report + readme) or [""])[0]
    metrics, na = {}, []
    for key in METRIC_KEYS:
        loom, llama = metric_pair(m, ALIASES[key])
        metrics[key] = {"loom": loom, "llamacpp": llama}
    decision = "decision model" in (m.get("workload") or "") or "/v1/systemone" in readme
    if decision:
        # d1-style: decode / c4 were measured in decisions per second, not tokens; they are not this table's rows
        for key in ("decode", "decode_2k", "decode_32k", "streams4"):
            metrics[key] = {"loom": None, "llamacpp": None}
            na.append(key)
    drafting_llama = None
    if src["kind"] == "local":
        dm = re.search(r"(draft-\w+)[ ,]*(n-max \d+)?", build_text)
        drafting_llama = ", ".join(x for x in dm.groups() if x) if dm else None
    ev["performance"] = {"llamacpp_build": build_tag(build_text), "date": date, "source": src,
                         "drafting": {"llamacpp": drafting_llama},
                         "runs": {"decode_prefill": None, "streams4": None}, "metrics": metrics, "not_applicable": na}

    gates = []

    def gate(name, status):
        if status:
            gates.append({"name": name, "status": status, "level": None, "time": date, "source": src, "detail": None})

    g = info.get("gates") if isinstance(info.get("gates"), str) else ""
    sm = re.search(r"GATES (?:PASS|FAIL) level=(\w+) \d+s (\{.*\})", g)
    if sm:
        for n, v in ast.literal_eval(sm.group(2)).items():
            gates.append({"name": n, "status": "pass" if v == "PASS" else "fail", "level": sm.group(1),
                          "time": date, "source": src, "detail": None})
    if decision:
        gate("decision", (m.get("correctness") or {}).get("verdict") and verdict(m["correctness"]["verdict"]))
    else:
        gg = m.get("golden_gate")
        st = verdict(gg.get("verdict") if isinstance(gg, dict) else gg) if gg else None
        if st is None and not sm:
            rm = re.search(r"(?i)(?:correctness gate|golden gate)\W{0,4}\s*(PASS|FAIL)|passes the correctness gate", report)
            st = None if not rm else ("pass" if (rm.group(1) or "PASS").upper() == "PASS" else "fail")
        # the old golden gate is compare.py's verdict against the recorded llama.cpp golden
        gate("golden", st)
        gate("compare", st)
    if not sm:
        cs = m.get("consistency")
        st = verdict(cs) if isinstance(cs, str) else None
        if st is None:
            rm = re.search(r"consistency(?:\.py)?`?\)?\s*(?:run )?(PASS|passed|FAIL|failed)", report)
            st = verdict(rm.group(1)) if rm else None
        gate("consistency", st)
    ev["gates"] = gates
    if not decision and re.search(r"(?i)greedy (decoding )?only|sampling (fields|parameters) (are )?(accepted but )?ignored", report):
        ev["features"] = {"sampling": {"status": "no", "value": None}}
    write_json(a.o, ev)
    print(f"wrote {a.o}: perf {build_tag(build_text)} {date}, gates "
          + ", ".join(f"{x['name']}={x['status']}" for x in gates)
          + (", sampling=no" if "features" in ev else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
