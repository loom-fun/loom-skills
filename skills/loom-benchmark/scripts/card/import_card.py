#!/usr/bin/env python3
"""One-time importer: a card published before loom-card/1 -> a make_info.py evidence file (loom-card-evidence/2).

usage: import_card.py REPO_DIR [--run-date YYYY-MM-DD] [--report-out REPORT.md] [--card-only] -o evidence.json
  REPO_DIR holds the card's pre-loom-card/1 README.md and info.json (from the repo's history on Hugging Face,
  never the weights).

It maps what the old ad-hoc formats recorded and nothing else; a value they did not record stays unknown
(rendered as not measured / no evidence):
  - README.md front matter (license, tags) and its quick-start command (--output, --memory_bytes);
  - info.json "metrics" in any of the three older layouts ({"loom", "llama_cpp"} objects, flat *_llamacpp keys,
    prefill_1000-style names) -> performance, with the llama.cpp build and the run they came from;
  - the gate verdicts the old formats stored (golden_gate / correctness / consistency fields, the factory
    REPORT.md's own PASS lines, or a "GATES PASS level=... {...}" summary) -> gates;
  - a factory REPORT.md stating greedy-only decoding -> sampling: not supported;
  - the report's own "llama.cpp flags" line -> the baseline's args, as stated;
  - versions (recovered, marked "recovered": true with the basis): the loom-skills commit is the head commit of
    the loom-factory run info.json names; the Loom SDK is the hrx-system CI artifact the report names (its
    commit and branch from the artifact's workflow run), with the version the report states.
--report-out writes the old info.json "report" (the narrative) to REPORT.md verbatim; pass make_info.py build
--report attached. --card-only keeps only the card block (license, tags, --output, --memory_bytes) for a card
whose measurements come from their raw logs instead.
Factory runs (info.json "run") are loom-fun/loom-skills Actions runs; their date and head commit come from
`gh run view` (--run-date overrides the date); artifacts from `gh api`. --lookups answers those calls from a
file instead (keys: the gh arguments joined by spaces).
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_lib import EVIDENCE_ID, build_tag, load_json, write_json  # noqa: E402

GH_REPO = "loom-fun/loom-skills"

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


LOOKUPS = {}   # --lookups: recorded answers instead of gh calls (tests, offline use)


def gh_json(*args):
    key = " ".join(args)
    if LOOKUPS:
        if key not in LOOKUPS:
            sys.exit(f"--lookups has no answer for: gh {key}")
        return LOOKUPS[key]
    return json.loads(subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout)


def run_info(run_url):
    rid = run_url.rstrip("/").split("/")[-1]
    return gh_json("run", "view", rid, "-R", GH_REPO, "--json", "createdAt,headSha")


def sdk_from_report(report, readme):
    """The Loom SDK a report names: an hrx-system CI wheel artifact (and the version, if stated), recovered."""
    text = report + "\n" + readme
    art = re.search(r"(?:CI artifact|wheel-)`?\s*(?:wheel-)?(\d{9,})", text)
    ver = re.search(r"hrx-loom`?\s+(\d+\.\d+\.\d+)", text)
    if not art:
        return "unknown"
    a = gh_json("api", f"repos/ROCm/hrx-system/actions/artifacts/{art.group(1)}")
    return {"hrx_loom": ver.group(1) if ver else None, "source": "ci", "artifact_id": int(art.group(1)),
            "hrx_system_sha": a["workflow_run"]["head_sha"], "branch": a["workflow_run"]["head_branch"],
            "wheel_sha256": None, "recovered": True,
            "basis": f"the card's report names hrx-system CI artifact {art.group(1)}; commit and branch from its workflow run"}


def verdict(text):
    m = re.search(r"\b(PASS|FAIL|passed|failed)\b", text or "")
    return None if not m else ("pass" if m.group(1) in ("PASS", "passed") else "fail")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_dir")
    ap.add_argument("--run-date")
    ap.add_argument("--report-out")
    ap.add_argument("--card-only", action="store_true")
    ap.add_argument("--lookups", help='JSON {"<gh arguments>": answer} used instead of calling gh')
    ap.add_argument("-o", required=True)
    a = ap.parse_args(argv)
    if a.lookups:
        LOOKUPS.update(load_json(a.lookups))
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
    if a.report_out and report:
        with open(a.report_out, "w", encoding="utf-8", newline="") as f:
            f.write(report)
    if a.card_only:
        write_json(a.o, ev)
        print(f"wrote {a.o}: card only", file=sys.stderr)
        return

    # where the old numbers came from: a loom-factory run, or (current cards) a local measurement
    run = info.get("run")
    sdk = sdk_from_report(report, readme)
    if run:
        ri = run_info(run)
        rid = run.rstrip("/").split("/")[-1]
        src = {"kind": "factory", "run": rid, "url": run, "device": None,
               "hrx_loom": sdk if isinstance(sdk, dict) else None}
        date = a.run_date or ri["createdAt"][:10]
        ev["versions"] = {"package": {
            "skills_commit": {"commit": ri["headSha"], "recovered": True,
                              "basis": f"head commit of the loom-factory run {rid} that info.json names"},
            "hrx_loom": sdk}}
    else:
        src = {"kind": "local", "run": None, "url": None, "device": None, "hrx_loom": sdk if isinstance(sdk, dict) else None}
        date = (re.search(r"\b(20\d\d-\d\d-\d\d)\b", json.dumps(m)) or [None, None])[1] or a.run_date
    build_text = " ".join(str(x) for x in (m.get("llama_cpp_build"), m.get("llamacpp_build"), m.get("measured"),
                                         info.get("measurement")) if x)
    build_text = build_text or (re.search(r"llama\.cpp[^\n]*?\bb1\d{3}\b", report + readme) or [""])[0]
    metrics = {}
    decision = "decision model" in (m.get("workload") or "") or "/v1/systemone" in readme
    for key in ALIASES:
        # d1-style: decode / c4 were measured in decisions per second, not tokens; they are not this table's rows
        if decision and key in ("decode", "decode_2k", "decode_32k", "streams4"):
            continue
        loom, llama = metric_pair(m, ALIASES[key])
        if loom is not None or llama is not None:
            metrics[key] = {"loom": loom, "baseline": llama}
    flags = re.search(r"llama\.cpp flags \| (.+?) \|\s*$", report, re.M)
    tag = build_tag(build_text)
    ev["performance"] = {
        "baseline": {"name": "llama.cpp", "version": f"llamacpp-rocm/{tag}" if tag else None,
                     "args": flags.group(1).replace("`", "") if flags else None},
        "session": {"date": date, "source": src, "runs": {"decode_prefill": None, "streams4": None}, "commands": [],
                    "programs_sha256": None, "metrics": metrics}}

    gates = []

    def gate(name, status):
        if status:
            gates.append({"name": name, "status": status, "level": None, "time": date, "source": src, "detail": None,
                          "programs_sha256": None})

    g = info.get("gates") if isinstance(info.get("gates"), str) else ""
    sm = re.search(r"GATES (?:PASS|FAIL) level=(\w+) \d+s (\{.*\})", g)
    if sm:
        for n, v in ast.literal_eval(sm.group(2)).items():
            gates.append({"name": n, "status": "pass" if v == "PASS" else "fail", "level": sm.group(1),
                          "time": date, "source": src, "detail": None, "programs_sha256": None})
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
        ev["card"]["features"] = {"sampling": {"status": "no", "value": None}}
    write_json(a.o, ev)
    print(f"wrote {a.o}: perf {tag} {date}, gates "
          + ", ".join(f"{x['name']}={x['status']}" for x in gates)
          + (", sampling=no" if "features" in ev["card"] else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
