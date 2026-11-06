#!/usr/bin/env python3
"""Write a card's info.json (schema loom-card/2) from measurement evidence.

info.json holds only what no package file records (see info_schema.json); render_card.py combines it with the
package's model.json, model.json.lock and llm_config.json. Both scripts are pure functions of their inputs (no
clock, no environment: the generator commit is an argument), so the same inputs give byte-identical files.

  make_info.py build PKG --repo loom-fun/NAME --generator-commit SHA [--evidence EV.json ...]
                         [--memory-bytes N] [--output DIR] [--license ID] [--report generated|attached] [-o info.json]
      PKG holds model.json (and model.json.lock, llm_config.json): card values the package already implies
      (tags, the default --output, a memory_bytes in model.json) are dropped. Evidence files apply in order.
      --generator-commit: the loom-skills commit of these card scripts (git rev-parse HEAD, or "unknown").

  make_info.py perf --loom-jsonl LOOM.jsonl --llama-jsonl LLAMA.jsonl [--c4-loom F] [--c4-llama F]
                    --baseline-version llamacpp-rocm/bNNNN [--baseline-args "ARGS"] [--metrics decode,...]
                    [--command CMD ...] [--package PKG] [--date YYYY-MM-DD] [SOURCE] -o perf_evidence.json
      perf.py logs (one engine per file; or --perf-log perf_log.jsonl with finalize.sh's labels final /
      llama-final) -> one measurement session: per-metric per-run values, their mean (decode, prefill) and the
      median of the c4_steady.py runs (4 streams). --metrics keeps only those metrics (a session that re-measured
      some rows). --package records which programs were measured (card_lib.programs_sha256).

  make_info.py gates FILE [FILE ...] [--time YYYY-MM-DDTHH:MM:SS] [--level L] [--spec-log F] [--package PKG]
                     [SOURCE] -o gates_evidence.json
      gates.sh's gates.json (verdicts, level, ts), a gates.out with its "GATES ..." line, or a kernel test
      log ending in "N/M passed" (make_kernel_tests.py) -> gate evidence. The spec gate's "acceptance N% of M
      proposals" (gates_spec.log or gates.out next to the file, or --spec-log) -> the spec_acceptance metric.

  SOURCE (perf and gates): --source-kind local|devlab|factory [--run ID --url URL] [--device auto|none|FILE.json]
      [--hrx-loom auto|none|FILE.json]. auto (the default) records this machine's device and the Loom SDK under
      $LOOM_ROOT (../machine_info.py): run perf and gates on the machine that measured.

Evidence file (loom-card-evidence/2), every block optional:
  {"evidence": "loom-card-evidence/2",
   "card": {"license", "tags", "output", "memory_bytes", "features"},
   "versions": {"package": {"skills_commit", "hrx_loom"}},          # recovered, for packages built unstamped
   "performance": {"baseline": {"name", "version", "args"},
                   "session": {"date", "source": SOURCE, "runs", "commands", "programs_sha256", "metrics"}},
   "gates": [{"name", "status", "level", "time", "source": SOURCE, "detail", "programs_sha256"}]}
  SOURCE: {"kind", "run", "url", "device", "hrx_loom"}
Merge rules: card keys and versions: the later file wins. performance: sessions append in order and per metric
the last session that measured it is the card's value; the baseline is the last one given. gates: per gate the
evidence with the latest time wins (equal times: the later file). Unknown metric and gate names are kept.
"""
import argparse
import ast
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from card_lib import (EVIDENCE_ID, MEDIAN_METRICS, METRIC_KEYS, SCHEMA_ID, build_tag, derived_tags, dumps,  # noqa: E402
                      load_json, load_package, package_facts, programs_sha256, require_valid, write_json)


# ---------------------------------------------------------------- sources

def source_arg(a):
    """The SOURCE block of a perf/gates evidence file from the command line."""
    def ident(v, auto):
        if v in (None, "auto"):
            return auto()
        if v == "none":
            return None
        return load_json(v)

    def auto_device():
        from machine_info import detect_device
        return detect_device()

    def auto_sdk():
        from machine_info import hrx_loom_identity
        return hrx_loom_identity()
    return {"kind": a.source_kind, "run": str(a.run) if a.run else None, "url": a.url,
            "device": ident(a.device, auto_device), "hrx_loom": ident(a.hrx_loom, auto_sdk)}


def source_id(src):
    base = src["kind"]
    if src.get("run"):
        base += f"-{src['run']}"
    job = re.search(r"/job/(\d+)", src.get("url") or "")
    if job:
        base += f"-job{job.group(1)}"
    return base


class Sources:
    """Interns SOURCE blocks: equal blocks share one id; different blocks with the same base id get -2, -3, ..."""

    def __init__(self):
        self.by_id = {}

    def add(self, src):
        src = {k: src.get(k) for k in ("kind", "run", "url", "device", "hrx_loom")}
        base = source_id(src)
        for n in range(1, 1000):
            sid = base if n == 1 else f"{base}-{n}"
            if sid not in self.by_id:
                self.by_id[sid] = src
                return sid
            if self.by_id[sid] == src:
                return sid
        raise SystemExit("too many distinct sources")


# ---------------------------------------------------------------- build

def cmd_build(a):
    facts = package_facts(load_package(a.pkg))
    card, versions, baseline, sessions, gates = {}, {}, None, [], {}
    sources = Sources()
    for i, path in enumerate(a.evidence or []):
        ev = load_json(path)
        if ev.get("evidence") != EVIDENCE_ID:
            sys.exit(f"{path}: not a {EVIDENCE_ID} file")
        c = ev.get("card") or {}
        card.update({k: v for k, v in c.items() if v is not None and k != "features"})
        if c.get("features"):
            card.setdefault("features", {}).update(c["features"])
        if (ev.get("versions") or {}).get("package"):
            versions.setdefault("package", {}).update(ev["versions"]["package"])
        perf = ev.get("performance") or {}
        if perf.get("baseline"):
            baseline = perf["baseline"]
        if perf.get("session"):
            s = dict(perf["session"])
            s["source"] = sources.add(s["source"])
            sessions.append({k: v for k, v in s.items() if v is not None or k in ("date",)})
        for g in ev.get("gates") or []:
            key = (g.get("time") or "", i)
            if g["name"] not in gates or key >= gates[g["name"]][0]:
                gates[g["name"]] = (key, g)
    for k, v in (("license", a.license), ("output", a.output), ("memory_bytes", a.memory_bytes)):
        if v is not None:
            card[k] = v
    # what the package's own files already say is not repeated
    tags = [t for t in card.get("tags") or [] if t not in derived_tags(facts["features"])]
    card.pop("tags", None)
    if tags:
        card["tags"] = tags
    if card.get("output") == facts["output"]:
        card.pop("output")
    if facts["memory_bytes"] is not None:
        card.pop("memory_bytes", None)
    if "license" in card and card["license"] is None:
        card.pop("license")
    if facts["generator"]:
        versions.pop("package", None)   # the build stamp in llm_config.json wins

    gate_out = {}
    for name in sorted(gates):
        g = gates[name][1]
        t = g.get("time")
        row = {"status": g["status"], "level": g.get("level"), "date": t[:10] if t else None,
               "source": sources.add(g["source"]), "detail": g.get("detail")}
        if g.get("programs_sha256"):
            row["programs_sha256"] = g["programs_sha256"]
        gate_out[name] = row
    info = {"schema": SCHEMA_ID, "repo": a.repo, "card": card,
            "versions": {"card_generator": a.generator_commit, **versions},
            "sources": sources.by_id, "gates": gate_out}
    if sessions:
        info["performance"] = {"baseline": baseline or {"name": "llama.cpp", "version": None, "args": None},
                               "sessions": sessions}
    if a.report:
        info["report"] = a.report
    require_valid(info)
    out = dumps(info)
    if a.o == "-":
        sys.stdout.write(out)
    else:
        with open(a.o, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        print(f"wrote {a.o}", file=sys.stderr)


# ---------------------------------------------------------------- perf logs -> evidence

DOC = {1000: "prefill_1k", 2000: "prefill_2k", 8000: "prefill_8k", 16000: "prefill_16k", 32000: "prefill_32k"}


def perf_rows(path, kind=None, label=None):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if kind and r.get("kind", kind) != kind:
            continue
        if label and r.get("label", label) != label:
            continue
        rows.append(r)
    return rows


def metric_samples(rows):
    """{metric: [values]} of perf.py rows (prefill samples bucketed by prompt tokens, as bench_common does)."""
    out = {}
    for r in rows:
        for k, m in (("decode_tps", "decode"), ("decode_tps_2k", "decode_2k"), ("decode_tps_32k", "decode_32k")):
            if r.get(k) is not None:
                out.setdefault(m, []).append(r[k])
        for p in r.get("prefill", []) + r.get("prefill_long", []):
            n = min(DOC, key=lambda x: abs(x - p["prompt_tokens"]))
            out.setdefault(DOC[n], []).append(p["prefill_tps"])
        if r.get("agg_tps") is not None:
            out.setdefault("streams4", []).append(r["agg_tps"])
    return out


def c4_samples(path):
    vals = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            r = json.loads(line)
            if r.get("agg_tps") is not None:
                vals.append(r["agg_tps"])
        else:
            vals.append(float(line))
    return vals


def summary(key, vals):
    if not vals:
        return None
    return round(statistics.median(vals) if key in MEDIAN_METRICS else statistics.mean(vals), 1)


def baseline_version(text):
    tag = build_tag(text)
    return f"llamacpp-rocm/{tag}" if tag and text == tag else text


def cmd_perf(a):
    if a.perf_log:
        loom_rows = perf_rows(a.perf_log, "loom", a.loom_label)
        llama_rows = perf_rows(a.perf_log, "llama", a.llama_label)
    else:
        loom_rows = perf_rows(a.loom_jsonl) if a.loom_jsonl else []
        llama_rows = perf_rows(a.llama_jsonl) if a.llama_jsonl else []
    ls, bs = metric_samples(loom_rows), metric_samples(llama_rows)
    if a.c4_loom:
        ls["streams4"] = c4_samples(a.c4_loom)
    if a.c4_llama:
        bs["streams4"] = c4_samples(a.c4_llama)
    keep = a.metrics.split(",") if a.metrics else METRIC_KEYS
    metrics = {}
    for k in keep:
        if k not in ls and k not in bs:
            continue
        m = {"loom": summary(k, ls.get(k)), "baseline": summary(k, bs.get(k))}
        if ls.get(k):
            m["loom_runs"] = ls[k]
        if bs.get(k):
            m["baseline_runs"] = bs[k]
        metrics[k] = m
    dp = [len(ls[k]) for k in keep if k in ls and k != "streams4"]
    runs = {"decode_prefill": max(dp) if dp else None,
            "streams4": len(ls["streams4"]) if "streams4" in keep and ls.get("streams4") else None}
    date = a.date or max([r["time"][:10] for r in loom_rows if r.get("time")] or [None], key=lambda x: x or "")
    commands = list(a.command or [])
    if a.commands_file:
        commands += [line.rstrip("\n") for line in open(a.commands_file, encoding="utf-8") if line.strip()]
    session = {"date": date, "source": source_arg(a), "runs": runs, "commands": commands, "metrics": metrics,
               "programs_sha256": programs_sha256(a.package) if a.package else None}
    perf = {"session": session}
    if a.baseline_version or a.baseline_args:
        perf["baseline"] = {"name": a.baseline_name, "version": baseline_version(a.baseline_version) if a.baseline_version else None,
                            "args": a.baseline_args}
    write_json(a.o, {"evidence": EVIDENCE_ID, "performance": perf})
    print(f"wrote {a.o}: " + ", ".join(f"{k} {m['loom']}/{m['baseline']}" for k, m in metrics.items()), file=sys.stderr)


# ---------------------------------------------------------------- gate results -> evidence

ACCEPT = re.compile(r"acceptance (\d+(?:\.\d+)?)% of (\d+) proposals")


def gates_from_file(path, time, level):
    text = open(path, encoding="utf-8").read()
    src_gates = []
    if text.lstrip().startswith("{"):
        g = json.loads(text)
        time = time or g.get("ts")
        level = level or g.get("level")
        for n, x in g.get("gates", {}).items():
            src_gates.append((n, "pass" if x.get("pass") else "fail", None))
        return src_gates, time, level
    m = re.search(r"^GATES (PASS|FAIL) level=(\w+) \d+s (\{.*\})\s*$", text, re.M)
    if m:
        level = level or m.group(2)
        for n, v in ast.literal_eval(m.group(3)).items():
            src_gates.append((n, "pass" if v == "PASS" else "fail", None))
        return src_gates, time, level
    m = re.search(r"^(\d+)/(\d+) passed\s*$", text, re.M)
    if m:
        ok = m.group(1) == m.group(2)
        return [("kernel_tests", "pass" if ok else "fail", f"{m.group(1)}/{m.group(2)}")], time, level
    sys.exit(f"{path}: no gates.json, GATES line or 'N/M passed' line found")


def acceptance(paths):
    """The spec gate's acceptance fraction from the first file that reports one, or None."""
    for p in paths:
        if p and os.path.isfile(p):
            found = ACCEPT.findall(open(p, encoding="utf-8", errors="replace").read())
            if found:
                return round(float(found[-1][0]) / 100, 3)
    return None


def cmd_gates(a):
    out, src, spec_date, logs = [], source_arg(a), None, [a.spec_log] if a.spec_log else []
    psha = programs_sha256(a.package) if a.package else None
    for path in a.files:
        gs, time, level = gates_from_file(path, a.time, a.level)
        if not time:
            sys.exit(f"{path}: no timestamp in the file; pass --time")
        for n, st, detail in gs:
            out.append({"name": n, "status": st, "level": None if n == "kernel_tests" else level, "time": time,
                        "source": src, "detail": detail, "programs_sha256": psha})
            if n == "spec":
                spec_date = time[:10]
        if not a.spec_log:
            d = os.path.dirname(os.path.abspath(path))
            logs += [os.path.join(d, "gates_spec.log"), path if path.endswith(".out") else os.path.join(d, "gates.out")]
    ev = {"evidence": EVIDENCE_ID, "gates": out}
    acc = acceptance(logs) if spec_date else None
    if acc is not None:
        ev["performance"] = {"session": {"date": spec_date, "source": src, "runs": {"decode_prefill": None, "streams4": None},
                                         "commands": [], "programs_sha256": psha,
                                         "metrics": {"spec_acceptance": {"loom": acc}}}}
    write_json(a.o, ev)
    print(f"wrote {a.o}: " + ", ".join(f"{g['name']}={g['status']}" for g in out)
          + (f", spec_acceptance={acc}" if acc is not None else ""), file=sys.stderr)


def add_source_args(p):
    p.add_argument("--source-kind", default="local", choices=["local", "devlab", "factory"])
    p.add_argument("--run")
    p.add_argument("--url")
    p.add_argument("--device", default="auto", help="auto (this machine), none, or a JSON file with {cpu, gpu, os}")
    p.add_argument("--hrx-loom", default="auto", help="auto (the SDK under $LOOM_ROOT), none, or a JSON file")
    p.add_argument("--package", help="the package measured (records card_lib.programs_sha256 of its programs)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("pkg")
    b.add_argument("--repo", required=True)
    b.add_argument("--generator-commit", required=True, help="loom-skills commit of the card scripts, or 'unknown'")
    b.add_argument("--evidence", nargs="*")
    b.add_argument("--memory-bytes", type=int)
    b.add_argument("--output")
    b.add_argument("--license")
    b.add_argument("--report", choices=["generated", "attached"])
    b.add_argument("-o", default="info.json")
    p = sub.add_parser("perf")
    p.add_argument("--perf-log")
    p.add_argument("--loom-label", default="final")
    p.add_argument("--llama-label", default="llama-final")
    p.add_argument("--loom-jsonl")
    p.add_argument("--llama-jsonl")
    p.add_argument("--c4-loom")
    p.add_argument("--c4-llama")
    p.add_argument("--baseline-name", default="llama.cpp")
    p.add_argument("--baseline-version", help="the baseline build, e.g. llamacpp-rocm/b1355 (a bare bNNNN tag is expanded)")
    p.add_argument("--baseline-args", help="the baseline server's exact arguments, model path, host and port left out")
    p.add_argument("--metrics", help="comma-separated metrics this session provides (default: all it measured)")
    p.add_argument("--command", action="append", help="a measuring command, for REPORT.md (repeat)")
    p.add_argument("--commands-file", help="measuring commands, one per line")
    p.add_argument("--date", help="YYYY-MM-DD (default: the latest Loom row's time)")
    add_source_args(p)
    p.add_argument("-o", required=True)
    g = sub.add_parser("gates")
    g.add_argument("files", nargs="+")
    g.add_argument("--time", help="ISO time of the run when the file has none (gates.out, kernel test log)")
    g.add_argument("--level")
    g.add_argument("--spec-log", help="the spec gate's log with 'acceptance N%% of M proposals' (default: next to FILE)")
    add_source_args(g)
    g.add_argument("-o", required=True)
    a = ap.parse_args(argv)
    {"build": cmd_build, "perf": cmd_perf, "gates": cmd_gates}[a.cmd](a)


if __name__ == "__main__":
    main()
