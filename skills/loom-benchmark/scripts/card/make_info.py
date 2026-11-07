#!/usr/bin/env python3
"""Write a card's info.json (schema loom-card/2) from measurement evidence.

info.json holds only what no package file records (see info_schema.json); render_card.py combines it with the
package's model.json, model.json.lock and llm_config.json. Both scripts are pure functions of their inputs (no
clock, no environment: the generator commit is an input, by default the checkout's HEAD), so the same inputs give
byte-identical files.

  make_info.py build PKG --repo loom-fun/NAME [--generator-commit SHA] [--evidence EV.json ...]
                         [--memory-bytes N] [--output DIR] [--license ID] [--report generated|attached] [-o info.json]
      PKG holds model.json (and model.json.lock, llm_config.json): card values the package already implies
      (tags, the default --output, a memory_bytes in model.json) are dropped. Evidence files apply in order.
      --generator-commit: the loom-skills commit of these card scripts (default: this checkout's HEAD, or
      "unknown" outside a git checkout).
      --from-info OLD.json: start from a published info.json (its sessions, gates, sources, roofline and DevLab
      sweeps) and apply the evidence files on top: how a card is regenerated when only new evidence arrives.
      --repo defaults to OLD.json's repo.

  make_info.py perf --loom-jsonl LOOM.jsonl --llama-jsonl LLAMA.jsonl [--c4-loom F] [--c4-llama F]
                    --baseline-version llamacpp-rocm/bNNNN [--baseline-args "ARGS"] [--metrics decode,...]
                    [--command CMD ...] [--package PKG] [--date YYYY-MM-DD] [SOURCE] -o perf_evidence.json
      perf.py logs (one engine per file; or --perf-log perf_log.jsonl with finalize.sh's labels final /
      llama-final) -> one measurement session: per-metric per-run values, their mean (decode, prefill) and the
      median of the c4_steady.py runs (4 streams). --metrics keeps only those metrics (a session that re-measured
      some rows). --package records which programs were measured (card_lib.programs_sha256). Rows that carry the
      drafted-token acceptance of their decode requests (perf.py "acceptance", c4_steady.py JSON lines) give each
      drafted metric its session's "acceptance" (mean of the runs).

  make_info.py gates FILE [FILE ...] [--time YYYY-MM-DDTHH:MM:SS] [--level L] [--spec-log F] [--package PKG]
                     [SOURCE] -o gates_evidence.json
      gates.sh's gates.json (verdicts, level, ts), a gates.out with its "GATES ..." line, or a kernel test
      log ending in "N/M passed" (make_kernel_tests.py) -> gate evidence. The spec gate's "acceptance N% of M
      proposals" (gates_spec.log or gates.out next to the file, or --spec-log) -> the spec_acceptance metric.

  make_info.py roofline --package PKG [--gguf G] [--cfg cfg.json] [--acceptance A | --evidence EV.json ...] [--skills-commit SHA]
                        -o roofline_evidence.json
      The hardware ceiling of every card metric that has one (loom-roofline/roofline.py; needs the `gguf` Python
      package: run it with the tools venv). --gguf defaults to the checkpoint model.json.lock pins, in the Hugging
      Face cache; --cfg to PKG/llm_config.json. A drafting package's decode and 4-stream rows are ceilings of
      speculative decoding: verify passes per second x (1 + acceptance x depth) tokens per pass. Per row, the
      acceptance measured in the same session as the card's value of that row (its "acceptance": perf.py and
      c4_steady.py read loom-serve's counters around the timed requests), from PKG/info.json's sessions with the
      --evidence files' sessions on top (a first card has no info.json yet); a row whose session recorded none
      falls back to the spec gate's: --acceptance, else the card's spec_acceptance (none: no ceiling). Each row
      records which ("acceptance_from": session | spec_gate): the gate's text can accept far more or less than the
      row's workload, which moves that row's ceiling by the same factor.

  make_info.py devlab --loom-jsonl PERF.jsonl [--c4-loom F] --run ID --url JOB_URL [--runner NAME] [--date D]
                      [--package PKG] -o devlab_evidence.json
      One DevLab sweep's Loom numbers (perf.py rows and c4_steady.py runs of a DevLab Dispatch job) for the card's
      DevLab column. pull_devlab.py writes the same block from a nightly-gates run.

  SOURCE (perf and gates): --source-kind local|devlab|factory [--run ID --url URL] [--device auto|none|FILE.json]
      [--hrx-loom auto|none|FILE.json]. auto (the default) records this machine's device and the Loom SDK under
      $LOOM_ROOT (../machine_info.py): run perf and gates on the machine that measured.

Evidence file (loom-card-evidence/2), every block optional:
  {"evidence": "loom-card-evidence/2",
   "card": {"license", "tags", "output", "memory_bytes", "features"},
   "versions": {"package": {"skills_commit", "hrx_loom"}},          # recovered, for packages built unstamped
   "performance": {"baseline": {"name", "version", "args"},
                   "session": {"date", "source": SOURCE, "runs", "commands", "programs_sha256",
                               "metrics": {METRIC: {"loom", "baseline", "loom_runs", "baseline_runs", "acceptance"}}}},
   "gates": [{"name", "status", "level", "time", "source": SOURCE, "detail", "programs_sha256"}],
   "roofline": {"device", "tool", "checkpoint", "drafting", "metrics"},
   "devlab": {"sweep": {"date", "source": SOURCE, "runner", "commands", "programs_sha256", "metrics"}}}
  SOURCE: {"kind", "run", "url", "device", "hrx_loom"}
Merge rules: card keys and versions: the later file wins. performance: sessions append in order and per metric
the last session that measured it is the card's value; the baseline is the last one given. gates: per gate the
latest evidence wins (gate_supersedes: a published date-only gate is older than new evidence of that day; equal
times: the later file), except that on the same programs_sha256 a later pass at a lower level (dev < fast < quick <
full) does not replace a pass at a higher level; a later failure always does. roofline: the later file wins. devlab: sweeps
append in order (the same source replaces its earlier sweep; the last DEVLAB_SWEEPS_KEPT are kept). Unknown metric
and gate names are kept.
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
from card_lib import (DEVLAB_SWEEPS_KEPT, DRAFTED_ROWS, EVIDENCE_ID, MEDIAN_METRICS, METRIC_KEYS,  # noqa: E402
                      ROOFLINE_METRIC, SCHEMA_ID, build_tag, card_metrics, derived_tags, dumps, load_json,
                      load_package, metric_applies, package_facts, programs_sha256, require_valid, write_json)


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

GATE_LEVELS = ("dev", "fast", "quick", "full")   # gates.sh LEVEL, weakest first


def gate_supersedes(new, cur):
    """Whether gate evidence `new` replaces `cur` on the card. Each is ((when, order), evidence, published row):
    `when` is an evidence time (YYYY-MM-DDTHH:MM:SS) or a published row's date (YYYY-MM-DD); order is the evidence
    file's position (-1: the published card). A later run wins, so a later failure always shows (a regression) and
    a pass on other programs replaces evidence on older ones; but on the same programs_sha256 a later pass at a
    lower level does not replace a pass at a higher level (a quick sweep after a full one keeps the full pass; a
    higher-level pass replaces a lower one whatever the order the evidence arrives in).
    Dates and times compare at the coarser precision (a published date-only gate vs a timestamped job of that day
    is a tie, broken by file order: the new evidence), so publishing a card never reorders its evidence."""
    def row(e):
        return e[2] if e[2] is not None else e[1]
    def when(e):
        t = (e[0][0] or "").replace(" ", "T")
        return t[:10], t[11:]
    (nd, nt), (cd, ct) = when(new), when(cur)
    later = (nd, nt if nt and ct else "", new[0][1]) >= (cd, ct if nt and ct else "", cur[0][1])
    n, c = row(new), row(cur)
    if n["status"] == c["status"] == "pass" and n.get("programs_sha256") and n.get("programs_sha256") == c.get("programs_sha256"):
        rank = lambda r: GATE_LEVELS.index(r["level"]) if r.get("level") in GATE_LEVELS else -1
        return rank(n) > rank(c) if rank(n) != rank(c) else later
    return later


def cmd_build(a):
    facts = package_facts(load_package(a.pkg))
    card, versions, baseline, sessions, gates = {}, {}, None, [], {}
    roofline, sweeps = None, []
    sources = Sources()
    if a.from_info:   # a published card: its data first, then the new evidence on top
        old = load_json(a.from_info)
        require_valid(old, a.from_info)
        a.repo = a.repo or old["repo"]
        if old["repo"] != a.repo:
            sys.exit(f"{a.from_info} is the card of {old['repo']}, not {a.repo}")
        card = json.loads(json.dumps(old["card"]))
        versions = {k: v for k, v in old["versions"].items() if k != "card_generator"}
        sources.by_id = dict(old["sources"])
        if old.get("performance"):
            baseline, sessions = old["performance"]["baseline"], list(old["performance"]["sessions"])
        for name, g in old["gates"].items():   # a published gate has a date, not a time: later evidence that day wins
            gates[name] = ((g.get("date") or "", -1), None, g)
        roofline = old.get("roofline")
        sweeps = list((old.get("devlab") or {}).get("sweeps", []))
        if a.report is None:
            a.report = old.get("report")
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
            new = ((g.get("time") or "", i), g, None)
            if g["name"] not in gates or gate_supersedes(new, gates[g["name"]]):
                gates[g["name"]] = new
        if ev.get("roofline"):
            roofline = ev["roofline"]
        sw = (ev.get("devlab") or {}).get("sweep")
        if sw:
            sw = dict(sw)
            sw["source"] = sources.add(sw["source"])
            sweeps = [s for s in sweeps if s["source"] != sw["source"]] + [sw]
    if not a.repo:
        sys.exit("make_info.py build: --repo is required without --from-info")
    if not a.generator_commit:
        a.generator_commit = skills_checkout(__file__)[0]
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
        if gates[name][2] is not None:   # kept from --from-info as published
            gate_out[name] = gates[name][2]
            continue
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
    if roofline:
        info["roofline"] = roofline
    if sweeps:
        info["devlab"] = {"sweeps": sweeps[-DEVLAB_SWEEPS_KEPT:]}
    used = ({s["source"] for s in sessions} | {g["source"] for g in gate_out.values()}
            | {s["source"] for s in info.get("devlab", {}).get("sweeps", [])})
    info["sources"] = {k: v for k, v in sources.by_id.items() if k in used}
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


DECODE = (("decode_tps", "decode"), ("decode_tps_2k", "decode_2k"), ("decode_tps_32k", "decode_32k"))


def metric_samples(rows):
    """{metric: [values]} of perf.py rows (prefill samples bucketed by prompt tokens, as bench_common does)."""
    out = {}
    for r in rows:
        for k, m in DECODE:
            if r.get(k) is not None:
                out.setdefault(m, []).append(r[k])
        for p in r.get("prefill", []) + r.get("prefill_long", []):
            n = min(DOC, key=lambda x: abs(x - p["prompt_tokens"]))
            out.setdefault(DOC[n], []).append(p["prefill_tps"])
        if r.get("agg_tps") is not None:
            out.setdefault("streams4", []).append(r["agg_tps"])
    return out


def acceptance_samples(rows, c4_path=None):
    """{metric: [drafted-token acceptance]} of perf.py rows ("acceptance" per decode request) and c4_steady.py JSON
    lines ("acceptance" over the 4 streams): the acceptance of the text each drafted metric decoded."""
    out = {}
    for r in rows:
        for k, m in DECODE:
            if (r.get("acceptance") or {}).get(k) is not None:
                out.setdefault(m, []).append(r["acceptance"][k])
    for line in open(c4_path, encoding="utf-8") if c4_path else []:
        if line.strip().startswith("{") and json.loads(line).get("acceptance") is not None:
            out.setdefault("streams4", []).append(json.loads(line)["acceptance"])
    return out


def with_acceptance(metrics, acc):
    """Adds each metric's session acceptance (mean of its runs) to metrics, in place."""
    for k, vals in acc.items():
        if k in metrics and vals:
            metrics[k]["acceptance"] = round(statistics.mean(vals), 3)
    return metrics


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
    with_acceptance(metrics, acceptance_samples(loom_rows, a.c4_loom))
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


# ---------------------------------------------------------------- roofline -> evidence

def skills_checkout(path):
    """(commit, dirty) of the loom-skills checkout holding path, or ("unknown", None)."""
    import subprocess
    d = os.path.dirname(os.path.abspath(path))
    try:
        sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", d, "status", "--porcelain", "--", os.path.basename(path)],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def cached_gguf(ck):
    """The checkpoint model.json.lock pins, in the Hugging Face cache, or None."""
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    p = os.path.join(hub, "models--" + ck["repo"].replace("/", "--"), "snapshots", ck["revision"] or "", ck["file"])
    return p if ck.get("revision") and os.path.exists(p) else None


def roofline_block(facts, m, acceptance, R, bw, flops, tool, checkpoint, row_acceptance=None):
    """info.json "roofline" from roofline.py's model shape m (R: the roofline module). A drafted row's ceiling counts
    row_acceptance[metric] (measured in the row's own session) when given, else `acceptance` (the spec gate's)."""
    ceil = R.ceilings(m, bw, flops, (1000, 2000, 8000, 16000, 32000))
    spec = facts["spec"]
    tpp = round(1 + acceptance * spec[1], 3) if spec and acceptance is not None else None
    metrics = {}
    for key, rk in ROOFLINE_METRIC.items():
        if not metric_applies(key, facts["kind"], facts["features"]):
            continue
        c = ceil[rk]
        row = {"ceiling": round(c, 1), "efficiency": round(R.efficiency(m, rk, c), 3)}
        if spec and key in DRAFTED_ROWS:
            row = {"ceiling": None, "single_pass": round(c, 1)}
            acc, how = (row_acceptance or {}).get(key), "session"
            if acc is None:
                acc, how = acceptance, "spec_gate"
            if acc is not None:
                t = round(1 + acc * spec[1], 3)
                d = R.drafted_row_ceiling(m, rk, spec[1], acc, bw)
                row.update(ceiling=round(d, 1), efficiency=round(R.efficiency(m, rk, d, tokens_per_pass=t), 3),
                           acceptance=acc, acceptance_from=how, tokens_per_pass=t)
        metrics[key] = row
    return {"device": {"name": R.DEVICE if (bw, flops) == (R.BW, R.FLOPS) else f"{R.DEVICE}, overridden",
                       "bandwidth_bytes_per_s": bw, "peak_flops": flops},
            "tool": tool, "checkpoint": checkpoint,
            "drafting": ({"method": spec[0], "depth": spec[1], "acceptance": acceptance, "tokens_per_pass": tpp}
                         if spec else None),
            "metrics": metrics}


def drafted_acceptance(facts, sessions, override=None):
    """(the spec gate's acceptance, {drafted metric: acceptance measured in its own session}) for the roofline, from
    measurement sessions combined as a build combines them (PKG/info.json's, then the new evidence's: per metric the
    last session that measured it). override: the gate's acceptance given on the command line."""
    if not facts["spec"]:
        return None, {}
    card = card_metrics({"performance": {"sessions": sessions}})
    gate = override if override is not None else card.get("spec_acceptance", ({}, 0))[0].get("loom")
    return gate, {k: card[k][0]["acceptance"] for k in DRAFTED_ROWS if card.get(k, ({}, 0))[0].get("acceptance") is not None}


def cmd_roofline(a):
    import hashlib
    rdir = os.path.join(HERE, "..", "..", "..", "loom-roofline")
    sys.path.insert(0, rdir)
    import roofline as R
    facts = package_facts(load_package(a.package))
    gguf = a.gguf or cached_gguf(facts["checkpoint"])
    if not gguf:
        sys.exit(f"{facts['checkpoint']['file']} is not in the Hugging Face cache at the locked revision; pass --gguf")
    cfg_p = a.cfg or os.path.join(a.package, "llm_config.json")
    cfg = load_json(cfg_p) if os.path.exists(cfg_p) else {}
    info_p = os.path.join(a.package, "info.json")
    sessions = list(((load_json(info_p).get("performance") or {}).get("sessions") or [])
                    if os.path.exists(info_p) else [])
    sessions += [s for s in ((load_json(ev).get("performance") or {}).get("session") for ev in a.evidence or []) if s]
    acc, row_acc = drafted_acceptance(facts, sessions, a.acceptance)
    script = os.path.join(rdir, "roofline.py")
    commit, dirty = skills_checkout(script)
    with open(script, "rb") as f:
        tool = {"script": "skills/loom-roofline/roofline.py", "sha256": hashlib.sha256(f.read()).hexdigest(),
                "skills_commit": a.skills_commit or commit}
    if dirty and not a.skills_commit:
        tool["dirty"] = True
    block = roofline_block(facts, R.model_shape(gguf, cfg), acc, R, a.bw, a.flops, tool, os.path.basename(gguf), row_acc)
    write_json(a.o, {"evidence": EVIDENCE_ID, "roofline": block})
    print(f"wrote {a.o}: " + ", ".join(f"{k} {r['ceiling']}" for k, r in block["metrics"].items())
          + "".join(f"; {k} acceptance {r['acceptance']} ({r['acceptance_from']})" for k, r in block["metrics"].items()
                    if r.get("acceptance_from")),
          file=sys.stderr)


# ---------------------------------------------------------------- DevLab sweep -> evidence

def devlab_sweep(loom_rows, c4_vals, source, runner=None, date=None, commands=None, psha=None, c4_path=None):
    """One DevLab sweep: the card's metrics from a DevLab job's perf.py rows and c4_steady.py runs (Loom only)."""
    ls = metric_samples(loom_rows)
    if c4_vals:
        ls["streams4"] = c4_vals
    metrics = {k: {"loom": summary(k, ls[k]), "loom_runs": ls[k]} for k in METRIC_KEYS if ls.get(k)}
    with_acceptance(metrics, acceptance_samples(loom_rows, c4_path))
    date = date or max([r["time"][:10] for r in loom_rows if r.get("time")] or [None], key=lambda x: x or "")
    return {"date": date, "source": source, "runner": runner, "commands": list(commands or []),
            "programs_sha256": psha, "metrics": metrics}


def cmd_devlab(a):
    a.source_kind = "devlab"
    sw = devlab_sweep(perf_rows(a.loom_jsonl, "loom"), c4_samples(a.c4_loom) if a.c4_loom else [], source_arg(a),
                      a.runner, a.date, a.command, programs_sha256(a.package) if a.package else None, a.c4_loom)
    write_json(a.o, {"evidence": EVIDENCE_ID, "devlab": {"sweep": sw}})
    print(f"wrote {a.o}: " + ", ".join(f"{k} {m['loom']}" for k, m in sw["metrics"].items()), file=sys.stderr)


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
    b.add_argument("--repo", help="loom-fun/NAME (default with --from-info: its repo)")
    b.add_argument("--generator-commit", help="loom-skills commit of the card scripts, or 'unknown' (default: this "
                   "checkout's HEAD)")
    b.add_argument("--evidence", nargs="*")
    b.add_argument("--memory-bytes", type=int)
    b.add_argument("--output")
    b.add_argument("--license")
    b.add_argument("--report", choices=["generated", "attached"])
    b.add_argument("--from-info", help="a published info.json to start from (regenerating a card with new evidence)")
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
    r = sub.add_parser("roofline")
    r.add_argument("--package", required=True, help="the package (model.json, model.json.lock, llm_config.json)")
    r.add_argument("--gguf", help="the checkpoint (default: model.json.lock's, in the Hugging Face cache)")
    r.add_argument("--cfg", help="the port's cfg.json (default: PKG/llm_config.json)")
    r.add_argument("--acceptance", type=float, help="the spec gate's acceptance, the fallback for drafted rows whose "
                   "session recorded none (default: the card's spec_acceptance, from the --evidence files or PKG/info.json)")
    r.add_argument("--evidence", nargs="*", help="evidence files whose sessions count on top of PKG/info.json's (perf: the "
                   "rows' acceptance; gates: spec_acceptance), e.g. a first card's, which has no info.json yet")
    r.add_argument("--skills-commit", help="record this loom-skills commit instead of the checkout's")
    r.add_argument("--bw", type=float, default=216e9, help="memory bandwidth, bytes/s (roofline.py's default)")
    r.add_argument("--flops", type=float, default=59e12, help="peak FLOP/s (roofline.py's default)")
    r.add_argument("-o", required=True)
    v = sub.add_parser("devlab")
    v.add_argument("--loom-jsonl", required=True, help="the DevLab job's perf.py log (Loom rows)")
    v.add_argument("--c4-loom", help="its c4_steady.py runs")
    v.add_argument("--runner", help="the DevLab runner's name")
    v.add_argument("--date", help="YYYY-MM-DD (default: the latest row's time)")
    v.add_argument("--command", action="append", help="a measuring command (repeat)")
    add_source_args(v)
    v.set_defaults(device="none", hrx_loom="none")
    v.add_argument("-o", required=True)
    a = ap.parse_args(argv)
    {"build": cmd_build, "perf": cmd_perf, "gates": cmd_gates, "roofline": cmd_roofline, "devlab": cmd_devlab}[a.cmd](a)


if __name__ == "__main__":
    main()
