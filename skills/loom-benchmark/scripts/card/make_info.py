#!/usr/bin/env python3
"""Write a card's info.json (schema loom-card/3) from measurement evidence.

info.json holds only what no package file records (see info_schema.json); render_card.py combines it with the
package's model.json, model.json.lock and llm_config.json. Both scripts are pure functions of their inputs (no
clock, no environment: the generator commit is an input, by default the checkout's HEAD), so the same inputs give
byte-identical files.

  make_info.py build PKG --repo loom-fun/NAME [--org ORG] [--generator-commit SHA] [--evidence EV.json ...]
                         [--memory-bytes N] [--output DIR] [--license ID] [--report generated|attached]
                         [--gpus ID,ID] [--gpu-memory GPU=N] [--served-context GPU=N] [--fit VERDICT.json]
                         [--from-info OLD.json] [--next-steps FILE] [--status paused|active] [-o PKG/info.json]
      PKG holds model.json (and model.json.lock, llm_config.json): card values the package already implies
      (tags, the default --output, a memory_bytes in model.json) are dropped. Evidence files apply in order.
      --generator-commit: the loom-skills commit of these card scripts (default: this checkout's HEAD, or
      "unknown" outside a git checkout).
      --from-info OLD.json: start from a published info.json (its sessions, gates, sources and roofline) and apply
      the evidence files on top: how a card is regenerated when only new evidence arrives. Performance sessions from
      anywhere but DevLab (a local machine's, from cards made before DevLab became the only performance source) and
      an older card's "devlab" sweeps are dropped.
      --reset-performance: drop OLD.json's performance sessions (its further-backend baselines stay): the card's
      performance is re-measured from scratch by the evidence given (a devlab-perf job on new programs or settings).
      --repo defaults to OLD.json's repo. --org ORG: the card's repo is ORG/NAME (NAME from --repo or OLD.json): how the
      staging flow (--org loom-fun-staging) gives the same repo names under another org; the Quick start fetches from it.
      GPUs (card_lib.GPUS: strix-halo, radeon-r9700): evidence says which GPU measured it (an evidence file's "gpu",
      else the GPU its source device names, else strix-halo), and info.json keeps every GPU's gates, performance and
      roofline under devices.<gpu> (an OLD.json of loom-card/2 is its strix-halo entry, unchanged). --gpus lists the
      GPUs the card covers: one without evidence gets state not_measured (the card says this package has no results
      there yet). --gpu-memory GPU=N: loom-serve --memory_bytes on that GPU (default: the package's, at most the GPU's
      memory); --served-context GPU=N: the tokens per request that GPU serves when its memory holds less KV cache than
      the native context needs. --fit VERDICT.json: both, and state unsupported for a package that does not fit, for
      radeon-r9700, from the fit table (a list of {slug, w: weight bytes, kv: KV bytes at ctx tokens, ctx, need}).
      --next-steps FILE: the card's next steps, one per line (a leading "- " is dropped; an empty FILE removes
      them); --status paused|active (active removes the status). Both are kept from OLD.json unless given.

  make_info.py perf [--gpu GPU] --loom-jsonl LOOM.jsonl --llama-jsonl LLAMA.jsonl [--c4-loom F] [--c4-llama F]
                    --baseline-version llamacpp-rocm/bNNNN [--baseline-args "ARGS"] [--metrics decode,...]
                    [--command CMD ...] [--package PKG] [--date YYYY-MM-DD] [SOURCE] -o perf_evidence.json
      perf.py logs (one engine per file; or --perf-log perf_log.jsonl with finalize.sh's labels final /
      llama-final) -> one measurement session: per-metric per-run values, their mean (decode, prefill) and the
      median of the c4_steady.py runs (4 streams). --metrics keeps only those metrics (a session that re-measured
      some rows). --package records which programs were measured (card_lib.programs_sha256). Rows that carry the
      drafted-token acceptance of their decode requests (perf.py "acceptance", c4_steady.py JSON lines) give each
      drafted metric its session's "acceptance" (mean of the runs).

  make_info.py gates [--gpu GPU] FILE [FILE ...] [--time YYYY-MM-DDTHH:MM:SS] [--level L] [--spec-log F] [--package PKG]
                     [SOURCE] -o gates_evidence.json
      gates.sh's gates.json (verdicts, level, ts), a gates.out with its "GATES ..." line, or a kernel test
      log ending in "N/M passed" (make_kernel_tests.py) -> gate evidence. The spec gate's "acceptance N% of M
      proposals" (gates_spec.log or gates.out next to the file, or --spec-log) -> the spec_acceptance metric, from a
      DevLab (or factory) source only, like every card performance value.

  make_info.py roofline [--gpu GPU] --package PKG [--gguf G] [--cfg cfg.json] [--acceptance A | --evidence EV.json ...] [--skills-commit SHA]
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

  SOURCE (perf and gates): --source-kind devlab|factory|local (perf: default devlab; gates: default local) [--run ID --url URL] [--device auto|none|FILE.json]
      [--hrx-loom auto|none|FILE.json]. auto records this machine's device and the Loom SDK under $LOOM_ROOT
      (../machine_info.py); on downloaded DevLab job files: --device none --hrx-loom none. Card performance comes
      only from AMD DevLab Dispatch (a devlab-perf job or a scripted DevLab port, kind devlab; the loom-factory
      workflow, kind factory): perf refuses --source-kind local. Gates may come from any source.

Evidence file (loom-card-evidence/2), every block optional:
  {"evidence": "loom-card-evidence/2", "gpu": "strix-halo" | "radeon-r9700",   # the GPU that produced it (perf, gates, roofline
                                      # --gpu; without: the GPU the SOURCE's device names, else strix-halo)
   "card": {"license", "tags", "output", "memory_bytes", "features"},
   "versions": {"package": {"skills_commit", "hrx_loom"}},          # recovered, for packages built unstamped
   "performance": {"baseline": {"name", "version", "args"},
                   "session": {"date", "source": SOURCE, "runs", "commands", "programs_sha256",
                               "metrics": {METRIC: {"loom", "baseline", "loom_runs", "baseline_runs", "acceptance"}}}},
   "gates": [{"name", "status", "level", "time", "source": SOURCE, "detail", "programs_sha256"}],
   "roofline": {"device", "tool", "checkpoint", "drafting", "metrics"}}
  SOURCE: {"kind", "run", "url", "device", "hrx_loom"}
Merge rules: card keys and versions: the later file wins. performance: sessions append in order and per metric
the last session that measured it is the card's value (card_lib card_metrics); the baseline is the last one given. gates: per gate the
latest evidence wins (gate_supersedes: a published date-only gate is older than new evidence of that day; equal
times: the later file), except that on the same programs_sha256 a later pass at a lower level (dev < fast < quick <
full) does not replace a pass at a higher level; a later failure always does. roofline: the later file wins. A "devlab"
block (an older Loom-only sweep) is ignored. Unknown metric and gate names are kept.
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
from card_lib import (DEFAULT_GPU, DRAFTED_ROWS, EVIDENCE_ID, GPUS, MEDIAN_METRICS, METRIC_KEYS, PERF_SOURCE_KINDS,  # noqa: E402
                      HF, ROOFLINE_METRIC, SCHEMA_ID, build_tag, card_metrics, derived_tags, dumps, gate_applies, gpu_from_device,
                      gpu_view, load_info, load_json, load_package, metric_applies, native_context, over_context, package_facts,
                      programs_sha256, require_valid, served_context, upstream_facts, with_gate_features, write_json,
                      SERVE_HEADROOM)


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


def gpu_key(a, src):
    """{"gpu": id} for an evidence file: --gpu, else the GPU the SOURCE's device names; {} when neither says (then
    build reads it as strix-halo, the only GPU before loom-card/3)."""
    gpu = getattr(a, "gpu", None) or gpu_from_device((src or {}).get("device"))
    return {"gpu": gpu} if gpu else {}


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


def evidence_gpu(ev, src=None):
    """The GPU an evidence block belongs to: the file's "gpu", else the GPU its SOURCE's device names, else strix-halo
    (every card measurement before loom-card/3 came from a Strix Halo)."""
    return ev.get("gpu") or gpu_from_device((src or {}).get("device")) or DEFAULT_GPU


def fit_settings(verdict_path, repo, gpu="radeon-r9700"):
    """(memory_bytes, served_context, required_bytes) of a package on a GPU from the fit table: the table's row of the
    repo's name. required_bytes is set (and nothing else) when the package does not fit the GPU's memory; memory_bytes
    and served_context only when the KV cache of the native context does not fit (the GPU's whole memory is the pool
    and bounds the context); all None when the native context fits."""
    rows = load_json(verdict_path)
    name = repo.split("/")[-1]
    row = next((r for r in rows if r["slug"] == name), None)
    if row is None:
        sys.exit(f"{verdict_path}: no row for {name}")
    mem = GPUS[gpu]["memory_bytes"]
    need = row["w"] + row["kv"] + SERVE_HEADROOM
    if need > mem and row["kv"] == 0:   # weights alone (no KV cache to cut): the model does not fit
        return None, None, int(need)
    if row["w"] + SERVE_HEADROOM >= mem:
        return None, None, int(need)
    if need <= mem:
        return None, None, None   # the native context fits: the package's own memory_bytes (at most the GPU's) serves it
    return mem, served_context(row["w"], row["kv"] / row["ctx"], mem), None


def cmd_build(a):
    pkg = load_package(a.pkg)
    upstream = None
    if a.upstream == "probe":
        ckpt = package_facts(pkg)["checkpoint"]
        legacy = "call_format" not in (pkg["cfg"].get("chat") or {})   # its chat block says nothing about tools: read the GGUF's template
        url = f"{HF}/{ckpt['repo']}/resolve/{ckpt.get('revision') or 'main'}/{ckpt['file']}" if legacy else None
        upstream = upstream_facts(ckpt["repo"], pkg["cfg"].get("model_id"), url)
    facts = package_facts(pkg, upstream)
    card, versions = {}, {}
    G = {}   # gpu -> {"baseline", "sessions", "gates", "baselines", "roofline", "meta"}: everything one GPU measured

    def gpu_state(gpu):
        return G.setdefault(gpu, {"baseline": None, "sessions": [], "gates": {}, "baselines": {}, "roofline": None, "meta": {}, "data": False})
    card_features = {}   # evidence-established feature statuses (import_card.py: greedy-only sampling)
    sources = Sources()
    if a.from_info:   # a published card: its data first, then the new evidence on top
        old = load_info(a.from_info)   # loom-card/2: its one GPU is strix-halo
        require_valid(old, a.from_info)
        a.repo = a.repo or old["repo"]
        if (old["repo"].split("/")[-1] if a.org else old["repo"]) != (a.repo.split("/")[-1] if a.org else a.repo):
            sys.exit(f"{a.from_info} is the card of {old['repo']}, not {a.repo}")
        card = json.loads(json.dumps(old["card"]))
        card_features = {k: v for k, v in card.pop("features", {}).items() if k == "sampling"}   # evidence; the rest is recomputed
        measured = {g["programs_sha256"] for d in old["devices"].values() for g in (d.get("gates") or {}).values()
                    if g.get("programs_sha256")}
        if card.get("memory_bytes") and a.memory_bytes is None and measured and programs_sha256(a.pkg) not in measured:
            sys.exit(f"{a.from_info}: its gates measured other programs than {a.pkg}; its card.memory_bytes "
                     f"({card['memory_bytes']}) is the old build's. Pass --memory-bytes (build.py prints the new one).")
        versions = {k: v for k, v in old["versions"].items() if k != "card_generator"}
        sources.by_id = dict(old["sources"])
        kind_of = lambda s: (old.get("sources") or {}).get(s["source"], {}).get("kind")
        for gpu, dev in old["devices"].items():
            st = gpu_state(gpu)
            st["meta"] = {k: dev[k] for k in ("state", "required_bytes", "memory_bytes", "served_context") if k in dev}
            st["data"] = "state" not in dev
            if dev.get("performance"):
                st["baseline"] = dev["performance"]["baseline"]
                st["sessions"] = [] if a.reset_performance else [s for s in dev["performance"]["sessions"]
                                                                if kind_of(s) in PERF_SOURCE_KINDS]
                dropped = len(dev["performance"]["sessions"]) - len(st["sessions"])
                if dropped:
                    print(f"--from-info ({gpu}): dropped {dropped} performance session(s)"
                          + ("" if a.reset_performance else " not measured on DevLab"), file=sys.stderr)
                st["baselines"] = json.loads(json.dumps(dev["performance"].get("baselines") or {}))
            for name, g in (dev.get("gates") or {}).items():   # a published gate has a date, not a time: later evidence that day wins
                if not gate_applies(name, facts["kind"], facts["features"]):   # e.g. spec after drafting was turned off: shown n/a
                    continue
                st["gates"][name] = ((g.get("date") or "", -1), None, g)
            st["roofline"] = dev.get("roofline")
        if a.report is None:
            a.report = old.get("report")
        if a.status is None:
            a.status = old.get("status")
        if a.next_steps is None:
            a.next_steps = old.get("next_steps")
    for i, path in enumerate(a.evidence or []):
        ev = load_json(path)
        if ev.get("evidence") != EVIDENCE_ID:
            sys.exit(f"{path}: not a {EVIDENCE_ID} file")
        c = ev.get("card") or {}
        card.update({k: v for k, v in c.items() if v is not None and k != "features"})
        if c.get("features"):
            card_features.update(c["features"])
        if (ev.get("versions") or {}).get("package"):
            versions.setdefault("package", {}).update(ev["versions"]["package"])
        perf = ev.get("performance") or {}
        if perf.get("session"):
            st = gpu_state(evidence_gpu(ev, perf["session"].get("source")))
        else:
            st = gpu_state(evidence_gpu(ev))
        st["data"] = True
        if ev.get("serve"):   # how this GPU serves the package: {"memory_bytes", "served_context"}
            st["meta"].update({k: v for k, v in ev["serve"].items() if v is not None})
        if perf.get("backend"):   # a further llama.cpp backend (make_info.py perf --backend): kept apart from the sessions
            sess = dict(perf["session"], source=sources.add(perf["session"]["source"]))
            st["baselines"][perf["backend"]] = {"baseline": perf["baseline"], "session": sess}
        else:
            if perf.get("baseline"):
                st["baseline"] = perf["baseline"]
            if perf.get("session"):
                s = dict(perf["session"])
                s["source"] = sources.add(s["source"])
                st["sessions"].append({k: v for k, v in s.items() if v is not None or k in ("date",)})
        for g in ev.get("gates") or []:
            gst = gpu_state(g.get("gpu") or evidence_gpu(ev, g.get("source")))
            gst["data"] = True
            new = ((g.get("time") or "", i), g, None)
            if g["name"] not in gst["gates"] or gate_supersedes(new, gst["gates"][g["name"]]):
                gst["gates"][g["name"]] = new
        if ev.get("roofline"):
            rst = gpu_state(ev.get("gpu") or gpu_from_device({"name": ev["roofline"]["device"]["name"]}) or DEFAULT_GPU)
            rst["roofline"], rst["data"] = ev["roofline"], True
        if ev.get("devlab"):
            print(f"{path}: ignored its Loom-only DevLab sweep (card performance is make_info.py perf on a devlab-perf job)", file=sys.stderr)
    if not a.repo:
        sys.exit("make_info.py build: --repo is required without --from-info")
    if a.org:   # the same repo name under another org (the staging flow)
        a.repo = f"{a.org}/{a.repo.split('/')[-1]}"
    if not a.generator_commit:
        a.generator_commit = skills_checkout(__file__)[0]
    for k, v in (("license", a.license), ("output", a.output), ("memory_bytes", a.memory_bytes)):
        if v is not None:
            card[k] = v
    # what the package's own files already say is not repeated
    gpus = [g for g in GPUS if g in G or g in (a.gpus or "").split(",")]
    tags = [t for t in card.get("tags") or [] if t not in derived_tags(facts["features"], gpus or (DEFAULT_GPU,))]
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

    # per-GPU serving settings: the command line, then the fit table
    for spec, key in ((a.gpu_memory, "memory_bytes"), (a.served_context, "served_context")):
        for item in spec or []:
            gpu, _, n = item.partition("=")
            if gpu not in GPUS or not n.isdigit():
                sys.exit(f"--{key.replace('_', '-')} {item}: expected GPU=N with GPU one of {', '.join(GPUS)}")
            gpu_state(gpu)["meta"][key] = int(n)
    if a.fit:
        for gpu in ("radeon-r9700",):
            if gpu in gpus or gpu in G:
                mem, ctx, need = fit_settings(a.fit, a.repo, gpu)
                m = gpu_state(gpu)["meta"]
                if need is not None:
                    m.update(state="unsupported", required_bytes=need)
                else:
                    m.pop("state", None)
                    m.pop("required_bytes", None)
                    for k in ("memory_bytes", "served_context"):
                        m.pop(k, None)
                    if mem is not None:
                        m["memory_bytes"] = mem
                    if ctx is not None:
                        m["served_context"] = ctx
    gpus = [g for g in GPUS if g in G or g in (a.gpus or "").split(",")]
    for g in (a.gpus or "").split(",") if a.gpus else []:
        if g and g not in GPUS:
            sys.exit(f"--gpus {g}: not one of {', '.join(GPUS)}")

    def gate_rows(gates):
        out = {}
        for name in sorted(gates):
            if gates[name][2] is not None:   # kept from --from-info as published
                out[name] = gates[name][2]
                continue
            g = gates[name][1]
            t = g.get("time")
            row = {"status": g["status"], "level": g.get("level"), "date": t[:10] if t else None,
                   "source": sources.add(g["source"]), "detail": g.get("detail")}
            if g.get("programs_sha256"):
                row["programs_sha256"] = g["programs_sha256"]
            out[name] = row
        return out
    gate_out = {g: gate_rows(G[g]["gates"]) for g in G}
    # every feature, so a dashboard reads them from info.json alone: the package's and upstream's facts, then the gates'
    # sampling proof, then evidence (an old report's greedy-only sampling, only without a sampling gate). The sampling
    # gate of any GPU proves it.
    feats = facts["features"]
    any_gates = {}
    for g in GPUS:
        for n, row in (gate_out.get(g) or {}).items():
            any_gates.setdefault(n, row)
    if "sampling" in card_features and "sampling" not in any_gates:
        feats["sampling"] = card_features["sampling"]
    card["features"] = {**with_gate_features(feats, facts["kind"], any_gates), **{k: v for k, v in card_features.items() if k != "sampling"}}
    devices = {}
    for gpu in gpus:
        st = G.get(gpu) or {"meta": {}, "sessions": [], "baselines": {}, "gates": {}, "baseline": None, "roofline": None, "data": False}
        d = dict(st["meta"])
        has = st["data"] or bool(gate_out.get(gpu) or st["sessions"] or st["baselines"] or st["roofline"])
        if has:
            d.pop("state", None)
            d.pop("required_bytes", None)
            d["gates"] = gate_out.get(gpu) or {}
            if st["sessions"]:
                d["performance"] = {"baseline": st["baseline"] or {"name": "llama.cpp", "version": None, "args": None},
                                    "sessions": st["sessions"]}
                if st["baselines"]:
                    d["performance"]["baselines"] = st["baselines"]
            if st["roofline"]:
                d["roofline"] = st["roofline"]
            cap = GPUS[gpu]["memory_bytes"]
            base_mem = facts["memory_bytes"] or card.get("memory_bytes")
            if cap and "memory_bytes" not in d and base_mem and base_mem > cap:
                d["memory_bytes"] = cap   # the package's budget is more than this GPU has
        elif "state" not in d:
            d["state"] = "not_measured"
        devices[gpu] = d
    info = {"schema": SCHEMA_ID, "repo": a.repo, "card": card,
            "versions": {"card_generator": a.generator_commit, **versions},
            "sources": sources.by_id, "devices": devices}
    if a.report:
        info["report"] = a.report
    if a.status and a.status != "active":
        info["status"] = a.status
    if a.next_steps:
        info["next_steps"] = a.next_steps
    used = {s["source"] for d in devices.values() for s in (d.get("performance") or {}).get("sessions", [])}
    used |= {b["session"]["source"] for d in devices.values() for b in ((d.get("performance") or {}).get("baselines") or {}).values()}
    used |= {g["source"] for d in devices.values() for g in (d.get("gates") or {}).values()}
    info["sources"] = {k: v for k, v in sources.by_id.items() if k in used}
    require_valid(info)
    out = dumps(info)
    if a.o == "-":
        sys.stdout.write(out)
    else:
        a.o = a.o or os.path.join(a.pkg, "info.json")
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
    """The aggregate rates of c4_steady.py's runs (its JSON lines, or one number per line). A run that measured
    nothing (a rate <= 0, or a window <= 0 s: every stream failed) is skipped, with a note: a crash, not data."""
    vals = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line) if line.startswith("{") else {"agg_tps": float(line)}
        if r.get("agg_tps") is None:
            continue
        if r["agg_tps"] <= 0 or r.get("window_s", 1) <= 0:
            print(f"{path}: skipped an invalid c4_steady run: {line[:120]}", file=sys.stderr)
            continue
        vals.append(r["agg_tps"])
    return vals


def summary(key, vals):
    if not vals:
        return None
    return round(statistics.median(vals) if key in MEDIAN_METRICS else statistics.mean(vals), 1)


def baseline_version(text, backend="rocm"):
    tag = build_tag(text)
    return f"llamacpp-{backend}/{tag}" if tag and text == tag else text


def refused_tokens(rows, package, measured):
    """{card metric: prompt tokens} of the long documents the Loom server refused (perf.py rows' over_context) and
    the session did not measure: each prompt re-rendered from its row's nonce and counted with the package's
    tokenizer (null without --package, or for a row from before perf.py logged its nonce)."""
    from bench_common import Package, long_prompt
    card_key = {v: k for k, v in ROOFLINE_METRIC.items()}
    pkg = Package(package) if package and os.path.exists(os.path.join(package, "tokenizer.json")) else None
    out = {}
    for r in rows:
        for k in r.get("over_context", []):
            if card_key[k] in measured:
                continue
            n = None
            if pkg and r.get("nonce"):
                think = {"on": True, "off": False}.get(r.get("thinking"))
                msgs = [{"role": "user", "content": long_prompt(k, r["nonce"])}]
                n = len(pkg.ids(pkg.render_messages(msgs, None, think)))
            prev = out.get(card_key[k])
            out[card_key[k]] = n if prev is None else (prev if n is None else min(prev, n))
    return out


def cmd_perf(a):
    if a.source_kind not in PERF_SOURCE_KINDS:
        sys.exit(f"perf --source-kind {a.source_kind}: card performance comes only from AMD DevLab Dispatch "
                 "(a devlab-perf job: loom-org SKILL.md step 3); measure locally only to develop")
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
    extra = a.backend != "rocm"   # a further llama.cpp backend: a baseline-only session (no Loom values)
    if extra and (loom_rows or a.c4_loom):
        sys.exit(f"perf --backend {a.backend}: the session holds the {a.backend} engine's runs only (no Loom files)")
    metrics = {}
    for k in keep:
        if k not in ls and k not in bs:
            continue
        m = {"baseline": summary(k, bs.get(k))} if extra else {"loom": summary(k, ls.get(k)), "baseline": summary(k, bs.get(k))}
        if ls.get(k):
            m["loom_runs"] = ls[k]
        if bs.get(k):
            m["baseline_runs"] = bs[k]
        metrics[k] = m
    with_acceptance(metrics, acceptance_samples(loom_rows, a.c4_loom))
    dp = [len(ls[k]) for k in keep if k in ls and k != "streams4"]
    if extra:
        ls = bs   # the runs counted are the baseline engine's
        dp = [len(bs[k]) for k in keep if k in bs and k != "streams4"]
    runs = {"decode_prefill": max(dp) if dp else None,
            "streams4": len(ls["streams4"]) if "streams4" in keep and ls.get("streams4") else None}
    date = a.date or max([r["time"][:10] for r in loom_rows + llama_rows if r.get("time")] or [None], key=lambda x: x or "")
    commands = list(a.command or [])
    if a.commands_file:
        commands += [line.rstrip("\n") for line in open(a.commands_file, encoding="utf-8") if line.strip()]
    session = {"date": date, "source": source_arg(a), "runs": runs, "metrics": metrics,
               "programs_sha256": programs_sha256(a.package) if a.package else None}
    if a.fetched_programs and a.package and a.fetched_programs != programs_sha256(a.package):
        sys.exit(f"the run fetched programs {a.fetched_programs[:12]}, the package has {programs_sha256(a.package)[:12]}: "
                 "the card was republished in between; re-run")
    if not extra:
        session["commands"] = commands
    over = refused_tokens(loom_rows, a.package, set(ls))
    if over and not extra:   # perf.py: the Loom server refused these documents as over its context (card_lib.over_context)
        session["over_context"] = over
    perf = {"session": session}
    if a.baseline_version or a.baseline_args:
        perf["baseline"] = {"name": a.baseline_name, "version": baseline_version(a.baseline_version, a.backend) if a.baseline_version else None,
                            "args": a.baseline_args}
    if extra:
        perf["backend"] = a.backend
        perf["baseline"] = dict(perf.get("baseline") or {"name": a.baseline_name, "version": None, "args": None})
        if a.baseline_sha256:
            perf["baseline"]["sha256"] = a.baseline_sha256
    if a.differs and "baseline" in perf:   # the main baseline too: a Loom-side note or a cell llama.cpp cannot run (rule: never silent)
        perf["baseline"]["differs"] = a.differs
    write_json(a.o, {"evidence": EVIDENCE_ID, **gpu_key(a, session["source"]), "performance": perf})
    print(f"wrote {a.o}: " + ", ".join(f"{k} {m.get('loom')}/{m['baseline']}" for k, m in metrics.items()), file=sys.stderr)


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
    ev = {"evidence": EVIDENCE_ID, **gpu_key(a, src), "gates": out}
    acc = acceptance(logs) if spec_date and a.source_kind in PERF_SOURCE_KINDS else None   # a card statistic: DevLab only
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


def roofline_block(facts, m, acceptance, R, bw, flops, tool, checkpoint, row_acceptance=None, skip=()):
    """info.json "roofline" from roofline.py's model shape m (R: the roofline module). A drafted row's ceiling counts
    row_acceptance[metric] (measured in the row's own session) when given, else `acceptance` (the spec gate's)."""
    ceil = R.ceilings(m, bw, flops, (1000, 2000, 8000, 16000, 32000))
    spec = facts["spec"]
    tpp = round(1 + acceptance * spec[1], 3) if spec and acceptance is not None else None
    metrics = {}
    for key, rk in ROOFLINE_METRIC.items():
        if not metric_applies(key, facts["kind"], facts["features"]):
            continue
        if key in skip:   # a document longer than the model's native context was never measured: no ceiling to show
            continue
        c = ceil[rk]
        basis = None
        if key == "streams4" and ceil.get("c4_steady_shared"):   # MoE: the 4 streams' experts overlap; the ceiling a run may not exceed
            c, basis = ceil["c4_steady_shared"], "shared-experts"
        row = {"ceiling": round(c, 1), "efficiency": round(R.efficiency(m, rk, c), 3)}
        if basis:
            row["basis"] = basis
        # benchmark rule: 4 streams are loaded without drafting, so that row's ceiling is always the plain one. A drafted row counts the
        # acceptance measured in the row's own session on THIS package; with none (or 0) the row shows the single-pass ceiling.
        if spec and key in DRAFTED_ROWS and key != "streams4":
            acc = (row_acceptance or {}).get(key)
            if acc:
                t = round(1 + acc * spec[1], 3)
                d = R.drafted_row_ceiling(m, rk, spec[1], acc, bw)
                row.update(ceiling=round(d, 1), efficiency=round(R.efficiency(m, rk, d, tokens_per_pass=t), 3),
                           acceptance=acc, acceptance_from="session", tokens_per_pass=t, single_pass=round(c, 1))
        metrics[key] = row
    preset = next((d["name"] for d in R.DEVICES.values() if (d["bw"], d["flops"]) == (bw, flops)), None)
    return {"device": {"name": preset or f"{R.DEVICE}, overridden",
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
    if a.device is None:
        a.device = GPUS[a.gpu or DEFAULT_GPU]["roofline"]   # the preset of the GPU (radeon-r9700: the DevLab R9700S runner)
    dev = R.DEVICES[a.device]
    a.bw, a.flops = a.bw or dev["bw"], a.flops or dev["flops"]
    facts = package_facts(load_package(a.package))
    gguf = a.gguf or cached_gguf(facts["checkpoint"])
    if not gguf:
        sys.exit(f"{facts['checkpoint']['file']} is not in the Hugging Face cache at the locked revision; pass --gguf")
    cfg_p = a.cfg or os.path.join(a.package, "llm_config.json")
    cfg = load_json(cfg_p) if os.path.exists(cfg_p) else {}
    info_p = os.path.join(a.package, "info.json")
    gpu = a.gpu or gpu_from_device({"name": dev["name"]}) or DEFAULT_GPU
    sessions = list(((gpu_view(load_info(info_p), gpu).get("performance") or {}).get("sessions") or [])
                    if os.path.exists(info_p) else [])
    for path in a.evidence or []:
        ev = load_json(path)
        s = (ev.get("performance") or {}).get("session")
        if s and evidence_gpu(ev, s.get("source")) == gpu:
            sessions.append(s)
    # a drafted ceiling counts acceptance measured on THIS package's programs only: another build's acceptance (an older drafter, depth
    # or n-gram package) says nothing about it; without a current one the card shows the single-pass ceiling
    here = programs_sha256(a.package)
    all_sessions = sessions
    sessions = [s for s in sessions if s.get("programs_sha256") == here]
    acc, row_acc = drafted_acceptance(facts, sessions, a.acceptance)
    script = os.path.join(rdir, "roofline.py")
    commit, dirty = skills_checkout(script)
    with open(script, "rb") as f:
        tool = {"script": "skills/loom-roofline/roofline.py", "sha256": hashlib.sha256(f.read()).hexdigest(),
                "skills_commit": a.skills_commit or commit}
    if dirty and not a.skills_commit:
        tool["dirty"] = True
    skip = over_context({"performance": {"sessions": all_sessions}}, a.served_context or native_context(facts["features"]))
    block = roofline_block(facts, R.model_shape(gguf, cfg), acc, R, a.bw, a.flops, tool, os.path.basename(gguf), row_acc, skip)
    write_json(a.o, {"evidence": EVIDENCE_ID, "gpu": gpu, "roofline": block})
    print(f"wrote {a.o}: " + ", ".join(f"{k} {r['ceiling']}" for k, r in block["metrics"].items())
          + "".join(f"; {k} acceptance {r['acceptance']} ({r['acceptance_from']})" for k, r in block["metrics"].items()
                    if r.get("acceptance_from")),
          file=sys.stderr)


def next_steps_file(path):
    with open(path, encoding="utf-8") as f:
        return [re.sub(r"^[-*]\s+", "", ln.strip()) for ln in f if ln.strip()]


def add_source_args(p, kind):
    p.add_argument("--source-kind", default=kind, choices=["devlab", "factory", "local"])
    p.add_argument("--run")
    p.add_argument("--url")
    p.add_argument("--device", default="auto", help="auto (this machine), none, or a JSON file with {cpu, gpu, os}")
    p.add_argument("--hrx-loom", default="auto", help="auto (the SDK under $LOOM_ROOT), none, or a JSON file")
    p.add_argument("--package", help="the package measured (records card_lib.programs_sha256 of its programs)")
    p.add_argument("--gpu", choices=sorted(GPUS), help="the GPU that measured (default: the one the source's device names, "
                   "else strix-halo; a DevLab job's downloaded files carry no device: pass it)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("pkg")
    b.add_argument("--repo", help="loom-fun/NAME (default with --from-info: its repo)")
    b.add_argument("--org", help="the Hugging Face org of the card's repo (the repo name stays): staging cards use loom-fun-staging")
    b.add_argument("--gpus", help="comma-separated GPU ids the card covers (card_lib.GPUS); one without evidence is not_measured")
    b.add_argument("--gpu-memory", action="append", metavar="GPU=BYTES", help="loom-serve --memory_bytes on that GPU (repeat)")
    b.add_argument("--served-context", action="append", metavar="GPU=TOKENS", help="tokens per request that GPU serves (repeat)")
    b.add_argument("--fit", metavar="VERDICT.json", help="the fit table: radeon-r9700's memory_bytes, served context or unsupported state")
    b.add_argument("--generator-commit", help="loom-skills commit of the card scripts, or 'unknown' (default: this "
                   "checkout's HEAD)")
    b.add_argument("--evidence", nargs="*")
    b.add_argument("--memory-bytes", type=int)
    b.add_argument("--output")
    b.add_argument("--license")
    b.add_argument("--report", choices=["generated", "attached"])
    b.add_argument("--from-info", help="a published info.json to start from (regenerating a card with new evidence)")
    b.add_argument("--reset-performance", action="store_true", help="drop --from-info's performance sessions (re-measured from scratch)")
    b.add_argument("--status", choices=["paused", "active"], help="paused: work on this model has stopped (active: clears it)")
    b.add_argument("--next-steps", type=next_steps_file, help="a file of next steps, one per line (empty: none)")
    b.add_argument("--upstream", choices=["probe", "none"], default=os.environ.get("LOOM_CARD_UPSTREAM", "probe"),
                   help="probe: ask Hugging Face what the upstream model has (vision: an mmproj or vision_config; audio: audio_* "
                   "config keys; an older package: the GGUF chat template), for the features that are n/a when it has none; "
                   "none: offline, they are not determined")
    b.add_argument("-o", help="where to write the card (default: PKG/info.json, the file render_card.py and publish_package.py read; never the current directory)")
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
    p.add_argument("--backend", default="rocm", help="the llama.cpp backend measured: rocm (the card's baseline, with Loom) "
                   "or another (vulkan): a baseline-only session recorded under performance.baselines.BACKEND")
    p.add_argument("--baseline-sha256", help="sha256 of the baseline release archive (--backend other than rocm)")
    p.add_argument("--differs", action="append", help="a server setting that differs from the main baseline's or that this backend could not run, with the reason, one line (repeat)")
    p.add_argument("--fetched-programs", help="programs_sha256 the DevLab job fetched (its programs_sha256.txt): refused if "
                   "it differs from --package's, i.e. the card was republished after the run fetched it")
    p.add_argument("--baseline-args", help="the baseline server's exact arguments, model path, host and port left out")
    p.add_argument("--metrics", help="comma-separated metrics this session provides (default: all it measured)")
    p.add_argument("--command", action="append", help="a measuring command, for REPORT.md (repeat)")
    p.add_argument("--commands-file", help="measuring commands, one per line")
    p.add_argument("--date", help="YYYY-MM-DD (default: the latest Loom row's time)")
    add_source_args(p, "devlab")
    p.add_argument("-o", required=True)
    g = sub.add_parser("gates")
    g.add_argument("files", nargs="+")
    g.add_argument("--time", help="ISO time of the run when the file has none (gates.out, kernel test log)")
    g.add_argument("--level")
    g.add_argument("--spec-log", help="the spec gate's log with 'acceptance N%% of M proposals' (default: next to FILE)")
    add_source_args(g, "local")
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
    r.add_argument("--device", default=None, help="roofline.py device preset for --bw / --flops (roofline.DEVICES; default: the "
                   "--gpu's: strix-halo, radeon-r9700 -> radeon-r9700s, the DevLab R9700S runner)")
    r.add_argument("--bw", type=float, default=None, help="memory bandwidth, bytes/s (default: the --device preset)")
    r.add_argument("--gpu", choices=sorted(GPUS), help="the GPU the ceilings are for (default strix-halo); sets --device")
    r.add_argument("--served-context", type=int, help="tokens per request the GPU serves, when less than the native context: "
                   "documents longer than it were refused, so they have no ceiling")
    r.add_argument("--flops", type=float, default=None, help="peak FLOP/s (default: the --device preset)")
    r.add_argument("-o", required=True)
    a = ap.parse_args(argv)
    {"build": cmd_build, "perf": cmd_perf, "gates": cmd_gates, "roofline": cmd_roofline}[a.cmd](a)


if __name__ == "__main__":
    main()
