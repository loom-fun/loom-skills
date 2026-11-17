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

      fleet/devices/<device>.json (loom-device-results/1; README there): a GPU with no DevLab runner (radeon-rx9070xt, gfx1150) is on the
      card where its file says this package's model fits: its devices entry gets serve_args, served_context (max_context below the
      native one), the evidence of the entry (a "machine" source) and, when the file asks, shares_section_with; its preset (chat-9070xt) is
      written into model.json. --device-results DIR (none: ignore the files; default $LOOM_DEVICE_RESULTS, else fleet/devices).
  make_info.py devices [DIR]: validate the device results files and summarize them.

  make_info.py perf [--gpu GPU] --loom-jsonl LOOM.jsonl --llama-jsonl LLAMA.jsonl [--c4-loom F] [--c4-llama F]
                    --baseline-version llamacpp-rocm/bNNNN [--baseline-args "ARGS"] [--metrics decode,...]
                    [--command CMD ...] [--package PKG] [--date YYYY-MM-DD] [SOURCE] -o perf_evidence.json
      perf.py logs (one engine per file; or --perf-log perf_log.jsonl with finalize.sh's labels final /
      llama-final) -> one measurement session ("aggregate": "median"): the rows split into serve sessions at devlab_perf.sh's
      {"session_end": V} lines (no markers: each row is a session); a session's value is the median of its samples (a row measured
      with perf.py --repeat-rows logs "samples"), the metric the median over the sessions (loom_runs: those values; runs.samples:
      the samples per session of the repeated rows); 4 streams: the median of the c4_steady.py runs. --metrics keeps only those metrics (a session that re-measured
      some rows). --package records which programs were measured (card_lib.programs_sha256). Rows that carry the
      drafted-token acceptance of their decode requests (perf.py "acceptance", c4_steady.py JSON lines) give each
      drafted metric its session's "acceptance" (median over the sessions).

  make_info.py gates [--gpu GPU] FILE [FILE ...] [--time YYYY-MM-DDTHH:MM:SS] [--level L] [--spec-log F] [--package PKG]
                     [SOURCE] -o gates_evidence.json
      gates.sh's gates.json (verdicts, level, ts), a gates.out with its "GATES ..." line, or a kernel test
      log ending in "N/M passed" (make_kernel_tests.py) -> gate evidence. A gates JSON may give a per-gate "detail"
      (logits_V.json of devlab_perf.sh: gate `logits`; a `gufo` gate in old evidence is dropped, card_lib.RETIRED_GATES). The spec gate's "acceptance N% of M
      proposals" (gates_spec.log or gates.out next to the file, or --spec-log) -> the spec_acceptance metric, from a
      DevLab (or factory) source only, like every card performance value.

  make_info.py equivalent --gpu GPU --proof PROOF.json --package PKG [--compared-programs SHA] [--proof-url U] [SOURCE] -o ev.json
      fleet/stage_equiv.py proved this package's default-path stages identical (after canonicalization) to the build whose
      sessions the card shows: those sessions stand for this package on GPU (remeasure block, result "identical", band 0, the
      proof named on the card). Refused unless the proof says equivalent.

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

  SOURCE (perf and gates): --source-kind devlab|factory|local|machine (perf: default devlab; gates: default local; machine: a named machine of a GPU with no DevLab runner, fleet/devices) [--run ID --url URL] [--device auto|none|FILE.json]
      [--hrx-loom auto|none|FILE.json]. auto records this machine's device and the Loom SDK under $LOOM_ROOT
      (../machine_info.py); on downloaded DevLab job files: --device none --hrx-loom none. Card performance comes
      only from AMD DevLab Dispatch (a devlab-perf job or a scripted DevLab port, kind devlab; the loom-factory
      workflow, kind factory), or from a named machine for a GPU with no DevLab runner (kind machine): perf refuses --source-kind local. Gates may come from any source.

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
import datetime
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from card_lib import (DEFAULT_GPU, DRAFTED_ROWS, EVIDENCE_ID, GPUS, METRIC_KEYS, PERF_SOURCE_KINDS,  # noqa: E402
                      HF, RETIRED_GATES, ROOFLINE_METRIC, SCHEMA_ID, build_tag, card_metrics, derived_tags, dumps, gate_applies, gpu_from_device,
                      gpu_view, load_info, load_json, load_package, metric_applies, native_context, over_context, package_facts,
                      programs_sha256, require_valid, fixed_pool_capacity, gpu_preset_execution, gpu_preset_name, with_gpu_preset, served_context, fit_served_context, upstream_facts, with_gate_features, write_json,
                      SERVE_HEADROOM, registry, fit_rows, load_device_results, validate_device_results, with_device_preset, DEVICE_RESULTS_ID)


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


def fit_tables(specs):
    """[(gpu, fit table path)] of the --fit arguments: GPU=PATH, 'auto' (every registry device whose fit table exists in this checkout), or a
    path (the device whose device.json fit_table has that file name, else radeon-r9700: the legacy single table)."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
    out = []
    for spec in specs or []:
        if spec == "auto":
            out += [(g, os.path.join(repo_root, registry.fit_table(g))) for g in GPUS if os.path.exists(os.path.join(repo_root, registry.fit_table(g)))]
        elif "=" in spec and spec.split("=")[0] in GPUS:
            out.append(tuple(spec.split("=", 1)))
        else:
            hit = [g for g in GPUS if os.path.basename(registry.fit_table(g)) == os.path.basename(spec)]
            out.append((hit[0] if hit else "radeon-r9700", spec))
    return out


def fit_settings(verdict_path, repo, gpu="radeon-r9700"):   # gpu: any device with a fit table (device.json fit_table)
    """(memory_bytes, served_context, required_bytes) of a package on a GPU from the fit table: the table's row of the
    repo's name. required_bytes is set (and nothing else) when the package does not fit the GPU's memory; memory_bytes
    and served_context only when the KV cache of the native context does not fit (the GPU's whole memory is the pool
    and bounds the context); all None when the native context fits."""
    rows = list(fit_rows(verdict_path, gpu).values())   # the settings of the scope rules are part of each row
    name = repo.split("/")[-1]
    row = next((r for r in rows if r["slug"] == name), None)
    if row is None:
        sys.exit(f"{verdict_path}: no row for {name}")
    mem = GPUS[gpu]["memory_bytes"]
    if mem is None:
        return None, None, None   # the device takes the package's own memory (Strix Halo): nothing to fit
    need = row["w"] + row["kv"] + SERVE_HEADROOM
    if need > mem and row["kv"] == 0:   # weights alone (no KV cache to cut): the model does not fit
        return None, None, int(need)
    if row["w"] + SERVE_HEADROOM >= mem:
        return None, None, int(need)
    ctx = fit_served_context(row, mem)
    if ctx is None:
        return None, None, None   # the native context fits: the package's own memory_bytes (at most the GPU's) serves it
    return mem, ctx, None


def write_gpu_preset(pkg_dir, model, verdict_path, repo, ctx, gpu="radeon-r9700"):
    """Add the `<first>-r9700` preset to the package's model.json from the scope rules' exec.* settings for the device (card_lib.fit_rows
    puts them on the fit row as "r9700_execution"; model nodes: skills/loom-kb/model/<slug>) (idempotent: the same
    settings give the same file); returns the execution the preset sets, or None when the row has no block (model.json is
    then left as it is, whatever presets a build gave it). The default preset, programs, weights and lock are untouched."""
    row = fit_rows(verdict_path, gpu).get(repo.split("/")[-1])   # the fit's sizing + the serving settings the scope rules give (exec.*)
    ex = gpu_preset_execution(row, ctx, gpu) or {}
    if ctx and "context_capacity" not in ex:   # a served context below the native one (fit ctx_cap or memory): the preset serves it
        ex.update(context_capacity=ctx)
    # the R9700 is served with fixed state backing (card_lib.GPUS "serve_args"): the pool is committed eagerly, so a default preset
    # whose pool is larger than the GPU holds beside the weights gets a smaller one (a native context that fits always fits it)
    base = next(iter(model["presets"].values()))["execution"]
    cap = fixed_pool_capacity(row, gpu=gpu)
    pool = ex.get("context_capacity") or base.get("context_capacity")   # SDK 0.2: the pool is sized by context_capacity (+ memory_bytes)
    if cap is not None and pool and pool > cap:
        ex["context_capacity"] = cap
    if not ex:
        return None
    doc = with_gpu_preset(model, ex, gpu)
    with open(os.path.join(pkg_dir, "model.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return doc["presets"][gpu_preset_name(doc, gpu)]["execution"]


def device_native_context(feats):
    return native_context(feats)


def write_device_preset(pkg_dir, gpu, doc, entry):
    """Add the GPU's `<first><suffix>` preset (chat-9070xt) to the package's model.json from its fleet/devices entry: a copy of the preset of
    the device whose programs it serves (doc["same_programs_as"]: the R9700's, when the package has one), with the entry's `execution` over
    it and context_capacity = the entry's max_context. Written when the file asks for it for every model (preset.always), or the entry has
    execution overrides or admits less context than the base preset serves; otherwise the GPU reuses the default preset. Returns the preset
    name written, else None. Idempotent; the default preset, programs, weights and lock are untouched."""
    path = os.path.join(pkg_dir, "model.json")
    model = load_json(path)
    first = next(iter(model["presets"]))
    other = doc.get("same_programs_as")
    base = f"{first}{GPUS[other].get('preset_suffix', '')}" if other in GPUS else None
    base_ex = (model["presets"].get(base) or model["presets"][first])["execution"]
    ex = dict(entry.get("execution") or {})
    if entry.get("max_context"):
        ex.setdefault("context_capacity", entry["max_context"])
    needed = ((doc.get("preset") or {}).get("always") or entry.get("execution")
              or (ex.get("context_capacity") or 0) < (base_ex.get("context_capacity") or 1 << 62))
    if not needed or not ex:
        return None
    new = with_device_preset(model, gpu, ex, base)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(new, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return gpu_preset_name(new, gpu)


def device_meta(gpu, doc, entry, feats):
    """The devices entry fields a fleet/devices entry gives its GPU on a card: how it is served and the context it admits."""
    args = list(entry["serve_args"]) if "serve_args" in entry else list(GPUS[gpu].get("serve_args") or [])
    m = {"serve_args": args}
    mem = next((int(x.split("=", 1)[1]) for x in args if x.startswith("--memory_bytes=") and x.split("=", 1)[1].isdigit()), None)
    if mem:
        m["memory_bytes"] = mem
    native = device_native_context(feats)
    if entry.get("max_context") and (not native or entry["max_context"] < native):
        m["served_context"] = entry["max_context"]
    return m


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
        return G.setdefault(gpu, {"baseline": None, "sessions": [], "gates": {}, "baselines": {}, "roofline": None, "meta": {}, "data": False,
                                  "remeasure": None})
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
                if name in RETIRED_GATES or not gate_applies(name, facts["kind"], facts["features"], facts.get("logits", True)):   # e.g. spec after drafting was turned off: shown n/a
                    continue
                st["gates"][name] = ((g.get("date") or "", -1), None, g)
            st["roofline"] = dev.get("roofline")
            st["remeasure"] = dev.get("remeasure")
        if a.report is None:
            a.report = old.get("report")
        if a.status is None:
            a.status = old.get("status")
        if a.next_steps is None:
            a.next_steps = old.get("next_steps")
    for spec in a.perf_from or []:   # GPU=INFO.json: that card's performance on GPU (sessions, baselines and their sources) as published
        gpu, _, path = spec.partition("=")
        if gpu not in GPUS or not path:
            sys.exit(f"--perf-from {spec}: expected GPU=INFO.json with GPU one of {', '.join(GPUS)}")
        other = load_info(path)
        require_valid(other, path)
        if other["repo"].split("/")[-1] != (a.repo or "").split("/")[-1]:
            sys.exit(f"--perf-from {path} is the card of {other['repo']}, not {a.repo}")
        dev = (other.get("devices") or {}).get(gpu) or {}
        if not dev.get("performance"):
            sys.exit(f"--perf-from {path}: no {gpu} performance")
        st = gpu_state(gpu)
        st["data"] = True
        osrc = lambda sid: sources.add(other["sources"][sid])
        perf = json.loads(json.dumps(dev["performance"]))
        st["baseline"] = perf["baseline"]
        st["sessions"] = [dict(s, source=osrc(s["source"])) for s in perf["sessions"]]
        st["baselines"] = {b: dict(x, session=dict(x["session"], source=osrc(x["session"]["source"]))) for b, x in (perf.get("baselines") or {}).items()}
    # fleet/devices data files: a GPU with no DevLab runner shows only where its file has data for this card's model (fits true); the
    # entry's evidence applies after the --evidence files, like any evidence
    slug = (a.repo or "").split("/")[-1]
    dev_serve = {}
    evs = [(path, None) for path in a.evidence or []]
    if slug and a.device_results != "none":
        for gpu, doc in load_device_results(a.device_results).items():
            entry = doc["models"].get(slug)
            if entry and entry["fits"]:
                dev_serve[gpu] = (doc, entry)
                evs += [(f"{gpu}:{slug}.evidence[{j}]", dict(e, gpu=gpu)) for j, e in enumerate(entry.get("evidence") or [])]
    for gpu in dev_serve:   # the device file is the only source of that GPU's data: an --from-info card's earlier copy of it is replaced, not doubled
        G.pop(gpu, None)
    for i, (path, ev) in enumerate(evs):
        ev = ev or load_json(path)
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
            if g["name"] in RETIRED_GATES:   # evidence from before the needle gates were removed
                continue
            gst = gpu_state(g.get("gpu") or evidence_gpu(ev, g.get("source")))
            gst["data"] = True
            new = ((g.get("time") or "", i), g, None)
            if g["name"] not in gst["gates"] or gate_supersedes(new, gst["gates"][g["name"]]):
                gst["gates"][g["name"]] = new
        if ev.get("remeasure"):   # make_info.py remeasure: a same-runner A/B of this package against the build the table shows
            mst = gpu_state(evidence_gpu(ev))
            prev = mst["remeasure"]
            mst["remeasure"] = dict(ev["remeasure"], source=sources.add(ev["remeasure"]["source"]))
            if ev["remeasure"].get("first_source"):
                mst["remeasure"]["first_source"] = sources.add(ev["remeasure"]["first_source"])
            if ev["remeasure"]["result"] == "identical":   # the proof covers the build it compared with, and what that build's own re-measure stood for
                chain = prev and prev["programs_sha256"] == ev["remeasure"]["compared_programs"]
                mst["remeasure"]["carried_programs"] = [ev["remeasure"]["compared_programs"]] + ([prev["compared_programs"]] if chain else [])
                if chain:   # the A/B that re-measured the compared build stays on the card
                    mst["remeasure"]["previous"] = prev
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
    for gpu, fit in fit_tables(a.fit):
        if gpu in gpus or gpu in G:
            mem, ctx, need = fit_settings(fit, a.repo, gpu)
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
                if need is None:
                    write_gpu_preset(a.pkg, pkg["model"], fit, a.repo, ctx, gpu)
    for gpu, (doc, entry) in dev_serve.items():   # how the GPU serves this package (fleet/devices), and its preset
        m = gpu_state(gpu)["meta"]
        for k in ("state", "required_bytes", "memory_bytes", "served_context"):
            m.pop(k, None)
        m.update(device_meta(gpu, doc, entry, facts["features"]))
        write_device_preset(a.pkg, gpu, doc, entry)
    for gpu, (doc, entry) in dev_serve.items():   # a GPU that shares another's performance section: that GPU is on the card
        share = doc.get("share_section_with")
        if share and (share in G or share in (a.gpus or "").split(",")):
            gpu_state(gpu)["meta"]["shares_section_with"] = share
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
            if st.get("remeasure") and st["remeasure"]["programs_sha256"] == programs_sha256(a.pkg):   # about this package only
                d["remeasure"] = st["remeasure"]
                for k, imp in (st["remeasure"].get("improved") or {}).items():   # confirmed faster in 2 runs: the card value is the runs' mean
                    row = next((s["metrics"][k] for s in reversed(st["sessions"]) if k in s["metrics"]), None)
                    if row is not None:
                        row["loom"], row["loom_runs"] = imp["loom"], imp["runs"]
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
    for d in devices.values():
        r = d.get("remeasure")
        while r:   # a chained re-measure (identical on top of a re-measured build) keeps its previous blocks' sources
            used |= {r["source"]} | ({r["first_source"]} if r.get("first_source") else set())
            r = r.get("previous")
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
    """{metric: [values]} of perf.py rows (prefill samples bucketed by prompt tokens, as bench_common does). A row measured
    with --repeat-rows carries r["samples"] {metric: [the repeats]} (card metric names): those replace its single value."""
    out = {}
    for r in rows:
        one = {}
        for k, m in DECODE:
            if r.get(k) is not None:
                one.setdefault(m, []).append(r[k])
        for p in r.get("prefill", []) + r.get("prefill_long", []):
            n = min(DOC, key=lambda x: abs(x - p["prompt_tokens"]))
            one.setdefault(DOC[n], []).append(p["prefill_tps"])
        if r.get("agg_tps") is not None:
            one.setdefault("streams4", []).append(r["agg_tps"])
        one.update({m: v for m, v in (r.get("samples") or {}).items() if v})
        for m, v in one.items():
            out.setdefault(m, []).extend(v)
    return out


def split_sessions(rows):
    """Rows split into serve sessions: devlab_perf.sh writes {"session_end": V} after each session's runs. Rows without any
    marker (older runs, a single local log): every row is its own session (a round)."""
    out, cur, marked = [], [], False
    for r in rows:
        if "session_end" in r:
            marked = True
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(r)
    if cur:
        out.append(cur)
    return out if marked else [[r] for s_ in out for r in s_]


def read_rows(path):
    """The JSON rows of a perf / c4 log (lines that are not JSON objects are skipped; a bare number is a c4 agg_tps)."""
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
        elif line:
            try:
                rows.append({"agg_tps": float(line)})
            except ValueError:
                pass
    return rows


def session_values(rows, fn=None):
    """({metric: [per-session value]}, {metric: [per-session sample counts]}) of rows split into sessions: a session's
    value is the MEDIAN of its samples (the one rule of every card value, the Strix A/B and the llama.cpp rows)."""
    fn = fn or metric_samples
    out, counts = {}, {}
    for sess in split_sessions(rows):
        for k, vals in fn(sess).items():
            if vals:
                out.setdefault(k, []).append(statistics.median(vals))
                counts.setdefault(k, []).append(len(vals))
    return out, counts


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
    """Adds each metric's session acceptance (median over the sessions of each session's median) to metrics, in place."""
    for k, vals in acc.items():
        if k in metrics and vals:
            metrics[k]["acceptance"] = round(statistics.median(vals), 3)
    return metrics


def loom_sessions(rows, c4_path=None):
    """The Loom (or one llama.cpp engine's) measurement as per-session values: ({metric: [per-session median]},
    {metric: [per-session sample counts]}, {metric: [per-session median acceptance]}); streams4 from the c4_steady.py log
    (a session = one serve session of its runs; the value of a session is the median of its runs). One function for
    the cards (cmd_perf) and the Strix A/B (strix_ab_compare.py)."""
    vals, counts = session_values(rows)
    acc, _ = session_values(rows, acceptance_samples)
    if c4_path:
        sess = c4_sessions(c4_path)
        if sess:
            vals["streams4"] = [statistics.median(v) for v, _ in sess]
            counts["streams4"] = [len(v) for v, _ in sess]
        a4 = [statistics.median(a) for _, a in sess if a]
        if a4:
            acc["streams4"] = a4
    return vals, counts, acc


def c4_sessions(path):
    """The aggregate rates of c4_steady.py's runs (its JSON lines, or one number per line), per serve session
    ({"session_end": V} markers; a file without any is one session). A run that measured nothing (a rate <= 0, or a
    window <= 0 s: every stream failed) is skipped, with a note: a crash, not data."""
    rows = read_rows(path)
    out = []
    for rs in (split_sessions(rows) if any("session_end" in r for r in rows) else [rows]):
        vals = []
        for r in rs:
            if r.get("agg_tps") is None:
                continue
            if r["agg_tps"] <= 0 or r.get("window_s", 1) <= 0:
                print(f"{path}: skipped an invalid c4_steady run: {json.dumps(r)[:120]}", file=sys.stderr)
                continue
            vals.append(r["agg_tps"])
        if vals:
            out.append((vals, [r["acceptance"] for r in rs if r.get("acceptance") is not None]))
    return out


def c4_samples(path):
    """All the runs of c4_sessions, flat."""
    return [v for vals, _ in c4_sessions(path) for v in vals]


def summary(key, vals):
    """A card value: the median of the per-session values (a session recorded before this rule has no "aggregate" key:
    it is the mean of N rounds and is never recomputed)."""
    if not vals:
        return None
    return round(statistics.median(vals), 1)


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


def runs_of(key, sessions, c4_path):
    """The per-run values a card metric records: the per-session medians; streams4: its runs (c4_steady.py), whose median
    is also the median of the session medians when the job had one session."""
    return c4_samples(c4_path) if key == "streams4" and c4_path else sessions[key]


def cmd_perf(a):
    if a.source_kind not in PERF_SOURCE_KINDS:
        sys.exit(f"perf --source-kind {a.source_kind}: card performance comes only from AMD DevLab Dispatch "
                 "(a devlab-perf job: loom-fleet reference/publish.md step 3); measure locally only to develop")
    if a.perf_log:
        loom_rows = perf_rows(a.perf_log, "loom", a.loom_label)
        llama_rows = perf_rows(a.perf_log, "llama", a.llama_label)
    else:
        loom_rows = perf_rows(a.loom_jsonl) if a.loom_jsonl else []
        llama_rows = perf_rows(a.llama_jsonl) if a.llama_jsonl else []
    ls, lcount, lacc = loom_sessions(loom_rows, a.c4_loom)
    bs, bcount, _ = loom_sessions(llama_rows, a.c4_llama)
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
            m["loom_runs"] = runs_of(k, ls, a.c4_loom)
        if bs.get(k):
            m["baseline_runs"] = runs_of(k, bs, a.c4_llama)
        metrics[k] = m
    with_acceptance(metrics, lacc)
    if extra:
        ls, lcount = bs, bcount   # the runs counted are the baseline engine's
    dp = [len(ls[k]) for k in keep if k in ls and k != "streams4"]
    runs = {"decode_prefill": max(dp) if dp else None,
            "streams4": len(runs_of("streams4", ls, a.c4_llama if extra else a.c4_loom)) if "streams4" in keep and ls.get("streams4") else None}
    rep = {k: max(lcount[k]) for k in keep if k in lcount and k != "streams4" and max(lcount[k]) > 1}
    if rep:
        runs["samples"] = rep   # rows measured several times per session (perf.py --repeat-rows): samples per session
    date = a.date or max([r["time"][:10] for r in loom_rows + llama_rows if r.get("time")] or [None], key=lambda x: x or "")
    commands = list(a.command or [])
    if a.commands_file:
        commands += [line.rstrip("\n") for line in open(a.commands_file, encoding="utf-8") if line.strip()]
    session = {"date": date, "source": source_arg(a), "runs": runs, "aggregate": "median", "metrics": metrics,
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
            src_gates.append((n, "pass" if x.get("pass") else "fail", x.get("detail")))   # logits_V.json / conformance_V.json carry a detail
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
        failed = re.findall(r"^FAIL (\S+)", text, re.M)
        return [("kernel_tests", "pass" if ok else "fail",
                 f"{m.group(1)}/{m.group(2)}" + (f"; failed: {', '.join(failed)}" if failed else ""))], time, level
    sys.exit(f"{path}: no gates.json, GATES line or 'N/M passed' line found")


def acceptance(paths):
    """The spec gate's acceptance fraction from the first file that reports one, or None."""
    for p in paths:
        if p and os.path.isfile(p):
            found = ACCEPT.findall(open(p, encoding="utf-8", errors="replace").read())
            if found:
                return round(float(found[-1][0]) / 100, 3)
    return None


def cmd_devices(a):
    """Validate the per-device results files and list, per device, the models it serves (a data file's author runs this before a PR)."""
    docs = load_device_results(a.dir)   # exits on an invalid file
    for gpu, doc in docs.items():
        fits = [s for s, m in doc["models"].items() if m["fits"]]
        measured = [s for s in fits if doc["models"][s].get("evidence")]
        print(f"{gpu}: {len(doc['models'])} models, {len(fits)} fit, {len(measured)} with evidence"
              + (f", shares the {doc['share_section_with']} section" if doc.get("share_section_with") else ""))
    if not docs:
        print("no device results files")


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


# ---------------------------------------------------------------- same-runner re-measure (A/B) -> evidence

def cmd_remeasure(a):
    """A strix_ab_compare.py result (the package re-measured against the build whose sessions the card shows, one
    runner, one job): every card metric within the band, else refused (the stop rule, not a card fact)."""
    x = load_json(a.cmp)
    a.gpu = a.gpu or DEFAULT_GPU
    if not a.package:
        sys.exit("remeasure: --package (the package re-measured) is required")
    improved = {}   # out of margin only by rows FASTER than the card in both runs (the ratchet): their new values go on the card
    if x["verdict"] == "out_of_margin" and not x.get("gate_failures"):
        if not a.first_cmp:
            sys.exit(f"{a.cmp}: verdict out_of_margin: --first-cmp (the first run's comparison) is needed to confirm the rows are faster in both runs")
        f = load_json(a.first_cmp)
        if (x.get("first_run") or {}).get("job") != f.get("job"):
            sys.exit(f"{a.first_cmp}: is not the first run {a.cmp} was compared with ({(x.get('first_run') or {}).get('job')})")
        slower = []
        for k in x["out_of_margin"]:
            if k.startswith("acceptance:"):   # informational: not a card row
                continue
            r1, r2 = f["metrics"][k], x["metrics"][k]
            if not (r1["ratio"] and r2["ratio"] and r1["ratio"] > 1 and r2["ratio"] > 1):
                slower.append(k)
            else:
                improved[k] = {"published": r2.get("card"), "loom": round((r1["new"] + r2["new"]) / 2, 1), "runs": [r1["new"], r2["new"]]}
        if slower:
            sys.exit(f"{a.cmp}: {', '.join(slower)} outside the band in both runs and not faster: refused (not within the band)")
        # out of margin only by acceptance rows (informational, not card rows): a plain within-band re-measure
    elif (x["verdict"] == "gate_fail" and x.get("gate_failures") and set(x["gate_failures"]) <= set(a.accept_gate_failure or [])
          and all(k.startswith("acceptance:") for k in x.get("out_of_band") or [])):
        pass   # every perf row within its band; the failed gates are named by the person who accepted them and stay in the card's gate table
    elif x["verdict"] != "pass" or x.get("gate_failures"):
        sys.exit(f"{a.cmp}: verdict {x['verdict']} (out of band {x.get('out_of_band')}, gate failures {x.get('gate_failures')}): not within the band")
    # a confirming re-run (strix_ab_compare.py --prev) may pass with a metric outside the band in the opposite direction to the
    # first run: the card names it
    exc = {k: m["ratio"] for k, m in x["metrics"].items() if m["status"] != "PASS" and k not in improved
           and not (x["verdict"] == "out_of_margin" and k.startswith("acceptance:"))}
    here = programs_sha256(a.package)
    if a.fetched_programs and a.fetched_programs != here:
        sys.exit(f"the run fetched programs {a.fetched_programs[:12]}, the package has {here[:12]}: re-run")
    old = a.compared_programs
    if not old:   # the compared build's programs at the revision the job fetched
        import tempfile
        from huggingface_hub import hf_hub_download, list_repo_files
        with tempfile.TemporaryDirectory() as t:
            for f in list_repo_files(a.compared_repo, revision=a.compared_revision):
                if f.startswith("program/") and f.endswith(".loom"):
                    hf_hub_download(a.compared_repo, f, revision=a.compared_revision, local_dir=t)
            old = programs_sha256(t)
    src = source_arg(a)
    date = a.date or x.get("date")
    if not date:
        sys.exit("--date is required (the job's day)")
    first_src = None
    if improved:
        if not (a.first_run and a.first_url):
            sys.exit("remeasure: --first-run and --first-url (the first run's links) are required with improved rows")
        first_src = dict(src, run=str(a.first_run), url=a.first_url)
    ev = {"evidence": EVIDENCE_ID, "gpu": a.gpu, "remeasure": {
        "date": date, "source": src, "band": x["band"], "programs_sha256": here, "compared_programs": old,
        "result": "improved" if improved else "within", **({"bands": x["bands"]} if x.get("bands") else {}),
        **({"improved": improved, "first_source": first_src} if improved else {}),
        "worst_metric": x.get("worst_metric"), "worst_ratio": x.get("worst_ratio"),
        "ratios": {k: m["ratio"] for k, m in x["metrics"].items()}, **({"exceptions": exc} if exc else {})}}
    write_json(a.o, ev)
    print(f"wrote {a.o}: {a.gpu} within +-{x['band']:.0%}" + (f" except {', '.join(improved)} (faster in both runs)" if improved else "") + f" (worst {x.get('worst_metric')} {x.get('worst_ratio')}); programs {here[:12]} ~ {old[:12]}", file=sys.stderr)


def cmd_equivalent(a):
    """Evidence that this package's default-path stages are identical (after canonicalization: fleet/stage_equiv.py) to the build
    whose performance sessions the card shows: those sessions stand for this package (the remeasure block with result
    "identical", band 0, and the proof). Refused unless the proof says equivalent."""
    a.gpu = a.gpu or DEFAULT_GPU
    if not a.package:
        sys.exit("equivalent: --package is required")
    x = load_json(a.proof)
    if x.get("equivalent") is not True:
        sys.exit(f"{a.proof}: not equivalent: no carry-over")
    here = programs_sha256(a.package)
    old = a.compared_programs or x.get("compared_programs")
    if not old:
        sys.exit("equivalent: --compared-programs (programs_sha256 of the build the card's table measured) is required")
    if old == here:
        sys.exit("equivalent: the compared build has the same programs: nothing to carry over")
    src = source_arg(a)
    date = a.date or (a.time or "")[:10] or datetime.date.today().isoformat()
    ev = {"evidence": EVIDENCE_ID, "gpu": a.gpu, "remeasure": {
        "date": date, "source": src, "band": 0, "programs_sha256": here, "compared_programs": old, "result": "identical",
        "proof": {"tool": x.get("tool") or "fleet/stage_equiv.py", "summary": x.get("summary") or "default-path stages identical after canonicalization",
                  "url": a.proof_url or x.get("url")}}}
    write_json(a.o, ev)
    print(f"wrote {a.o}: {a.gpu} programs {here[:12]} carry the sessions measured on {old[:12]} (default path identical)", file=sys.stderr)


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
    p.add_argument("--source-kind", default=kind, choices=["devlab", "factory", "local", "machine"])
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
    b.add_argument("--fit", action="append", metavar="VERDICT.json", help="a fit table (repeat; GPU=PATH names the device): the device's memory_bytes, served context or "
                   "unsupported state. A path takes the device whose device.json fit_table has that file name, else radeon-r9700; 'auto' = the table of every "
                   "registry device that has one in this checkout")
    b.add_argument("--generator-commit", help="loom-skills commit of the card scripts, or 'unknown' (default: this "
                   "checkout's HEAD)")
    b.add_argument("--evidence", nargs="*")
    b.add_argument("--device-results", default=None, help="the directory of per-device results files (default: $LOOM_DEVICE_RESULTS, else this checkout's "
                   "fleet/devices; none: no device files): a GPU with no DevLab runner is on the card where its file says this model fits")
    b.add_argument("--memory-bytes", type=int)
    b.add_argument("--output")
    b.add_argument("--license")
    b.add_argument("--report", choices=["generated", "attached"])
    b.add_argument("--from-info", help="a published info.json to start from (regenerating a card with new evidence)")
    b.add_argument("--reset-performance", action="store_true", help="drop --from-info's performance sessions (re-measured from scratch)")
    b.add_argument("--perf-from", action="append", metavar="GPU=INFO.json", help="take that GPU's performance (sessions, baselines, their sources) "
                   "from another card's info.json (the staging flow: the loom-fun card's Strix table, with a remeasure evidence)")
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
    m = sub.add_parser("remeasure", help="a same-runner A/B of this package against the build the card's table shows (strix_ab_compare.py --json)")
    m.add_argument("--cmp", required=True, help="strix_ab_compare.py --json output (verdict pass)")
    m.add_argument("--compared-programs", help="programs_sha256 of the build compared with (default: computed from --compared-repo at --compared-revision)")
    m.add_argument("--compared-repo", help="ORG/NAME the old build was fetched from (e.g. loom-fun/NAME)")
    m.add_argument("--compared-revision", help="its commit the job fetched (model_fun.json's document URL)")
    m.add_argument("--fetched-programs", help="programs_sha256 the job fetched for the package (programs_sha256.txt)")
    m.add_argument("--date", help="YYYY-MM-DD of the job")
    m.add_argument("--first-cmp", help="the first run's strix_ab_compare.py --json, when --cmp is the confirming re-run with verdict out_of_margin: "
                   "rows faster than the card in both runs get the mean of the two runs' medians as the card value")
    m.add_argument("--first-run", help="the first run's Actions run id (with --first-cmp)")
    m.add_argument("--accept-gate-failure", action="append", help="a failed gate of the cmp (e.g. ktest:staging) a person accepted: the re-measure "
                   "goes on the card (rows within the band) and the gate stays failed in the gate table")
    m.add_argument("--first-url", help="the first run's job URL (with --first-cmp)")
    add_source_args(m, "devlab")
    m.add_argument("-o", required=True)
    e = sub.add_parser("equivalent", help="the package's default-path stages are identical to the build the card's table measured (fleet/stage_equiv.py): no re-measure")
    e.add_argument("--proof", required=True, help="stage_equiv.py's JSON: {equivalent: true, tool, summary, url, compared_programs}")
    e.add_argument("--compared-programs", help="programs_sha256 of the build the card's table measured")
    e.add_argument("--proof-url", help="link to the proof's run")
    e.add_argument("--date", help="YYYY-MM-DD of the proof")
    e.add_argument("--time")
    add_source_args(e, "local")
    e.add_argument("-o", required=True)
    dv = sub.add_parser("devices", help="validate fleet/devices/*.json (loom-device-results/1) and summarize them")
    dv.add_argument("dir", nargs="?", default=None, help="the directory (default: $LOOM_DEVICE_RESULTS, else fleet/devices)")
    a = ap.parse_args(argv)
    {"devices": cmd_devices, "build": cmd_build, "equivalent": cmd_equivalent, "perf": cmd_perf, "gates": cmd_gates, "roofline": cmd_roofline, "remeasure": cmd_remeasure}[a.cmd](a)


if __name__ == "__main__":
    main()
