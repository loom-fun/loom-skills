#!/usr/bin/env python3
"""Write a card's info.json (schema loom-card/1) from the package and measurement evidence.

The card is data, not prose: this script collects it, render_card.py turns it into README.md. Both are pure
functions of their inputs (no clock, no randomness, sorted keys), so the same inputs give byte-identical files.

  make_info.py build PKG --repo loom-fun/NAME [--evidence EV.json ...] [--memory-bytes N] [--output DIR]
                         [--license ID] [-o info.json]
      PKG holds model.json, model.json.lock and llm_config.json (a built package, or the files downloaded from
      the card's repo). Checkpoint, drafter, projector, service, context and features come from them;
      performance, gates and anything the package cannot say come from the evidence files, applied in order.

  make_info.py perf --loom-jsonl LOOM.jsonl --llama-jsonl LLAMA.jsonl [--c4-loom F] [--c4-llama F]
                    --llamacpp-build bNNNN [--llamacpp-drafting "draft-dflash, n-max 3"] [--date YYYY-MM-DD]
                    [--source-kind local|devlab|factory --run ID --url URL] -o perf_evidence.json
      perf.py logs (one engine per file; or --perf-log perf_log.jsonl with finalize.sh's labels final /
      llama-final) -> per-metric means; c4_steady.py results (a number per line, or JSON lines with agg_tps)
      -> median.

  make_info.py gates FILE [FILE ...] [--time YYYY-MM-DDTHH:MM:SS] [--level L] [--source-kind ... --run --url]
                     -o gates_evidence.json
      gates.sh's gates.json (verdicts, level, ts), a gates.out with its "GATES ..." line, or a kernel test
      log ending in "N/M passed" (make_kernel_tests.py) -> gate evidence.

Evidence file (loom-card-evidence/1), every block optional:
  {"evidence": "loom-card-evidence/1",
   "card": {"license", "tags", "output", "memory_bytes"},
   "performance": {same shape as info.json's performance; "metrics" may omit rows},
   "performance_overrides": [{"metric", "value", "date", "source"}],     # a Loom value from another run
   "gates": [{"name", "status", "level", "time", "source", "detail"}],
   "features": {"<feature>": {"status", "value"}}}
Merge rules: card keys and features: the later file wins. performance: the later block replaces the earlier.
gates: per gate the evidence with the latest time wins (equal times: the later file).
"""
import argparse
import ast
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_lib import (EVIDENCE_ID, GATE_NAMES, METHOD, METRIC_KEYS, SCHEMA_ID, build_tag, dumps, load_json,  # noqa: E402
                      quant_of, require_valid, write_json)

BASE_TAGS = ["loom", "loom-serve", "gguf", "strix-halo"]


def source(kind, run=None, url=None):
    return {"kind": kind, "run": str(run) if run else None, "url": url}


# ---------------------------------------------------------------- package -> facts

def package_sources(model, lock):
    """{parameter name: source block} for every file model.json loads, resolved through its origins and lock."""
    origins = model.get("origins", {})
    locked = {f["origin"]: f for f in (lock or {}).get("files", [])}
    resolved = {k: (v.get("resolved") or {}).get("revision") for k, v in (lock or {}).get("origins", {}).items()}
    out = {}
    for variant in model.get("variants", {}).values():
        for service in variant.get("services", {}).values():
            for pname, param in service.get("parameters", {}).items():
                for f in param.get("files", []):
                    origin = f.get("origin") or ""
                    m = re.match(r"@([^/]+)/(.+)$", origin)
                    if not m or m.group(1) not in origins or pname in out:
                        continue
                    o = origins[m.group(1)]
                    lk = locked.get(origin, {})
                    out[pname] = {"repo": o["repository"], "revision": resolved.get(m.group(1)) or o.get("revision"),
                                  "file": m.group(2), "quant": quant_of(m.group(2)),
                                  "sha256": lk.get("sha256"), "size": lk.get("size")}
    return out


def drafter_setting(cfg, execution):
    """(method, depth) of the package's default speculative decoding, or None."""
    draft = cfg.get("draft") or {}
    depth = execution.get("draft_depth", draft.get("depth"))
    if not draft or not depth:
        return None
    return draft.get("kind") or "draft", int(depth)


def derive_features(model, cfg, kind, execution, srcs, spec):
    chat = cfg.get("chat") or {}
    na = {"status": "n/a", "value": None}
    f = {}
    if kind == "decision":
        f["api"] = {"status": "yes", "value": {"endpoint": "/v1/systemone", "style": "systemone"}}
        for k in ("tool_calling", "thinking", "reasoning_effort", "speculative_decoding", "image_input", "sampling",
                  "max_output_tokens"):
            f[k] = dict(na)
    else:
        f["api"] = {"status": "yes", "value": {"endpoint": "/v1/chat/completions", "style": "openai"}}
        # the starter's tool-call glue: a call format (parser) and the tool declaration block (renderer)
        tools = "call_format" in chat or bool(chat.get("tools_prefix"))
        f["tool_calling"] = {"status": "yes" if tools else "no", "value": None}
        mode = chat.get("thinking")
        if mode not in ("always", "toggle"):
            mode = "always" if chat.get("starts_in_reasoning") else None
        if mode:
            default = chat.get("thinking_default", True) if mode == "toggle" else True
            f["thinking"] = {"status": "yes", "value": {"mode": mode, "default": bool(default)}}
        else:
            f["thinking"] = {"status": "no", "value": None}
        levels = list((chat.get("reasoning_instructions") or {}).keys())
        f["reasoning_effort"] = ({"status": "yes", "value": {"levels": levels,
                                                             "default": chat.get("reasoning_effort_default")}}
                                 if levels else {"status": "no", "value": None})
        if spec:
            d = srcs.get("draft")
            f["speculative_decoding"] = {"status": "yes", "value": {"method": spec[0], "depth": spec[1],
                                                                    "drafter": d["repo"] if d else None}}
        else:
            f["speculative_decoding"] = {"status": "no", "value": None}
        v = srcs.get("vision")
        f["image_input"] = ({"status": "yes", "value": {"projector": v["file"]}} if v and cfg.get("vision")
                            else {"status": "no", "value": None})
        f["sampling"] = {"status": "unknown", "value": None}
        mt = execution.get("max_tokens")
        f["max_output_tokens"] = {"status": "yes", "value": {"max_tokens": mt}} if mt else {"status": "unknown", "value": None}
    audio = any(s.get("kind") in ("asr", "tts", "audio", "speech") for s in model.get("services", {}).values())
    f["audio"] = {"status": "yes" if audio else "no", "value": None}
    ctx = execution.get("context_capacity")
    f["context"] = {"status": "yes", "value": {"tokens": ctx}} if ctx else {"status": "unknown", "value": None}
    rows, pending = execution.get("rows"), execution.get("pending_requests")
    f["concurrency"] = ({"status": "yes", "value": {"rows": rows, "pending_requests": pending}} if rows or pending
                        else {"status": "unknown", "value": None})
    return f


def gate_applies(name, kind, feats):
    if name == "kernel_tests" or name == "consistency":
        return True
    if kind == "decision":
        return name == "decision"
    if name == "decision":
        return False
    if name == "vision":
        return feats["image_input"]["status"] == "yes"
    if name == "spec":
        return feats["speculative_decoding"]["status"] == "yes"
    if name == "agent":
        return feats["tool_calling"]["status"] == "yes"
    if name == "agent_think_off":
        th = feats["thinking"]
        return feats["tool_calling"]["status"] == "yes" and th["status"] == "yes" and th["value"]["mode"] == "toggle"
    return True


def empty_performance():
    return {"llamacpp_build": None, "drafting": {"loom": None, "llamacpp": None}, "date": None, "source": None,
            "runs": {"decode_prefill": None, "streams4": None},
            "metrics": {k: {"loom": None, "llamacpp": None, "loom_source": None} for k in METRIC_KEYS},
            "not_applicable": []}


def cmd_build(a):
    pkg = a.pkg
    model = load_json(os.path.join(pkg, "model.json"))
    lock = load_json(os.path.join(pkg, "model.json.lock")) if os.path.exists(os.path.join(pkg, "model.json.lock")) else {}
    cfgp = os.path.join(pkg, "llm_config.json")
    cfg = load_json(cfgp) if os.path.exists(cfgp) else {}
    presets = model.get("presets", {})
    preset_name = next(iter(presets), None)   # loom-fetch's default: the first preset
    preset = presets.get(preset_name, {}) if preset_name else {}
    execution = preset.get("execution", {})
    svc_kind = model.get("services", {}).get(preset.get("service"), {}).get("kind")
    kind = "decision" if svc_kind == "decision" else "chat"
    srcs = package_sources(model, lock)
    if "model" not in srcs:
        sys.exit("model.json has no checkpoint file with an @origin (parameters.model)")
    spec = drafter_setting(cfg, execution) if kind == "chat" else None
    feats = derive_features(model, cfg, kind, execution, srcs, spec)

    card, perf, overrides, gates, feat_over = {}, None, [], {}, {}
    for i, path in enumerate(a.evidence or []):
        ev = load_json(path)
        if ev.get("evidence") != EVIDENCE_ID:
            sys.exit(f"{path}: not a {EVIDENCE_ID} file")
        card.update({k: v for k, v in (ev.get("card") or {}).items() if v is not None})
        if ev.get("performance"):
            perf = ev["performance"]
        overrides += ev.get("performance_overrides") or []
        for g in ev.get("gates") or []:
            if g["name"] not in GATE_NAMES:
                sys.exit(f"{path}: unknown gate {g['name']} (known: {GATE_NAMES})")
            key = (g.get("time") or "", i)
            if g["name"] not in gates or key >= gates[g["name"]][0]:
                gates[g["name"]] = (key, g)
        feat_over.update(ev.get("features") or {})
    for k, v in (("license", a.license), ("output", a.output), ("memory_bytes", a.memory_bytes)):
        if v is not None:
            card[k] = v

    # Sampling is a feature the gate proves; a legacy card may state it is greedy-only.
    if kind == "chat":
        sg = gates.get("sampling", (None, {}))[1]
        if sg.get("status") == "pass":
            feats["sampling"] = {"status": "yes", "value": None}
        elif sg.get("status") == "fail":
            feats["sampling"] = {"status": "no", "value": None}
    for k, v in feat_over.items():
        if k not in feats:
            sys.exit(f"unknown feature {k}")
        feats[k] = {"status": v["status"], "value": v.get("value")}

    p = empty_performance()
    if perf:
        for k in ("llamacpp_build", "date", "source"):
            p[k] = perf.get(k)
        p["drafting"].update(perf.get("drafting") or {})
        p["runs"].update(perf.get("runs") or {})
        for k, m in (perf.get("metrics") or {}).items():
            if k not in METRIC_KEYS:
                sys.exit(f"unknown metric {k} (known: {METRIC_KEYS})")
            p["metrics"][k].update({x: m.get(x) for x in ("loom", "llamacpp", "loom_source") if x in m})
        p["not_applicable"] = list(perf.get("not_applicable") or [])
    if "drafting" not in (perf or {}) or "loom" not in (perf or {}).get("drafting", {}):
        p["drafting"]["loom"] = f"{METHOD.get(spec[0], spec[0])}, depth {spec[1]}" if spec else None
    for o in overrides:
        p["metrics"][o["metric"]]["loom"] = o["value"]
        p["metrics"][o["metric"]]["loom_source"] = {"date": o["date"], "source": o["source"]}
    if kind == "decision":
        p["not_applicable"] = sorted(set(p["not_applicable"]) | {"decode", "decode_2k", "decode_32k", "streams4"},
                                     key=METRIC_KEYS.index)

    gate_rows = []
    for name in GATE_NAMES:
        if name in gates:
            g = gates[name][1]
            t = g.get("time")
            gate_rows.append({"name": name, "status": g["status"], "level": g.get("level"),
                              "date": t[:10] if t else None, "source": g.get("source"), "detail": g.get("detail")})
        else:
            gate_rows.append({"name": name, "status": "unknown" if gate_applies(name, kind, feats) else "n/a",
                              "level": None, "date": None, "source": None, "detail": None})

    tags = list(BASE_TAGS)
    if feats["speculative_decoding"]["status"] == "yes":
        tags += ["speculative-decoding", feats["speculative_decoding"]["value"]["method"]]
    if feats["image_input"]["status"] == "yes":
        tags += ["vision", "image-text-to-text"]
    for t in card.get("tags") or []:
        if t not in tags:
            tags.append(t)

    ckpt = srcs["model"]
    name = re.sub(r"[-_]gguf$", "", ckpt["repo"].split("/")[-1], flags=re.I)
    install = card.get("output") or model.get("model", {}).get("id", name).split("/")[-1]
    info = {
        "schema": SCHEMA_ID, "repo": a.repo, "name": name, "license": card.get("license"), "tags": tags,
        "checkpoint": ckpt, "drafter": srcs.get("draft"), "projector": srcs.get("vision"),
        "service": {"kind": kind, "endpoint": feats["api"]["value"]["endpoint"]},
        "serve": {"output": install, "memory_bytes": card.get("memory_bytes"), "device": "amdgpu",
                  "preset": preset_name if len(presets) > 1 else None},
        "context": {"tokens": execution.get("context_capacity"), "rows": execution.get("rows"),
                    "pool": execution.get("pool_capacity"), "max_tokens": execution.get("max_tokens"),
                    "pending_requests": execution.get("pending_requests")},
        "performance": p, "features": feats, "gates": gate_rows,
    }
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


def summarize(samples, c4):
    vals = {k: round(statistics.mean(v), 1) for k, v in samples.items() if k != "streams4"}
    s4 = c4 if c4 else samples.get("streams4", [])
    if s4:
        vals["streams4"] = round(statistics.median(s4), 1)
    return vals


def cmd_perf(a):
    if a.perf_log:
        loom_rows = perf_rows(a.perf_log, "loom", a.loom_label)
        llama_rows = perf_rows(a.perf_log, "llama", a.llama_label)
    else:
        loom_rows = perf_rows(a.loom_jsonl) if a.loom_jsonl else []
        llama_rows = perf_rows(a.llama_jsonl) if a.llama_jsonl else []
    ls, lls = metric_samples(loom_rows), metric_samples(llama_rows)
    c4l = c4_samples(a.c4_loom) if a.c4_loom else None
    c4r = c4_samples(a.c4_llama) if a.c4_llama else None
    loom, llama = summarize(ls, c4l), summarize(lls, c4r)
    runs = max([len(v) for k, v in ls.items() if k.startswith("decode")] or [0]) or None
    s4runs = len(c4l) if c4l else (len(ls.get("streams4", [])) or None)
    date = a.date or max([r["time"][:10] for r in loom_rows if r.get("time")] or [None], key=lambda x: x or "")
    perf = {"llamacpp_build": a.llamacpp_build, "date": date,
            "drafting": {"llamacpp": a.llamacpp_drafting},
            "source": source(a.source_kind, a.run, a.url),
            "runs": {"decode_prefill": runs, "streams4": s4runs},
            "metrics": {k: {"loom": loom.get(k), "llamacpp": llama.get(k)} for k in METRIC_KEYS}}
    if a.loom_drafting is not None:
        perf["drafting"]["loom"] = a.loom_drafting or None
    write_json(a.o, {"evidence": EVIDENCE_ID, "performance": perf})
    print(f"wrote {a.o}: " + ", ".join(f"{k} {loom.get(k)}/{llama.get(k)}" for k in METRIC_KEYS), file=sys.stderr)


# ---------------------------------------------------------------- gate results -> evidence

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


def cmd_gates(a):
    out = []
    for path in a.files:
        gs, time, level = gates_from_file(path, a.time, a.level)
        if not time:
            sys.exit(f"{path}: no timestamp in the file; pass --time")
        for n, st, detail in gs:
            if n not in GATE_NAMES:
                sys.exit(f"{path}: unknown gate {n}")
            out.append({"name": n, "status": st, "level": None if n == "kernel_tests" else level, "time": time,
                        "source": source(a.source_kind, a.run, a.url), "detail": detail})
    write_json(a.o, {"evidence": EVIDENCE_ID, "gates": out})
    print(f"wrote {a.o}: " + ", ".join(f"{g['name']}={g['status']}" for g in out), file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("pkg")
    b.add_argument("--repo", required=True)
    b.add_argument("--evidence", nargs="*")
    b.add_argument("--memory-bytes", type=int)
    b.add_argument("--output")
    b.add_argument("--license")
    b.add_argument("-o", default="info.json")
    p = sub.add_parser("perf")
    p.add_argument("--perf-log")
    p.add_argument("--loom-label", default="final")
    p.add_argument("--llama-label", default="llama-final")
    p.add_argument("--loom-jsonl")
    p.add_argument("--llama-jsonl")
    p.add_argument("--c4-loom")
    p.add_argument("--c4-llama")
    p.add_argument("--llamacpp-build", required=True, type=lambda s: build_tag(s) or s)
    p.add_argument("--llamacpp-drafting", help="llama.cpp's speculative setting, e.g. 'draft-dflash, n-max 3' (omit: off)")
    p.add_argument("--loom-drafting", help="Loom's setting while measured if not the package's ('' = off)")
    p.add_argument("--date", help="YYYY-MM-DD (default: the latest Loom row's time)")
    p.add_argument("--source-kind", default="local", choices=["local", "devlab", "factory"])
    p.add_argument("--run")
    p.add_argument("--url")
    p.add_argument("-o", required=True)
    g = sub.add_parser("gates")
    g.add_argument("files", nargs="+")
    g.add_argument("--time", help="ISO time of the run when the file has none (gates.out, kernel test log)")
    g.add_argument("--level")
    g.add_argument("--source-kind", default="local", choices=["local", "devlab", "factory"])
    g.add_argument("--run")
    g.add_argument("--url")
    g.add_argument("-o", required=True)
    a = ap.parse_args(argv)
    {"build": cmd_build, "perf": cmd_perf, "gates": cmd_gates}[a.cmd](a)


if __name__ == "__main__":
    main()
