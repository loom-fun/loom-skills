"""Shared definitions for the loom-fun card generator (schema loom-card/2).

A card is rendered from the files of the package's repository:
  model.json.lock   sources, revisions, file names, sha256 and sizes
  model.json        services, presets (context, rows, max_tokens, drafting depth), origins
  llm_config.json   chat features, the drafter, and "generator" (the loom-skills commit and the Loom SDK that
                    generated the programs, stamped by loom-port-llm's build.py)
  info.json         only what has no other home: measurements, gate evidence, the device and SDK that measured,
                    the baseline, dates and versions (schema info_schema.json)
make_info.py writes info.json, render_card.py renders README.md (and a generated REPORT.md). Both are pure
functions of their inputs: no clock, no environment, sorted JSON keys. The fixed row sets below are the card
template: every card shows the same performance, feature and gate rows, in this order. info.json may hold
other metric and gate names; readers ignore names they don't know.
"""
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_ID = "loom-card/2"
EVIDENCE_ID = "loom-card-evidence/2"
SCHEMA_PATH = os.path.join(HERE, "info_schema.json")

# (key, label): the performance table's rows. spec_acceptance is a fraction (0-1), the others tokens per second.
METRICS = [
    ("decode", "decode tok/s, short prompt"),
    ("decode_2k", "decode tok/s, 2k context"),
    ("decode_32k", "decode tok/s, 32k context"),
    ("prefill_1k", "prefill tok/s, 1k document"),
    ("prefill_2k", "prefill tok/s, 2k document"),
    ("prefill_8k", "prefill tok/s, 8k document"),
    ("prefill_16k", "prefill tok/s, 16k document"),
    ("prefill_32k", "prefill tok/s, 32k document"),
    ("streams4", "4 streams, aggregate decode tok/s"),
    ("spec_acceptance", "speculative decoding, drafted tokens accepted"),
]
METRIC_KEYS = [k for k, _ in METRICS]
# how a metric's per-run samples become the card value
MEDIAN_METRICS = {"streams4"}
# card metric -> loom-roofline/roofline.py metric (the hardware ceiling). spec_acceptance has none.
ROOFLINE_METRIC = {"decode": "decode_tps", "decode_2k": "decode_tps_2k", "decode_32k": "decode_tps_32k",
                   "prefill_1k": "prefill_1000", "prefill_2k": "prefill_2000", "prefill_8k": "prefill_8000",
                   "prefill_16k": "prefill_16000", "prefill_32k": "prefill_32000", "streams4": "c4_steady"}
# decode context (tokens of KV read per decoded token) and streams of the drafted rows: speculative decoding commits
# several tokens per verify pass, so their ceiling is the pass rate x the expected tokens per pass
DRAFTED_ROWS = {"decode": (0, 1), "decode_2k": (2000, 1), "decode_32k": (32000, 1), "streams4": (600, 4)}
# Performance guardrails (loom-org org_status.py). Far from roofline (calibrated in loom-org SKILL.md, "Far from
# roofline: how the threshold was set"): Loom below FAR_FRACTION x the efficiency tuned Loom ports have demonstrated
# for the metric kind
FAR_FRACTION = 0.5
# A card keeps both kinds of performance session, each labeled by its source: local sessions (Loom and llama.cpp on
# one quiet Strix Halo: the headline values, judged against the roofline) and DevLab sessions (source kind "devlab":
# a scripted DevLab port, or a devlab-perf job on the published programs), shown next to them with the local / DevLab
# ratio. The two may differ a lot (loom-org SKILL.md, "Local and DevLab performance"): the ratio is data, never
# judged. spec_acceptance (the spec gate's drafted-token acceptance) is a drafting statistic of model, drafter and
# text, not a timing: it is a headline value from any source, DevLab's nightly spec gate included (pull_devlab.py),
# and its session's programs_sha256 decides whether it is stale like any other card value. info.json "devlab"
# (Loom-only sweeps older cards recorded) is still read and validated, never rendered or judged.
TIMED = set(ROOFLINE_METRIC)

# (key, label): the features table's rows.
FEATURES = [
    ("api", "API"),
    ("tool_calling", "tool calling"),
    ("thinking", "reasoning / thinking"),
    ("reasoning_effort", "reasoning effort"),
    ("speculative_decoding", "speculative decoding"),
    ("image_input", "image input"),
    ("audio", "audio input / output"),
    ("sampling", "sampling"),
    ("context", "context size"),
    ("concurrency", "concurrency"),
    ("max_output_tokens", "max output tokens"),
]
FEATURE_KEYS = [k for k, _ in FEATURES]

# (name, description): the correctness table's rows (loom-benchmark gates.sh gate names).
GATES = [
    ("kernel_tests", "Unit tests of the generated kernels against reference implementations"),
    ("golden", "llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents"),
    ("compare", "Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie"),
    ("vision", "Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties"),
    ("consistency", "Concurrent requests give the same output as the same requests run alone"),
    ("spec", "Speculative decoding on gives exactly the output of speculative decoding off"),
    ("agent", "Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp"),
    ("agent_think_off", "The tool-calling loop with thinking off"),
    ("sampling", "temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp"),
    ("needle", "Needle-in-a-haystack prompt at up to half the context, compared with llama.cpp; over-capacity prompt refused"),
    ("needle_full", "Needle-in-a-haystack prompt at the full context is served"),
    ("decision", "Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts"),
]
GATE_NAMES = [n for n, _ in GATES]
# speculative decoding methods (llm_config.json draft.kind) as the card names them
METHOD = {"dflash": "DFlash", "mtp": "MTP", "ngram": "n-gram prompt lookup", "assist": "assistant drafter"}
BASE_TAGS = ["loom", "loom-serve", "gguf", "strix-halo"]

HF = "https://huggingface.co"
LOOM_URL = "https://github.com/ROCm/hrx-system"
SKILLS_URL = "https://github.com/loom-fun/loom-skills"


def dumps(obj):
    """The one JSON encoding of info.json and evidence files: sorted keys, 1-space indent, trailing newline."""
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(dumps(obj))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def quant_of(filename):
    """The quantization in a GGUF file name (Q4_K_M, UD-Q5_K_M, Q8_0, BF16, ...; any case, as gguf-my-repo writes q4_k_m), or None."""
    stem = os.path.basename(filename)
    stem = stem[:-5] if stem.lower().endswith(".gguf") else stem
    found = re.findall(r"(?<![A-Za-z0-9])((?:UD-)?I?Q[0-9](?:_[0-9A-Z]+)*|BF16|F16|F32)(?![a-z0-9])", stem.upper())
    return found[-1] if found else None


def build_tag(text):
    """A lemonade-sdk/llamacpp-rocm release tag (b1355) found in free text, or None."""
    m = re.search(r"\bb(1[0-9]{3,})\b", text or "")
    return f"b{m.group(1)}" if m else None


def programs_sha256(pkg):
    """sha256 over PKG/program/*.loom (name and content), each file without the generator's first-line
    '// Generated by ... from <cfg path>' comment, which names a local path. None without program files."""
    d = os.path.join(pkg, "program")
    if not os.path.isdir(d):
        return None
    h = hashlib.sha256()
    names = sorted(f for f in os.listdir(d) if f.endswith(".loom"))
    for n in names:
        with open(os.path.join(d, n), "rb") as f:
            data = f.read()
        if data.startswith(b"// Generated by"):
            data = data.split(b"\n", 1)[1] if b"\n" in data else b""
        h.update(n.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest() if names else None


# ---------------------------------------------------------------- the package's own files -> card facts

def load_package(pkg):
    """model.json, model.json.lock and llm_config.json of a package directory ({} for a missing optional file)."""
    def opt(name):
        p = os.path.join(pkg, name)
        return load_json(p) if os.path.exists(p) else {}
    model = load_json(os.path.join(pkg, "model.json"))
    return {"model": model, "lock": opt("model.json.lock"), "cfg": opt("llm_config.json")}


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


def package_facts(pkg_files):
    """Everything the card shows that the package's own files determine."""
    model, lock, cfg = pkg_files["model"], pkg_files["lock"], pkg_files["cfg"]
    presets = model.get("presets", {})
    preset_name = next(iter(presets), None)   # loom-fetch's default: the first preset
    preset = presets.get(preset_name, {}) if preset_name else {}
    execution = preset.get("execution", {})
    svc_kind = model.get("services", {}).get(preset.get("service"), {}).get("kind")
    kind = "decision" if svc_kind == "decision" else "chat"
    srcs = package_sources(model, lock)
    if "model" not in srcs:
        raise SystemExit("model.json has no checkpoint file with an @origin (parameters.model)")
    spec = drafter_setting(cfg, execution) if kind == "chat" else None
    ckpt = srcs["model"]
    return {"kind": kind, "execution": execution, "preset": preset_name if len(presets) > 1 else None,
            "checkpoint": ckpt, "drafter": srcs.get("draft"), "projector": srcs.get("vision"), "spec": spec,
            "name": re.sub(r"[-_]gguf$", "", ckpt["repo"].split("/")[-1], flags=re.I),
            "output": model.get("model", {}).get("id", "").split("/")[-1] or None,
            "memory_bytes": execution.get("memory_bytes"),
            "generator": cfg.get("generator"),
            "features": derive_features(model, cfg, kind, execution, srcs, spec)}


def tool_calling(chat):
    """Whether a package's chat block serves tool calls: the starter's tool-call glue, a call format (parser) and the
    tool declaration block (renderer). call_format "none" is a chat-only block (a template that ignores tools).
    gates.sh skips the agent gates on the same test, so the card shows them as not applicable."""
    return chat.get("call_format") not in (None, "none") or bool(chat.get("tools_prefix"))


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
        f["tool_calling"] = {"status": "yes" if tool_calling(chat) else "no", "value": None}
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
        f["sampling"] = {"status": "unknown", "value": None}   # proven by the sampling gate (evidence)
        mt = execution.get("max_tokens")
        f["max_output_tokens"] = {"status": "yes", "value": {"max_tokens": mt}} if mt else {"status": "unknown", "value": None}
    audio = any(s.get("kind") in ("asr", "tts", "audio", "speech") for s in model.get("services", {}).values())
    f["audio"] = {"status": "yes" if audio else "no", "value": None}
    ctx = execution.get("context_capacity")
    f["context"] = ({"status": "yes", "value": {"tokens": ctx, "native": cfg.get("context_length")}} if ctx
                    else {"status": "unknown", "value": None})
    rows, pending = execution.get("rows"), execution.get("pending_requests")
    f["concurrency"] = ({"status": "yes", "value": {"rows": rows, "pending_requests": pending}} if rows or pending
                        else {"status": "unknown", "value": None})
    return f


def derived_tags(features):
    tags = list(BASE_TAGS)
    if features["speculative_decoding"]["status"] == "yes":
        tags += ["speculative-decoding", features["speculative_decoding"]["value"]["method"]]
    if features["image_input"]["status"] == "yes":
        tags += ["vision", "image-text-to-text"]
    return tags


def gate_applies(name, kind, feats):
    if name in ("kernel_tests", "consistency"):
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


def over_context(info, native):
    """The metrics whose document is longer than the model's native context (`native`: the GGUF context_length,
    llm_config.json): a performance session's Loom server refused it (session "over_context", from perf.py's HTTP
    400) at a counted prompt longer than `native`, and no session measured it. A refusal at or below the native
    context (a package built smaller), an uncounted one, or an unknown native context exempts nothing, and a measured
    value always counts."""
    sessions = (info or {}).get("performance", {}).get("sessions", [])
    measured = {k for s in sessions for k, m in s["metrics"].items() if m.get("loom") is not None}
    return {k for s in sessions for k, n in (s.get("over_context") or {}).items()
            if native and n and n > native} - measured


def metric_applies(key, kind, feats, info=None):
    if kind == "decision" and key in ("decode", "decode_2k", "decode_32k", "streams4", "spec_acceptance"):
        return False   # a decision model answers in one forward pass: no decode
    if key == "spec_acceptance":
        return feats["speculative_decoding"]["status"] == "yes"
    return key not in over_context(info, native_context(feats))   # with info.json: longer than the native context


def native_context(feats):
    """The model's native context (the GGUF context_length llm_config.json records), or None."""
    return (feats["context"]["value"] or {}).get("native")


def is_devlab(info, session):
    """Whether a performance session ran on DevLab (its source kind is "devlab"). The source is an id into
    info["sources"], or the source itself in an evidence file's session (make_info.py roofline reads those)."""
    src = session.get("source")
    src = src if isinstance(src, dict) else (info.get("sources") or {}).get(src)
    return (src or {}).get("kind") == "devlab"


def card_metrics(info, devlab=False):
    """{metric: (metric entry, session index)}: per metric the last session that measured it, among the local sessions
    (the card's headline value; spec_acceptance from any session) or, with devlab=True, among the DevLab sessions
    (timed metrics only: the card's DevLab columns)."""
    out = {}
    for i, s in enumerate((info.get("performance") or {}).get("sessions", [])):
        dl = is_devlab(info, s)
        for k, m in s["metrics"].items():
            if (dl and k in TIMED) if devlab else (not dl or k not in TIMED):
                out[k] = (m, i)
    return out


def stale_metrics(info, programs, kind=None, feats=None, devlab=False):
    """{metric: programs_sha256 measured} for the card values (card_metrics(info, devlab)) whose session recorded
    programs other than `programs` (card_lib.programs_sha256 of the published package): that value describes another
    build. {} when `programs` is unknown; a session that recorded no programs is not flagged."""
    if not programs:
        return {}
    sessions = (info.get("performance") or {}).get("sessions", [])
    out = {}
    for k, (_, i) in card_metrics(info, devlab).items():
        sha = sessions[i].get("programs_sha256")
        if sha and sha != programs and (kind is None or metric_applies(k, kind, feats, info)):
            out[k] = sha
    return out


def devlab_metrics(info, kind=None, feats=None):
    """{metric: source id} for the timed metrics only DevLab measured: the headline value needs a local session."""
    local, sessions = card_metrics(info), (info.get("performance") or {}).get("sessions", [])
    return {k: sessions[i]["source"] for k, (_, i) in card_metrics(info, devlab=True).items()
            if k not in local and (kind is None or metric_applies(k, kind, feats, info))}


def devlab_perf_metrics(info, kind=None, feats=None, programs=None):
    """{metric: "missing" | programs measured} for the timed metrics with a local value but no DevLab value on the
    package's programs (`programs`, when known): a DevLab perf session fills the card's DevLab columns."""
    have, stale = card_metrics(info, devlab=True), stale_metrics(info, programs, kind, feats, devlab=True)
    return {k: stale.get(k, "missing") for k in card_metrics(info)
            if k in TIMED and (k not in have or k in stale) and (kind is None or metric_applies(k, kind, feats, info))}


def perf_flags(info, kind=None, feats=None, programs=None):
    """The performance guardrails of one card (loom-org org_status.py). The local / DevLab ratio is never a flag.
    far          {metric: [Loom / roofline, demonstrated efficiency]} where Loom's local value is below FAR_FRACTION x
                 the efficiency Loom ports have demonstrated for that kind of metric (info.json roofline)
    devlab       {metric: source id} where only DevLab measured the metric (devlab_metrics): a local session gives the
                 headline value (org_status: remeasure, devlab-primary)
    devlab_perf  {metric: "missing" | programs measured} where the DevLab columns lack the value or have it on other
                 programs (devlab_perf_metrics; org_status: missing devlab-perf, low priority)
    stale        {metric: programs measured} where the local value was measured on programs other than `programs`,
                 the package's own (stale_metrics; only with `programs`): it describes another build, and far is
                 judged on it, so a fresh local session comes first"""
    loom = {k: m.get("loom") for k, (m, _) in card_metrics(info).items()}
    if kind is not None:
        loom = {k: v for k, v in loom.items() if metric_applies(k, kind, feats, info)}
    far = {}
    for k, r in ((info.get("roofline") or {}).get("metrics") or {}).items():
        if loom.get(k) and r.get("ceiling") and r.get("efficiency"):
            frac = loom[k] / r["ceiling"]
            if frac < FAR_FRACTION * r["efficiency"]:
                far[k] = [round(frac, 3), r["efficiency"]]
    out = {"far": far, "devlab": devlab_metrics(info, kind, feats),
           "devlab_perf": devlab_perf_metrics(info, kind, feats, programs)}
    if programs is not None:
        out["stale"] = stale_metrics(info, programs, kind, feats)
    return out


def remeasure_reasons(flags):
    """Why a card needs a fresh local session before any performance judgement (org_status's remeasure action):
    "stale" (a local value measured on other programs), "devlab-primary" (a metric only DevLab measured)."""
    return [r for r, k in (("stale", "stale"), ("devlab-primary", "devlab")) if (flags or {}).get(k)]


def generator_commit(gen, versions):
    """(loom-skills commit that generated the programs, how it is known): llm_config.json's build stamp
    ("generator".skills_commit, build.py's generator_stamp(); "stamp"), else info.json's recovered
    versions.package.skills_commit ("recovered"); (None, None) when neither says."""
    if gen and gen.get("skills_commit") and gen["skills_commit"] != "unknown":
        return gen["skills_commit"], "stamp"
    rec = ((versions or {}).get("package") or {}).get("skills_commit")
    if not gen and isinstance(rec, dict):
        return rec["commit"], "recovered"
    return None, None


# ---- validation: the subset of JSON Schema info_schema.json uses (no third-party dependency) ----

def _types(t):
    return t if isinstance(t, list) else [t]


def _is(v, t):
    return {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str),
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": isinstance(v, bool), "null": v is None}[t]


def _check(v, s, root, path, errs):
    if "$ref" in s:
        s = root["$defs"][s["$ref"].split("/")[-1]]
    if "oneOf" in s:
        ok = [sub for sub in s["oneOf"] if not _check(v, sub, root, path, [])]
        if len(ok) != 1:
            errs.append(f"{path}: matches {len(ok)} of the oneOf alternatives")
        return errs
    if "const" in s and v != s["const"]:
        errs.append(f"{path}: must be {s['const']!r}")
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} not in {s['enum']}")
    if "type" in s and not any(_is(v, t) for t in _types(s["type"])):
        errs.append(f"{path}: {type(v).__name__} is not {s['type']}")
        return errs
    if isinstance(v, str) and "pattern" in s and not re.search(s["pattern"], v):
        errs.append(f"{path}: {v!r} does not match {s['pattern']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]:
            errs.append(f"{path}: {v} < {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            errs.append(f"{path}: {v} > {s['maximum']}")
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                errs.append(f"{path}: missing {k}")
        props = s.get("properties", {})
        for k, x in v.items():
            if k in props:
                _check(x, props[k], root, f"{path}.{k}", errs)
            elif "additionalProperties" in s and isinstance(s["additionalProperties"], dict):
                _check(x, s["additionalProperties"], root, f"{path}.{k}", errs)
            elif s.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {k}")
    if isinstance(v, list) and "items" in s:
        for i, x in enumerate(v):
            _check(x, s["items"], root, f"{path}[{i}]", errs)
    return errs


def validate(info):
    """Errors of info against info_schema.json, plus the cross-references the schema can't express (empty: valid)."""
    schema = load_json(SCHEMA_PATH)
    errs = _check(info, schema, schema, "info", [])
    if errs:
        return errs
    ids = set(info.get("sources", {}))
    p = info.get("performance") or {}
    for i, s in enumerate(p.get("sessions", [])):
        if s["source"] not in ids:
            errs.append(f"info.performance.sessions[{i}].source: no source {s['source']!r}")
    for n, g in info.get("gates", {}).items():
        if g["source"] not in ids:
            errs.append(f"info.gates.{n}.source: no source {g['source']!r}")
    for i, s in enumerate((info.get("devlab") or {}).get("sweeps", [])):
        if s["source"] not in ids:
            errs.append(f"info.devlab.sweeps[{i}].source: no source {s['source']!r}")
    return errs


def require_valid(info, what="info.json"):
    errs = validate(info)
    if errs:
        raise SystemExit(f"{what} is not valid {SCHEMA_ID}:\n  " + "\n  ".join(errs))
