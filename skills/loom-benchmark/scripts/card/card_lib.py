"""Shared definitions for the loom-fun card generator (schema loom-card/3; loom-card/2 cards are read as one GPU).

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
SCHEMA_ID = "loom-card/3"
LEGACY_SCHEMA_IDS = ("loom-card/2",)   # one GPU (Strix Halo): normalize() reads it as devices["strix-halo"]
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
# Card performance has one source: devlab-perf jobs on AMD DevLab Dispatch Strix Halo runners (Loom, served with the
# active wait, and llama.cpp on the same runner, the published programs; loom-org SKILL.md step 3). Per metric the last
# session that measured it is the card's value, judged against the roofline. spec_acceptance (drafted-token
# acceptance) is a drafting statistic of model, drafter and text, not a timing; its session's programs_sha256 decides
# whether it is stale like any other card value.
TIMED = set(ROOFLINE_METRIC)
PERF_SOURCE_KINDS = ("devlab", "factory")   # sources a card's performance sessions may come from

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
BASE_TAGS = ["loom", "loom-serve", "gguf"]

# The GPUs a card can cover, in the order the card shows them. id: the key of info.json "devices"; name: the card's
# name for the GPU; arch: its ISA; roofline: the loom-roofline roofline.py --device preset a card's ceilings use
# (radeon-r9700s is the AMD DevLab Dispatch R9700S runner every R9700 measurement runs on); memory_bytes: the
# memory loom-serve may use there (None: the package's own value); detect: what machine_info.py's GPU string says.
GPUS = {
    "strix-halo": {"name": 'AMD Ryzen AI Max+ 395 "Strix Halo"', "short": "Strix Halo", "arch": "gfx1151", "memory": None,
                   "roofline": "strix-halo", "memory_bytes": None, "runner": "Strix Halo", "tag": "strix-halo",
                   "detect": re.compile(r"strix|8060s|gfx1151", re.I)},
    "radeon-r9700": {"name": "AMD Radeon AI PRO R9700", "short": "Radeon AI PRO R9700", "arch": "gfx1201", "memory": "32 GB",
                     "roofline": "radeon-r9700s", "memory_bytes": 34_000_000_000, "runner": "R9700", "tag": "radeon-r9700",
                     "detect": re.compile(r"r9700|gfx1201|radeon ai pro", re.I)},
}
DEFAULT_GPU = "strix-halo"   # a card of loom-card/2 and evidence that names no GPU
# The R9700 pool: the board reports 31.86 GiB (34.2 GB) of VRAM and the driver reserves part of it, so loom-serve gets
# 34.0 GB (31.67 GiB): below what the board reports, above the 33.2 GB the largest package that fits at its native
# context needs (fit/verdict.json: weights + KV + SERVE_HEADROOM).
# The tokens a card row or gate needs one request to hold: on a GPU that serves a shorter context the cell reads
# "doesn't fit" (the user's rule: an accepted outcome, not a failure).
METRIC_CONTEXT = {"decode_2k": 2000, "decode_32k": 32000, "prefill_1k": 1000, "prefill_2k": 2000, "prefill_8k": 8000,
                  "prefill_16k": 16000, "prefill_32k": 32000}
GATE_CONTEXT = {"golden": 32000, "needle_full": "native"}   # golden's longest document; needle_full: the whole native context
SERVE_HEADROOM = 2_000_000_000   # bytes beyond weights and KV cache a GPU needs (the fit table's reserve)

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


_QUANT = r"(?<![A-Za-z0-9])((?:UD-)?I?Q[0-9]%s|BF16|F16|F32)(?![a-z0-9])"


def quant_of(filename):
    """The quantization in a GGUF file name (Q4_K_M, UD-Q5_K_M, Q8_0, BF16, ...), or None. A name written entirely in
    lower case (gguf-my-repo: q4_k_m) is read as upper case, its quant needing an underscore part (q4-k-m is none)."""
    stem = os.path.basename(filename)
    stem = stem[:-5] if stem.lower().endswith(".gguf") else stem
    found = re.findall(_QUANT % r"(?:_[0-9A-Z]+)*", stem)
    if not found and stem == stem.lower():
        found = re.findall(_QUANT % r"(?:_[0-9A-Z]+)+", stem.upper())
    return found[-1] if found else None


def build_tag(text):
    """A lemonade-sdk/llamacpp-rocm release tag (b1355) found in free text, or None."""
    m = re.search(r"\bb(1[0-9]{3,})\b", text or "")
    return f"b{m.group(1)}" if m else None


def programs_sha256(pkg):
    """sha256 over PKG/program/**/*.loom (path relative to program/, and content), each file without the generator's
    first-line '// Generated by ... from <cfg path>' comment, which names a local path. None without program files.
    (A catalog package keeps its programs in subdirectories; a generated one is flat.)"""
    d = os.path.join(pkg, "program")
    if not os.path.isdir(d):
        return None
    h = hashlib.sha256()
    names = sorted(os.path.relpath(os.path.join(r, f), d).replace(os.sep, "/")
                   for r, _, fs in os.walk(d) for f in fs if f.endswith(".loom"))
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


def package_facts(pkg_files, upstream=None):
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
            "occ_min": int((cfg.get("drafter") or cfg.get("draft") or {}).get("occ_min") or 0),   # occupancy gate: streams >= occ_min run undrafted
            "name": re.sub(r"[-_]gguf$", "", ckpt["repo"].split("/")[-1], flags=re.I),
            "output": model.get("model", {}).get("id", "").split("/")[-1] or None,
            "memory_bytes": execution.get("memory_bytes"),
            "generator": cfg.get("generator"), "serve": cfg.get("serve"), "sdk": cfg.get("sdk"),
            "features": derive_features(model, cfg, kind, execution, srcs, spec, upstream)}


def tool_calling(chat):
    """Whether a package's chat block serves tool calls: the starter's tool-call glue, a call format (parser) and the
    tool declaration block (renderer). call_format "none" is a chat-only block (a template that ignores tools).
    gates.sh skips the agent gates on the same test, so the card shows them as not applicable."""
    return chat.get("call_format") not in (None, "none") or bool(chat.get("tools_prefix"))


UPSTREAM_DRAFTERS = ("mtp", "dflash", "assist")   # drafter kinds that need an upstream drafter or head (ngram needs none)
HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
GGUF_SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # value type -> bytes


def _http_json(url):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except Exception:
        return None


def gguf_chat_template(url, limit=64 << 20):
    """The `tokenizer.chat_template` of a GGUF file, streamed from its header (the template follows the vocabulary:
    a few MB; the stream is dropped at the template), or None when the file has none or cannot be read."""
    import struct
    import urllib.request
    try:
        r = urllib.request.urlopen(url, timeout=60)
    except Exception:
        return None
    used = [0]

    def take(n):
        buf = b""
        while len(buf) < n:
            chunk = r.read(min(n - len(buf), 1 << 20))
            if not chunk:
                raise EOFError
            buf += chunk
        used[0] += n
        if used[0] > limit:
            raise EOFError
        return buf

    def string():
        return take(struct.unpack("<Q", take(8))[0])

    def skip(t):
        if t == 8:
            string()
        elif t == 9:
            et, n = struct.unpack("<IQ", take(12))
            if et in GGUF_SCALARS:
                for _ in range(n * GGUF_SCALARS[et] // (1 << 20)):
                    take(1 << 20)
                take(n * GGUF_SCALARS[et] % (1 << 20))
            else:
                for _ in range(n):
                    skip(et)
        else:
            take(GGUF_SCALARS[t])
    try:
        with r:
            if take(4) != b"GGUF":
                return None
            _, _, n_kv = struct.unpack("<IQQ", take(20))
            for _ in range(n_kv):
                key = string().decode()
                (t,) = struct.unpack("<I", take(4))
                if key == "tokenizer.chat_template" and t == 8:
                    return string().decode("utf-8", "replace")
                skip(t)
    except (EOFError, struct.error, KeyError, UnicodeDecodeError, OSError):
        return None
    return None


def template_facts(template):
    """What a chat template supports, from its text: {"tools", "thinking", "reasoning_effort"} (None: no template)."""
    if template is None:
        return None
    return {"tools": "tools" in template, "thinking": "enable_thinking" in template or "<think>" in template,
            "reasoning_effort": "reasoning_effort" in template}


def upstream_facts(ckpt_repo, model_id, ckpt_url=None, fetch_json=None, fetch_template=None):
    """What the upstream model has, from upstream files alone, failing closed: each of "vision", "audio" is True (the
    evidence is there), False (the files were read and say no) or None (not determined: nothing readable to say),
    and "template" is template_facts() of the GGUF's chat template when ckpt_url is given (else absent).
    Vision: the GGUF repo ships an mmproj, or the base model's config.json declares vision_config. Audio: config.json
    declares an audio_* / speech_* key. The base model is llm_config's model_id when it is org/name, else the GGUF
    repo's base_model tag. An unreadable config.json (gated, missing) falls back to the base model's API config: a
    *ForCausalLM / *LMHeadModel architecture is text only; anything else is not determined."""
    get = fetch_json or _http_json
    meta = get(f"{HF}/api/models/{ckpt_repo}")
    if meta is None:
        raise SystemExit(f"cannot read {HF}/api/models/{ckpt_repo} (upstream facts for the card's features; "
                         f"offline: --upstream none)")
    tag = next((t[len("base_model:"):] for t in meta.get("tags", [])
                if t.startswith("base_model:") and t.count(":") == 1 and HF_ID.match(t[len("base_model:"):])), None)
    cfg, bmeta = None, None
    for base in dict.fromkeys(b for b in (model_id if HF_ID.match(model_id or "") else None, tag) if b):
        cfg = cfg or get(f"{HF}/{base}/resolve/main/config.json")
        bmeta = bmeta or get(f"{HF}/api/models/{base}")
    archs = ((bmeta or {}).get("config") or {}).get("architectures") or []
    text_only = bool(archs) and all(re.search(r"(ForCausalLM|LMHeadModel)$", x) for x in archs)
    mmproj = any("mmproj" in s["rfilename"].lower() for s in meta.get("siblings", []))
    keys = list(cfg) if isinstance(cfg, dict) else None

    def has(found):
        return True if found else (False if keys is not None or text_only else None)
    out = {"vision": has(mmproj or keys is not None and "vision_config" in keys),
           "audio": has(keys is not None and any("audio" in k or "speech" in k for k in keys))}
    if ckpt_url:
        out["template"] = template_facts((fetch_template or gguf_chat_template)(ckpt_url))
    return out


def derive_features(model, cfg, kind, execution, srcs, spec, upstream=None):
    """Each feature is yes (the package has it), no (the upstream model has it, the package lacks it: the only red),
    n/a (the upstream model has none) or unknown (not determined: shown as a neutral ?). The chat block is derived
    from the upstream chat template, so tool calling, thinking and reasoning effort absent from it are absent
    upstream; a block older than the template glue (no call_format) says nothing about tools, so upstream's chat
    template (upstream["template"], from the GGUF) decides, and without it they are not determined. Vision and audio
    the package lacks follow upstream["vision"] / ["audio"]: True no, False n/a, None (or no probe) not determined."""
    chat = cfg.get("chat") or {}
    upstream = upstream or {}
    na = {"status": "n/a", "value": None}
    unknown = {"status": "unknown", "value": None}
    f = {}

    def lacks(has):   # the package lacks a feature: no when upstream has it
        return dict(unknown) if has is None else {"status": "no", "value": None} if has else dict(na)
    if kind == "decision":
        f["api"] = {"status": "yes", "value": {"endpoint": "/v1/systemone", "style": "systemone"}}
        for k in ("tool_calling", "thinking", "reasoning_effort", "speculative_decoding", "image_input", "sampling",
                  "max_output_tokens"):
            f[k] = dict(na)
    else:
        f["api"] = {"status": "yes", "value": {"endpoint": "/v1/chat/completions", "style": "openai"}}
        glue = "call_format" in chat   # a chat block from the template glue; older ones record nothing about tools
        tpl = upstream.get("template")

        def from_template(k):   # a legacy block: upstream's template says whether the feature exists
            return lacks(None if tpl is None else tpl[k])
        f["tool_calling"] = ({"status": "yes", "value": None} if tool_calling(chat) else dict(na) if glue
                             else from_template("tools"))
        mode = chat.get("thinking")
        if mode not in ("always", "toggle"):
            mode = "always" if chat.get("starts_in_reasoning") else None
        if mode:
            default = chat.get("thinking_default", True) if mode == "toggle" else True
            f["thinking"] = {"status": "yes", "value": {"mode": mode, "default": bool(default)}}
        else:
            f["thinking"] = dict(na) if "thinking" in chat or glue else from_template("thinking")
        levels = list((chat.get("reasoning_instructions") or {}).keys())
        f["reasoning_effort"] = ({"status": "yes", "value": {"levels": levels,
                                                             "default": chat.get("reasoning_effort_default")}}
                                 if levels else dict(na) if glue else from_template("reasoning_effort"))
        if spec:
            d = srcs.get("draft")
            f["speculative_decoding"] = {"status": "yes", "value": {"method": spec[0], "depth": spec[1],
                                                                    "drafter": d["repo"] if d else None}}
        else:
            kinds = sorted({t["kind"] for t in (cfg.get("drafter") or {}).get("tuned", []) if t["kind"] in UPSTREAM_DRAFTERS})
            # a drafter exists upstream and Loom supports it, but the package does not enable it (value.method null: no
            # spec gate applies); why is not stated unless the card's own data says so
            f["speculative_decoding"] = {"status": "yes", "value": {"method": None, "available": kinds}} if kinds else dict(na)
        v = srcs.get("vision")
        f["image_input"] = ({"status": "yes", "value": {"projector": v["file"]}} if v and cfg.get("vision")
                            else lacks(upstream.get("vision")))
        f["sampling"] = dict(unknown)   # proven by the sampling gate (evidence)
        mt = execution.get("max_tokens")
        f["max_output_tokens"] = {"status": "yes", "value": {"max_tokens": mt}} if mt else dict(unknown)
    audio = any(s.get("kind") in ("asr", "tts", "audio", "speech") for s in model.get("services", {}).values())
    f["audio"] = {"status": "yes", "value": None} if audio else lacks(upstream.get("audio"))
    ctx = execution.get("context_capacity")
    f["context"] = ({"status": "yes", "value": {"tokens": ctx, "native": cfg.get("context_length")}} if ctx
                    else dict(unknown))
    rows, pending = execution.get("rows"), execution.get("pending_requests")
    f["concurrency"] = ({"status": "yes", "value": {"rows": rows, "pending_requests": pending}} if rows or pending
                        else dict(unknown))
    return f


def with_gate_features(feats, kind, gates):
    """feats with the sampling feature the sampling gate proves (a chat model; no gate: the feature as it was)."""
    feats = {k: dict(v) for k, v in feats.items()}
    sg = gates.get("sampling")
    if kind == "chat" and sg:
        feats["sampling"] = {"status": "yes" if sg["status"] == "pass" else "no", "value": None}
    return feats


def drafting(feats):
    """Whether the package drafts by default (a speculative_decoding yes with a method: not one that is only available)."""
    sd = feats["speculative_decoding"]
    return sd["status"] == "yes" and bool(sd["value"].get("method"))


def derived_tags(features, gpus=(DEFAULT_GPU,)):
    tags = list(BASE_TAGS) + [GPUS[g]["tag"] for g in GPUS if g in gpus]
    if drafting(features):
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
        return drafting(feats)
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
    sessions = (single(info or {})).get("performance", {}).get("sessions", [])
    measured = {k for s in sessions for k, m in s["metrics"].items() if m.get("loom") is not None}
    return {k for s in sessions for k, n in (s.get("over_context") or {}).items()
            if native and n and n > native} - measured


def metric_applies(key, kind, feats, info=None, context=None):
    if kind == "decision" and key in ("decode", "decode_2k", "decode_32k", "streams4", "spec_acceptance"):
        return False   # a decision model answers in one forward pass: no decode
    if key == "spec_acceptance":
        return drafting(feats)
    # with info.json: longer than the native context (context: the context a GPU serves, when its memory holds less)
    native = native_context(feats)
    return key not in over_context(info, context or native) and not beyond_context(info, key, context, native)


def beyond_context(info, key, context, native):
    """Whether a card row (or gate, when `key` is a gate name) needs more tokens than the `context` a GPU serves, a
    context shorter than the model's `native` one. A GPU that serves the native context has no such rows (what the
    server refused there stays n/a), and a row a session measured fits."""
    if not context or not native or context >= native:
        return False
    need = {**METRIC_CONTEXT, **GATE_CONTEXT}.get(key)
    need = native if need == "native" else need
    if not need or need <= context:
        return False
    sessions = (single(info or {})).get("performance", {}).get("sessions", [])
    return not any(m.get("loom") is not None for s in sessions for k, m in s["metrics"].items() if k == key)


def native_context(feats):
    """The model's native context (the GGUF context_length llm_config.json records), or None."""
    return (feats["context"]["value"] or {}).get("native")


def card_metrics(info):
    """{metric: (metric entry, session index)}: per metric the last performance session that measured it (the card's
    value)."""
    out = {}
    for i, s in enumerate((single(info).get("performance") or {}).get("sessions", [])):
        for k, m in s["metrics"].items():
            out[k] = (m, i)
    return out


def stale_metrics(info, programs, kind=None, feats=None):
    """{metric: programs_sha256 measured} for the card values (card_metrics) whose session recorded programs other
    than `programs` (card_lib.programs_sha256 of the published package): that value describes another build. {} when
    `programs` is unknown; a session that recorded no programs is not flagged."""
    if not programs:
        return {}
    sessions = (single(info).get("performance") or {}).get("sessions", [])
    out = {}
    for k, (_, i) in card_metrics(info).items():
        sha = sessions[i].get("programs_sha256")
        if sha and sha != programs and (kind is None or metric_applies(k, kind, feats, info)):
            out[k] = sha
    return out


def perf_flags(info, kind=None, feats=None, programs=None):
    """The performance guardrails of one card (loom-org org_status.py).
    far          {metric: [Loom / roofline, demonstrated efficiency]} where Loom's card value is below FAR_FRACTION x
                 the efficiency Loom ports have demonstrated for that kind of metric (info.json roofline)
    stale        {metric: programs measured} where the card value was measured on programs other than `programs`,
                 the package's own (stale_metrics; only with `programs`): it describes another build, and far is
                 judged on it, so a fresh devlab-perf job comes first"""
    info = single(info)
    loom = {k: m.get("loom") for k, (m, _) in card_metrics(info).items()}
    if kind is not None:
        loom = {k: v for k, v in loom.items() if metric_applies(k, kind, feats, info)}
    far = {}
    for k, r in ((info.get("roofline") or {}).get("metrics") or {}).items():
        if loom.get(k) and r.get("ceiling") and r.get("efficiency"):
            frac = loom[k] / r["ceiling"]
            if frac < FAR_FRACTION * r["efficiency"]:
                far[k] = [round(frac, 3), r["efficiency"]]
    out = {"far": far}
    if programs is not None:
        out["stale"] = stale_metrics(info, programs, kind, feats)
    return out


def remeasure_reasons(flags):
    """Why a card needs a fresh devlab-perf job before any performance judgement (org_status's remeasure action):
    "stale" (a card value measured on other programs)."""
    return ["stale"] if (flags or {}).get("stale") else []


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

# ---- GPUs: loom-card/3 keeps gates, performance and roofline per GPU under info["devices"]

def gpu_from_device(device):
    """The GPU id a machine_info.py device block ({cpu, gpu, os}) names, else None."""
    text = " ".join(str(v) for v in (device or {}).values())
    return next((g for g, x in GPUS.items() if x["detect"].search(text)), None)


def normalize(info):
    """info as loom-card/3 (a copy): a loom-card/2 card's top-level gates, performance and roofline become the
    strix-halo entry of devices. A loom-card/3 card is returned as it is. Everything that reads a card goes through
    this (load_info), so one code path serves both."""
    if info.get("schema") not in LEGACY_SCHEMA_IDS:
        return info
    out = {k: v for k, v in info.items() if k not in ("gates", "performance", "roofline")}
    out["schema"] = SCHEMA_ID
    out["devices"] = {DEFAULT_GPU: {k: info[k] for k in ("gates", "performance", "roofline") if k in info}}
    return out


def load_info(path):
    return normalize(load_json(path))


def has_results(dev):
    """Whether a devices entry holds results (not a placeholder state)."""
    return dev is not None and not dev.get("state")


def card_gpus(info):
    """The GPU ids of a card in display order: every devices entry (results, or a state)."""
    return [g for g in GPUS if g in (info.get("devices") or {})]


def gpu_view(info, gpu):
    """A loom-card/2-shaped view of one GPU (top-level gates, performance, roofline of that GPU, plus its state,
    memory_bytes and served_context): what the single-GPU functions below take."""
    d = (info.get("devices") or {}).get(gpu) or {}
    out = {k: v for k, v in info.items() if k != "devices"}
    out["gates"] = d.get("gates") or {}
    for k in ("performance", "roofline", "state", "required_bytes", "memory_bytes", "served_context"):
        if d.get(k) is not None:
            out[k] = d[k]
    out["gpu"] = gpu
    return out


def primary_gpu(info):
    """The GPU a single-GPU reader (loom-org's org_status, the card's dashboards) reads: strix-halo when it has
    results, else the first GPU that has."""
    devs = info.get("devices") or {}
    for g in GPUS:
        if has_results(devs.get(g)):
            return g
    return DEFAULT_GPU


def single(info):
    """info itself when it is single-GPU shaped (a gpu_view, or loom-card/2), else its primary GPU's view."""
    if "devices" not in info:
        return info
    return gpu_view(info, primary_gpu(info))


def gpu_memory(info, facts_memory, gpu):
    """The loom-serve --memory_bytes of a GPU: the GPU's own recorded value, else the package's (model.json, then
    card.memory_bytes), at most the GPU's memory."""
    d = (info.get("devices") or {}).get(gpu) or {}
    if d.get("memory_bytes"):
        return d["memory_bytes"]
    base = facts_memory or (info.get("card") or {}).get("memory_bytes")
    cap = GPUS[gpu]["memory_bytes"]
    return min(base, cap) if base and cap else base


def gpu_context(info, gpu, native):
    """Tokens per request a GPU serves: its served_context when its memory holds less KV cache than the native
    context needs, else the native context."""
    return ((info.get("devices") or {}).get(gpu) or {}).get("served_context") or native


def served_context(weights, kv_bytes_per_token, memory_bytes, headroom=SERVE_HEADROOM):
    """Tokens of KV cache one request can address when loom-serve is given memory_bytes: what the budget leaves
    after the weights and the headroom, in whole 1024-token blocks (the elastic pool's address range follows the
    budget; docs/serving: --memory_bytes tightens it)."""
    if kv_bytes_per_token <= 0:
        return None
    return max(0, int((memory_bytes - weights - headroom) // kv_bytes_per_token) // 1024 * 1024)


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


# ---- free text: the only places a card shows words nobody templated (everything else is render_card.py's template)
# A caveat is one short factual line ("not supported: <reason>", "MTP head present but not yet supported by Loom (<reason>)",
# "llama.cpp loads <repo> (a conversion of the same <drafter> drafter)", "vision projector (--mmproj): not loaded, ...").
# Narrative (first person, history, what was tried, ideas, run/PR/agent chatter, speculation, local paths) is not a caveat.
NARRATIVE = re.compile(
    r"(?:\b(?:I|I'm|I've|I'll)\b|\b(?:we|we've|we'll|our|ours|my|me)\b"
    r"|\b(?:tried|trying|previously|earlier|formerly|used to|no longer|so far|for now|ruled out|remaining ideas?|ideas?"
    r"|workarounds?|attempt(?:ed|s)?|turns? out|seems?|appears? to|probably|likely|maybe|perhaps|hopefully|should|"
    r"investigat\w*|debug\w*|campaign|orchestrator|tech[- ]?lead|TODO|WIP|hack|guardrail)\b"
    r"|\bafter (?:we|the fix)\b|\bsince (?:then|we)\b|\bPR ?#?\d+|(?<![\w`])#\d+\b)",
    re.IGNORECASE)
LOCAL_PATH = re.compile(r"/home/|/tmp/|~/")   # in free text only: the Quick start legitimately installs under ~/
CAVEAT_MAX, STEP_MAX, DETAIL_MAX = 240, 140, 60


def narrative_hits(text):
    """The narrative phrases in text (empty: none)."""
    return sorted({m.group(0) for m in NARRATIVE.finditer(text)})


def text_errors(label, s, limit):
    """Errors of one free-text line: one short factual sentence, no narrative."""
    out = []
    if not isinstance(s, str) or not s.strip():
        return [f"{label}: empty"]
    if len(s) > limit:
        out.append(f"{label}: {len(s)} characters (at most {limit}): a caveat is one short factual line")
    if "\n" in s or re.search(r"[.!?]\s+[A-Z]", s):
        out.append(f"{label}: more than one sentence")
    if LOCAL_PATH.search(s):
        out.append(f"{label}: a local path")
    if narrative_hits(s):
        out.append(f"{label}: narrative ({', '.join(narrative_hits(s))}): state the fact (not supported: <reason>), not the history")
    return out


def free_text_errors(info):
    info = normalize(info)
    errs = []
    for gpu, dev in (info.get("devices") or {}).items():
        at = f"info.devices.{gpu}"
        p = dev.get("performance") or {}
        for i, u in enumerate((p.get("baseline") or {}).get("differs") or []):
            errs += text_errors(f"{at}.performance.baseline.differs[{i}]", u, CAVEAT_MAX)
        for b, x in (p.get("baselines") or {}).items():
            for i, u in enumerate(x["baseline"].get("differs") or []):
                errs += text_errors(f"{at}.performance.baselines.{b}.baseline.differs[{i}]", u, CAVEAT_MAX)
        for n, g in (dev.get("gates") or {}).items():
            if g.get("detail"):
                errs += text_errors(f"{at}.gates.{n}.detail", g["detail"], DETAIL_MAX)
    steps = info.get("next_steps") or []
    if steps and info.get("status") != "paused":
        errs.append("info.next_steps: only a paused card (status: paused) shows next steps")
    if len(steps) > 5:
        errs.append("info.next_steps: at most 5 steps")
    for i, u in enumerate(steps):
        errs += text_errors(f"info.next_steps[{i}]", u, STEP_MAX)
    card = info.get("card") or {}
    for k in ("license", "output"):
        if card.get(k) and not re.fullmatch(r"[A-Za-z0-9._+/-]+", card[k]):
            errs.append(f"info.card.{k}: {card[k]!r} is not a plain identifier")
    for t in card.get("tags") or []:
        if not re.fullmatch(r"[A-Za-z0-9._+-]+", t):
            errs.append(f"info.card.tags: {t!r} is not a plain tag")
    return errs


def validate(info):
    """Errors of info (loom-card/3, or loom-card/2 read as one GPU) against info_schema.json, plus the
    cross-references the schema can't express (empty: valid)."""
    info = normalize(info)
    schema = load_json(SCHEMA_PATH)
    errs = _check(info, schema, schema, "info", [])
    if errs:
        return errs
    ids = set(info.get("sources", {}))
    for gpu, dev in info["devices"].items():
        at = f"info.devices.{gpu}"
        if gpu not in GPUS:
            errs.append(f"{at}: unknown GPU (known: {', '.join(GPUS)})")
        if dev.get("state") and (dev.get("gates") or dev.get("performance") or dev.get("roofline")):
            errs.append(f"{at}: state {dev['state']} cannot come with results")
        if dev.get("state") == "unsupported" and not dev.get("required_bytes"):
            errs.append(f"{at}: state unsupported needs required_bytes")
        p = dev.get("performance") or {}
        for i, s in enumerate(p.get("sessions", [])):
            if s["source"] not in ids:
                errs.append(f"{at}.performance.sessions[{i}].source: no source {s['source']!r}")
        for b, x in (p.get("baselines") or {}).items():
            if x["session"]["source"] not in ids:
                errs.append(f"{at}.performance.baselines.{b}.session.source: no source {x['session']['source']!r}")
        for n, g in (dev.get("gates") or {}).items():
            if g["source"] not in ids:
                errs.append(f"{at}.gates.{n}.source: no source {g['source']!r}")
    return errs + free_text_errors(info)


def require_valid(info, what="info.json"):
    errs = validate(info)
    if errs:
        raise SystemExit(f"{what} is not valid {SCHEMA_ID}:\n  " + "\n  ".join(errs))
