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
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "loom-kb"))   # skills/loom-kb: devices, quants, methodology as data
import registry  # noqa: E402
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
# Sampling rule (methodology.md rule 5): every card value is the MEDIAN over sessions of each session's median sample. A fast model
# (published or expected decode above FAST_DECODE_TPS tok/s: its rate is bimodal per serve session) gets SESSIONS_FAST sessions and
# REPEAT_FAST samples per session of decode_2k, decode_32k and prefill_8k (perf.py --repeat-rows); the others SESSIONS sessions.
_S = registry.methodology()["sessions"]
FAST_DECODE_TPS, SESSIONS, SESSIONS_FAST, REPEAT_FAST = _S["fast_decode_tps"], _S["normal"], _S["fast"], _S["fast_repeat_rows"]


def sampling_plan(decode_tps, fast_tps=FAST_DECODE_TPS):
    """(sessions per 1-stream row, perf.py --repeat-rows) for a model whose published/expected decode rate is decode_tps (None: unknown = not fast)."""
    return (SESSIONS_FAST, REPEAT_FAST) if (decode_tps or 0) > fast_tps else (SESSIONS, 1)


def published_decode(slug, org="loom-fun"):
    """The published card's Strix Halo decode tok/s of org/slug (None when it cannot be read)."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"https://huggingface.co/{org}/{slug}/resolve/main/info.json", timeout=30) as r:
            info = json.load(r)
    except Exception:  # noqa: BLE001
        return None
    perf = info.get("performance") or ((info.get("devices") or {}).get("strix-halo") or {}).get("performance") or {}
    v = None
    for s in perf.get("sessions", []):
        v = (s["metrics"].get("decode") or {}).get("loom") or v
    return v


# card metric -> loom-roofline/roofline.py metric (the hardware ceiling). spec_acceptance has none.
ROOFLINE_METRIC = {"decode": "decode_tps", "decode_2k": "decode_tps_2k", "decode_32k": "decode_tps_32k",
                   "prefill_1k": "prefill_1000", "prefill_2k": "prefill_2000", "prefill_8k": "prefill_8000",
                   "prefill_16k": "prefill_16000", "prefill_32k": "prefill_32000", "streams4": "c4_steady"}
# decode context (tokens of KV read per decoded token) and streams of the drafted rows: speculative decoding commits
# several tokens per verify pass, so their ceiling is the pass rate x the expected tokens per pass
DRAFTED_ROWS = {"decode": (0, 1), "decode_2k": (2000, 1), "decode_32k": (32000, 1), "streams4": (600, 4)}
# Performance guardrails (loom-fleet org_status.py). Far from roofline (calibrated in loom-fleet reference/populate-org.md, "Far from
# roofline: how the threshold was set"): Loom below FAR_FRACTION x the efficiency tuned Loom ports have demonstrated
# for the metric kind
FAR_FRACTION = 0.5
# Card performance has one source: devlab-perf jobs on AMD DevLab Dispatch Strix Halo runners (Loom, served with the
# active wait, and llama.cpp on the same runner, the published programs; loom-fleet reference/publish.md step 3). Per metric the last
# session that measured it is the card's value, judged against the roofline. spec_acceptance (drafted-token
# acceptance) is a drafting statistic of model, drafter and text, not a timing; its session's programs_sha256 decides
# whether it is stale like any other card value.
TIMED = set(ROOFLINE_METRIC)
PERF_SOURCE_KINDS = ("devlab", "factory", "machine")   # sources a card's performance sessions may come from (machine: a GPU with no DevLab runner, measured on a named machine: fleet/devices data files)

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
    ("logits", "Host logits readback (llm.logits on): at T=0 every returned row's argmax is the device's token, the top-k matches llama.cpp, greedy text is unchanged"),
    ("decision", "Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts"),
]
GATE_NAMES = [n for n, _ in GATES]
# Gates that no longer exist (the needle gates were removed; gates.sh now only probes that the native context is admitted,
# which is a check, not a gate; `gufo`, Gufo's conformance suite, is a test of another project's server that a card does not claim: only
# the optional logits readback is). Published info.json files and old evidence still carry them: make_info.py drops them,
# normalize() strips them for every reader, so a card never shows one. The schema still accepts them (unknown gate names).
RETIRED_GATES = ("needle", "needle_full", "gufo")
# speculative decoding methods (llm_config.json draft.kind) as the card names them
METHOD = {"dflash": "DFlash", "mtp": "MTP", "ngram": "n-gram prompt lookup", "assist": "assistant drafter"}
BASE_TAGS = ["loom", "loom-serve", "gguf"]

# The GPUs a card can cover, in the order the card shows them. id: the key of info.json "devices"; name: the card's
# name for the GPU; arch: its ISA; roofline: the loom-roofline roofline.py --device preset a card's ceilings use
# (radeon-r9700s is the AMD DevLab Dispatch R9700S runner every R9700 measurement runs on); memory_bytes: the
# memory loom-serve may use there (None: the package's own value); detect: what machine_info.py's GPU string says.
# The card's view of the registry's device rows (loom-kb/device/<family>/<device>/device.json):


def _gpu_row(dev):
    c = dev["card"]
    row = {"name": dev["name"], "short": dev["short"], "arch": dev["gfx"], "memory": dev["memory"], "roofline": dev["roofline"]["default"],
           "memory_bytes": dev["memory_bytes"], "runner": c["runner"], "tag": c["tag"],
           "detect": re.compile(c["detect"]["pattern"], re.I if "i" in c["detect"].get("flags", "") else 0)}
    if c.get("serve_args"):
        row["serve_args"] = list(c["serve_args"])
    if "preload_max_weight_bytes" in c:   # absent: the GPU is served as before; null: --parameter_mode=preload for every package; N: up to N weight bytes
        row["preload_max_weight_bytes"] = c["preload_max_weight_bytes"]
    if c.get("title"):   # the card's section heading when it is not "<short> (<arch>, <memory>)"
        row["title"] = c["title"]
    if dev.get("preset_suffix"):
        row["preset_suffix"] = dev["preset_suffix"]
    return row


GPUS = {k: _gpu_row(d) for k, d in registry.devices().items()}
# The order a card shows its GPUs: device.json "card.order", else the registry's position (registry order puts gfx1150, a gfx11 device,
# before the gfx12 ones; the card reads Strix Halo, R9700, gfx1201 16 GB, gfx1150).
GPU_ORDER = sorted(GPUS, key=lambda g: (registry.devices()[g]["card"].get("order", list(GPUS).index(g)), list(GPUS).index(g)))
DEFAULT_GPU = "strix-halo"   # a card of loom-card/2 and evidence that names no GPU
# The R9700 pool: the board reports 31.86 GiB (34.2 GB) of VRAM and the driver reserves part of it, so loom-serve gets
# 34.0 GB (31.67 GiB): below what the board reports, above the 33.2 GB the largest package that fits at its native
# context needs (fit/verdict.json: weights + KV + SERVE_HEADROOM).
# The tokens a card row or gate needs one request to hold: on a GPU that serves a shorter context the cell reads
# "doesn't fit" (the user's rule: an accepted outcome, not a failure).
METRIC_CONTEXT = {"decode_2k": 2000, "decode_32k": 32000, "prefill_1k": 1000, "prefill_2k": 2000, "prefill_8k": 8000,
                  "prefill_16k": 16000, "prefill_32k": 32000}
GATE_CONTEXT = {"golden": 32000}   # golden's longest document
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
    return {"kind": kind, "execution": execution, "preset": preset_name if len(presets) > 1 else None, "presets": list(presets),
            "checkpoint": ckpt, "drafter": srcs.get("draft"), "projector": srcs.get("vision"), "spec": spec,
            "occ_min": int((cfg.get("drafter") or cfg.get("draft") or {}).get("occ_min") or 0),   # occupancy gate: streams >= occ_min run undrafted
            "name": re.sub(r"[-_]gguf$", "", ckpt["repo"].split("/")[-1], flags=re.I),
            "output": model.get("model", {}).get("id", "").split("/")[-1] or None,
            "memory_bytes": execution.get("memory_bytes"), "weight_bytes": package_weight_bytes(model, lock),
            "generator": cfg.get("generator"), "serve": cfg.get("serve"), "sdk": cfg.get("sdk"),
            "logits": declares_logits(model),   # model.json declares the host logits readback capability (llm.logits)
            "features": derive_features(model, cfg, kind, execution, srcs, spec, upstream)}


def declares_logits(model):
    """True when a model.json (any variant's service config) carries the `llm.logits` capability key (#736: its presence = the package supports it)."""
    if isinstance(model, dict):
        return "llm.logits" in model or any(declares_logits(v) for v in model.values())
    return isinstance(model, list) and any(declares_logits(v) for v in model)


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
    tags = list(BASE_TAGS) + [GPUS[g]["tag"] for g in GPU_ORDER if g in gpus]
    if drafting(features):
        tags += ["speculative-decoding", features["speculative_decoding"]["value"]["method"]]
    if features["image_input"]["status"] == "yes":
        tags += ["vision", "image-text-to-text"]
    return tags


def gate_applies(name, kind, feats, logits=True):
    if name in ("kernel_tests", "consistency"):
        return True
    if kind == "decision":
        return name == "decision"
    if name == "decision":
        return False
    if name == "logits":   # the host logits readback is an LLM text feature (llm.logits): a package without the capability has none
        return logits
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
    view = single(info)
    sessions = (view.get("performance") or {}).get("sessions", [])
    rm = view.get("remeasure") or {}
    same = {programs} | ({rm["compared_programs"], *rm.get("carried_programs", [])} if rm.get("programs_sha256") == programs else set())   # a same-runner A/B within the band, or identical stages
    out = {}
    for k, (_, i) in card_metrics(info).items():
        sha = sessions[i].get("programs_sha256")
        if sha and sha not in same and (kind is None or metric_applies(k, kind, feats, info)):
            out[k] = sha
    return out


def perf_flags(info, kind=None, feats=None, programs=None):
    """The performance guardrails of one card (loom-fleet org_status.py).
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
    this (load_info), so one code path serves both. Retired gates (RETIRED_GATES) are dropped from either."""
    if info.get("schema") in LEGACY_SCHEMA_IDS:
        out = {k: v for k, v in info.items() if k not in ("gates", "performance", "roofline")}
        out["schema"] = SCHEMA_ID
        out["devices"] = {DEFAULT_GPU: {k: info[k] for k in ("gates", "performance", "roofline") if k in info}}
        info = out
    elif not any(g in RETIRED_GATES for d in (info.get("devices") or {}).values() for g in (d.get("gates") or {})):
        return info
    out = dict(info)
    out["devices"] = {gpu: ({**d, "gates": {n: g for n, g in d["gates"].items() if n not in RETIRED_GATES}}
                            if d.get("gates") else d) for gpu, d in (info.get("devices") or {}).items()}
    return out


def load_info(path):
    return normalize(load_json(path))


def has_results(dev):
    """Whether a devices entry holds results (not a placeholder state)."""
    return dev is not None and not dev.get("state")


def card_gpus(info):
    """The GPU ids of a card in display order: every devices entry (results, or a state)."""
    return [g for g in GPU_ORDER if g in (info.get("devices") or {})]


def gpu_view(info, gpu):
    """A loom-card/2-shaped view of one GPU (top-level gates, performance, roofline of that GPU, plus its state,
    memory_bytes and served_context): what the single-GPU functions below take."""
    d = (info.get("devices") or {}).get(gpu) or {}
    out = {k: v for k, v in info.items() if k != "devices"}
    out["gates"] = d.get("gates") or {}
    for k in ("performance", "roofline", "state", "required_bytes", "memory_bytes", "served_context", "remeasure"):
        if d.get(k) is not None:
            out[k] = d[k]
    out["gpu"] = gpu
    return out


def primary_gpu(info):
    """The GPU a single-GPU reader (loom-fleet org_status, the card's dashboards) reads: strix-halo when it has
    results, else the first GPU that has."""
    devs = info.get("devices") or {}
    for g in GPU_ORDER:
        if has_results(devs.get(g)):
            return g
    return DEFAULT_GPU


def single(info):
    """info itself when it is single-GPU shaped (a gpu_view, or loom-card/2), else its primary GPU's view."""
    if "devices" not in info:
        return info
    return gpu_view(info, primary_gpu(info))


def gpu_preset(facts, gpu):
    """The preset a GPU's Quick start fetches: `<first preset>-r9700` on the R9700 when the package has one (build.py
    --prefill-r9700: a larger prefill chunk, same programs), else the default preset (None when the package has one)."""
    first = (facts.get("presets") or [None])[0]
    suffix = (GPUS.get(gpu) or {}).get("preset_suffix", "")
    alt = f"{first}{suffix}"
    if suffix and first and alt in facts["presets"]:
        return alt
    return facts.get("preset")


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


def serves(info, gpu):
    """Whether the card shows a Quick start block for a GPU: the package fits it (not unsupported) and the GPU has results, a DevLab
    runner, or serve flags recorded for it (fleet/devices: a fit without measurements yet)."""
    dev = (info.get("devices") or {}).get(gpu)
    return dev is not None and dev.get("state") != "unsupported" and (has_results(dev) or has_runner(gpu) or "serve_args" in dev)


def info_serve_args(info, gpu, weight_bytes=None):
    """The loom-serve flags of a GPU on this card: the ones its devices entry records (a data file's, per model: used as they are), else the
    GPU's own for a package of weight_bytes (card_serve_args)."""
    dev = (info.get("devices") or {}).get(gpu) or {}
    return list(dev["serve_args"]) if "serve_args" in dev else card_serve_args(gpu, weight_bytes)


def gpu_serve_args(gpu):
    """The loom-serve flags a GPU is served with in place of --memory_bytes (card_lib.GPUS "serve_args"), [] when it takes the budget."""
    return list(GPUS[gpu].get("serve_args") or [])


PRELOAD_FLAG = "--parameter_mode=preload"


def preload_mode(gpu, weight_bytes):
    """How a GPU loads a package's weights on the host (device.json card.preload_max_weight_bytes): "preload" (--parameter_mode=preload: the stock-host
    fix for ROCm/hrx-system#3332, the host copy of the weights must fit beside the GPU's memory), "default" (the GPU has a limit and the package is
    above it: the card says how to start on a stock host), or None (the GPU has no such field, or the weights are unknown: unchanged)."""
    row = GPUS[gpu]
    if "preload_max_weight_bytes" not in row:
        return None
    limit = row["preload_max_weight_bytes"]
    if limit is None:
        return "preload"
    if not weight_bytes:
        return None
    return "preload" if weight_bytes <= limit else "default"


def card_serve_args(gpu, weight_bytes=None):
    """The loom-serve flags a package of weight_bytes (the parameter files' sizes: package_weight_bytes) is served with on a GPU, as its card's Quick
    start prints them: the GPU's own flags plus --parameter_mode=preload where preload_mode says so."""
    return gpu_serve_args(gpu) + ([PRELOAD_FLAG] if preload_mode(gpu, weight_bytes) == "preload" else [])


def package_weight_bytes(model, lock):
    """Bytes of every parameter file model.json loads (model, drafter, projector ...; each origin once), from model.json.lock sizes; None when a size is missing."""
    locked = {f["origin"]: f.get("size") for f in (lock or {}).get("files", [])}
    seen, total = set(), 0
    for variant in model.get("variants", {}).values():
        for service in variant.get("services", {}).values():
            for param in service.get("parameters", {}).values():
                for f in param.get("files", []):
                    o = f.get("origin")
                    if not o or o in seen:
                        continue
                    seen.add(o)
                    if not isinstance(locked.get(o), int):
                        return None
                    total += locked[o]
    return total or None


def package_serve_args(gpu, pkg_dir):
    """card_serve_args for the package in a directory (model.json + model.json.lock): what its card's Quick start serves it with on the GPU
    (devlab_perf.sh serve_alone serves with the same flags). A package whose lock sizes are unknown has unknown weights: only a GPU with no weight limit preloads."""
    pf = load_package(pkg_dir)
    return card_serve_args(gpu, package_weight_bytes(pf["model"], pf["lock"]))


def has_runner(gpu):
    """True when the device has a DevLab runner (device.json runner.labels): without one nothing can be measured on it."""
    return bool(registry.device(gpu)["runner"]["labels"])


def fixed_pool_capacity(row, memory_bytes=None, gpu="radeon-r9700"):
    """KV pool positions a fixed-backing server (no --memory_bytes budget: the pool is committed eagerly) can hold on the GPU (default the R9700) beside
    the package's weights/workspace/recurrent state (fit row "w") and SERVE_HEADROOM, rounded down to 1024; None when the row has no sizing."""
    if not row or not row.get("kvpp") or "w" not in row:
        return None
    mem = memory_bytes or GPUS[gpu]["memory_bytes"]
    return max(0, int((mem - row["w"] - SERVE_HEADROOM) // row["kvpp"]) // 1024 * 1024)


# --- the execution settings of a device's preset: scope.json rules (skills/loom-kb exec.* choices), not the fit table --------------
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
EXEC_KEYS = ("prefill_capacity", "context_capacity", "rows", "draft_depth", "cache_target_bytes")   # model.json execution fields, in this order (SDK 0.2 has no pool_capacity: context_capacity and memory_bytes size the pool)
_RESOLVERS = {}


def _resolver(kb_root):
    kb_root = os.path.abspath(kb_root)
    if kb_root not in _RESOLVERS:
        import importlib.util
        spec = importlib.util.spec_from_file_location("loom_kb_resolve_" + str(len(_RESOLVERS)), os.path.join(kb_root, "resolve.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _RESOLVERS[kb_root] = mod
    return _RESOLVERS[kb_root]


def scope_execution(slug, gpu, root=None, cfg=None):
    """{"prefill_capacity": N, "context_capacity": N, ..., "context_cap": N} the scope.json rules (model nodes, family, facet ...) set for
    SLUG's preset on device GPU, resolved from the checked-in build input fleet/<slug>/llm_config.json (or CFG) of the checkout ROOT (default:
    this one; pass a git export of another ref to read that ref's rules). Empty when no rule sets one or the model is not in the fleet."""
    root = os.path.abspath(root or REPO)
    kb = os.path.join(root, "skills", "loom-kb")
    if cfg is None:
        p = os.path.join(root, "fleet", slug, "llm_config.json")
        if not os.path.isfile(p) or not os.path.isfile(os.path.join(kb, "resolve.py")):
            return {}
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
    got = _resolver(kb).device_execution(cfg, gpu, kb, slug=slug)
    return {k[len("exec."):]: v["value"] for k, v in got.items()}


def fit_rows(path, gpu, root=None):
    """{slug: row} of a device's fit table (fit.py: sizing and verdict data) with the serving settings of the scope rules put back as the
    row keys the readers know: "ctx_cap" (exec.context_cap) and "<runner key>_execution" (prefill_capacity, context_capacity,
    rows, draft_depth). A row that still carries such a block of its own (a table of a device without rules) keeps it unless a rule sets one.
    ROOT: the checkout whose rules apply; default: this one when PATH is one of its own tables (.github/devlab/), else none (a table elsewhere,
    e.g. a test fixture, is standalone data: pass root= to read this checkout's rules for it)."""
    with open(path, encoding="utf-8") as f:
        rows = {r["slug"]: dict(r) for r in json.load(f)}
    xkey = registry.fit_execution_key(gpu)
    if root is None and os.path.realpath(path).startswith(os.path.join(REPO, ".github", "devlab") + os.sep):
        root = REPO
    for slug, row in rows.items():
        got = scope_execution(slug, gpu, root) if root else {}
        if got.get("context_cap"):
            row["ctx_cap"] = got["context_cap"]
        ex = {k: got[k] for k in EXEC_KEYS if k in got}
        if ex:
            row[xkey] = ex
    return rows


def gpu_preset_name(model_or_presets, gpu="radeon-r9700"):
    """`<first preset><device preset suffix>` (`<first>-r9700`): the preset a package carries for a GPU when its serving settings differ
    from model.json's default preset (Strix Halo's). The suffix is the device's "preset_suffix" (device.json)."""
    presets = model_or_presets.get("presets", model_or_presets) if isinstance(model_or_presets, dict) else model_or_presets
    return f"{next(iter(presets))}{GPUS[gpu]['preset_suffix']}"


def gpu_preset_execution(row, served_ctx=None, gpu="radeon-r9700"):
    """The execution overrides a fit-table row's "<runner>_execution" ("r9700_execution": registry.fit_execution_key) gives the GPU's preset (prefill_capacity, rows, draft_depth,
    context_capacity ...), plus the served context (context_capacity = served_ctx) when the R9700
    serves a shorter context and the row does not say otherwise. None: the row has no block, so the preset is not touched."""
    ex = (row or {}).get(registry.fit_execution_key(gpu))
    if not ex:
        return None
    out = {}
    if served_ctx:
        out.update(context_capacity=served_ctx)
    out.update(ex)
    return out


def with_gpu_preset(model, execution, gpu="radeon-r9700"):
    """model.json's document with the GPU's `<first>-r9700` preset: a copy of the first preset (or of the preset a build already
    made, e.g. build.py --prefill-r9700) with `execution` over its execution. The programs, weights and the default preset
    (Strix Halo) are untouched."""
    doc = json.loads(json.dumps(model))
    name = gpu_preset_name(doc, gpu)
    first = next(iter(doc["presets"]))
    alt = doc["presets"].get(name) or json.loads(json.dumps(doc["presets"][first]))
    alt["execution"].update(execution)
    ex = alt["execution"]
    if "pool_capacity" in ex and ex.get("context_capacity"):   # a v0.1 preset still has one: it never exceeds the context this preset serves
        ex["pool_capacity"] = min(ex["pool_capacity"], ex["context_capacity"])
    doc["presets"][name] = alt
    return doc


def fit_served_context(row, memory_bytes):
    """Tokens one request holds on a GPU, from a fit-table row: None when the native context fits (the package's own
    context), else what the memory leaves after the weights; a row's optional "ctx_cap" (a measured limit, e.g. the
    retained prefix state of the 8k/16k rows leaves no room for a 32k request) lowers it further."""
    cap = row.get("ctx_cap")
    if row["w"] + row["kv"] + SERVE_HEADROOM <= memory_bytes and not (cap and cap < row["ctx"]):
        return None
    ctx = served_context(row["w"], row["kv"] / row["ctx"], memory_bytes) if row["kv"] else row["ctx"]
    return min(ctx, cap) if cap else ctx


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
        if dev.get("shares_section_with") and dev["shares_section_with"] not in info["devices"]:
            errs.append(f"{at}: shares_section_with {dev['shares_section_with']!r} is not a GPU of this card")
        if dev.get("shares_section_with") and (dev.get("gates") or dev.get("performance") or dev.get("roofline")):
            errs.append(f"{at}: a GPU that shares another's section has no results of its own")
        if dev.get("state") == "unsupported" and not dev.get("required_bytes"):
            errs.append(f"{at}: state unsupported needs required_bytes")
        p = dev.get("performance") or {}
        for i, s in enumerate(p.get("sessions", [])):
            if s["source"] not in ids:
                errs.append(f"{at}.performance.sessions[{i}].source: no source {s['source']!r}")
        for b, x in (p.get("baselines") or {}).items():
            if x["session"]["source"] not in ids:
                errs.append(f"{at}.performance.baselines.{b}.session.source: no source {x['session']['source']!r}")
        if dev.get("remeasure") and dev["remeasure"]["source"] not in ids:
            errs.append(f"{at}.remeasure.source: no source {dev['remeasure']['source']!r}")
        fs = (dev.get("remeasure") or {}).get("first_source")
        if fs and fs not in ids:
            errs.append(f"{at}.remeasure.first_source: no source {fs!r}")
        for n, g in (dev.get("gates") or {}).items():
            if g["source"] not in ids:
                errs.append(f"{at}.gates.{n}.source: no source {g['source']!r}")
    return errs + free_text_errors(info)


def require_valid(info, what="info.json"):
    errs = validate(info)
    if errs:
        raise SystemExit(f"{what} is not valid {SCHEMA_ID}:\n  " + "\n  ".join(errs))


# ---- per-device results files: fleet/devices/<name>.json (schema fleet/devices/device_results.schema.json, README.md there)
# A GPU with no DevLab runner (the gfx1201 16 GB class, gfx1150) records, per model, whether it fits, the largest context it admits, how to serve
# it and, when measured, evidence (make_info.py perf/gates output with a "machine" source). make_info.py build turns the entry of a card's
# model into the card's devices entry, the GPU's preset and its evidence; a GPU shows only where its file has data for that card.
DEVICE_RESULTS_ID = "loom-device-results/1"
DEVICE_RESULTS_DIR = os.path.join(REPO, "fleet", "devices")
DEVICE_SCHEMA_PATH = os.path.join(DEVICE_RESULTS_DIR, "device_results.schema.json")


LEGACY_DEVICE_IDS = {"radeon-rx9070xt": "gfx1201-16gb"}   # the RX 9070 XT file (fleet/devices/rx9070xt.json) is the 16 GB class's data: read as it


def validate_device_results(doc, what="device results"):
    """Errors of one fleet/devices file against its schema, plus the rules the schema cannot say (empty: valid)."""
    if doc.get("device") in LEGACY_DEVICE_IDS:
        doc = dict(doc, device=LEGACY_DEVICE_IDS[doc["device"]])
    schema = load_json(DEVICE_SCHEMA_PATH)
    errs = _check(doc, schema, schema, what, [])
    if errs:
        return errs
    if doc["device"] not in GPUS:
        errs.append(f"{what}: unknown device {doc['device']!r} (known: {', '.join(GPUS)})")
    share = doc.get("share_section_with")
    if share and share not in GPUS:
        errs.append(f"{what}: share_section_with {share!r} is not a device")
    for slug, m in doc["models"].items():
        at = f"{what}.models.{slug}"
        if m["fits"] and not m.get("max_context"):
            errs.append(f"{at}: fits without max_context")
        if not m["fits"] and (m.get("evidence") or m.get("serve_args") or m.get("execution")):
            errs.append(f"{at}: a model that does not fit has no serve_args, execution or evidence")
        if share and m.get("evidence"):
            errs.append(f"{at}: evidence on a device that shares {share}'s section (the shared section is {share}'s)")
        for i, ev in enumerate(m.get("evidence") or []):
            if ev.get("evidence") != EVIDENCE_ID:
                errs.append(f"{at}.evidence[{i}]: not a {EVIDENCE_ID} document")
            src = ((ev.get("performance") or {}).get("session") or {}).get("source")
            srcs = [src] if src else []
            srcs += [g.get("source") for g in ev.get("gates") or []]
            for s in srcs:
                if s and s.get("kind") not in ("machine", "local", "devlab", "factory"):
                    errs.append(f"{at}.evidence[{i}]: unknown source kind {s.get('kind')!r}")
                if src and s is src and s.get("kind") != "machine":
                    errs.append(f"{at}.evidence[{i}]: performance of a device without a DevLab runner comes from a source of kind machine (make_info.py perf --source-kind machine)")
            for k in ev:
                if k not in ("evidence", "gpu", "performance", "gates", "roofline"):
                    errs.append(f"{at}.evidence[{i}]: key {k!r} not allowed (a device file carries performance, gates and roofline only)")
    return errs


def load_device_results(directory=None):
    """{GPU id: validated document} of every *.json in fleet/devices (or DIRECTORY); None/"none": no data files (DIRECTORY "none" disables)."""
    directory = directory or os.environ.get("LOOM_DEVICE_RESULTS") or DEVICE_RESULTS_DIR
    out = {}
    if directory == "none" or not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name.endswith(".schema.json"):
            continue
        path = os.path.join(directory, name)
        doc = load_json(path)
        errs = validate_device_results(doc, name)
        if errs:
            raise SystemExit(f"{path} is not valid {DEVICE_RESULTS_ID}:\n  " + "\n  ".join(errs))
        doc["device"] = LEGACY_DEVICE_IDS.get(doc["device"], doc["device"])
        if doc["device"] in out:
            raise SystemExit(f"{path}: device {doc['device']} is also in another file of {directory}")
        out[doc["device"]] = doc
    return out


def with_device_preset(model, gpu, execution, base=None):
    """model.json's document with the GPU's `<first><suffix>` preset: a copy of preset BASE (a name: the R9700's for an RDNA4 GPU of the
    same programs; default the first preset) with EXECUTION over its execution. Programs, weights, tokenizer and the other presets are untouched."""
    doc = json.loads(json.dumps(model))
    name = gpu_preset_name(doc, gpu)
    first = next(iter(doc["presets"]))
    alt = json.loads(json.dumps(doc["presets"].get(base) or doc["presets"][first]))
    alt["execution"].update(execution)
    ex = alt["execution"]
    if "pool_capacity" in ex and ex.get("context_capacity"):
        ex["pool_capacity"] = min(ex["pool_capacity"], ex["context_capacity"])
    doc["presets"][name] = alt
    return doc
