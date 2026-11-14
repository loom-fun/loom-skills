#!/usr/bin/env python3
"""Render a loom-fun card's README.md (and a generated REPORT.md) from the files of its repository (schema loom-card/3).

usage: render_card.py DIR [-o README.md]           (-o - prints to stdout; default: DIR/README.md)
       render_card.py --check DIR README.md        exit 1 unless README.md is exactly what DIR renders to

DIR holds model.json, model.json.lock, llm_config.json and info.json (a package, or those files downloaded from
the card's repo). Sources, revisions and files come from model.json.lock; context, rows, presets and drafting from
model.json; features, the drafter and the generating commit and SDK from llm_config.json; measurements, gate
evidence, device, baseline and dates from info.json. With info.json "report": "generated" it also writes
DIR/REPORT.md (methodology, commands, per-run numbers); with "attached", DIR/REPORT.md must already exist.

A pure function of those files (and DIR/program/*.loom, whose hash REPORT.md compares with the measured one):
no clock, no environment. Every sentence is part of this template; only the filled-in values differ between
cards. Never edit a rendered README by hand: change the data (make_info.py) or this template, and re-render.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_lib import (DEFAULT_GPU, FEATURES, GATE_NAMES, GATES, GPUS, HF, LOOM_URL, METHOD, METRIC_KEYS, METRICS, SCHEMA_ID, SKILLS_URL,  # noqa: E402
                      TIMED, card_gpus, card_metrics, derived_tags, beyond_context, gate_applies, generator_commit, gpu_context, gpu_memory,
                      gpu_document_errors, gpu_view, has_results, load_info, load_json, load_package, metric_applies, native_context, over_context,
                      package_facts, programs_sha256, require_valid, stale_metrics, with_gate_features)
from matched_spec import TIE_BAND  # noqa: E402

YES, NO, NA, MISSING = "✅", "❌", "—", "—"
FIT = "doesn't fit"   # a row or gate that needs more context than the GPU's memory serves: accepted, not a failure
UNKNOWN = "?"   # a feature that could not be determined: neither supported, nor n/a, nor a failure
SUP = "¹²³⁴⁵⁶⁷⁸⁹"
ABOVE = "‡"   # a Loom value above its modeled ceiling
STALE = "†"   # a Loom value measured on programs other than this repository's (card_lib.stale_metrics)
SHORT = {"decode": "decode", "decode_2k": "decode at 2k", "decode_32k": "decode at 32k", "streams4": "4 streams"}


def mark(i):
    return SUP[i - 1] if i <= len(SUP) else f"[{i}]"


def num(key, v):
    if v is None:
        return MISSING
    return f"{round(v * 100)} %" if key == "spec_acceptance" else f"{v:.1f}"


def count(v):
    return f"{v:,}"


def text(v):
    return "not recorded" if v in (None, "") else v


def hf(repo):
    return f"[{repo}]({HF}/{repo})"


def short(sha):
    return f"`{sha[:7]}`"


def src_text(s):
    if not s:
        return "not recorded"
    name = {"local": "local run", "devlab": "DevLab Dispatch run", "factory": "loom-factory run"}.get(s["kind"], f"{s['kind']} run")
    if s["kind"] == "local":
        return name
    label = f"{name} {s['run']}" if s.get("run") else name
    return f"[{label}]({s['url']})" if s.get("url") else label


def device_text(d):
    return ", ".join(d[k] for k in ("cpu", "gpu", "os") if d.get(k)) if d else "not recorded"


def sdk_text(x):
    """An hrx-loom identity (loom-install.json or a recovered one) in one neutral phrase."""
    if not isinstance(x, dict):
        return "unknown"
    v = f"`{x['hrx_loom']}`" if x.get("hrx_loom") else "version unknown"
    detail = []
    if x.get("hrx_system_sha"):
        detail.append(f"hrx-system {short(x['hrx_system_sha'])}")
    elif x.get("fingerprint"):
        detail.append(f"install fingerprint `{x['fingerprint']}`")
    else:
        detail.append("hrx-system commit unknown")
    if x.get("recovered"):
        detail.append("recovered")
    return f"{v} ({', '.join(detail)})"


def skills_text(gen, versions):
    """The loom-skills commit that generated the programs: llm_config.json's build stamp, else the recovered one."""
    commit, how = generator_commit(gen, versions)
    if how == "stamp":
        out = short(commit)
        if gen.get("dirty"):
            out += " (with local changes)"
        if (gen.get("kit") or {}).get("modified"):
            out += " (kit edited after copying)"
        return out
    return f"{short(commit)} (recovered)" if how == "recovered" else "unknown"


def package_sdk(gen, versions):
    if gen:
        return sdk_text(gen.get("hrx_loom"))
    return sdk_text((versions.get("package") or {}).get("hrx_loom"))


LEARN_MORE = "Learn more on the [Loom ecosystem website](https://loom-fun.github.io/loom-dev/)."


def required_sdk(facts, versions, sources):
    """The hrx-loom build the Quick start must install: llm_config.json "sdk" (an explicit requirement, possibly a CI
    build), else the PyPI release that generated the programs, else the one the card's package record or its
    measurement sessions name (only when every named session agrees). A derived pin is only used when that build is a
    PyPI release (a CI build is not installable by version, so it needs the explicit "sdk"). Never hardcoded per model."""
    pin = facts.get("sdk")
    if pin and (pin.get("source") == "ci" or pin.get("hrx_loom")):
        return pin
    gen = (facts.get("generator") or {}).get("hrx_loom")
    pypi = lambda x: isinstance(x, dict) and x.get("hrx_loom") and x.get("source") == "pypi"
    for x in (gen, (versions.get("package") or {}).get("hrx_loom")):
        if pypi(x):
            return x
    seen = {x["hrx_loom"] for x in (s.get("hrx_loom") for s in sources.values()) if pypi(x)}
    return {"hrx_loom": seen.pop(), "source": "pypi"} if len(seen) == 1 else None


def margin(loom, base):
    if loom is None or base is None or not base:
        return MISSING
    return f"{round((loom / base - 1) * 100):+d} %"


LLAMA_METHOD = {"draft-dflash": "DFlash", "draft-mtp": "MTP", "ngram-simple": "n-gram", "ngram-mod": "n-gram"}


def llama_draft(args):
    """(method, n-max or None) of a llama.cpp server's arguments, (None, None) without speculative decoding."""
    m = re.search(r"--spec-type (\S+)", args or "")
    n = re.search(r"--spec-draft-n-max (\d+)", args or "")
    return (LLAMA_METHOD.get(m.group(1), m.group(1)) if m else None), (int(n.group(1)) if n else None)


def methodology(base, extra, spec, drafter, runs_dec, runs_c4):
    """The card's methodology: the same short template on every card, filled from the card's own data (baseline server
    arguments, drafting setting, runs). Anything that differs between the engines is stated as it is, not smoothed over."""
    vk = next((x["baseline"] for b, x in extra.items() if b == "vulkan"), None)
    engines = [(base["name"] + " ROCm", base)] + ([(base["name"] + " Vulkan", vk)] if vk else [])
    ver = ", ".join(f"{n} `{b['version']}`" if b.get("version") else f"{n} (build not recorded)" for n, b in engines)
    loom_m = METHOD.get(spec[0], spec[0]) if spec else None
    ll = [(n, *llama_draft(b.get("args"))) for n, b in engines]
    if spec is None and all(m is None for _, m, _ in ll):
        draft = "no drafting on either engine (no MTP head in the GGUF and no drafter published for this exact model)"
        depth = ""
    elif spec and all(m is not None and (m == loom_m or (loom_m == "assistant drafter" and m == "MTP")) for _, m, _ in ll):
        d = f" ({drafter})" if drafter else ""
        draft = f"{loom_m}{d} on both engines"
        depth = "; draft depth: Loom " + str(spec[1]) + ", llama.cpp --spec-draft-n-max " + " / ".join(
            (str(n) if n else "llama.cpp default") for _, _, n in ll[:1]) + " (each engine its own)"
    else:
        draft = ("Loom " + (f"{loom_m} depth {spec[1]}" if spec else "no drafting") + "; "
                 + "; ".join(f"{n} " + (f"{m}" + (f" n-max {k}" if k else "") if m else "no drafting") for n, m, k in ll)
                 + " (NOT the same choice on both engines)")
        depth = ""
    out = [f"- Compared: Loom against {ver}; the same GGUF, the same DevLab runner and job, the same prompts; Loom served with "
           f"the active wait. Values are the mean of {runs_dec} rounds (4 streams: median of {runs_c4}); Loom is timed by the client, "
           "llama.cpp by its server.",
           f"- Drafting, 1-stream rows (decode, decode_2k, decode_32k, prefill): {draft}{depth}. Each engine is loaded once for "
           "all these rows.",
           "- 4 streams: both engines loaded with drafting off, 4 parallel requests, aggregate tokens per second.",
           "- Rows: decode = short-context decode; decode_2k / decode_32k = decode after a ~2k / ~32k-token context; "
           "prefill_Nk = prompt processing of an N k-token prompt; streams4 = 4-stream aggregate decode. A margin within "
           f"+-{round(TIE_BAND * 100)} % (the measured same-runner noise) is a tie."]
    for n, b in engines:
        out.append(f"- {n} server arguments: " + (f"`{b['args']}`." if b.get("args") else "not recorded."))
    ns = [u if u.startswith("Loom:") else f"{n}: {u}" for n, b in engines for u in (b.get("differs") or [])]
    if ns:
        out.append("- Not supported / not measured: " + "; ".join(ns) + ".")
    return out


def drafting(spec):
    return f"on ({METHOD.get(spec[0], spec[0])}, depth {spec[1]})" if spec else "off"


def feature_detail(key, f):
    st, v = f["status"], f["value"] or {}
    if st == "no":
        return {"sampling": "greedy decoding only"}.get(key, "not supported by Loom (the upstream model has it)")
    if st == "unknown":
        return "not determined"
    if st == "n/a":
        return "not applicable"
    if key == "api":
        return ("OpenAI-style `/v1/chat/completions`" if v.get("style") == "openai"
                else f"`{v.get('endpoint')}`: choice, score and noul decisions")
    if key == "tool_calling":
        return "OpenAI-style `tools` and `tool_calls`"
    if key == "thinking":
        if v.get("mode") == "always":
            return "always on"
        if v.get("default"):
            return 'on by default; `chat_template_kwargs: {"enable_thinking": false}` turns it off'
        return 'off by default; `chat_template_kwargs: {"enable_thinking": true}` turns it on'
    if key == "reasoning_effort":
        d = f" (default {v['default']})" if v.get("default") else ""
        return "`reasoning_effort`: " + ", ".join(v.get("levels") or []) + d
    if key == "speculative_decoding" and not v.get("method"):
        return f"available, disabled ({', '.join(v.get('available') or [])})"
    if key == "speculative_decoding":
        out = f"{METHOD.get(v.get('method'), v.get('method'))}, depth {v.get('depth')}, on by default"
        return out + (f"; drafter {hf(v['drafter'])}" if v.get("drafter") else "")
    if key == "image_input":
        return f"OpenAI-style `image_url` parts; projector `{v.get('projector')}`"
    if key == "sampling":
        return "`temperature`, `top_p`, `top_k`, `seed`"
    if key == "context":
        return f"{count(v['tokens'])} tokens per request"
    if key == "concurrency":
        parts = []
        if v.get("rows"):
            parts.append(f"{v['rows']} concurrent requests")
        if v.get("pending_requests"):
            parts.append(f"{v['pending_requests']} pending requests")
        return ", ".join(parts)
    if key == "max_output_tokens":
        return f"{count(v['max_tokens'])} (preset `max_tokens`)"
    return "supported"


def load(d):
    info = load_info(os.path.join(d, "info.json"))
    require_valid(info)
    facts = package_facts(load_package(d))
    feats = {k: dict(v) for k, v in facts["features"].items()}
    feats = with_gate_features(feats, facts["kind"], all_gates(info))
    for k, v in (info["card"].get("features") or {}).items():
        if k in feats:
            feats[k] = {"status": v["status"], "value": v.get("value")}
    return info, facts, feats


def all_gates(info):
    """Every gate's evidence on the card, the first GPU (card order) that has it: what the package's features need."""
    out = {}
    for g in card_gpus(info):
        for n, row in (info["devices"][g].get("gates") or {}).items():
            out.setdefault(n, row)
    return out


def pct(v, ceiling):
    return f"{round(v / ceiling * 100)} %" if v is not None and ceiling else MISSING


def pct_of(a):
    return f"{round(a * 100)} %"


def row_acceptance(dr, row):
    """(acceptance, "session" | "gate") a drafted roofline row's ceiling counts, or (None, None) without a ceiling.
    Rows written before per-row acceptance used the spec gate's (drafting.acceptance)."""
    if row.get("ceiling") is None:
        return None, None
    if row.get("acceptance") is not None:
        return row["acceptance"], "session" if row.get("acceptance_from") == "session" else "gate"
    return dr.get("acceptance"), "gate"


def drafted_note(dr, metrics):
    """How the drafted rows' ceilings count speculative decoding: per row, the acceptance measured in the row's own session on this
    package; a row without one shows the single-pass ceiling. 4 streams run without drafting (the benchmark rule)."""
    drafted = {k: m for k, m in metrics.items() if k in SHORT and k != "streams4" and m.get("acceptance_from") == "session"}
    single = [k for k in SHORT if k in metrics and k != "streams4" and k not in drafted]
    out = ""
    if drafted:
        out += (f"; decode rows with speculative decoding: verify passes × (1 + acceptance × depth {dr['depth']}) tokens per pass, "
                "the drafter's cost not counted, with the acceptance measured in the row's own session: "
                + ", ".join(f"{SHORT[k]} {pct_of(m['acceptance'])}" for k, m in drafted.items()))
    if single:
        out += ("; " if out else "; ") + "single-pass ceiling (no acceptance measured on this package) for " + ", ".join(SHORT[k] for k in single)
    return out + "; 4 streams run without drafting, so its ceiling is single-pass"


def basis_note(metrics):
    """A mixture-of-experts model's 4-stream ceiling is the bound with the four streams on the same experts."""
    if any(m.get("basis") == "shared-experts" for m in metrics.values()):
        return "; the 4-stream ceiling of this mixture-of-experts model assumes the four streams route to the same experts (the bound a run may not exceed)"
    return ""


def roofline_note(info):
    """The footnote that says what the roofline column is."""
    r = info.get("roofline")
    if r:
        t, d, dr = r["tool"], r["device"], r.get("drafting")
        commit = short(t["skills_commit"]) if t["skills_commit"] != "unknown" else "unknown commit"
        out = (f"Roofline: the hardware ceiling for this GGUF on {d['name']} at {d['bandwidth_bytes_per_s'] / 1e9:.0f} GB/s "
               f"and {d['peak_flops'] / 1e12:.0f} TFLOP/s, from loom-roofline `roofline.py` (loom-skills {commit}"
               + (", with local changes" if t.get("dirty") else "") + ")")
        if dr:
            out += drafted_note(dr, r["metrics"])
        out += basis_note(r["metrics"]) + "."
    else:
        out = "Roofline: not computed."
    return out


def gib(n):
    return f"{n / 2**30:.0f} GiB"


def gpu_title(g):
    """Strix Halo (gfx1151); Radeon AI PRO R9700 (gfx1201, 32 GB)"""
    return f"{GPUS[g]['short']} ({GPUS[g]['arch']}" + (f", {GPUS[g]['memory']}" if GPUS[g]["memory"] else "") + ")"


def gpu_arch(g):
    return GPUS[g]["arch"] + (f", {GPUS[g]['memory']}" if GPUS[g]["memory"] else "")


def gpu_line(info, g):
    """The one factual line of a GPU the card has no results for: not yet re-measured, or the package does not fit."""
    dev = info["devices"][g]
    if dev.get("state") == "unsupported":
        return (f"{GPUS[g]['name']} ({gpu_arch(g)}): this package does not fit its "
                f"{GPUS[g]['memory'] or 'memory'} (it needs about {dev['required_bytes'] / 1e9:.0f} GB).")
    return f"{GPUS[g]['name']} ({gpu_arch(g)}): the results for this package are not yet re-measured."


def single_layout(info):
    """True for a card of one GPU with results (every other GPU, if any, is one the package does not fit): the card
    reads exactly as the cards of one GPU always did. Otherwise each GPU gets its own table and column."""
    gpus = card_gpus(info)
    res = [g for g in gpus if has_results(info["devices"][g])]
    return len(res) == 1 and all(info["devices"][g].get("state") == "unsupported" for g in gpus if g not in res)


def serve_command(info, facts, feats, gpu, output):
    serve = ["loom-serve", f"--model={output}/model.json", "'--device=amdgpu://?completion_spin_ns=4294967295'"]
    pkg_serve = (facts.get("serve") or {}).get("args")   # llm_config.json "serve": flags the package must be served with
    memory = gpu_memory(info, facts["memory_bytes"], gpu)
    if pkg_serve:
        serve += pkg_serve   # e.g. --pool_backing=fixed, which takes no --memory_bytes
    elif memory:
        serve.append(f"--memory_bytes={memory}")
    return " ".join(serve)


def perf_section(L, info, facts, feats, d, g, here, ctx):
    """One GPU's performance table, methodology and footnotes (appended to L). ctx: the tokens per request that GPU serves."""
    kind = facts["kind"]
    view = gpu_view(info, g)
    sources, perf = info["sources"], view.get("performance") or {"baseline": {"name": "llama.cpp", "version": None, "args": None}, "sessions": []}
    base, sessions = perf["baseline"], perf["sessions"]
    bname = base["name"]
    shown = card_metrics(view)
    rows = {k: shown[k] for k, _ in METRICS if k in shown and metric_applies(k, kind, feats, view, ctx)}
    counts = {}
    for k, (_, i) in rows.items():
        counts[i] = counts.get(i, 0) + 1
    primary = min(counts, key=lambda i: (-counts[i], i)) if counts else None
    pkey = (sessions[primary]["source"], sessions[primary]["date"]) if primary is not None else None
    notes = []
    roof = (view.get("roofline") or {}).get("metrics") or {}
    stale = stale_metrics(view, here, kind, feats)
    # further llama.cpp backends (performance.baselines): one column each, after the roofline
    extra = {b: x for b, x in (perf.get("baselines") or {}).items()}
    xnames = {b: f"{x['baseline']['name']} {b.capitalize()}" for b, x in extra.items()}
    L += [f"| metric | {bname} | Loom | margin | roofline | Loom % of roofline "
          + "".join(f"| {n} " for n in xnames.values()) + "|",
          "|---|---|---|---|---|---|" + "---|" * len(extra)]

    def marked(key, v, i, stale_keys):
        """A value with its session's footnote mark (sessions other than the primary one) and the stale mark."""
        out = num(key, v)
        if i is not None and (sessions[i]["source"], sessions[i]["date"]) != pkey:
            nk = (sessions[i]["source"], sessions[i]["date"])
            if nk not in notes:
                notes.append(nk)
            out += f" {mark(notes.index(nk) + 1)}"
        return out + (f" {STALE}" if key in stale_keys else "")
    above = []
    for key, label in METRICS:
        if not metric_applies(key, kind, feats, view, ctx):
            cell = FIT if beyond_context(view, key, ctx, native_context(feats)) else "n/a"
            L.append(f"| {label} |" + f" {cell} |" * (5 + len(extra)))
            continue
        m, i = shown.get(key, ({"loom": None, "baseline": None}, None))
        loom = marked(key, m.get("loom"), i, stale)
        ceiling = (roof.get(key) or {}).get("ceiling")
        if key == "spec_acceptance":
            cells = " | ".join(["n/a"] * (2 + len(extra)))
        else:
            over100 = m.get("loom") is not None and ceiling and m["loom"] > ceiling
            if over100 and ABOVE not in above:
                above.append(ABOVE)
            cells = " | ".join([num(key, ceiling), pct(m.get('loom'), ceiling) + (f" {ABOVE}" if over100 else "")]
                               + [num(key, x['session']['metrics'].get(key, {}).get('baseline')) for x in extra.values()])
        L.append(f"| {label} | {num(key, m.get('baseline'))} | {loom} | "
                 f"{MISSING if key == 'spec_acceptance' else margin(m.get('loom'), m.get('baseline'))} | {cells} |")

    def runs_of(keys, med):
        vals = set()
        for k in keys:
            if k in rows:
                m, i = rows[k]
                n = len(m.get("loom_runs") or []) or sessions[i]["runs"]["streams4" if med else "decode_prefill"]
                if n:
                    vals.add(n)
        return "–".join(str(v) for v in sorted(vals)) if vals else "not recorded"
    tps_keys = [k for k, _ in METRICS if k not in ("streams4", "spec_acceptance")]
    psrc = sources[sessions[primary]["source"]] if primary is not None else None
    drafter = None
    if facts["drafter"]:
        drafter = f"{facts['drafter']['repo']} / {facts['drafter']['file']}"
    L += ["", "**Methodology**", "", *methodology(base, extra, facts["spec"], drafter, runs_of(tps_keys, False), runs_of(["streams4"], True)),
          f"- Measured on AMD DevLab Dispatch {GPUS[g]['runner']} runners (`devlab-perf` jobs); runners differ between sessions."]
    if psrc:
        L.append(f"- Measured {text(sessions[primary]['date'])}; device {device_text(psrc['device'])}; "
                 f"hrx-loom {sdk_text(psrc['hrx_loom'])}; source: {src_text(psrc)}.")
    for i, (sid, date) in enumerate(notes, 1):
        s = sources[sid]
        L.append(f"- {mark(i)} Measured {text(date)}; device {device_text(s['device'])}; hrx-loom {sdk_text(s['hrx_loom'])}; "
                 f"source: {src_text(s)}.")
    for b, x in extra.items():
        xb, xs = x["baseline"], x["session"]
        L.append(f"- {xnames[b]}: measured {text(xs['date'])}; source: {src_text(sources[xs['source']])}."
                 + (f" Archive sha256 {short(xb['sha256'])}." if xb.get("sha256") else ""))
    for sha in sorted(set(stale.values())):
        L.append(f"- {STALE} Measured on programs other than this repository's (sha256 {short(sha)}, this repository's "
                 f"{short(here)}): the value describes another build of the package until a session re-measures it.")
    if metric_applies("spec_acceptance", kind, feats):
        L.append("- Drafted tokens accepted: the share of the drafter's proposed tokens the model accepted during the "
                 "`spec` gate's requests.")
    if above:
        bw = ((view.get("roofline") or {}).get("device") or {}).get("bandwidth_bytes_per_s") or 216e9
        L.append(f"- {ABOVE} measured above the modeled ceiling; the roofline is an estimate ({bw / 1e9:.0f} GB/s effective bandwidth).")
    L.append("- " + roofline_note(view))
    native = native_context(feats)
    nofit_keys = [k for k in METRIC_KEYS if beyond_context(view, k, ctx, native)]
    nofit = ", ".join(nofit_keys)
    oc = ", ".join(sorted(over_context(view, ctx) - set(nofit_keys)))
    served = ctx != native
    if nofit:   # the user's rule: what does not fit this GPU's memory is not expected to be done
        L.append(f"- {FIT} ({nofit}): the document is longer than the {ctx:,}-token context this GPU serves; not expected "
                 "to be measured, not a failure.")
    L += [f"- {MISSING} not measured; n/a not applicable" + (
        f" ({oc}: the document is longer than the " + (f"{ctx:,}-token context this GPU serves" if served else
                                                       f"model's native {ctx}-token context") + ")" if oc else "") + ".", ""]


def render(d):
    info, facts, feats = load(d)
    c, card, kind = facts["checkpoint"], info["card"], facts["kind"]
    sources, versions = info["sources"], info["versions"]
    gpus = card_gpus(info)
    solo = single_layout(info)
    live = [g for g in gpus if has_results(info["devices"][g])]
    named = [g for g in gpus if info["devices"][g].get("state") != "unsupported"]
    output = card.get("output") or facts["output"] or facts["name"]
    native = native_context(feats)

    L = ["---", f"base_model: {c['repo']}"]
    if card.get("license"):
        L.append(f"license: {card['license']}")
    tags = derived_tags(feats, named) + [t for t in card.get("tags") or [] if t not in derived_tags(feats, named)]
    L += ["tags:", *[f"- {t}" for t in tags], "---", ""]

    quant = f" ({c['quant']})" if c["quant"] else ""
    where = " and ".join(GPUS[g]["name"] for g in named) if named else GPUS[DEFAULT_GPU]["name"]
    L += [f"# {facts['name']}{quant} for AMD Loom", "",
          f"A [Loom]({LOOM_URL}) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint "
          f"{hf(c['repo'])} (`{c['file']}`), generated on an {where} with the "
          f"[loom-skills]({SKILLS_URL}) workflow. {LEARN_MORE}" if solo or len(named) < 2 else
          f"A [Loom]({LOOM_URL}) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint "
          f"{hf(c['repo'])} (`{c['file']}`), for the {where} GPUs, generated with the "
          f"[loom-skills]({SKILLS_URL}) workflow. {LEARN_MORE}", ""]

    for g, dev in info["devices"].items():   # a GPU's model document is a file of the package that differs from model.json only in execution
        md = dev.get("model_document")
        if md:
            path = os.path.join(d, md["file"])
            if not os.path.isfile(path) or not os.path.isfile(path + ".lock"):
                sys.exit(f"{g}: info.json names {md['file']} but {path} or its .lock is missing (make_info.py build --fit writes both)")
            errs = gpu_document_errors(load_json(os.path.join(d, "model.json")), load_json(path))
            if errs or any(load_json(path)["presets"][n]["execution"].get(k) != v for n in load_json(path)["presets"] for k, v in md["execution"].items()):
                sys.exit(f"{md['file']}: " + "; ".join(errs or ["its execution settings are not info.json's"]))

    def fetch_for(g=None):   # the loom-fetch line: model.json, or the GPU's own document when the package has one
        doc = ((info["devices"].get(g) or {}).get("model_document") or {}).get("file") if g else None
        f = ["loom-fetch", f"{HF}/{info['repo']}/resolve/main/{doc or 'model.json'}"]
        if facts["preset"]:
            f.append(f"--preset={facts['preset']}")
        return " ".join(f + ["--materialize=copy", f"--output={output}"])
    api = ("answers OpenAI-style `/v1/chat/completions` requests" if kind == "chat"
           else f"answers SystemOne decision requests at `{feats['api']['value']['endpoint']}`")
    pin = required_sdk(facts, versions, sources)
    if pin and pin.get("source") == "ci":
        root = f"~/loom-{pin['artifact_id']}"
        install = [f"git clone {SKILLS_URL}",
                   f"loom-skills/skills/loom-setup/scripts/install_loom.sh {root} ci:{pin['artifact_id']}",
                   f"source {root}/.venv/bin/activate"]
    elif pin and pin.get("hrx_loom"):
        install = [f'pip install "hrx-loom[hf]=={pin["hrx_loom"]}"']
    else:
        install = ['pip install "hrx-loom[hf]"']   # no SDK version recorded anywhere (cannot happen for a built card)
    serving = [g for g in gpus if info["devices"][g].get("state") != "unsupported"] or [DEFAULT_GPU]
    own_doc = any((info["devices"].get(g) or {}).get("model_document") for g in serving)
    if len(serving) == 1:
        L += ["## Quick start", "", "```bash", *install, fetch_for(serving[0]), serve_command(info, facts, feats, serving[0], output), "```", ""]
    elif own_doc:   # a GPU with its own model document fetches it: the fetch line is per GPU too
        L += ["## Quick start", "", "```bash", *install, "```", ""]
        for g in serving:
            L += [f"Fetch and serve on the {GPUS[g]['name']} ({gpu_arch(g)}):", "", "```bash", fetch_for(g),
                  serve_command(info, facts, feats, g, output), "```", ""]
    else:   # the same install and fetch; the serve line differs by the GPU's memory
        L += ["## Quick start", "", "```bash", *install, fetch_for(), "```", ""]
        for g in serving:
            L += [f"Serve on the {GPUS[g]['name']} ({gpu_arch(g)}):", "", "```bash",
                  serve_command(info, facts, feats, g, output), "```", ""]
    if pin and pin.get("source") == "ci":
        sha = (pin.get("hrx_system_sha") or "")[:8]
        L += [f"This package needs an hrx-loom development build (hrx-loom {pin.get('hrx_loom')}, ROCm/hrx-system "
              f"`{sha}`, CI artifact `{pin['artifact_id']}`); the PyPI release does not run it yet. `install_loom.sh` "
              "downloads that wheel with an authenticated GitHub CLI (`gh auth login`). "
              f"**These instructions expire on {pin.get('expires') or 'the artifact expiry date'}**, when GitHub "
              "deletes the CI artifact; after that the package's SDK pin has to be refreshed to a newer build "
              "(or a PyPI release that has the features), and this card republished.", ""]
    L += [f"The server listens on `127.0.0.1:8080` and {api}.", ""]
    for g in serving:   # a GPU whose memory holds less KV cache than the native context needs serves a shorter context
        sc = info["devices"][g].get("served_context")
        if sc:
            L += [f"On the {GPUS[g]['name']}, `--memory_bytes={gpu_memory(info, facts['memory_bytes'], g)}` bounds the memory pool: "
                  f"this package serves up to {sc:,} tokens per request there (its native context is {native:,}); "
                  + ("the programs and weights are the same on every GPU." if info["devices"][g].get("model_document") else "the package files are the same on every GPU."), ""]
    for g in serving:   # a GPU's own model document: what it changes (the same programs, weights and tokenizer)
        md = info["devices"][g].get("model_document")
        if md:
            L += [f"On the {GPUS[g]['name']}, the Quick start fetches `{md['file']}`: `model.json` with these execution settings for "
                  f"its {GPUS[g]['memory'] or 'memory'}: " + ", ".join(f"`{k}` {v:,}" if isinstance(v, int) and not isinstance(v, bool) else f"`{k}` {v}"
                                                                   for k, v in sorted(md["execution"].items())) + ". "
                  "The programs, weights and tokenizer are the same.", ""]
    for g in gpus:
        if info["devices"][g].get("state") == "unsupported":
            L += [gpu_line(info, g), ""]

    # ---- performance
    here = programs_sha256(d)
    first = next(iter(live), None)
    bname = ((info["devices"][first].get("performance") or {}).get("baseline") or {}).get("name", "llama.cpp") if first else "llama.cpp"
    L += [f"## Performance vs {bname}", ""]
    if solo:
        perf_section(L, info, facts, feats, d, live[0], here, gpu_context(info, live[0], native))
    else:
        for g in gpus:
            L += [f"### {gpu_title(g)}", ""]
            if has_results(info["devices"][g]):
                perf_section(L, info, facts, feats, d, g, here, gpu_context(info, g, native))
            else:
                L += [gpu_line(info, g), ""]

    # ---- features
    L += ["## Features", "", "| feature | status | detail |", "|---|---|---|"]
    for key, label in FEATURES:
        f = feats[key]
        st = {"yes": YES, "no": NO, "unknown": UNKNOWN, "n/a": NA}[f["status"]]
        detail = feature_detail(key, f)
        if key == "context" and not solo:
            cut = [f"{GPUS[g]['short']} up to {info['devices'][g]['served_context']:,} (its memory)" for g in gpus
                   if info["devices"][g].get("served_context")]
            if cut:
                detail += "; " + "; ".join(cut)
        L.append(f"| {label} | {st} | {detail} |")
    L.append("")

    # ---- correctness
    cols = [live[0]] if solo else gpus
    head = "| gate | description | result |" if solo else "| gate | description | " + " | ".join(GPUS[g]["short"] for g in cols) + " |"
    L += ["## Correctness", "", head, "|---|---|" + "---|" * len(cols)]
    ev = []

    def gate_cell(g, name, desc):
        dev = info["devices"][g]
        if not has_results(dev):
            return MISSING + (" not yet re-measured" if dev.get("state") == "not_measured" else " does not fit")
        gt = dev["gates"].get(name)
        if gt is None and beyond_context(gpu_view(info, g), name, gpu_context(info, g, native), native):
            return f"{FIT} (needs more context than the {gpu_context(info, g, native):,} tokens this GPU serves)"
        if gt is None:
            return f"{NO} no evidence" if gate_applies(name, kind, feats) else f"{NA} not applicable"
        key = (gt["source"], gt["level"], gt["date"])
        gsha = gt.get("programs_sha256")
        if gsha and gsha != here and gt["status"] == "pass":   # a pass on other programs says nothing about this package
            return (f"{UNKNOWN} not re-run on this package (the last pass was on a previous build, sha256 {short(gsha)}; "
                    f"the nightly gates re-run it)")
        if gsha and gsha != here:   # a failure is never hidden
            if key not in ev:
                ev.append(key)
            return f"{NO} fail on a previous build (sha256 {short(gsha)}), not re-run on this package {mark(ev.index(key) + 1)}"
        if key not in ev:
            ev.append(key)
        res = (f"{YES} pass" if gt["status"] == "pass" else f"{NO} fail")
        return res + (f" ({gt['detail']})" if gt["detail"] else "") + f" {mark(ev.index(key) + 1)}"
    for name, desc in GATES:
        L.append(f"| `{name}` | {desc} | " + " | ".join(gate_cell(g, name, desc) for g in cols) + " |")
    L.append("")
    for i, (sid, level, date) in enumerate(ev, 1):
        s = sources[sid]
        lv = f", LEVEL={level}" if level else ""
        L.append(f"- {mark(i)} {src_text(s)}{lv}, {text(date)}; device {device_text(s['device'])}; "
                 f"hrx-loom {sdk_text(s['hrx_loom'])}.")
    L += ["- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.", ""]

    # ---- files
    L += ["## Files", "",
          "- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).",
          *([f"- `{md['file']}` / `{md['file']}.lock`: {GPUS[g]['short']}'s own card: `model.json` with that GPU's execution settings."
             for g in gpus for md in [info["devices"][g].get("model_document")] if md]),
          "- `program/*.loom`, `weights.loom`: the generated Loom programs.",
          "- `tokenizer.json`: the tokenizer.",
          "- `llm_config.json`: the configuration the programs were generated from.",
          "- `info.json`: the measurements, gate evidence and versions this card shows; everything else comes from the "
          "files above."]
    if info.get("report") == "generated":
        L.append("- `REPORT.md`: how the values above were measured: setup, commands and the value of every run.")
    elif info.get("report") == "attached":
        L.append("- `REPORT.md`: the report written when this package was made.")
    L += ["", f"Versions: card schema `{SCHEMA_ID}`; loom-skills {skills_text(facts['generator'], versions)}; "
              f"hrx-loom {package_sdk(facts['generator'], versions)}; card generator "
              f"{short(versions['card_generator']) if versions['card_generator'] != 'unknown' else 'unknown'}.", "",
          "Source files, fetched by `loom-fetch` from their own repositories:", ""]
    for role, x in (("checkpoint", c), ("drafter", facts["drafter"]), ("projector", facts["projector"])):
        if x:
            rev = f" @ `{x['revision'][:7]}`" if x["revision"] else ""
            L.append(f"- {role}: {hf(x['repo'])}{rev}, `{x['file']}`")
    paused, steps = info.get("status") == "paused", [f"- {x}" for x in info.get("next_steps") or []]
    if paused or steps:   # last: what would be done if work continued
        L += ["", "## Next steps", ""]
        if paused:
            L += ["Work on this model is paused."] + ([""] if steps else [])
        L += steps
    return "\n".join(L) + "\n"


def method_text(gpus):
    """The report's method bullets; the runner is the one GPU's (the same text a one-GPU card always had)."""
    runner = " or ".join(GPUS[g]["runner"] for g in gpus) if gpus else GPUS[DEFAULT_GPU]["runner"]
    return [
    "- `perf.py` (loom-benchmark) sends real-text requests: a short prompt for decode, decode after 2k and 32k tokens "
    "of context, and prefill of documents of about 1k, 2k, 8k, 16k and 32k tokens. Each run gives one value per "
    "metric; the card shows the mean of the runs.",
    "- `c4_steady.py` runs 4 concurrent streams and reports the aggregate decode rate over the window in which all "
    "4 are decoding; the card shows the median of the runs.",
    "- One engine runs at a time on the same machine, with the same GGUF. Loom rates are timed by the client; "
    "llama.cpp rates come from its server-side `timings`.",
    f"- Every performance value comes from a `devlab-perf` job (loom-skills) on an AMD DevLab Dispatch {runner} runner: "
    "Loom, served with the active wait (`--device=amdgpu://?completion_spin_ns=4294967295`), and llama.cpp on the same "
    "runner, with the published programs.",
    "- Drafted tokens accepted is the `spec` gate's count: accepted over proposed draft tokens across its requests.",
    "- Gates are loom-benchmark's `gates.sh` at the LEVEL shown; DevLab runs are the nightly-gates workflow on AMD "
    "DevLab Dispatch.",
    "- Roofline: per metric, the rate at the hardware limit computed from the GGUF's tensors by loom-roofline "
    "`roofline.py`: decode is memory bandwidth / bytes read per token (weights, active experts, KV cache), prefill is "
    "peak FLOP/s / flops per token, 4 streams reads the weights once for 4 tokens. With speculative decoding, decode "
    "and 4 streams are verify passes per second x (1 + acceptance x depth) tokens per pass, with the acceptance measured "
    "in the row's own session (loom-serve's drafted-token counters around perf.py's decode request, or around "
    "c4_steady.py's streams), else the `spec` gate's.",
    ]


def report_gpu(L, info, facts, g, here, h, single):
    """One GPU's part of REPORT.md: baseline, sessions, further baselines, roofline and gate evidence, under headings
    of level h ("##" for a one-GPU card, "###" under the GPU's own heading)."""
    sources = info["sources"]
    view = gpu_view(info, g)
    perf = view.get("performance") or {"baseline": {"name": "llama.cpp", "version": None, "args": None}, "sessions": []}
    base = perf["baseline"]

    def prog(sha):
        if not sha:
            return "not recorded"
        rel = "" if here is None else (", the same as this repository's programs" if sha == here
                                       else ", not the same as this repository's programs")
        return f"sha256 `{sha}`{rel}"
    labels = dict(METRICS)
    for i, s in enumerate(perf["sessions"], 1):
        src = sources[s["source"]]
        L += [f"{h} Session {i}: {src_text(src)}, {text(s['date'])}", "",
              f"- Device: {device_text(src['device'])}.",
              f"- Loom SDK: hrx-loom {sdk_text(src['hrx_loom'])}.",
              f"- Programs measured: {prog(s.get('programs_sha256'))}."]
        if s.get("commands"):
            L += ["- Commands:", "", "```bash", *s["commands"], "```"]
        L += ["", "| metric | engine | runs | value |", "|---|---|---|---|"]
        order = [x for x, _ in METRICS]
        for k, m in sorted(s["metrics"].items(), key=lambda kv: (order.index(kv[0]) if kv[0] in labels else len(order), kv[0])):
            for eng, v, runs in (("Loom", m.get("loom"), m.get("loom_runs")), (base["name"], m.get("baseline"), m.get("baseline_runs"))):
                if v is None and not runs:
                    continue
                r = ", ".join(str(x) for x in runs) if runs else MISSING
                acc = (f" ({pct_of(m['acceptance'])} of drafted tokens accepted)"
                       if eng == "Loom" and m.get("acceptance") is not None else "")
                L.append(f"| {labels.get(k, k)} | {eng} | {r} | {num(k, v)}{acc} |")
        L.append("")
    for b, x in (perf.get("baselines") or {}).items():
        xs, xb = x["session"], x["baseline"]
        src = sources[xs["source"]]
        L += [f"{h} {xb['name']} {b.capitalize()}: {src_text(src)}, {text(xs['date'])}", "",
              f"- Build: {('`' + xb['version'] + '`') if xb['version'] else 'not recorded'}; server arguments "
              + (f"`{xb['args']}`" if xb["args"] else "not recorded") + ".",
              f"- Device: {device_text(src['device'])}."]
        L += [f"- Differs from {base['name']}: {u}." for u in xb.get("differs") or []]
        L += ["", "| metric | runs | value |", "|---|---|---|"]
        order = [k for k, _ in METRICS]
        for k, m in sorted(xs["metrics"].items(), key=lambda kv: (order.index(kv[0]) if kv[0] in labels else len(order), kv[0])):
            L.append(f"| {labels.get(k, k)} | {', '.join(str(v) for v in m.get('baseline_runs') or []) or MISSING} | {num(k, m.get('baseline'))} |")
        L.append("")
    r = view.get("roofline")
    if r:
        t, d, dr = r["tool"], r["device"], r.get("drafting")
        L += [f"{h} Roofline", "",
              f"- Device: {d['name']}, {d['bandwidth_bytes_per_s'] / 1e9:.0f} GB/s, {d['peak_flops'] / 1e12:.0f} TFLOP/s.",
              f"- Tool: `{t['script']}` sha256 `{t['sha256']}`, loom-skills "
              + (f"`{t['skills_commit']}`" if t["skills_commit"] != "unknown" else "commit unknown")
              + (" (with local changes)" if t.get("dirty") else "") + ".",
              f"- Checkpoint: `{text(r.get('checkpoint'))}`."]
        if dr:
            L.append(f"- Speculative decoding: {METHOD.get(dr['method'], dr['method'])}, depth {dr['depth']}; a drafted row's "
                     "ceiling counts the acceptance measured in its own session on this package; a row without one shows the single-pass "
                     "ceiling; 4 streams run without drafting.")
        L += ["", "| metric | ceiling | without drafting | " + ("acceptance | " if dr else "") + "demonstrated efficiency |",
              "|---|---|---|" + ("---|" if dr else "") + "---|"]
        for k, row in sorted(r["metrics"].items(), key=lambda kv: [x for x, _ in METRICS].index(kv[0])
                             if kv[0] in labels else len(METRICS)):
            if k in labels:
                acc = ""
                if dr:
                    a_k, how = row_acceptance(dr, row) if "single_pass" in row else (None, None)
                    acc = (f"{pct_of(a_k)} ({'session' if how == 'session' else '`spec` gate'})" if how else MISSING) + " | "
                L.append(f"| {labels[k]} | {num(k, row.get('ceiling'))} | {num(k, row.get('single_pass'))} | {acc}"
                         + (f"{round(row['efficiency'] * 100)} %" if row.get("efficiency") is not None else MISSING) + " |")
        L.append("")
    L += [f"{h} Gate evidence", ""]
    groups = []
    gates = view["gates"]
    for name in GATE_NAMES + sorted(set(gates) - set(GATE_NAMES)):
        gt = gates.get(name)
        if not gt:
            continue
        key = (gt["source"], gt["level"], gt["date"], gt.get("programs_sha256"))
        if key not in [k for k, _ in groups]:
            groups.append((key, []))
        dict(groups)[key].append(f"`{name}` {gt['status']}" + (f" ({gt['detail']})" if gt["detail"] else ""))
    for (sid, level, date, sha), names in groups:
        s = sources[sid]
        L += [f"{'###' if h == '##' else '####'} {src_text(s)}" + (f", LEVEL={level}" if level else "") + f", {text(date)}", "",
              f"- Gates: {', '.join(names)}.",
              f"- Device: {device_text(s['device'])}.",
              f"- Loom SDK: hrx-loom {sdk_text(s['hrx_loom'])}.",
              f"- Programs: {prog(sha)}.", ""]
    if not groups:
        L += ["No gate evidence is recorded.", ""]


def render_report(d):
    info, facts, feats = load(d)
    versions = info["versions"]
    c = facts["checkpoint"]
    here = programs_sha256(d)
    gpus = card_gpus(info)
    live = [g for g in gpus if has_results(info["devices"][g])]
    solo = single_layout(info)
    base_line = lambda b: (f"- Baseline: {b['name']} " + (f"`{b['version']}`" if b["version"] else "(build not recorded)")
                           + ", server arguments " + (f"`{b['args']}`" if b["args"] else "not recorded")
                           + " (model path, host and port left out).")
    L = [f"# {facts['name']} for AMD Loom: measurement report", "",
         f"How the values on this repository's model card were measured. Generated from `info.json` (schema "
         f"`{SCHEMA_ID}`) by the loom-skills card generator; the card's README.md is rendered from the same data.", "",
         "## Setup", "",
         f"- Package: this repository. Programs generated by loom-skills {skills_text(facts['generator'], versions)} with "
         f"hrx-loom {package_sdk(facts['generator'], versions)}.",
         f"- Checkpoint: {hf(c['repo'])}" + (f" @ `{c['revision']}`" if c["revision"] else "") + f", `{c['file']}`"
         + (f", sha256 `{c['sha256']}`" if c["sha256"] else "") + "."]
    if facts["drafter"]:
        x = facts["drafter"]
        L.append(f"- Drafter: {hf(x['repo'])}" + (f" @ `{x['revision']}`" if x["revision"] else "") + f", `{x['file']}`; "
                 f"Loom speculative decoding {drafting(facts['spec'])}.")
    if solo:
        perf = gpu_view(info, live[0]).get("performance") or {"baseline": {"name": "llama.cpp", "version": None, "args": None}}
        L.append(base_line(perf["baseline"]))
    L += ["- Programs measured: the sha256 below is card_lib.programs_sha256, over `program/*.loom` without each file's "
          "first-line generator comment.", "",
          "## Method", "", *method_text(live if not solo else live), ""]
    if solo:
        report_gpu(L, info, facts, live[0], here, "##", True)
    else:
        for g in gpus:
            L += [f"## {GPUS[g]['name']} ({gpu_arch(g)})", ""]
            if not has_results(info["devices"][g]):
                L += [gpu_line(info, g), ""]
                continue
            perf = gpu_view(info, g).get("performance") or {"baseline": {"name": "llama.cpp", "version": None, "args": None}}
            L += [base_line(perf["baseline"]), ""]
            report_gpu(L, info, facts, g, here, "###", False)
    while L[-1] == "":
        L.pop()
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("readme", nargs="?", help="with --check: the README to compare")
    ap.add_argument("-o")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    out = render(a.dir)
    info = load_json(os.path.join(a.dir, "info.json"))
    if a.check:
        checks = [(a.readme, out)]
        if info.get("report") == "generated":
            checks.append((os.path.join(os.path.dirname(os.path.abspath(a.readme)), "REPORT.md"), render_report(a.dir)))
        bad = 0
        for path, want in checks:
            with open(path, encoding="utf-8", newline="") as f:
                same = f.read() == want
            print(("OK   " if same else "DIFF ") + path)
            bad |= not same
        sys.exit(bad)
    if info.get("report") == "attached" and not os.path.isfile(os.path.join(a.dir, "REPORT.md")):
        sys.exit(f"info.json says REPORT.md is attached, but {a.dir}/REPORT.md is missing")
    dest = a.o or os.path.join(a.dir, "README.md")
    if dest == "-":
        sys.stdout.write(out)
    else:
        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        print(f"wrote {dest}", file=sys.stderr)
    if info.get("report") == "generated":
        rp = os.path.join(a.dir, "REPORT.md") if dest == "-" else os.path.join(os.path.dirname(os.path.abspath(dest)), "REPORT.md")
        with open(rp, "w", encoding="utf-8", newline="\n") as f:
            f.write(render_report(a.dir))
        print(f"wrote {rp}", file=sys.stderr)


if __name__ == "__main__":
    main()
