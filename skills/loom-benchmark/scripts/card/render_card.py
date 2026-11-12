#!/usr/bin/env python3
"""Render a loom-fun card's README.md (and a generated REPORT.md) from the files of its repository (schema loom-card/2).

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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_lib import (FEATURES, GATE_NAMES, GATES, HF, LOOM_URL, METHOD, METRICS, SCHEMA_ID, SKILLS_URL, TIMED,  # noqa: E402
                      card_metrics, derived_tags, gate_applies, generator_commit, load_json,
                      load_package, metric_applies, native_context, over_context, package_facts, programs_sha256,
                      require_valid, stale_metrics, with_gate_features)

YES, NO, NA, MISSING = "✅", "❌", "—", "—"
UNKNOWN = "?"   # a feature that could not be determined: neither supported, nor n/a, nor a failure
SUP = "¹²³⁴⁵⁶⁷⁸⁹"
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
    info = load_json(os.path.join(d, "info.json"))
    require_valid(info)
    facts = package_facts(load_package(d))
    feats = {k: dict(v) for k, v in facts["features"].items()}
    feats = with_gate_features(feats, facts["kind"], info["gates"])
    for k, v in (info["card"].get("features") or {}).items():
        if k in feats:
            feats[k] = {"status": v["status"], "value": v.get("value")}
    return info, facts, feats


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
    """How the drafted rows' ceilings count speculative decoding: per row, the acceptance measured in the row's own
    session (perf.py, c4_steady.py), else the spec gate's (another text: it can accept more or less than the row's)."""
    rows = {k: row_acceptance(dr, metrics[k]) for k in SHORT if "single_pass" in metrics.get(k, {})}
    session = [k for k, (_, how) in rows.items() if how == "session"]
    gate = [k for k, (_, how) in rows.items() if how == "gate"]
    if not session and not gate:
        return "; decode and 4 streams with speculative decoding have no ceiling until acceptance is measured"
    out = (f"; decode and 4 streams with speculative decoding: verify passes × (1 + acceptance × depth {dr['depth']}) "
           "tokens per pass, the drafter's cost not counted")
    if session:
        out += ("; acceptance measured in the row's own session: "
                + ", ".join(f"{SHORT[k]} {pct_of(rows[k][0])}" for k in session))
    if gate:
        acc = rows[gate[0]][0]
        out += (f"; {'for ' + ', '.join(SHORT[k] for k in gate) + ', ' if session else ''}the `spec` gate's acceptance "
                f"{pct_of(acc)} (× {round(1 + acc * dr['depth'], 3):g} tokens per pass), as the row's session recorded "
                "none: the gate's text can accept more or less than the row's")
    return out


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
        out += "."
    else:
        out = "Roofline: not computed."
    return out


def render(d):
    info, facts, feats = load(d)
    c, card, kind = facts["checkpoint"], info["card"], facts["kind"]
    sources, versions = info["sources"], info["versions"]
    perf = info.get("performance") or {"baseline": {"name": "llama.cpp", "version": None, "args": None}, "sessions": []}
    base, sessions = perf["baseline"], perf["sessions"]
    output = card.get("output") or facts["output"] or facts["name"]
    memory = facts["memory_bytes"] or card.get("memory_bytes")

    L = ["---", f"base_model: {c['repo']}"]
    if card.get("license"):
        L.append(f"license: {card['license']}")
    tags = derived_tags(feats) + [t for t in card.get("tags") or [] if t not in derived_tags(feats)]
    L += ["tags:", *[f"- {t}" for t in tags], "---", ""]

    quant = f" ({c['quant']})" if c["quant"] else ""
    L += [f"# {facts['name']}{quant} for AMD Loom", "",
          f"A [Loom]({LOOM_URL}) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint "
          f"{hf(c['repo'])} (`{c['file']}`), generated on an AMD Ryzen AI Max+ 395 \"Strix Halo\" with the "
          f"[loom-skills]({SKILLS_URL}) workflow. {LEARN_MORE}", ""]

    fetch = ["loom-fetch", f"{HF}/{info['repo']}/resolve/main/model.json"]
    if facts["preset"]:
        fetch.append(f"--preset={facts['preset']}")
    fetch += ["--materialize=copy", f"--output={output}"]
    serve = ["loom-serve", f"--model={output}/model.json", "'--device=amdgpu://?completion_spin_ns=4294967295'"]
    pkg_serve = (facts.get("serve") or {}).get("args")   # llm_config.json "serve": flags the package must be served with
    if pkg_serve:
        serve += pkg_serve   # e.g. --pool_backing=fixed, which takes no --memory_bytes
    elif memory:
        serve.append(f"--memory_bytes={memory}")
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
    L += ["## Quick start", "", "```bash", *install, " ".join(fetch), " ".join(serve), "```", ""]
    if pin and pin.get("source") == "ci":
        sha = (pin.get("hrx_system_sha") or "")[:8]
        L += [f"This package needs an hrx-loom development build (hrx-loom {pin.get('hrx_loom')}, ROCm/hrx-system "
              f"`{sha}`, CI artifact `{pin['artifact_id']}`); the PyPI release does not run it yet. `install_loom.sh` "
              "downloads that wheel with an authenticated GitHub CLI (`gh auth login`). "
              f"**These instructions expire on {pin.get('expires') or 'the artifact expiry date'}**, when GitHub "
              "deletes the CI artifact; after that the package's SDK pin has to be refreshed to a newer build "
              "(or a PyPI release that has the features), and this card republished.", ""]
    L += [f"The server listens on `127.0.0.1:8080` and {api}.", ""]

    # ---- performance
    bname = base["name"]
    shown = card_metrics(info)
    rows = {k: shown[k] for k, _ in METRICS if k in shown and metric_applies(k, kind, feats, info)}
    counts = {}
    for k, (_, i) in rows.items():
        counts[i] = counts.get(i, 0) + 1
    primary = min(counts, key=lambda i: (-counts[i], i)) if counts else None
    pkey = (sessions[primary]["source"], sessions[primary]["date"]) if primary is not None else None
    notes = []
    roof = (info.get("roofline") or {}).get("metrics") or {}
    here = programs_sha256(d)
    stale = stale_metrics(info, here, kind, feats)
    # further llama.cpp backends (performance.baselines): one column each, after the roofline
    extra = {b: x for b, x in (perf.get("baselines") or {}).items()}
    xnames = {b: f"{x['baseline']['name']} {b.capitalize()}" for b, x in extra.items()}
    L += [f"## Performance vs {bname}", "",
          f"| metric | {bname} | Loom | margin | roofline | Loom % of roofline "
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
    for key, label in METRICS:
        if not metric_applies(key, kind, feats, info):
            L.append(f"| {label} |" + " n/a |" * (5 + len(extra)))
            continue
        m, i = shown.get(key, ({"loom": None, "baseline": None}, None))
        loom = marked(key, m.get("loom"), i, stale)
        ceiling = (roof.get(key) or {}).get("ceiling")
        if key == "spec_acceptance":
            cells = " | ".join(["n/a"] * (2 + len(extra)))
        else:
            cells = " | ".join([num(key, ceiling), pct(m.get('loom'), ceiling)]
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
    L += ["", f"- {bname}: build {('`' + base['version'] + '`') if base['version'] else 'not recorded'}, server arguments "
              + (f"`{base['args']}`." if base["args"] else "not recorded."),
          f"- Loom: speculative decoding {drafting(facts['spec'])}.",
          "- Same GGUF on both engines, tokens per second. Loom: client-side timing; "
          f"{bname}: server-side timings. Runs per value: decode and prefill {runs_of(tps_keys, False)} (mean), "
          f"4 streams {runs_of(['streams4'], True)} (median).",
          "- Measured on AMD DevLab Dispatch Strix Halo runners (`devlab-perf` jobs): Loom, served with the active wait, "
          "and llama.cpp on the same runner; runners differ between sessions."]
    if psrc:
        L.append(f"- Measured {text(sessions[primary]['date'])}; device {device_text(psrc['device'])}; "
                 f"hrx-loom {sdk_text(psrc['hrx_loom'])}; source: {src_text(psrc)}.")
    for i, (sid, date) in enumerate(notes, 1):
        s = sources[sid]
        L.append(f"- {mark(i)} Measured {text(date)}; device {device_text(s['device'])}; hrx-loom {sdk_text(s['hrx_loom'])}; "
                 f"source: {src_text(s)}.")
    for b, x in extra.items():
        xb, xs = x["baseline"], x["session"]
        L.append(f"- {xnames[b]}: build {('`' + xb['version'] + '`') if xb['version'] else 'not recorded'}"
                 + (f" (archive sha256 {short(xb['sha256'])})" if xb.get("sha256") else "")
                 + ", server arguments " + (f"`{xb['args']}`" if xb["args"] else "not recorded")
                 + f"; same GGUF, prompts and settings as {bname}; measured {text(xs['date'])} in its own DevLab job "
                 f"(another runner than the other columns: compare the engines through ratios, not across jobs); "
                 f"source: {src_text(sources[xs['source']])}."
                 + "".join(f" Differs from {bname}: {u}." for u in xb.get("differs") or []))
    for sha in sorted(set(stale.values())):
        L.append(f"- {STALE} Measured on programs other than this repository's (sha256 {short(sha)}, this repository's "
                 f"{short(here)}): the value describes another build of the package until a session re-measures it.")
    if metric_applies("spec_acceptance", kind, feats):
        L.append("- Drafted tokens accepted: the share of the drafter's proposed tokens the model accepted during the "
                 "`spec` gate's requests.")
    L.append("- " + roofline_note(info))
    oc = ", ".join(sorted(over_context(info, native_context(feats))))
    L += [f"- {MISSING} not measured; n/a not applicable" + (f" ({oc}: the document is longer than the model's native "
                                                             f"{native_context(feats)}-token context)" if oc else "") + ".", ""]

    # ---- features
    L += ["## Features", "", "| feature | status | detail |", "|---|---|---|"]
    for key, label in FEATURES:
        f = feats[key]
        st = {"yes": YES, "no": NO, "unknown": UNKNOWN, "n/a": NA}[f["status"]]
        L.append(f"| {label} | {st} | {feature_detail(key, f)} |")
    L.append("")

    # ---- correctness
    L += ["## Correctness", "", "| gate | description | result |", "|---|---|---|"]
    ev = []
    for name, desc in GATES:
        g = info["gates"].get(name)
        if g is None:
            res = f"{NO} no evidence" if gate_applies(name, kind, feats) else f"{NA} not applicable"
        else:
            key = (g["source"], g["level"], g["date"])
            if key not in ev:
                ev.append(key)
            res = (f"{YES} pass" if g["status"] == "pass" else f"{NO} fail")
            res += (f" ({g['detail']})" if g["detail"] else "") + f" {mark(ev.index(key) + 1)}"
        L.append(f"| `{name}` | {desc} | {res} |")
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


METHOD_TEXT = [
    "- `perf.py` (loom-benchmark) sends real-text requests: a short prompt for decode, decode after 2k and 32k tokens "
    "of context, and prefill of documents of about 1k, 2k, 8k, 16k and 32k tokens. Each run gives one value per "
    "metric; the card shows the mean of the runs.",
    "- `c4_steady.py` runs 4 concurrent streams and reports the aggregate decode rate over the window in which all "
    "4 are decoding; the card shows the median of the runs.",
    "- One engine runs at a time on the same machine, with the same GGUF. Loom rates are timed by the client; "
    "llama.cpp rates come from its server-side `timings`.",
    "- Every performance value comes from a `devlab-perf` job (loom-skills) on an AMD DevLab Dispatch Strix Halo runner: "
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


def render_report(d):
    info, facts, feats = load(d)
    sources, versions = info["sources"], info["versions"]
    perf = info.get("performance") or {"baseline": {"name": "llama.cpp", "version": None, "args": None}, "sessions": []}
    base, c = perf["baseline"], facts["checkpoint"]
    here = programs_sha256(d)

    def prog(sha):
        if not sha:
            return "not recorded"
        rel = "" if here is None else (", the same as this repository's programs" if sha == here
                                       else ", not the same as this repository's programs")
        return f"sha256 `{sha}`{rel}"
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
    L += [f"- Baseline: {base['name']} " + (f"`{base['version']}`" if base["version"] else "(build not recorded)")
          + ", server arguments " + (f"`{base['args']}`" if base["args"] else "not recorded")
          + " (model path, host and port left out).",
          "- Programs measured: the sha256 below is card_lib.programs_sha256, over `program/*.loom` without each file's "
          "first-line generator comment.", "",
          "## Method", "", *METHOD_TEXT, ""]
    labels = dict(METRICS)
    for i, s in enumerate(perf["sessions"], 1):
        src = sources[s["source"]]
        L += [f"## Session {i}: {src_text(src)}, {text(s['date'])}", "",
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
        L += [f"## {xb['name']} {b.capitalize()}: {src_text(src)}, {text(xs['date'])}", "",
              f"- Build: {('`' + xb['version'] + '`') if xb['version'] else 'not recorded'}; server arguments "
              + (f"`{xb['args']}`" if xb["args"] else "not recorded") + ".",
              f"- Device: {device_text(src['device'])}."]
        L += [f"- Differs from {base['name']}: {u}." for u in xb.get("differs") or []]
        L += ["", "| metric | runs | value |", "|---|---|---|"]
        order = [k for k, _ in METRICS]
        for k, m in sorted(xs["metrics"].items(), key=lambda kv: (order.index(kv[0]) if kv[0] in labels else len(order), kv[0])):
            L.append(f"| {labels.get(k, k)} | {', '.join(str(v) for v in m.get('baseline_runs') or []) or MISSING} | {num(k, m.get('baseline'))} |")
        L.append("")
    r = info.get("roofline")
    if r:
        t, d, dr = r["tool"], r["device"], r.get("drafting")
        L += ["## Roofline", "",
              f"- Device: {d['name']}, {d['bandwidth_bytes_per_s'] / 1e9:.0f} GB/s, {d['peak_flops'] / 1e12:.0f} TFLOP/s.",
              f"- Tool: `{t['script']}` sha256 `{t['sha256']}`, loom-skills "
              + (f"`{t['skills_commit']}`" if t["skills_commit"] != "unknown" else "commit unknown")
              + (" (with local changes)" if t.get("dirty") else "") + ".",
              f"- Checkpoint: `{text(r.get('checkpoint'))}`."]
        if dr:
            L.append(f"- Speculative decoding: {METHOD.get(dr['method'], dr['method'])}, depth {dr['depth']}; a drafted row's "
                     "ceiling counts the acceptance measured in its own session, else the `spec` gate's ("
                     + (f"{pct_of(dr['acceptance'])}, {dr['tokens_per_pass']:g} tokens per verify pass)."
                        if dr.get("tokens_per_pass") else "not measured)."))
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
    L += ["## Gate evidence", ""]
    groups = []
    for name in GATE_NAMES + sorted(set(info["gates"]) - set(GATE_NAMES)):
        g = info["gates"].get(name)
        if not g:
            continue
        key = (g["source"], g["level"], g["date"], g.get("programs_sha256"))
        if key not in [k for k, _ in groups]:
            groups.append((key, []))
        dict(groups)[key].append(f"`{name}` {g['status']}" + (f" ({g['detail']})" if g["detail"] else ""))
    for (sid, level, date, sha), names in groups:
        s = sources[sid]
        L += [f"### {src_text(s)}" + (f", LEVEL={level}" if level else "") + f", {text(date)}", "",
              f"- Gates: {', '.join(names)}.",
              f"- Device: {device_text(s['device'])}.",
              f"- Loom SDK: hrx-loom {sdk_text(s['hrx_loom'])}.",
              f"- Programs: {prog(sha)}.", ""]
    if not groups:
        L += ["No gate evidence is recorded.", ""]
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
