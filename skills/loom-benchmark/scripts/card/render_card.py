#!/usr/bin/env python3
"""Render a loom-fun card's README.md from its info.json (schema loom-card/1) alone.

usage: render_card.py info.json [-o README.md]      (-o - prints to stdout; default: README.md next to info.json)
       render_card.py --check info.json README.md    exit 1 unless README.md is exactly what info.json renders to

A pure function: the same info.json always gives the same bytes. Every sentence is part of this template;
only the filled-in values differ between cards. Never edit a rendered README by hand: change the data
(make_info.py) or this template, and re-render.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_lib import FEATURES, GATES, HF, LOOM_URL, METHOD, METRICS, SCHEMA_ID, SKILLS_URL, load_json, require_valid  # noqa: E402

YES, NO, NA, MISSING = "✅", "❌", "—", "—"
SUP = "¹²³⁴⁵⁶⁷⁸⁹"


def mark(i):
    return SUP[i - 1] if i <= len(SUP) else f"[{i}]"


def num(v):
    return MISSING if v is None else f"{v:.1f}"


def count(v):
    return f"{v:,}"


def text(v):
    return "not recorded" if v in (None, "") else v


def hf(repo):
    return f"[{repo}]({HF}/{repo})"


def src_text(s):
    if not s:
        return "not recorded"
    name = {"local": "local run", "devlab": "DevLab Dispatch run", "factory": "loom-factory run"}[s["kind"]]
    if s["kind"] == "local":
        return name
    label = f"{name} {s['run']}" if s.get("run") else name
    return f"[{label}]({s['url']})" if s.get("url") else label


def margin(loom, llama):
    if loom is None or llama is None or not llama:
        return MISSING
    return f"{round((loom / llama - 1) * 100):+d} %"


def drafting(setting):
    return f"on ({setting})" if setting else "off"


def feature_detail(key, f, info):
    st, v = f["status"], f["value"] or {}
    if st == "no":
        return {"sampling": "greedy decoding only"}.get(key, "not supported")
    if st == "unknown":
        return "not verified"
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


def render(info):
    require_valid(info)
    c, p, s = info["checkpoint"], info["performance"], info["serve"]
    L = ["---", f"base_model: {c['repo']}"]
    if info["license"]:
        L.append(f"license: {info['license']}")
    L += ["tags:", *[f"- {t}" for t in info["tags"]], "---", ""]

    quant = f" ({c['quant']})" if c["quant"] else ""
    L += [f"# {info['name']}{quant} for AMD Loom", "",
          f"A [Loom]({LOOM_URL}) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint "
          f"{hf(c['repo'])} (`{c['file']}`), generated on an AMD Ryzen AI Max+ 395 \"Strix Halo\" with the "
          f"[loom-skills]({SKILLS_URL}) workflow.", ""]

    fetch = ["loom-fetch", f"{HF}/{info['repo']}/resolve/main/model.json"]
    if s["preset"]:
        fetch.append(f"--preset={s['preset']}")
    fetch += ["--materialize=copy", f"--output={s['output']}"]
    serve = ["loom-serve", f"--model={s['output']}/model.json", f"--device={s['device']}"]
    if s["memory_bytes"]:
        serve.append(f"--memory_bytes={s['memory_bytes']}")
    api = ("answers OpenAI-style `/v1/chat/completions` requests" if info["service"]["kind"] == "chat"
           else f"answers SystemOne decision requests at `{info['service']['endpoint']}`")
    L += ["## Quick start", "", "```bash", 'pip install "hrx-loom[hf]"', " ".join(fetch), " ".join(serve), "```", "",
          f"The server listens on `127.0.0.1:8080` and {api}.", ""]

    L += ["## Performance vs llama.cpp", "", "| metric | llama.cpp | Loom | margin |", "|---|---|---|---|"]
    notes = []
    for key, label in METRICS:
        m = p["metrics"][key]
        if key in p["not_applicable"]:
            L.append(f"| {label} | n/a | n/a | n/a |")
            continue
        loom = num(m["loom"])
        if m["loom_source"]:
            notes.append(m["loom_source"])
            loom += f" {mark(len(notes))}"
        L.append(f"| {label} | {num(m['llamacpp'])} | {loom} | {margin(m['loom'], m['llamacpp'])} |")
    r = p["runs"]
    runs = (f"decode and prefill {text(r['decode_prefill'])} (mean), 4 streams {text(r['streams4'])} (median)")
    build = f"ROCm build {p['llamacpp_build']}" if p["llamacpp_build"] else "ROCm build not recorded"
    L += ["", f"- llama.cpp: {build} (lemonade-sdk/llamacpp-rocm), speculative decoding "
              f"{drafting(p['drafting']['llamacpp'])}.",
          f"- Loom: speculative decoding {drafting(p['drafting']['loom'])}.",
          "- Same GGUF on both engines, tokens per second. Loom: client-side timing; llama.cpp: server-side timings. "
          f"Runs per value: {runs}.",
          f"- Measured {text(p['date'])}; source: {src_text(p['source'])}."]
    for i, n in enumerate(notes, 1):
        L.append(f"- {mark(i)} Loom value from {src_text(n['source'])}, {n['date']}.")
    L += [f"- {MISSING} not measured; n/a not applicable.", ""]

    L += ["## Features", "", "| feature | status | detail |", "|---|---|---|"]
    for key, label in FEATURES:
        f = info["features"][key]
        st = {"yes": YES, "no": NO, "unknown": NO, "n/a": NA}[f["status"]]
        L.append(f"| {label} | {st} | {feature_detail(key, f, info)} |")
    L.append("")

    L += ["## Correctness", "", "| gate | description | result |", "|---|---|---|"]
    desc = dict(GATES)
    ev = []
    for g in info["gates"]:
        if g["status"] == "n/a":
            res = f"{NA} not applicable"
        elif g["status"] == "unknown":
            res = f"{NO} no evidence"
        else:
            key = (src_text(g["source"]), g["level"], g["date"])
            if key not in ev:
                ev.append(key)
            res = (f"{YES} pass" if g["status"] == "pass" else f"{NO} fail")
            res += (f" ({g['detail']})" if g["detail"] else "") + f" {mark(ev.index(key) + 1)}"
        L.append(f"| `{g['name']}` | {desc[g['name']]} | {res} |")
    L.append("")
    for i, (src, level, date) in enumerate(ev, 1):
        lv = f", LEVEL={level}" if level else ""
        L.append(f"- {mark(i)} {src}{lv}, {text(date)}.")
    L += ["- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.", ""]

    L += ["## Files", "",
          "- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).",
          "- `program/*.loom`, `weights.loom`: the generated Loom programs.",
          "- `tokenizer.json`: the tokenizer.",
          "- `llm_config.json`: the configuration the programs were generated from.",
          f"- `info.json`: the data this card is rendered from (schema `{SCHEMA_ID}`).", "",
          "Source files, fetched by `loom-fetch` from their own repositories:", ""]
    for role, x in (("checkpoint", c), ("drafter", info["drafter"]), ("projector", info["projector"])):
        if x:
            rev = f" @ `{x['revision'][:7]}`" if x["revision"] else ""
            L.append(f"- {role}: {hf(x['repo'])}{rev}, `{x['file']}`")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("info")
    ap.add_argument("readme", nargs="?", help="with --check: the README to compare")
    ap.add_argument("-o")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    out = render(load_json(a.info))
    if a.check:
        with open(a.readme, encoding="utf-8", newline="") as f:
            same = f.read() == out
        print(("OK   " if same else "DIFF ") + a.readme)
        sys.exit(0 if same else 1)
    dest = a.o or os.path.join(os.path.dirname(os.path.abspath(a.info)), "README.md")
    if dest == "-":
        sys.stdout.write(out)
    else:
        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        print(f"wrote {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
