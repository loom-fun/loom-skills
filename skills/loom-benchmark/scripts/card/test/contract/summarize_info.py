"""Write one summary per loom model, named after its checkpoint: summaries/<checkpoint file stem>.json.

The models are every public repo in the loom-fun org.

Reads each repo's info.json (schema loom-card/2 or loom-card/3, loom-skills/skills/loom-benchmark/scripts/card/info_schema.json)
with the model.json and model.json.lock next to it, and writes the same keys for every model. A loom-card/3 card keeps
its performance and gates per GPU (devices.<gpu>): the top-level keys are the primary GPU's (Strix Halo when it has
results, else the first GPU that has), identical to what a loom-card/2 card gives, and "primary_device" and "devices"
(the other GPUs: their results, or the state that says why they have none) are added. A value that isn't
known is null, never left out. Each summary records the repo and revision it was made from: that is the mapping
from a summary back to its Hugging Face repo. When two repos ship checkpoints with the same file name, both
summaries are prefixed with the checkpoint's uploader (unsloth-Qwen3-Coder-30B-A3B-Instruct-Q4_K_M). Summaries of
repos that are no longer in the org, or are private, are deleted.

    python summarize_info.py [--org loom-fun] [--out summaries]
    python summarize_info.py --check    # list repo -> summary file and what a run would change, write nothing
    python summarize_info.py --skip-unusable    # write the rest when some repos have no usable card yet

Without --skip-unusable a repo with no usable card stops the run before anything is written. With it, that repo is
reported and left as it is: its summary, if it has one, is kept as it was.
"""
import argparse
import glob
import json
import os
import re
import sys

from huggingface_hub import HfApi, hf_hub_download

SCHEMA = "loom-summary/1"
CARD_SCHEMAS = ("loom-card/2", "loom-card/3")
LEGACY_SCHEMAS = ("loom-card/2",)   # one GPU: Strix Halo, gates/performance/roofline at the top level
DEFAULT_GPU = "strix-halo"
GPUS = ("strix-halo", "radeon-r9700")   # display order; must match loom-skills card_lib.GPUS
RETIRED_GATES = ("needle", "needle_full")   # tolerated in old cards, dropped on read (card_lib.RETIRED_GATES)
METRICS = ["decode", "decode_2k", "decode_32k", "prefill_1k", "prefill_2k", "prefill_8k", "prefill_16k",
           "prefill_32k", "streams4", "spec_acceptance"]
TIMED = set(METRICS) - {"spec_acceptance"}
GATES = ["kernel_tests", "golden", "compare", "vision", "consistency", "spec", "agent", "agent_think_off", "sampling",
         "decision"]
METHOD = {"dflash": "DFlash", "mtp": "MTP", "ngram": "n-gram prompt lookup", "assist": "assistant drafter"}
DEVICE = "amdgpu"
_QUANT = r"(?<![A-Za-z0-9])((?:UD-)?I?Q[0-9]%s|BF16|F16|F32)(?![a-z0-9])"

# The rules below must match loom-skills card_lib.py, which renders the same info.json into the repo's README.


def stem(file):
    name = file.replace("\\", "/").rsplit("/", 1)[-1]
    name = name[:-len(".gguf")] if name.lower().endswith(".gguf") else name
    return re.sub(r"-\d{5}-of-\d{5}$", "", name)


def quant_of(file):
    s = stem(file)
    found = re.findall(_QUANT % r"(?:_[0-9A-Z]+)*", s)
    if not found and s == s.lower():
        found = re.findall(_QUANT % r"(?:_[0-9A-Z]+)+", s.upper())
    return found[-1] if found else None


def normalize(info):
    """info as loom-card/3 (a copy; card_lib.normalize): a loom-card/2 card's top-level gates, performance and roofline
    become the strix-halo entry of devices, and retired gates are dropped from either."""
    if info.get("schema") in LEGACY_SCHEMAS:
        out = {k: v for k, v in info.items() if k not in ("gates", "performance", "roofline")}
        out["devices"] = {DEFAULT_GPU: {k: info[k] for k in ("gates", "performance", "roofline") if k in info}}
        info = out
    out = dict(info)
    out["devices"] = {gpu: ({**d, "gates": {n: g for n, g in d["gates"].items() if n not in RETIRED_GATES}}
                            if d.get("gates") else d) for gpu, d in (info.get("devices") or {}).items()}
    return out


def has_results(dev):
    return dev is not None and not dev.get("state") and bool((dev.get("performance") or {}).get("sessions"))


def primary_gpu(info):
    devs = info["devices"]
    return next((g for g in GPUS if has_results(devs.get(g))), None)


def gpu_view(info, gpu):
    """A loom-card/2-shaped view of one GPU (card_lib.gpu_view): its gates and performance at the top level."""
    d = info["devices"].get(gpu) or {}
    out = {k: v for k, v in info.items() if k != "devices"}
    out["gates"] = d.get("gates") or {}
    for k in ("performance", "roofline", "state", "required_bytes", "memory_bytes", "served_context"):
        if d.get(k) is not None:
            out[k] = d[k]
    return out


def package_sources(model, lock):
    """{parameter name (model, draft, vision): source} for the files model.json loads from its origins."""
    origins = model.get("origins", {})
    locked = {f["origin"]: f for f in lock.get("files", [])}
    resolved = {k: (v.get("resolved") or {}).get("revision") for k, v in lock.get("origins", {}).items()}
    out = {}
    for variant in model.get("variants", {}).values():
        for service in variant.get("services", {}).values():
            for pname, param in service.get("parameters", {}).items():
                for f in param.get("files", []):
                    m = re.match(r"@([^/]+)/(.+)$", f.get("origin") or "")
                    if not m or m.group(1) not in origins or pname in out:
                        continue
                    o, lk = origins[m.group(1)], locked.get(f["origin"], {})
                    out[pname] = {"repo": o["repository"], "file": m.group(2), "quant": quant_of(m.group(2)),
                                  "revision": resolved.get(m.group(1)) or o.get("revision"),
                                  "sha256": lk.get("sha256"), "size": split_size(f["origin"], locked)}
    return out


def split_size(origin, locked):
    """Size of a file, or of every shard together when it is the first of a split GGUF (model.json names only that one)."""
    m = re.match(r"(.+)-00001-of-(\d{5})\.gguf$", origin)
    if not m:
        return locked.get(origin, {}).get("size")
    shards = [locked.get(f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf", {}).get("size") for i in range(1, int(m.group(2)) + 1)]
    return None if None in shards else sum(shards)


def source(s, full=False):
    if s is None:
        return None
    out = {"repo": s["repo"], "file": s["file"], "quant": s["quant"]}
    if full:
        out.update(revision=s["revision"], sha256=s["sha256"], size=s["size"])
    return out


def evidence(e):
    return None if e is None else {"kind": e["kind"], "run": e["run"], "url": e["url"]}


def value(feats, key):
    return feats.get(key, {}).get("value") or {}


def drafts(feats):
    sd = feats.get("speculative_decoding", {})
    return sd.get("status") == "yes" and bool((sd.get("value") or {}).get("method"))


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
        return drafts(feats)
    if name == "agent":
        return feats["tool_calling"]["status"] == "yes"
    if name == "agent_think_off":
        return (feats["tool_calling"]["status"] == "yes" and feats["thinking"]["status"] == "yes"
                and value(feats, "thinking").get("mode") == "toggle")
    return True


def metric_applies(key, kind, feats, sessions):
    if kind == "decision" and key in ("decode", "decode_2k", "decode_32k", "streams4", "spec_acceptance"):
        return False
    if key == "spec_acceptance":
        return drafts(feats)
    native = value(feats, "context").get("native")
    measured = {k for s in sessions for k, m in s["metrics"].items() if m.get("loom") is not None}
    refused = {k for s in sessions for k, n in (s.get("over_context") or {}).items() if native and n and n > native}
    return key not in refused - measured


def headline(card, devlab=False):
    """{metric: (entry, session index)}: per metric the last local session that measured it, the README's headline
    value (spec_acceptance from any session), or with devlab=True the last DevLab session, the README's DevLab
    columns (timed metrics only)."""
    out = {}
    for i, s in enumerate(card["performance"]["sessions"]):
        dl = card["sources"][s["source"]]["kind"] == "devlab"
        for k, m in s["metrics"].items():
            if (dl and k in TIMED) if devlab else (not dl or k not in TIMED):
                out[k] = (m, i)
    return out


def measured(card, kind, feats, keys, devlab=False):
    """(date, source, metrics) of the headline (or DevLab) values: every metric in `keys`, null when not measured.
    The date and source are those of the session that gave the most timed values."""
    sessions = card["performance"]["sessions"]
    shown = {k: v for k, v in headline(card, devlab).items() if metric_applies(k, kind, feats, sessions)}
    counts = {}
    for k, (_, i) in shown.items():
        if k in TIMED:
            counts[i] = counts.get(i, 0) + 1
    primary = sessions[min(counts, key=lambda i: (-counts[i], i))] if counts else None
    metrics = {}
    for k in keys + sorted(set(shown) - set(keys)):
        m = shown.get(k, ({}, None))[0]
        loom, base = m.get("loom"), m.get("baseline")
        metrics[k] = {"loom": loom, "llamacpp": base,
                      "speedup": round(loom / base, 2) if loom is not None and base else None}
    return (primary["date"] if primary else None,
            evidence(card["sources"][primary["source"]]) if primary else None, metrics)


def llamacpp_drafting(args):
    t = re.search(r"--spec-type\s+(\S+)", args or "")
    if not t:
        return None
    n = re.search(r"--spec-draft-n-max\s+(\d+)", args)
    return f"{t.group(1)}, n-max {n.group(1)}" if n else t.group(1)


def device_results(card, kind, feats, sd):
    """The performance and gates of one GPU's view of the card (the keys "performance" and "gates" of a summary)."""
    perf, sessions = card["performance"], card["performance"]["sessions"]
    date, src, metrics = measured(card, kind, feats, METRICS)
    dl_date, dl_src, dl_metrics = measured(card, kind, feats, [k for k in METRICS if k in TIMED], devlab=True)
    gates = {}
    for g in GATES + sorted(set(card["gates"]) - set(GATES)):
        if g in card["gates"]:
            gates[g] = card["gates"][g]["status"]
        elif g in GATES:
            gates[g] = "unknown" if gate_applies(g, kind, feats) else "n/a"
    counts = {s: sum(1 for v in gates.values() if v == s) for s in ("pass", "fail", "unknown", "n/a")}
    build = re.search(r"\bb(1[0-9]{3,})\b", perf["baseline"]["version"] or "")
    return {
        "date": date,
        "llamacpp_build": f"b{build.group(1)}" if build else None,
        "drafting": {"loom": f"{METHOD.get(sd['method'], sd['method'])}, depth {sd['depth']}" if drafts(feats)
                     else None,
                     "llamacpp": llamacpp_drafting(perf["baseline"]["args"])},
        "source": src,
        "metrics": metrics,
        "not_applicable": [k for k in METRICS if not metric_applies(k, kind, feats, sessions)],
        "devlab": {"date": dl_date, "source": dl_src, "metrics": dl_metrics},
    }, {"summary": counts, "status": gates}


def summarize(card, model, lock, revision, license):
    """The summary of a card: loom-card/2, or loom-card/3 (the primary GPU's results at the top level, the other
    GPUs under "devices")."""
    card = normalize(card)
    gpu = primary_gpu(card)
    if gpu is None:
        raise ValueError("card has no GPU with performance results")
    out = summarize_gpu(gpu_view(card, gpu), model, lock, revision, license)
    if card.get("schema") not in LEGACY_SCHEMAS:
        out["primary_device"] = gpu
        out["devices"] = {}
        for g in (g for g in GPUS if g in card["devices"] and g != gpu):
            d = card["devices"][g]
            if not has_results(d):
                out["devices"][g] = {"state": d.get("state") or "not_measured", "required_bytes": d.get("required_bytes")}
                continue
            view = gpu_view(card, g)
            kind = "decision" if out["service"]["kind"] == "decision" else "chat"
            feats = card["card"]["features"]
            perf, gates = device_results(view, kind, feats, value(feats, "speculative_decoding"))
            out["devices"][g] = {"state": None, "memory_bytes": d.get("memory_bytes"),
                                 "context_tokens": d.get("served_context") or out["serve"]["context_tokens"],
                                 "performance": perf, "gates": gates}
    return out


def summarize_gpu(card, model, lock, revision, license):
    srcs = package_sources(model, lock)
    if "model" not in srcs:
        raise ValueError("model.json loads no checkpoint from an @origin (parameters.model)")
    ckpt = srcs["model"]
    preset = next(iter(model.get("presets", {}).values()), {})
    execution = preset.get("execution", {})
    kind = "decision" if model.get("services", {}).get(preset.get("service"), {}).get("kind") == "decision" else "chat"
    feats = card["card"]["features"]
    sd = value(feats, "speculative_decoding")
    perf, gates = device_results(card, kind, feats, sd)
    return {
        "schema": SCHEMA,
        "checkpoint": stem(ckpt["file"]),
        "name": re.sub(r"[-_]gguf$", "", ckpt["repo"].split("/")[-1], flags=re.I),
        "repo": card["repo"],
        "revision": revision,
        "status": card.get("status"),
        "license": card["card"].get("license") or license,
        "service": {"kind": kind, "endpoint": value(feats, "api").get("endpoint")},
        "weights": {
            "checkpoint": source(ckpt, full=True),
            "drafter": source(srcs.get("draft")),
            "projector": source(srcs.get("vision")),
        },
        "serve": {
            "device": DEVICE,
            "memory_bytes": execution.get("memory_bytes") or card["card"].get("memory_bytes"),
            "context_tokens": value(feats, "context").get("tokens"),
            "rows": value(feats, "concurrency").get("rows"),
            "max_tokens": value(feats, "max_output_tokens").get("max_tokens"),
        },
        "features": {k: v["status"] for k, v in sorted(feats.items())},
        "performance": perf,
        "gates": gates,
    }


def load(api, repo_id):
    info = api.model_info(repo_id, files_metadata=False)
    names = {s.rfilename for s in info.siblings}
    missing = [f for f in ("info.json", "model.json") if f not in names]
    if missing:
        return info, None, f"no {', '.join(missing)}"

    def get(name):
        if name not in names:
            return {}
        with open(hf_hub_download(repo_id, name, revision=info.sha), encoding="utf-8") as f:
            return json.load(f)
    card = get("info.json")
    if card.get("schema") not in CARD_SCHEMAS:
        return info, None, f"info.json schema is {card.get('schema')!r}, expected one of {CARD_SCHEMAS}"
    license = (info.card_data or {}).get("license") if info.card_data else None
    try:
        return info, summarize(card, get("model.json"), get("model.json.lock"), info.sha, license), None
    except (KeyError, TypeError, ValueError) as e:
        return info, None, f"{type(e).__name__}: {e}"


def file_names(summaries):
    """{repo: summary file name}: the checkpoint stem, prefixed with the checkpoint's uploader when it is not unique."""
    by_stem = {}
    for s in summaries:
        by_stem.setdefault(s["checkpoint"].lower(), []).append(s)
    names = {}
    for group in by_stem.values():
        for s in group:
            if len(group) > 1:
                s["checkpoint"] = f"{s['weights']['checkpoint']['repo'].split('/')[0]}-{s['checkpoint']}"
            names[s["repo"]] = s["checkpoint"] + ".json"
    seen = {}
    for repo, name in names.items():
        if name.lower() in seen:
            sys.exit(f"{repo} and {seen[name.lower()]} map to the same summary {name}")
        seen[name.lower()] = repo
    return names


def existing(out):
    """{repo: (file name, revision)} for the summaries already in `out`."""
    found = {}
    for path in glob.glob(os.path.join(out, "*.json")):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        if s.get("schema") == SCHEMA:
            found[s["repo"]] = (os.path.basename(path), s.get("revision"))
    return found


def list_repos(api, org):
    """Every public model repo in `org`, sorted case-insensitively."""
    return sorted({m.id for m in api.list_models(author=org) if not m.private}, key=str.lower)


def check(api, org, out):
    have = existing(out)
    repos = list_repos(api, org)
    for repo in repos:
        sha = api.model_info(repo, files_metadata=False).sha
        name, rev = have.get(repo, (None, None))
        state = "new" if name is None else "current" if rev == sha else f"stale {rev[:7]} -> {sha[:7]}"
        print(f"{state:20} {repo} -> {name or '?'}")
    for repo, (name, _) in sorted(have.items()):
        if repo not in repos:
            print(f"{'gone':20} {repo} -> {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--org", default="loom-fun")
    p.add_argument("--out", default="summaries")
    p.add_argument("--check", action="store_true", help="list repo -> summary file and its state, write nothing")
    p.add_argument("--skip-unusable", action="store_true",
                   help="skip repos with no usable card (keeping their summary) instead of writing nothing")
    a = p.parse_args()

    api = HfApi()
    if a.check:
        return check(api, a.org, a.out)

    summaries, failed, skipped = [], [], set()
    for repo in list_repos(api, a.org):
        _, summary, why = load(api, repo)
        if summary is None:
            failed.append(f"{repo}: {why}")
            skipped.add(repo)
        else:
            summaries.append(summary)
    if failed and not a.skip_unusable:
        sys.exit("not written, these repos have no usable card:\n  " + "\n  ".join(failed))
    if failed:
        print("skipped, these repos have no usable card:\n  " + "\n  ".join(failed), file=sys.stderr)

    names = file_names(summaries)
    old = {repo: v for repo, v in existing(a.out).items() if repo not in skipped}
    os.makedirs(a.out, exist_ok=True)
    for s in summaries:
        with open(os.path.join(a.out, names[s["repo"]]), "w", encoding="utf-8", newline="\n") as f:
            json.dump(s, f, indent=2)
            f.write("\n")
        print(f"{s['repo']} -> {os.path.join(a.out, names[s['repo']])}")
    written = {n.lower() for n in names.values()}
    for repo, (name, _) in sorted(old.items()):
        if name.lower() not in written:
            os.remove(os.path.join(a.out, name))
            print(f"removed {name} ({repo})")


if __name__ == "__main__":
    main()
