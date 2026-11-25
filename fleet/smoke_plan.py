#!/usr/bin/env python3
"""Release smoke plan: which published cards the DevLab smoke (.github/workflows/devlab-smoke.yml) runs, on which GPU and OS.

usage: smoke_plan.py --org ORG [--eligible eligible.json | --slugs SLUG ...] [--gpus strix-halo,radeon-r9700] [--windows strix-halo]
                     [--windows-skip SLUG=REASON ...] [--hook-job NAME:DEVICE:SLUG:HOOK[:K=V,...] ...] [--batch 6]
                     [--cache DIR] --out smoke.json [--md plan.md]

Every card is read at an exact revision (eligible.json "staging_revision", else the repo head when the plan is made), anonymously:
README.md, llm_config.json and info.json. The sample (release-checklist.md sections 2 and 3):
  * strata: fleet/sample_plan.py with granularity "arch" (arch, draft kind, dense/MoE, projector) over the text cards that have a
    Quick start block for the GPU, one representative per stratum, the smallest weights (a functional check: the stratum's programs,
    template and serve path matter, runner time is the cost); image cards are one stratum per GPU (a text-prompt generation card);
  * stock host: every Linux job serves under an 8 MiB `ulimit -l` in a non-root shell, so the smoke is also the stock-host Quick start
    check; every card whose serve line on that GPU carries other memory flags than the GPU's usual line is added, and a Quick start
    block of a runner-less GPU with the same ISA (gfx1201-16gb on the R9700) is run once per distinct memory-flag set on that runner;
  * Windows: the same strata on each --windows GPU's Windows runners (`runner.windows_labels` in device.json), the card's own block
    for that GPU in PowerShell; --windows-skip records a card left out with its reason (a known Windows limit the card does not claim
    to work around); a card whose README names Windows is never skipped.
Cards are packed into jobs of --batch cards per (OS, GPU), heaviest first, so a runner pickup serves several smokes.
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import sample_plan as SP  # noqa: E402
import render_card  # noqa: E402

registry = SP.registry
HF = "https://huggingface.co"
MEMORY_FLAG = re.compile(r"^--(pool_backing|memory_bytes|\w*slab_bytes|parameter_mode|kv\w*|cache\w*)=")
MEMORY_WRAP = ("systemd-run", "ulimit", "prlimit")


# ---- reading a card --------------------------------------------------------------------------------------------------------
def quick_start_blocks(readme):
    """{summary label | None: [command lines]} of the card's Quick start: one entry per <details> block, or None for a single block."""
    section = readme.partition("\n## Quick start\n")[2].split("\n## ", 1)[0]
    if not section:
        return {}
    details = re.findall(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", section, re.S)
    out = {}
    for label, body in (details or [(None, section)]):
        blocks = re.findall(r"^```(?:bash|sh)\n(.*?)^```$", body, re.M | re.S)
        if blocks:
            lab = label.replace("&quot;", '"').replace("&amp;", "&") if label else None
            out[lab] = [ln.strip() for ln in blocks[0].splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return out


def memory_flags(lines):
    """The memory-related part of a block's serve line: flags that size or back memory, and launcher wrappers (systemd-run, ulimit)."""
    serve = next((ln for ln in lines if "loom-serve" in ln), "")
    toks = serve.split()
    return tuple(sorted({t.strip("'\"") for t in toks if MEMORY_FLAG.match(t.strip("'\""))} | {w for w in MEMORY_WRAP if w in toks}))


def label_device(label):
    """Device key of a Quick start summary (render_card.gpu_label), else None."""
    return next((g for g in render_card.GPUS if render_card.gpu_label(g) == label), None)


def card_kind(cfg):
    return "llm" if cfg else "image"


def claimed_runner_devices(info):
    """Devices of info.json with results (no 'state') and a DevLab runner: where a single-block Quick start applies."""
    devs = (info or {}).get("devices") or {}
    return [g for g, d in devs.items() if g in render_card.GPUS and not (d or {}).get("state") and registry.device(g)["runner"]["labels"]]


def device_blocks(readme, info):
    """{device: [lines]}: each block mapped to its device; a single block belongs to every claimed runner device (or Strix Halo)."""
    blocks = quick_start_blocks(readme)
    if list(blocks) == [None]:
        return {g: blocks[None] for g in (claimed_runner_devices(info) or [render_card.DEFAULT_GPU])}
    return {label_device(lab): ln for lab, ln in blocks.items() if label_device(lab)}


def features(info):
    f = ((info or {}).get("card") or {}).get("features") or {}
    return {k: (v or {}).get("status") for k, v in f.items()}


def mentions_windows(readme):
    return bool(re.search(r"\bwindows\b", readme, re.I))


# ---- the plan --------------------------------------------------------------------------------------------------------------
class CardFleet(SP.Fleet):
    """sample_plan's Fleet over the cards' own llm_config.json files at the planned revisions."""

    def __init__(self, cards, fit_path, baselines_path):
        super().__init__(None, fit_path, baselines_path)
        self.cards = cards

    def config(self, slug):
        return self.cards[slug].get("config")


def runner_for(device, os_name):
    run = registry.device(device)["runner"]
    labels = run.get("windows_labels") if os_name == "windows" else run.get("labels")
    return list(labels or [])


def host_of(device):
    return same_isa_runner(device) or device


def same_isa_runner(device):
    """A runner device with the same ISA as a runner-less device (its Quick start block runs there), else None."""
    dev = registry.device(device)
    if dev["runner"]["labels"]:
        return device
    for k, d in registry.devices().items():
        if d["runner"]["labels"] and d.get("gfx") == dev.get("gfx") and (d.get("memory_bytes") or 0) >= (dev.get("memory_bytes") or 0):
            return k
    return None


def entry(cards, slug, block_device, why):
    c = cards[slug]
    single = None in quick_start_blocks(c["readme"])
    return {"card": c["repo"], "slug": slug, "id": slug if block_device == host_of(block_device) else f"{slug}--{block_device}",
            "revision": c["revision"], "block": None if single else render_card.gpu_label(block_device),
            "block_device": block_device, "kind": c["kind"],
            "tools": c["features"].get("tool_calling") == "yes" and c["kind"] == "llm",
            "thinking": c["features"].get("thinking") == "yes", "reasoning_effort": c.get("reasoning_levels"),
            "weights": c.get("w"), "why": why}


def plan(cards, gpus, windows=(), windows_skip=None, fit=None, baselines=None):
    """-> (rows, skipped): rows = [{os, device, ...entry}] for each card to smoke; skipped = [{os, device, slug, reason}]."""
    windows_skip = windows_skip or {}
    fleet = CardFleet(cards, fit or os.path.join(ROOT, ".github", "devlab", "r9700_fit.json"),
                      baselines or os.path.join(ROOT, "fleet", "baselines.json"))
    for slug in cards:
        cards[slug]["w"] = fleet.weights(slug)
    rows, skipped = [], []

    def sample(gpu, os_name, eligible):
        text = [s for s in eligible if cards[s]["kind"] == "llm"]
        impact = {"models": {s: {"model_json": True} for s in text}}
        strata, sk = SP.plan(impact, [gpu], fleet, granularity="arch", prefer="lightest")
        skipped.extend({"os": os_name, "device": gpu, "slug": x["slug"], "reason": x["reason"]} for x in sk)
        out = []
        for st in strata:
            r = st["representatives"][0]
            out.append(entry(cards, r["slug"], gpu, f"stratum {st['key'].rsplit('|gpu=', 1)[0]} ({len(st['members'])} cards): "
                             + "; ".join(r["why"])))
        images = sorted(s for s in eligible if cards[s]["kind"] == "image" and cards[s].get("image_request"))
        if images:
            out.append(entry(cards, images[0], gpu, f"image stratum ({sum(cards[s]['kind'] == 'image' for s in eligible)} cards): "
                             "a text-prompt generation card"))
        return out

    for gpu in gpus:
        eligible = sorted(s for s, c in cards.items() if gpu in c["blocks"])
        picked = sample(gpu, "linux", eligible)
        have = {e["slug"] for e in picked}
        for kind in ("llm", "image"):   # the usual serve line of each kind of card (an image server has its own flags)
            sigs = {}
            for s in eligible:
                if cards[s]["kind"] == kind:
                    sigs.setdefault(memory_flags(cards[s]["blocks"][gpu]), []).append(s)
            usual = max(sorted(sigs), key=lambda k: len(sigs[k])) if sigs else ()
            for sig, members in sorted(sigs.items()):
                if sig == usual:
                    continue
                for s in members:
                    if s not in have:
                        picked.append(entry(cards, s, gpu, f"serve line memory flags {list(sig) or 'none'} differ from the usual "
                                            f"{list(usual) or 'none'} of {kind} cards"))
                        have.add(s)
        rows += [{"os": "linux", "device": gpu, **e} for e in picked]
    # runner-less GPUs with a runner of the same ISA: one card per distinct memory-flag set of their block
    for dev in render_card.GPUS:
        host = same_isa_runner(dev)
        if dev in gpus or host not in gpus:
            continue
        sigs = {}
        for s, c in sorted(cards.items()):
            if dev in c["blocks"] and c["kind"] == "llm":
                sigs.setdefault(memory_flags(c["blocks"][dev]), []).append(s)
        for sig, members in sorted(sigs.items()):
            pick = sorted(members, key=lambda s: (cards[s].get("w") or 1e15) if (cards[s].get("w") or 0) >= SP.SUB_1B_BYTES else 1e16)[0]
            e = entry(cards, pick, dev, f"{dev} Quick start (no runner of its own) on a {host} runner: memory flags {list(sig)}")
            rows.append({"os": "linux", "device": host, **e})
    for gpu in windows:
        eligible = []
        for s in sorted(cards):
            if gpu not in cards[s]["blocks"]:
                continue
            if s in windows_skip and not mentions_windows(cards[s]["readme"]):
                skipped.append({"os": "windows", "device": gpu, "slug": s, "reason": windows_skip[s]})
                continue
            eligible.append(s)
        rows += [{"os": "windows", "device": gpu, **e} for e in sample(gpu, "windows", eligible)]
    return rows, skipped


def minutes(row):
    """Rough runner minutes of one smoke: install + fetch + load + three requests, growing with the weights."""
    return round(4 + 0.35 * ((row.get("weights") or 8e9) / 1e9), 1)


def jobs(rows, batch, hooks=()):
    """Pack rows into jobs of `batch` cards per (os, device), heaviest first; each hook is its own job."""
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["os"], r["device"]), []).append(r)
    for (os_name, dev), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: -minutes(r))
        n = max(1, -(-len(rs) // batch))
        bins = [[] for _ in range(n)]
        for r in rs:   # longest-processing-time packing: each card into the lightest job
            min(bins, key=lambda b: sum(minutes(x) for x in b)).append(r)
        short = registry.device(dev)["runner"]["key"]
        for i, b in enumerate(bins, 1):
            out.append({"name": f"{short}-{os_name}-{i}", "device": dev, "os": os_name, "runs_on": runner_for(dev, os_name),
                        "stock_memlock": os_name == "linux", "minutes": round(sum(minutes(x) for x in b), 1),
                        "cards": [{k: v for k, v in x.items() if k not in ("os", "device")} for x in b]})
    out += list(hooks)
    return out


# ---- reading the Hub (anonymous) -------------------------------------------------------------------------------------------
def _get(url, cache=None, binary=False):
    path = None
    if cache:
        path = os.path.join(cache, urllib.parse.quote(url, safe=""))
        if os.path.exists(path):
            data = open(path, "rb").read()
            return data if binary else data.decode()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "loom-skills-smoke-plan"}), timeout=120) as r:
        data = r.read()
    if path and "/resolve/" in url and re.search(r"/resolve/[0-9a-f]{40}/", url):   # only immutable (revision-pinned) files are cached
        os.makedirs(cache, exist_ok=True)
        open(path, "wb").write(data)
    return data if binary else data.decode()


def head(repo):
    return json.loads(_get(f"{HF}/api/models/{repo}/revision/main"))["sha"]


def org_repos(org):
    return sorted(m["id"] for m in json.loads(_get(f"{HF}/api/models?author={org}&limit=1000")))


def image_request(readme):
    """The JSON body of the card's first `curl ... /v1/images/generations -d '{...}'` example with a text prompt, else None."""
    m = re.search(r"/v1/images/generations.*?-d '(\{.*?\})'", readme, re.S)
    if not m:
        return None
    try:
        body = json.loads(m.group(1))
    except ValueError:
        return None
    return body if isinstance(body.get("prompt"), str) else None


def load_card(repo, revision, cache=None):
    base = f"{HF}/{repo}/resolve/{revision}"
    readme = _get(f"{base}/README.md", cache)
    out = {"repo": repo, "revision": revision, "readme": readme}
    for name in ("llm_config.json", "info.json"):
        try:
            out[name] = json.loads(_get(f"{base}/{name}", cache))
        except Exception:   # an image card has no llm_config.json
            out[name] = None
    return describe(out)


def describe(c):
    """Add the plan's view of a loaded card: config, kind, blocks per device, features, image request."""
    c["config"] = c.get("llm_config.json")
    c["kind"] = card_kind(c["config"])
    c["blocks"] = device_blocks(c["readme"], c.get("info.json"))
    c["features"] = features(c.get("info.json"))
    rf = (((c.get("info.json") or {}).get("card") or {}).get("features") or {}).get("reasoning_effort") or {}
    c["reasoning_levels"] = ((rf.get("value") or {}).get("levels") or None) if rf.get("status") == "yes" else None
    c["image_request"] = image_request(c["readme"]) if c["kind"] == "image" else None
    return c


def table(rows, skipped, job_list):
    L = ["| OS | GPU | card | revision | Quick start block | why |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['os']} | {r['device']} | {r['slug']} | `{r['revision'][:8]}` | {r['block'] or '(single)'} | {r['why']} |")
    for s in skipped:
        L.append(f"| {s['os']} | {s['device']} | {s['slug']} | | skipped | {s['reason']} |")
    L += ["", f"{len(rows)} smokes in {len(job_list)} jobs; ~{round(sum(j.get('minutes', 0) for j in job_list) / 60, 1)} runner-h estimated"]
    return "\n".join(L)


def parse_hook(spec):
    """NAME:DEVICE:SLUG:HOOK[:K=V,...] -> a job that smokes one card on DEVICE (Linux) and then runs HOOK with env K=V."""
    name, dev, slug, hook, *rest = spec.split(":")
    env = dict(kv.split("=", 1) for kv in (rest[0].split(",") if rest and rest[0] else []))
    return name, dev, slug, hook, env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--org", required=True, help="the org the cards are read from (the staging org before a release)")
    ap.add_argument("--eligible", help="eligible.json: {slug: {staging_revision}}; the plan uses exactly these cards and revisions")
    ap.add_argument("--slugs", nargs="*", help="cards to plan from (default: every model repo of --org at its head)")
    ap.add_argument("--gpus", default="strix-halo,radeon-r9700")
    ap.add_argument("--windows", default="strix-halo", help="GPUs sampled on their Windows runners ('' for none)")
    ap.add_argument("--windows-skip", action="append", default=[], metavar="SLUG=REASON")
    ap.add_argument("--hook-job", action="append", default=[], metavar="NAME:DEVICE:SLUG:HOOK[:K=V,...]")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--cache", help="directory for revision-pinned Hub files (re-planning reads no file twice)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    gpus = [g for g in a.gpus.split(",") if g]
    if a.eligible:
        el = json.load(open(a.eligible))
        revs = {s: r["staging_revision"] for s, r in el.items()}
    else:
        slugs = a.slugs or [r.split("/", 1)[1] for r in org_repos(a.org)]
        revs = {s: head(f"{a.org}/{s}") for s in slugs}
    cards = {s: load_card(f"{a.org}/{s}", rev, a.cache) for s, rev in sorted(revs.items())}
    skip = dict(x.split("=", 1) for x in a.windows_skip)
    rows, skipped = plan(cards, gpus, [g for g in a.windows.split(",") if g], skip)
    hooks = []
    for spec in a.hook_job:
        name, dev, slug, hook, env = parse_hook(spec)
        e = entry(cards, slug, dev, f"hook {os.path.basename(hook)}")
        hooks.append({"name": name, "device": dev, "os": "linux", "runs_on": runner_for(dev, "linux"), "stock_memlock": True,
                      "minutes": minutes(e) + 30, "cards": [e], "hook": hook, "env": env})
    job_list = jobs(rows, a.batch, hooks)
    doc = {"_comment": "generated by fleet/smoke_plan.py; run by .github/workflows/devlab-smoke.yml (label devlab-smoke)",
           "org": a.org, "skipped": skipped, "jobs": job_list}
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    md = table(rows, skipped, job_list)
    if a.md:
        open(a.md, "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
