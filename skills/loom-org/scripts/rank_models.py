#!/usr/bin/env python3
"""Rank the LLMs a Loom org should host: the most-used first-party models whose Q4_K_M fits the device.

usage: rank_models.py [--top 50] [--max-gb 80] [--pages 4] [--trending-pages 2] [-o ranked.json]

1. Reads the Hub's GGUF repos twice: by downloads (--pages x 250) and by trending score (--trending-pages x 250),
   so a model released last week is seen even before its downloads add up.
2. Keeps LLM pipelines (text, image-text, any-to-any, audio-text, or none declared) and counts each repo's
   downloads (the Hub's rolling 30 days) toward the model it quantizes (its base_model). A repo that declares a
   finetune/merge/adapter relation, or whose name doesn't contain the base's name, is a derivative and doesn't count.
3. Groups variants of one model (-GGUF, -BF16, -qat-*, -mobile-*, letter case) into one entry.
4. Keeps a model when its own repo declares no base_model or one from its own org (Base -> Instruct), or when an
   organization (not a user account) published it on another lab's base; those get "derived_from" for the review
   (e.g. NVIDIA's Bolt-E4B on Gemma 4: kept; a user's finetune: dropped, it shares a listed model's port). Also
   drops non-LLM architectures, base (pretrain) checkpoints and names matching --exclude.
5. Ranks by downloads per day since the model's first GGUF appeared (at most 30 days): a fair comparison between a
   new release and a model that has been out for months.
6. For the top entries, finds a plain (non-UD) Q4_K_M in the order ggml-org, unsloth, bartowski, lmstudio-community,
   the publisher's own GGUF repo, then any repo that counted; keeps it if its real size is <= --max-gb.

The output is a draft. Review it by hand before acting on it (loom-org SKILL.md, "Pick the models").
"""
import argparse
import collections
import concurrent.futures as cf
import datetime
import json
import re
import sys
import os
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hub import HF, get_json  # noqa: E402

LLM_PIPES = {"text-generation", "image-text-to-text", "any-to-any", "audio-text-to-text", None}
PREFER = ["ggml-org", "unsloth", "bartowski", "lmstudio-community"]
EXCLUDE = (r"abliterat|uncensor|heretic|obliterat|distill|merge|-base$|embed|rerank|music|tts|asr|translat|"
           r"guard|reward|nvfp4|-int4|-fp8|-awq|-gptq")
VARIANT = re.compile(r"(-gguf|-bf16|-fp16|-qat.*|-mobile.*|-unquantized)$", re.I)
# GGUF architectures that are not chat LLMs (encoders, diffusion/video/audio generators)
NON_LLM_ARCH = {"t5", "t5encoder", "bert", "nomic-bert", "jina-bert-v2", "clip", "ltxv", "wan", "flux", "sd3",
                "mert2", "acestep-lm", "audiocpp", "controlvector", "gemma-embedding", "deberta", "cosmos", "seedvr"}


def get(url):
    return get_json(url)


def listing(sort, skip):
    q = urllib.parse.urlencode([("filter", "gguf"), ("sort", sort), ("direction", "-1"), ("limit", 250),
                                ("skip", skip)] + [("expand[]", e) for e in
                               ("downloads", "cardData", "pipeline_tag", "gguf", "createdAt")])
    return get(f"{HF}/api/models?{q}") or []


def base_of(card, any_form=False):
    """The single declared base_model as org/name; with any_form, also one written without an org."""
    bm = (card or {}).get("base_model")
    if isinstance(bm, list):
        bm = bm[0] if len(bm) == 1 else (bm[0] if any_form and bm else None)
    if not isinstance(bm, str):
        return None
    return bm if "/" in bm or any_form else None


def group_key(model_id):
    org, name = model_id.split("/", 1)
    return f"{org.lower()}/{VARIANT.sub('', name).lower()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--max-gb", type=float, default=80.0, help="largest Q4_K_M (GB, 1e9) that fits comfortably")
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--trending-pages", type=int, default=2)
    ap.add_argument("--exclude", default=EXCLUDE, help="regex of model names to drop (case-insensitive)")
    ap.add_argument("-o", "--output")
    a = ap.parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    excl = re.compile(a.exclude, re.I)

    jobs = [("downloads", i * 250) for i in range(a.pages)] + [("trendingScore", i * 250) for i in range(a.trending_pages)]
    with cf.ThreadPoolExecutor(8) as ex:
        pages = list(ex.map(lambda j: listing(*j), jobs))
    repos = {m["id"]: m for p in pages for m in p}
    print(f"{len(repos)} GGUF repos read", file=sys.stderr)

    groups = collections.defaultdict(lambda: {"dl": 0, "repos": [], "first": None, "names": collections.Counter(),
                                              "arch": collections.Counter(), "params": 0})
    for m in repos.values():
        if m.get("pipeline_tag") not in LLM_PIPES:
            continue
        card = m.get("cardData") or {}
        bm, rel = base_of(card), card.get("base_model_relation")
        target = bm or m["id"]
        if bm:
            short = VARIANT.sub("", bm.split("/")[-1]).lower()[:12]
            if rel not in (None, "quantized") or short not in m["id"].lower().replace("_", "-"):
                continue  # a derivative, not a quant of bm
        g = groups[group_key(target)]
        g["dl"] += m.get("downloads") or 0
        g["repos"].append(m["id"])
        g["names"][target] += (m.get("downloads") or 0) + 1
        made = datetime.datetime.fromisoformat(m["createdAt"].replace("Z", "+00:00"))
        g["first"] = made if g["first"] is None else min(g["first"], made)
        gg = m.get("gguf") or {}
        if gg.get("architecture") not in (None, "clip"):
            g["arch"][gg["architecture"]] += 1
        if (gg.get("total") or 0) > g["params"]:
            g["params"] = gg["total"]

    rows = []
    for g in groups.values():
        # the model's own name: prefer a spelling without a -GGUF/-qat/... suffix
        plain = [n for n, _ in g["names"].most_common() if not VARIANT.search(n)]
        name = plain[0] if plain else g["names"].most_common(1)[0][0]
        arch = g["arch"].most_common(1)[0][0] if g["arch"] else None
        if excl.search(name.split("/")[-1]) or arch in NON_LLM_ARCH:
            continue
        age = max(1, (now - g["first"]).days)
        rows.append({"model": name, "rate": round(g["dl"] / min(30, age)), "dl30": g["dl"],
                     "first_gguf": g["first"].date().isoformat(), "params_b": round(g["params"] / 1e9, 1),
                     "arch": arch, "repos": g["repos"]})
    rows.sort(key=lambda r: -r["rate"])
    # cheap pre-filter: Q4_K_M is ~4.85 bits/weight; skip what can't fit before any per-model lookups
    rows = [r for r in rows if r["params_b"] * 4.85 / 8 <= a.max_gb * 1.1][: a.top * 3]

    is_org = {}

    def publisher_is_org(org):
        if org not in is_org:
            is_org[org] = get(f"{HF}/api/organizations/{org}/overview") is not None
        return is_org[org]

    def provenance(r):
        """(keep, derived_from): the publisher's own model, or a derivative an organization published (a lab's
        product built on another lab's base, e.g. NVIDIA's Bolt on Gemma 4). A user account's finetune is dropped."""
        org = r["model"].split("/")[0]
        info = get(f"{HF}/api/models/{r['model']}?expand[]=cardData&expand[]=pipeline_tag")
        if info is None:   # a GGUF-only repo named as its own base: the publisher's own
            return True, None
        if info.get("pipeline_tag") not in LLM_PIPES:
            return False, None   # the source model says it is not a chat LLM (video, translation, encoder, ...)
        bm = base_of(info.get("cardData"), any_form=True)
        if bm is None or ("/" in bm and bm.split("/")[0].lower() == org.lower()):
            return True, None
        return publisher_is_org(org), bm

    with cf.ThreadPoolExecutor(12) as ex:
        prov = list(ex.map(provenance, rows))
    kept = []
    for r, (keep, derived) in zip(rows, prov):
        if keep:
            r["derived_from"] = derived
            kept.append(r)
    rows = kept

    def q4(r):
        org, name = r["model"].split("/", 1)
        base = VARIANT.sub("", name)
        cands = [f"{p}/{base}-GGUF" for p in PREFER] + [f"bartowski/{org}_{base}-GGUF", f"{org}/{base}-GGUF"]
        cands += [x for x in r["repos"] if x not in cands]
        for repo in cands:
            tree = get(f"{HF}/api/models/{repo}/tree/main?recursive=1")
            if not tree:
                continue
            files = [f for f in tree if f.get("type") == "file" and f["path"].endswith(".gguf")
                     and re.search(r"(?<!UD-)Q4_K_M", f["path"], re.I)
                     and not re.search(r"mmproj|dflash|mtp|draft", f["path"], re.I)]
            if files:
                return repo, sorted(f["path"] for f in files), round(sum(f["size"] for f in files) / 1e9, 1)
        return None, None, None

    out = []
    for i in range(0, len(rows), 12):
        batch = rows[i:i + 12]
        with cf.ThreadPoolExecutor(12) as ex:
            found = list(ex.map(q4, batch))
        for r, (repo, files, gb) in zip(batch, found):
            if repo and gb <= a.max_gb:
                r.update(q4_repo=repo, q4_files=files, q4_gb=gb)
                out.append(r)
            else:
                print(f"skip {r['model']}: " + (f"Q4_K_M {gb} GB" if repo else "no plain Q4_K_M"), file=sys.stderr)
        if len(out) >= a.top:
            break
    out = out[: a.top]
    for i, r in enumerate(out, 1):
        r["rank"] = i
        print(f"{i:3} {r['rate']:>8}/day {r['first_gguf']} {r['model'][:52]:52} {r['arch'] or '?':13} "
              f"{r['params_b']:6}B  {r['q4_gb']:5} GB  {r['q4_repo']}"
              + (f"  (derived from {r['derived_from']})" if r.get("derived_from") else ""))
    if a.output:
        with open(a.output, "w") as f:
            json.dump({"generated": now.isoformat(timespec="seconds"), "max_gb": a.max_gb, "models": out}, f, indent=1)


if __name__ == "__main__":
    main()
