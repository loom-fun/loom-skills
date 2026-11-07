#!/usr/bin/env python3
"""Find the published loom-fun packages a port (or an optimization) should start from, instead of from scratch.

Only CATEGORY A packages count: cards made by the current skill, i.e. info.json has "schema": "loom-card/2" AND
"report": "generated" (fresh measurements through make_info.py -> render_card.py). Cards imported from old runs
("report": "attached", import_card.py) are listed as skipped, never used: their programs predate the current
starter and their numbers were not re-measured.

  find_reference.py rank GGUF [--cfg cfg.json] [--repo ORG/REPO] [--org loom-fun] [--top 5] [-o reference.json]
      Profiles the target from the GGUF's metadata (architecture, family, size, layer pattern: gated delta-net
      hybrid / MoE / sliding window, MTP block, chat template's tool-call format and thinking switch; with --cfg,
      inspect_gguf.py's exact values; with --repo, whether the repo ships a vision projector) and ranks every
      category A repo of the org: same architecture and family first, then same architecture, then shared building
      blocks (hybrid, MoE routing, vision tower family, chat/tool format, drafter type), nearest size last.
      Writes the ranking (with the reasons) to -o and prints it.
  find_reference.py apply CFG reference.json [--pick ORG/REPO]
      Starts the port's cfg.json from the best reference (or --pick), when it has the same architecture:
      fills every key the port's inspect_gguf.py did not set (hand-reviewed features), takes its chat block when
      the port's own derivation failed (port.sh then verifies it with chat_probe.py --verify against this model's
      template), and records cfg["reference"] (repo, commit, score, reasons, generator stamp, what was taken, and
      the reference's measured drafting table as a prior for draft/tune.py). build.py copies cfg into
      llm_config.json, so the published package says which reference it was derived from. Keys the GGUF determines
      (geometry, formats, eos ids) always stay the port's own; differences are printed as review items.
  find_reference.py fetch ORG/REPO DIR
      The reference's package without weights (model.json, lock, llm_config.json, info.json, README/REPORT,
      program/, *.loom policies, kit/ if published) into DIR, to read or diff (e.g. a program against your build).
  find_reference.py example-cfg EXAMPLE [-o FILE]
      A reference example's reviewed config (reference/examples.json): its category A package's llm_config.json at
      the pinned revision, without the package's own keys (EXCLUDE below); reference/apply_example.sh merges it.
  find_reference.py show reference.json|ORG/REPO
      What an optimization should start from: the reference's drafting choice and measured table, execution
      settings (prefill chunk, pool), its llama.cpp baseline arguments, per-metric margins and gate verdicts.

Standard library only (+ gguf for a local GGUF). Downloads are cached in ~/.cache/loom-skills/references/<repo>/<sha>.
"""
import argparse
import json
import os
import re
import sys
import urllib.request

HF = "https://huggingface.co"
CACHE = os.path.join(os.path.expanduser(os.environ.get("XDG_CACHE_HOME", "~/.cache")), "loom-skills", "references")
SCHEMA = "loom-card/2"
# Keys inspect_gguf.py derives from the GGUF (or that name this package's own files): never taken from a reference.
OWN = {"name", "gguf", "tensors", "model_id", "draft", "drafter", "assistant", "vision", "generator", "reference",
       "mtp", "hidden", "layers", "heads", "kv_heads", "kv_heads_global", "head_dim", "ffn", "vocab", "context_length", "rms_eps",
       "rope_theta", "eos_id", "extra_eos_ids", "bos_id", "recurrent_layers", "kv_layers", "gdn", "moe",
       "sliding_layers", "nope_layers", "global_layers", "nextn_layers", "mrope"}


# What apply_example.sh never copies from an example's config into a port's: the port's own GGUF, tensor table and
# name, and what names one package's files (model id, drafter, vision projector, build stamps).
EXCLUDE = {"gguf", "tensors", "name", "model_id", "draft", "drafter", "assistant", "vision", "generator", "reference"}
EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "examples.json")


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "loom-skills find_reference"})
    tok = os.environ.get("HF_TOKEN")
    if tok and url.startswith(HF):
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def cached(repo, sha, name):
    """repo file at commit sha, cached; None when the repo has no such file."""
    d = os.path.join(CACHE, repo.replace("/", "--"), sha)
    p = os.path.join(d, name)
    if os.path.exists(p):
        return open(p, "rb").read()
    if os.path.exists(p + ".missing"):
        return None
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        data = get(f"{HF}/{repo}/resolve/{sha}/{name}")
    except urllib.error.HTTPError as e:
        if e.code in (404, 401, 403):
            open(p + ".missing", "w").close()
            return None
        raise
    with open(p, "wb") as f:
        f.write(data)
    return data


def jfile(repo, sha, name):
    b = cached(repo, sha, name)
    try:
        return json.loads(b) if b else None
    except ValueError:
        return None


# ---------------------------------------------------------------- profiles

def call_format(tpl):
    """The tool-call format a chat template writes (chat_probe.py's names; a heuristic, exact with --cfg)."""
    if not tpl:
        return None
    if "<function=" in tpl and "<tool_call>" in tpl:
        return "qwen_xml"
    if "<tool_call>" in tpl:
        return "hermes"
    if "<|tool_call_start|>" in tpl:
        return "pythonic"
    if "<|turn>" in tpl or "<|tool_call>" in tpl:
        return "gemma4"
    if "<arg_key>" in tpl or "<|observation|>" in tpl:
        return "glm"
    if "<|channel|>" in tpl:
        return "atem"
    return None


def family_of(name):
    """'Qwen3.5-9B' -> 'qwen3.5'; 'ornith-1.5' -> 'ornith-1.5' (size and quant suffixes dropped)."""
    n = (name or "").lower().replace(" ", "-").replace("_", "-")
    n = re.sub(r"-(\d+(\.\d+)?[bm]|e\d+b|a\d+(\.\d+)?b|\d+gb|instruct|it|chat|base|gguf|preview)(?=-|$)", "", n)
    return n.strip("-") or None


def gguf_profile(path, repo=None):
    from gguf import GGUFReader
    r = GGUFReader(path)

    def v(k, default=None):
        f = r.fields.get(k)
        if f is None:
            return default
        try:
            return f.contents()
        except Exception:  # noqa: BLE001
            return default
    arch = v("general.architecture")
    a = lambda k, d=None: v(f"{arch}.{k}", d)  # noqa: E731
    bases = [v(f"general.base_model.{i}.name") for i in range(int(v("general.base_model.count", 0) or 0))]
    tpl = v("tokenizer.chat_template") or ""
    kv = a("attention.head_count_kv")
    p = {"arch": arch, "name": v("general.name"),
         "families": sorted({f for f in [family_of(v("general.basename")), family_of(v("general.name"))]
                             + [family_of(b) for b in bases] if f}),
         "layers": a("block_count"), "hidden": a("embedding_length"), "heads": a("attention.head_count"),
         "kv_heads": max(kv) if isinstance(kv, list) else kv, "head_dim": a("attention.key_length"),
         "experts": a("expert_count") or 0, "hybrid": bool(a("full_attention_interval") or a("ssm.state_size")
                                                            or a("shortconv.l_cache")),
         "shortconv": bool(a("shortconv.l_cache")),
         "sliding_window": a("attention.sliding_window") or 0, "mtp": bool(a("nextn_predict_layers")),
         "call_format": call_format(tpl), "thinking": "toggle" if "enable_thinking" in tpl else None,
         "vision": None, "context_length": a("context_length")}
    if repo:
        try:
            m = json.loads(get(f"{HF}/api/models/{repo}"))
            files = [s["rfilename"] for s in m["siblings"]]
            p["vision"] = "mmproj" if any("mmproj" in f.lower() for f in files) else None
            # The model card's base_model chain (a GGUF repo -> the model -> what it was fine-tuned from): a
            # fine-tune is in its base's family (two hops at most).
            seen, todo = set(), [m]
            for _ in range(3):
                nxt = []
                for x in todo:
                    b = (x.get("cardData") or {}).get("base_model") or []
                    for name in [b] if isinstance(b, str) else b:
                        if name not in seen:
                            seen.add(name)
                            try:
                                nxt.append(json.loads(get(f"{HF}/api/models/{name}")))
                            except Exception:  # noqa: BLE001
                                pass
                todo = nxt
            p["families"] = sorted(set(p["families"]) | {family_of(n.split("/")[-1]) for n in seen} - {None})
        except Exception:  # noqa: BLE001
            pass
    return p


def cfg_profile(c, p=None):
    """A package's llm_config.json (or a port's cfg.json) -> the same profile keys."""
    p = dict(p or {})
    chat = c.get("chat") or {}
    moe = c.get("moe") or {}
    p.update({k: v for k, v in {
        "arch": c.get("arch"), "name": c.get("name"), "layers": c.get("layers"), "hidden": c.get("hidden"),
        "heads": c.get("heads"), "kv_heads": c.get("kv_heads"), "head_dim": c.get("head_dim"),
        "experts": moe.get("experts") or c.get("experts") or 0,
        "hybrid": bool(c.get("recurrent_layers") or c.get("gdn") or c.get("ssm_heads") or c.get("sconv") or c.get("shortconv_layers")),
        "shortconv": bool(c.get("shortconv_layers")),
        "sliding_window": c.get("sliding_window") or 0, "mtp": bool(c.get("mtp") or c.get("nextn_layers")),
        "call_format": chat.get("call_format"), "thinking": chat.get("thinking"),
        "context_length": c.get("context_length")}.items() if v is not None})
    if c.get("vision"):
        p["vision"] = (c["vision"] or {}).get("family") or "mmproj"
    p.setdefault("families", [f for f in [family_of(c.get("name"))] if f])
    if c.get("name") and family_of(c["name"]) not in p["families"]:
        p["families"] = sorted(set(p["families"]) | {family_of(c["name"])})
    return p


def score(t, r, ref_cfg):
    """(score, reasons): same arch and family first, then same arch, then shared building blocks, nearest size."""
    s, why = 0.0, []
    same_arch = t.get("arch") and t["arch"] == r.get("arch")
    if same_arch:
        s += 100; why.append(f"same architecture ({t['arch']})")
    fam = sorted(set(t.get("families") or []) & set(r.get("families") or []))
    if fam and same_arch:
        s += 50; why.append(f"same family ({fam[0]})")
    blocks = [("hybrid", "recurrent hybrid layers"), ("mtp", "MTP block")]
    for k, label in blocks:
        if t.get(k) and r.get(k):
            s += 8; why.append(label)
    if t.get("shortconv") and r.get("shortconv"):   # the same recurrent-layer kernels (LFM short conv): the kit's core
        s += 20; why.append("LFM short-conv layers")
    if t.get("experts") and r.get("experts"):
        s += 12; why.append(f"MoE routing ({r['experts']} experts)")
    if t.get("sliding_window") and r.get("sliding_window"):
        s += 4; why.append("sliding-window attention")
    if t.get("call_format") and t["call_format"] == r.get("call_format"):
        s += 10; why.append(f"tool-call format {t['call_format']}")
    if t.get("thinking") and t["thinking"] == r.get("thinking"):
        s += 2; why.append("thinking switch")
    if t.get("vision") and r.get("vision"):
        s += 5; why.append(f"vision tower ({r['vision']})")
    dk = ((ref_cfg.get("drafter") or {}).get("kind") or (ref_cfg.get("draft") or {}).get("kind"))
    if dk and dk != "none":
        if (dk == "mtp" and t.get("mtp")) or dk in ("dflash", "ngram"):
            s += 2; why.append(f"drafter type {dk}")
    if t.get("head_dim") and t["head_dim"] == r.get("head_dim"):
        s += 2; why.append(f"head_dim {t['head_dim']}")
    if t.get("hidden") and r.get("hidden"):   # nearest size: up to 5 points, log-distance of the hidden size
        import math
        s += max(0.0, 5 - 3 * abs(math.log2(t["hidden"] / r["hidden"])))
    return round(s, 2), why


# ---------------------------------------------------------------- org listing

def candidates(org):
    rows, skipped = [], []
    for m in json.loads(get(f"{HF}/api/models?author={org}&full=true&limit=1000")):
        repo, sha = m["id"], m.get("sha")
        files = {s["rfilename"] for s in m.get("siblings") or []}
        if not sha or "info.json" not in files or "llm_config.json" not in files:
            skipped.append({"repo": repo, "why": "no info.json/llm_config.json"})
            continue
        info = jfile(repo, sha, "info.json") or {}
        if info.get("schema") != SCHEMA or info.get("report") != "generated":
            skipped.append({"repo": repo, "why": f"category B (schema {info.get('schema')}, report {info.get('report')})"})
            continue
        cfg = jfile(repo, sha, "llm_config.json") or {}
        rows.append({"repo": repo, "sha": sha, "info": info, "cfg": cfg, "files": sorted(files),
                     "last_modified": m.get("lastModified")})
    return rows, skipped


def plain_q4km(name):
    n = os.path.basename(name or "").upper()
    return "Q4_K_M" in n and "UD-" not in n


def gate_summary(info):
    g = info.get("gates") or {}
    bad = sorted(k for k, v in g.items() if (v or {}).get("status") != "pass")
    return {"gates": len(g), "failing": bad, "levels": sorted({(v or {}).get("level") for v in g.values() if (v or {}).get("level")})}


def cmd_rank(a):
    t = gguf_profile(a.gguf, a.repo)
    if a.cfg and os.path.exists(a.cfg):
        t = cfg_profile(json.load(open(a.cfg, encoding="utf-8")), t)
    rows, skipped = candidates(a.org)
    ranked = []
    for r in rows:
        rp = cfg_profile(r["cfg"])
        sc, why = score(t, rp, r["cfg"])
        gen = r["cfg"].get("generator") or {}
        ranked.append({"repo": r["repo"], "sha": r["sha"], "score": sc, "reasons": why, "arch": rp.get("arch"),
                       "name": rp.get("name"), "category": "A",
                       "generator": {"skills_commit": gen.get("skills_commit"), "kit": gen.get("kit"),
                                     "hrx_loom": (gen.get("hrx_loom") or {}).get("hrx_loom") if isinstance(gen.get("hrx_loom"), dict) else gen.get("hrx_loom")},
                       "kit_example": ((gen.get("kit") or {}).get("example"))
                       or next((k for k, e in examples().items() if e.get("card") == r["repo"]), None),
                       "kit_published": any(f.startswith("kit/") for f in r["files"]),
                       "drafter": {k: (r["cfg"].get("drafter") or {}).get(k) for k in ("kind", "depth")},
                       "quality": gate_summary(r["info"]),
                       # still a reference, but loom-org's rule says the repo itself needs an update (plain Q4_K_M only)
                       "plain_q4_k_m": plain_q4km(r["cfg"].get("gguf"))})
    ranked.sort(key=lambda x: -x["score"])
    out = {"target": t, "org": a.org, "rule": "category A only: info.json schema loom-card/2 and report generated",
           "ranked": ranked[:a.top], "skipped": skipped}
    if a.o:
        json.dump(out, open(a.o, "w", encoding="utf-8"), indent=1)
    print(f"target: {t['arch']} {t.get('name')} (families {t.get('families')}, L={t.get('layers')} H={t.get('hidden')}, "
          f"hybrid={t.get('hybrid')} experts={t.get('experts')} mtp={t.get('mtp')} calls={t.get('call_format')})")
    for x in ranked[:a.top]:
        q = x["quality"]
        print(f"  {x['score']:6.1f}  {x['repo']}  [{'; '.join(x['reasons'])}]{'' if x['plain_q4_k_m'] else '  (not plain Q4_K_M)'}  generator {str(x['generator']['skills_commit'])[:9]}"
              f"{' kit ' + x['kit_example'] if x['kit_example'] else ''}; gates {q['gates']} ({'all pass' if not q['failing'] else 'failing: ' + ','.join(q['failing'])})")
    print(f"  ({len(skipped)} repos skipped: not category A)")
    best = next((x for x in ranked if x["arch"] == t.get("arch")), None)
    print("REFERENCE_RESULT " + json.dumps({"best": best and best["repo"], "score": best and best["score"],
                                             "same_arch": bool(best), "kit_example": best and best["kit_example"]}))


# ---------------------------------------------------------------- apply / fetch / show

def pick(ref, repo=None):
    for x in ref["ranked"]:
        if (repo and x["repo"] == repo) or (not repo and x["arch"] == ref["target"].get("arch")):
            return x
    return None


def cmd_apply(a):
    ref = json.load(open(a.reference, encoding="utf-8"))
    x = pick(ref, a.pick)
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    if not x:
        print("reference: no category A package of this architecture; starting from the kit alone")
        return
    rc = jfile(x["repo"], x["sha"], "llm_config.json") or {}
    if rc.get("arch") != cfg.get("arch"):
        print(f"reference: {x['repo']} is {rc.get('arch')}, this port {cfg.get('arch')}: not merged (read it with show/fetch)")
        return
    taken, review = [], []
    for k, v in rc.items():
        if k in OWN or k == "chat":
            continue
        if k not in cfg:
            cfg[k] = v; taken.append(k)
        elif cfg[k] != v:
            review.append(k)
    chat_ok = bool(cfg.get("chat")) and not a.chat_failed
    if not chat_ok and rc.get("chat"):
        cfg["chat"] = rc["chat"]; taken.append("chat")
    elif rc.get("chat") and cfg.get("chat") != rc["chat"]:
        review.append("chat")
    d = rc.get("drafter") or {}
    cfg["reference"] = {"repo": x["repo"], "revision": x["sha"], "category": "A", "score": x["score"],
                        "reasons": x["reasons"], "generator": x["generator"], "kit_example": x["kit_example"],
                        "taken": taken, "differs": review,
                        "prior": {"drafter": {"kind": d.get("kind"), "depth": d.get("depth"), "tuned": d.get("tuned")}}}
    json.dump(cfg, open(a.cfg, "w", encoding="utf-8"), indent=1)
    print(f"reference: {x['repo']}@{x['sha'][:9]} (score {x['score']}): took {taken or 'nothing (inspect derived every key)'}; "
          f"differs (this GGUF's own value kept, review if unexpected): {review or 'none'}; "
          f"drafting prior {d.get('kind')} depth {d.get('depth')}")


def cmd_fetch(a):
    m = json.loads(get(f"{HF}/api/models/{a.repo}"))
    sha = m["sha"]
    names = [s["rfilename"] for s in m["siblings"] if not s["rfilename"].endswith((".gguf", ".safetensors"))
             and s["rfilename"] != ".gitattributes"]
    for n in names:
        b = cached(a.repo, sha, n)
        if b is not None:
            p = os.path.join(a.dir, n)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            open(p, "wb").write(b)
    print(f"fetched {a.repo}@{sha[:9]}: {len(names)} files into {a.dir}")


def examples():
    return {k: v for k, v in json.load(open(EXAMPLES, encoding="utf-8")).items() if not k.startswith("_")}


def example_cfg(name):
    """(config, card) of a reference example: its cfg.json, else its category A package's llm_config.json."""
    name = name[:-len("-example")] if name.endswith("-example") else name
    local = os.path.join(os.path.dirname(EXAMPLES), f"{name}-example", "cfg.json")
    if os.path.isfile(local):
        return json.load(open(local, encoding="utf-8")), None
    e = examples().get(name) or {}
    if not e.get("card"):
        sys.exit(f"example {name}: no cfg.json and no category A package in {EXAMPLES}")
    info = jfile(e["card"], e["revision"], "info.json") or {}
    if info.get("schema") != SCHEMA or info.get("report") != "generated":
        sys.exit(f"{e['card']}@{e['revision'][:9]} is not category A (schema {info.get('schema')}, report {info.get('report')})")
    cfg = jfile(e["card"], e["revision"], "llm_config.json")
    if not cfg or cfg.get("arch") != e["arch"]:
        sys.exit(f"{e['card']}@{e['revision'][:9]}: llm_config.json missing or not arch {e['arch']}")
    return cfg, f"{e['card']}@{e['revision'][:9]}"


def cmd_example_cfg(a):
    cfg, card = example_cfg(a.example)
    out = {k: v for k, v in cfg.items() if k not in EXCLUDE}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.o:
        open(a.o, "w", encoding="utf-8").write(text)
        print(f"{a.example}: reviewed config from {card or 'the example cfg.json'} ({len(out)} keys)", file=sys.stderr)
    else:
        print(text)


def cmd_show(a):
    if os.path.exists(a.what):
        x = pick(json.load(open(a.what, encoding="utf-8")))
        if not x:
            sys.exit("no same-architecture category A reference")
        repo, sha = x["repo"], x["sha"]
    else:
        repo = a.what; sha = json.loads(get(f"{HF}/api/models/{repo}"))["sha"]
    info, cfg, card = (jfile(repo, sha, n) or {} for n in ("info.json", "llm_config.json", "model.json"))
    ex = next(iter((card.get("presets") or {}).values()), {}).get("execution", {})
    d = cfg.get("drafter") or {}
    print(f"{repo}@{sha[:9]} ({'category A' if info.get('schema') == SCHEMA and info.get('report') == 'generated' else 'NOT category A'})")
    print(f"  generator: {json.dumps(cfg.get('generator'))}")
    print(f"  drafting: {d.get('kind')} depth {d.get('depth')}; tuned: " + ", ".join(
        f"{r['kind']}/{r['depth']}={r['decode_tps']}" for r in d.get("tuned") or []))
    print(f"  execution: {json.dumps(ex)}")
    perf = info.get("performance") or {}
    print(f"  llama.cpp baseline: {json.dumps(perf.get('baseline'))}")
    for s in perf.get("sessions") or []:
        for k, m in (s.get("metrics") or {}).items():
            if m.get("loom") and m.get("baseline"):
                print(f"    {k}: loom {m['loom']} vs {m['baseline']} ({100 * (m['loom'] / m['baseline'] - 1):+.0f} %)")
    print(f"  gates: {json.dumps(gate_summary(info))}")
    if cfg.get("reference"):
        print(f"  itself derived from: {cfg['reference'].get('repo')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank")
    r.add_argument("gguf"); r.add_argument("--cfg"); r.add_argument("--repo")
    r.add_argument("--org", default="loom-fun"); r.add_argument("--top", type=int, default=5); r.add_argument("-o")
    p = sub.add_parser("apply")
    p.add_argument("cfg"); p.add_argument("reference"); p.add_argument("--pick")
    p.add_argument("--chat-failed", action="store_true", help="the port's own chat derivation failed: take the reference's")
    f = sub.add_parser("fetch")
    f.add_argument("repo"); f.add_argument("dir")
    e = sub.add_parser("example-cfg")
    e.add_argument("example"); e.add_argument("-o")
    s = sub.add_parser("show")
    s.add_argument("what")
    a = ap.parse_args(argv)
    {"rank": cmd_rank, "apply": cmd_apply, "fetch": cmd_fetch, "show": cmd_show, "example-cfg": cmd_example_cfg}[a.cmd](a)


if __name__ == "__main__":
    main()
