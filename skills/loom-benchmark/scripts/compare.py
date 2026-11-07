"""Correctness gate: greedy Loom generations vs the llama.cpp golden.

usage: python compare.py PACKAGE GOLDEN.json [--url http://127.0.0.1:8160] [--out compare.json]

Per prompt: length of the token prefix matching the golden, and at the first
divergence the golden log-prob gap  logp(golden token) - logp(loom token).
Greedy decoding diverges at near-ties under any change of summation order, so
the gate is about *where* divergences happen, not whether:
  PASS   every divergence is a near-tie (bench_common.near_tie: gap <= 1.0 with Loom's token in the golden
         top-5, or Loom picked llama.cpp's second choice where its first has p < 0.5; with --llama, gap <= 1.0
         plus llama.cpp's own rounding noise at that position, bench_common.llama_views)
  FAIL   any divergence with a large gap, or a token outside the golden top-5
(Two llama.cpp builds disagree with each other in the same way.)
"""
import argparse
import json
import time

import requests

from bench_common import Package, gap_spread, llama_greedy, llama_views, near_tie

ap = argparse.ArgumentParser()
ap.add_argument("package")
ap.add_argument("golden")
ap.add_argument("--url", default="http://127.0.0.1:8160")
ap.add_argument("--out", default="compare.json")
ap.add_argument("--max-gap", type=float, default=1.0)
ap.add_argument("--alt", default=None, help="second golden recording; per prompt the better-matching one is used")
ap.add_argument("--llama", default=None, help="llama.cpp URL: a prompt that would FAIL against the golden is re-recorded "
                "once (llama.cpp is not run-to-run deterministic at near-ties) and scored against the better recording; "
                "replaces a full second recording (--alt)")
a = ap.parse_args()
pkg = Package(a.package)
rows = []
alt = {g["prompt"]: g for g in json.load(open(a.alt))} if a.alt else {}


THINK_END = pkg.tok.token_to_id(pkg.chat.get("think_end", "</think>"))


def strip_ws_after_think(g):
    """Whitespace after </think> belongs to neither channel: depending on the chat format Loom's control program
    drops it or raw_from_chat re-adds it from think_close, so it is dropped on both sides (the golden's token and
    its top-5 entry here; Loom's reconstructed ids at the call site) before matching. llama.cpp's own ids stay in
    llama_ids, with the dropped position in cut."""
    ids, top = list(g["ids"]), list(g["top"])
    for i in range(len(ids) - 1):
        if ids[i] == THINK_END and pkg.tok.decode([ids[i + 1]]).strip() == "":
            del ids[i + 1]
            if i + 1 < len(top):
                del top[i + 1]
            return dict(g, ids=ids, top=top, llama_ids=g["ids"], cut=i + 1)
    return dict(g, ids=ids, top=top)


def match(got, g):
    gold = g["ids"]
    n = 0
    while n < min(len(got), len(gold)) and got[n] == gold[n]:
        n += 1
    gap, near = None, True
    if n < min(len(got), len(gold)):
        if n >= len(g["top"]):
            gap = 0.0  # no top-5 recorded here (byte-split token); counted as a near-tie
        else:
            top = dict((int(i), lp) for i, lp in g["top"][n])
            gap = (top[gold[n]] - top[got[n]]) if got[n] in top else "not in golden top-5"
            near = near_tie(top, gold[n], got[n], a.max_gap)
            # A split divergence: Loom's token spans llama.cpp's token plus more text ('[A' vs '[' then 'a').
            # The choice was made at the next position, so compare there: the rest of Loom's token against
            # llama.cpp's next token (Qwen3.5-0.8B under weaver: '[A-Za-z' vs '[a-zA-Z', gap 0.043, was
            # reported as "not in golden top-5").
            if got[n] not in top and n + 1 < min(len(gold), len(g["top"])):
                lt, gt = pkg.tok.decode([got[n]]), pkg.tok.decode([gold[n]])
                if len(lt) > len(gt) and lt.startswith(gt):
                    top1 = dict((int(i), lp) for i, lp in g["top"][n + 1])
                    rest = next((i for i in top1 if pkg.tok.decode([i]) == lt[len(gt):]), None)
                    if rest is not None and rest != gold[n + 1]:
                        gap = top1[gold[n + 1]] - top1[rest] if gold[n + 1] in top1 else gap
                        near = near_tie(top1, gold[n + 1], rest, a.max_gap)
    return n, gap, near


alt = {k: strip_ws_after_think(v) for k, v in alt.items()}
for g in json.load(open(a.golden)):
    g = strip_ws_after_think(g)
    t0 = time.perf_counter()
    r = requests.post(a.url + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                      json={"model": pkg.model_id, "messages": [{"role": "user", "content": g["prompt"]}],
                            "max_tokens": len(g["ids"]), "temperature": 0}).json()
    if "choices" not in r:  # e.g. 503 capacity errors: raise --memory_bytes (KV pool must fit)
        print("ERROR", r.get("error"), flush=True)
        rows.append({"prompt": g["prompt"][:80], "error": r.get("error"), "bad": True, "matched": 0, "full_match": False, "divergence_gap": None})
        continue
    ch = r["choices"][0]
    text = pkg.raw_from_chat(ch["message"], ch["finish_reason"] == "stop")
    # Same rule on Loom's side: raw_from_chat re-adds think_close's trailing whitespace (Qwen: "\n</think>\n\n"),
    # which the golden no longer has, so every reasoning answer would diverge right after </think>.
    got = strip_ws_after_think({"ids": pkg.ids(text), "top": []})["ids"]
    n, gap, near = match(got, g)
    # llama.cpp can emit a non-canonical split (e.g. '[' 'A' where the HF tokenizer gives '[A');
    # identical text is a match, whatever the token boundaries.
    gold_text = pkg.tok.decode(g["ids"])
    if n < min(len(got), len(g["ids"])) and (text.startswith(gold_text) or gold_text.startswith(text)):
        n, gap, near = min(len(got), len(g["ids"])), None, True
    if g["prompt"] in alt:  # llama.cpp disagreed with itself? take the recording Loom follows further
        n2, gap2, near2 = match(got, alt[g["prompt"]])
        if n2 > n:
            g, n, gap, near = alt[g["prompt"]], n2, gap2, near2
    ids_p = pkg.ids(pkg.render(g["prompt"]))
    if not near and a.llama and g["prompt"] not in alt:  # on-demand second recording of this prompt only
        r2, _, cp = llama_greedy(a.llama, ids_p, len(g["ids"]), pkg, "rec2")
        if "tokens" in r2:
            g2 = strip_ws_after_think({"prompt": g["prompt"], "prompt_tokens": len(ids_p), "ids": r2["tokens"],
                                       "top": [[(t["id"], t["logprob"]) for t in x["top_logprobs"]] for x in cp]})
            n2, gap2, near2 = match(got, g2)
            print(f"       re-recorded (llama.cpp run-to-run check): {n2} matched, gap {gap2}", flush=True)
            # Further match wins; at the same position the smaller gap wins (llama.cpp itself moved there).
            better_gap = isinstance(gap2, float) and (isinstance(gap, str) or gap2 < gap)
            if n2 > n or (n2 == n and (better_gap or near2)):
                g, n, gap, near = g2, n2, gap2, near2
    noise, top = None, dict((int(i), lp) for i, lp in g["top"][n]) if n < min(len(got), len(g["top"])) else {}
    if not near and a.llama and top and got[n] in top:
        # Judged again with llama.cpp's own rounding noise at the divergence (bench_common.llama_views), on its own
        # tokens before it (the whitespace strip_ws_after_think dropped included).
        views = llama_views(a.llama, ids_p + g.get("llama_ids", g["ids"])[:n + (g.get("cut", n + 1) <= n)], pkg)
        noise = round(gap_spread((top, *views), g["ids"][n], got[n]), 3)
        near = near_tie(top, g["ids"][n], got[n], a.max_gap, views)
        print(f"       llama.cpp's own noise here: the gap moves by {noise} across {len(views)} more views", flush=True)
    gold = g["ids"]
    full = n >= min(len(gold), len(got)) and abs(len(got) - len(gold)) <= 1
    bad = not near
    rows.append({"prompt": g["prompt"][:80], "prompt_tokens": g.get("prompt_tokens"), "golden_tokens": len(gold), "matched": n,
                 "full_match": full, "divergence_gap": gap, "noise": noise, "bad": bad, "seconds": round(time.perf_counter() - t0, 2), "loom_text": text[:160]})
    print(f"{g.get('prompt_tokens', 0):6d} {n:3d}/{len(gold):3d} gap={gap if not isinstance(gap, float) else round(gap, 3)} {'BAD ' if bad else ''}{g['prompt'][:50]!r}", flush=True)
json.dump(rows, open(a.out, "w"), indent=1)
gaps = [r["divergence_gap"] for r in rows if isinstance(r["divergence_gap"], float)]
summary = {"prompts": len(rows), "full_match": sum(r["full_match"] for r in rows),
           "median_matched": sorted(r["matched"] for r in rows)[len(rows) // 2],
           "max_gap": round(max(gaps), 3) if gaps else 0.0, "bad": sum(r["bad"] for r in rows)}
summary["verdict"] = "PASS" if summary["bad"] == 0 else "FAIL"
print(json.dumps(summary))
from campaign import log_gate, run_audit
log_gate("compare", summary["verdict"], golden=a.golden, **{k: summary[k] for k in ("prompts", "full_match", "max_gap", "bad")})
run_audit()
