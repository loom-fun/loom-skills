"""Speculative-decoding equivalence gate: a drafting package must produce exactly the target's own output.

Drafting only changes how many tokens one target pass verifies; every committed token is the target's own
choice (greedy, or the same counter-based draw at the same position with a fixed seed). So the gate compares the
package as built (drafting on) with the same package and the same compiled program with drafting off (a copy of
model.json whose execution has draft_depth 0) on identical requests; a package with image input (/v1/models lists
"image") adds image questions (greedy and seeded sampling), since M-RoPE positions flow through verify and drafting.

usage:
  spec_check.py card PKG                 -> writes PKG/model_target.json (draft_depth 0, one epoch per turn)
  spec_check.py record URL MODEL OUT [--lean]
                                         -> completions of the fixed request set (greedy + seeded sampling) as JSON lines;
                                            --lean (gates.sh LEVEL=dev): 5 requests (6 with images) of at most 128 tokens (greedy, seeded,
                                            tool call, thinking, one image when the package takes images)
  spec_check.py compare TARGET.jsonl DRAFT.jsonl [--metrics URL | --metrics-file FILE] [--pkg PKG --llama URL]
                                         -> exit 1 unless every completion is identical or (with --llama) every
                                            greedy divergence is a near-tie; prints acceptance if --metrics
gates.sh / port.sh run: record on the drafting server, stop it, serve model_target.json, record, compare.
A difference is either a bug (verification, acceptance or state replay) or a near-tie flipped by rounding: wide
verification (more than 4 tokens per epoch) takes the f16 WMMA GEMM where 1-token decode takes the f32 matvec,
the same mixed-epoch effect consistency.py documents. With --pkg/--llama a greedy divergence passes when llama.cpp
puts both tokens in its top-10 within the near-tie rule (bench_common.near_tie); a sampled divergence passes when
the same prompt's greedy pair is identical or a near-tie. Everything must still be deterministic.
"""
import json
import os
import sys
import urllib.request

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
    "List the first ten prime numbers separated by commas.",
    "Explain how a hash map works, in about 150 words.",
    "Count from 1 to 60, separated by spaces.",
]
DOC = ("Speculative decoding drafts several tokens cheaply and lets the large model verify them in one pass. "
       "Each verified token is exactly what the large model would have produced on its own. ") * 40
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def image_requests(model):
    """Packages with image input: drafting must not change image answers either (M-RoPE positions in verify/draft)."""
    from vision_golden import builtin_images, data_url
    ims = builtin_images()
    out = []
    for name, samp in (("shapes", {"temperature": 0}), ("wide", {"temperature": 0}),
                       ("shapes", {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "seed": 7})):
        im, q = ims[name]
        out.append({"model": model, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(im)}},
                                                                             {"type": "text", "text": q + " Answer in detail."}]}],
                    "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}, **samp})
    return out


def has_images(url):
    try:
        d = json.load(urllib.request.urlopen(url + "/v1/models", timeout=30))
        return any("image" in (m.get("input_modalities") or []) for m in d.get("data", []))
    except Exception:  # noqa: BLE001
        return False


def requests_set(model, images=False):
    out = []
    for i, p in enumerate(PROMPTS):
        for samp in ({"temperature": 0}, {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "seed": 1000 + i}):
            out.append({"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 256,
                        "chat_template_kwargs": {"enable_thinking": False}, **samp})
    out.append({"model": model, "messages": [{"role": "user", "content": DOC + "\n\nSummarize the text above in three sentences."}],
                "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    out.append({"model": model, "messages": [{"role": "user", "content": "What's the weather in Paris and in Rome?"}], "tools": TOOLS,
                "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    out.append({"model": model, "messages": [{"role": "user", "content": "Think step by step: what is 17 * 23?"}],
                "max_tokens": 300, "temperature": 0})
    return out + (image_requests(model) if images else [])


def lean_set(model, images=False):
    """A few requests that still cross every drafting path: greedy, seeded sampling, a tool call, a thinking turn
    and (with image input) one image; 128 tokens each (several verify epochs at depth 3)."""
    full = requests_set(model, images)
    k = 3 * images   # the image requests come last
    pick = [full[0], full[2], full[3], full[-2 - k], full[-1 - k]] + full[-3:-2] * images
    return [dict(b, max_tokens=min(b["max_tokens"], 128)) for b in pick]


def record(url, model, path, lean=False):
    with open(path, "w") as f:
        for i, body in enumerate((lean_set if lean else requests_set)(model, has_images(url))):
            r = post(url, body)
            c = r["choices"][0]
            m = c["message"]
            f.write(json.dumps({"i": i, "request": body, "message": m, "finish": c["finish_reason"], "n": r["usage"]["completion_tokens"],
                                "content": m.get("content"), "reasoning": m.get("reasoning_content"),
                                "calls": [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []]}) + "\n")
            f.flush()


def card(pkg):
    m = json.load(open(f"{pkg}/model.json"))
    for p in m["presets"].values():
        ex = p.get("execution", {})
        ex["draft_depth"] = 0
        ex.pop("continuation_epochs", None)
    json.dump(m, open(f"{pkg}/model_target.json", "w"), indent=1)
    print(f"{pkg}/model_target.json")


def near_tie_divergence(x, y, pkg, llama):
    """(ok, message) for a greedy divergence: llama.cpp's log-probs at the first differing token."""
    from bench_common import Package, llama_greedy, near_tie
    P = Package(pkg)
    req = x["request"]
    think = (req.get("chat_template_kwargs") or {}).get("enable_thinking")
    prompt = P.render_messages(req["messages"], req.get("tools"), enable_thinking=think)
    t1 = P.ids(P.raw_from_chat(x["message"], x["finish"] == "stop", think, prompt))
    t2 = P.ids(P.raw_from_chat(y["message"], y["finish"] == "stop", think, prompt))
    k = next((j for j, (u, v) in enumerate(zip(t1, t2)) if u != v), min(len(t1), len(t2)))
    if k >= min(len(t1), len(t2)):
        return False, f"one completion is a prefix of the other at token {k} (not a near-tie)"
    g = llama_greedy(llama, P.ids(prompt) + t1[:k], 1, P, n_probs=10)[0]
    top = {t["id"]: t["logprob"] for t in g["completion_probabilities"][0]["top_logprobs"]}
    ok = near_tie(top, t1[k], t2[k])
    gap = abs(top[t1[k]] - top[t2[k]]) if t1[k] in top and t2[k] in top else None
    return ok, f"first divergence at token {k}: {P.tok.decode(t1[k:k + 1])!r} vs {P.tok.decode(t2[k:k + 1])!r}, llama.cpp gap {gap}"


def compare(a_path, b_path, metrics=None, pkg=None, llama=None):
    a = [json.loads(l) for l in open(a_path)]
    b = [json.loads(l) for l in open(b_path)]
    bad = ties = 0
    greedy_ok = {}
    for x, y in zip(a, b):
        same = all(x[k] == y[k] for k in ("content", "reasoning", "calls", "n"))
        req = x.get("request") or {}
        key = json.dumps(req.get("messages"))
        greedy = not req.get("temperature")
        image = any(isinstance(m.get("content"), list) for m in req.get("messages") or [])   # judged exact (no llama.cpp near-tie)
        if same:
            greedy_ok.setdefault(key, True)
            continue
        s1, s2 = (x["reasoning"] or "") + (x["content"] or ""), (y["reasoning"] or "") + (y["content"] or "")
        k = next((j for j, (c1, c2) in enumerate(zip(s1, s2)) if c1 != c2), min(len(s1), len(s2)))
        print(f"DIFF request {x['i']} at char {k}: target {s1[max(0, k - 40):k + 40]!r} | draft {s2[max(0, k - 40):k + 40]!r}")
        ok = False
        if pkg and llama and greedy and "message" in x and not image:
            ok, why = near_tie_divergence(x, y, pkg, llama)
            print(f"   {'near-tie' if ok else 'NOT a near-tie'}: {why}")
            greedy_ok[key] = ok
        elif pkg and llama and not greedy and not image:
            ok = greedy_ok.get(key, False)
            print(f"   sampled: {'accepted (its greedy pair is identical or a near-tie)' if ok else 'not accepted'}")
        ties += ok
        bad += not ok
    if len(a) != len(b):
        bad += 1
        print(f"DIFF: {len(a)} vs {len(b)} records")
    acc = ""
    if metrics:
        from bench_common import draft_counters
        text = (urllib.request.urlopen(metrics + "/metrics", timeout=30).read().decode() if metrics.startswith("http")
                else open(metrics).read())
        vals = draft_counters(text)
        if vals.get("draft_proposed_tokens"):
            acc = f", acceptance {vals['draft_accepted_inputs'] / vals['draft_proposed_tokens']:.0%} of {int(vals['draft_proposed_tokens'])} proposals"
    print(f"SPEC_CHECK {'PASS' if not bad else 'FAIL'}: {len(a) - bad - ties}/{len(a)} completions identical, {ties} near-tie{acc}")
    return 0 if not bad else 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "card":
        card(sys.argv[2])
    elif cmd == "record":
        record(sys.argv[2], sys.argv[3], sys.argv[4], "--lean" in sys.argv[5:])
    elif cmd == "compare":
        flag = "--metrics" if "--metrics" in sys.argv else ("--metrics-file" if "--metrics-file" in sys.argv else None)
        m = sys.argv[sys.argv.index(flag) + 1] if flag else None
        opt = lambda f: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else None
        sys.exit(compare(sys.argv[2], sys.argv[3], m, opt("--pkg"), opt("--llama")))
