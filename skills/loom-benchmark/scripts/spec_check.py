"""Speculative-decoding equivalence gate: a drafting package must produce exactly the target's own output.

Drafting only changes how many tokens one target pass verifies; every committed token is the target's own
choice (greedy, or the same counter-based draw at the same position with a fixed seed). So the gate compares the
package as built (drafting on) with the same package and the same compiled program with drafting off (a copy of
model.json whose execution has draft_depth 0) on identical requests.

usage:
  spec_check.py card PKG                 -> writes PKG/model_target.json (draft_depth 0, one epoch per turn)
  spec_check.py record URL MODEL OUT     -> completions of the fixed request set (greedy + seeded sampling) as JSON lines
  spec_check.py compare TARGET.jsonl DRAFT.jsonl [--metrics URL | --metrics-file FILE]
                                         -> exit 1 unless every completion is identical; prints acceptance if --metrics
gates.sh / port.sh run: record on the drafting server, stop it, serve model_target.json, record, compare.
Near-identical is not identical: a difference means verification, acceptance or state replay is wrong
(or, rarely, a near-tie that multi-token verification rounds differently from 1-token decode: inspect it).
"""
import json
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


def requests_set(model):
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
    return out


def record(url, model, path):
    with open(path, "w") as f:
        for i, body in enumerate(requests_set(model)):
            r = post(url, body)
            c = r["choices"][0]
            m = c["message"]
            f.write(json.dumps({"i": i, "finish": c["finish_reason"], "n": r["usage"]["completion_tokens"],
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


def compare(a_path, b_path, metrics=None):
    a = [json.loads(l) for l in open(a_path)]
    b = [json.loads(l) for l in open(b_path)]
    bad = 0
    for x, y in zip(a, b):
        same = all(x[k] == y[k] for k in ("content", "reasoning", "calls", "n"))
        if not same:
            bad += 1
            s1, s2 = (x["reasoning"] or "") + (x["content"] or ""), (y["reasoning"] or "") + (y["content"] or "")
            k = next((j for j, (c1, c2) in enumerate(zip(s1, s2)) if c1 != c2), min(len(s1), len(s2)))
            print(f"DIFF request {x['i']} at char {k}: target {s1[max(0, k - 40):k + 40]!r} | draft {s2[max(0, k - 40):k + 40]!r}")
    if len(a) != len(b):
        bad += 1
        print(f"DIFF: {len(a)} vs {len(b)} records")
    acc = ""
    if metrics:
        text = (urllib.request.urlopen(metrics + "/metrics", timeout=30).read().decode() if metrics.startswith("http")
                else open(metrics).read())
        vals = {}
        for line in text.splitlines():
            for k in ("draft_proposed_tokens", "draft_accepted_inputs"):
                if f'field="{k}"' in line:
                    vals[k] = float(line.split()[-1])
        if vals.get("draft_proposed_tokens"):
            acc = f", acceptance {vals['draft_accepted_inputs'] / vals['draft_proposed_tokens']:.0%} of {int(vals['draft_proposed_tokens'])} proposals"
    print(f"SPEC_CHECK {'PASS' if not bad else 'FAIL'}: {len(a) - bad}/{len(a)} completions identical{acc}")
    return 0 if not bad else 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "card":
        card(sys.argv[2])
    elif cmd == "record":
        record(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "compare":
        flag = "--metrics" if "--metrics" in sys.argv else ("--metrics-file" if "--metrics-file" in sys.argv else None)
        m = sys.argv[sys.argv.index(flag) + 1] if flag else None
        sys.exit(compare(sys.argv[2], sys.argv[3], m))
