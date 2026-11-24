#!/usr/bin/env python3
"""Score the exact nine 32K prompts from the Radeon attention ablation."""

import hashlib
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skills/loom-benchmark/scripts"))
from bench_common import (Package, exact_gaps, llama_greedy, long_prompt, near_tie, stream,
                          text_divergence, thinking_extra, top_logprobs)


def prompts():
    for session in range(3):
        for repeat in range(3):
            nonce = f"179547623{session}.06706{repeat:02d}"
            yield f"{session}-{repeat}", long_prompt("decode_tps_32k", nonce)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture(url, package, out):
    rows = {}
    for case, prompt in prompts():
        reply = stream(url, package.model_id, [{"role": "user", "content": prompt}], 128,
                       extra={**thinking_extra("off"), "cache_prompt": False}, timeout=240)
        if (reply["status"], reply["prompt_tokens"], reply["tokens"], reply["finish"]) != (200, 33404, 128, "length"):
            raise ValueError(f"{case}: unexpected Loom reply: {reply}")
        reply["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        reply["output_sha256"] = hashlib.sha256(reply["text"].encode()).hexdigest()
        rows[case] = reply
    write(out, rows)


def record(url, package, ids, count, salt, n_probs=5):
    reply, tokens, tops = llama_greedy(url, ids, count, package, salt=salt, n_probs=n_probs)
    if reply.get("_suspect") or reply.get("timings", {}).get("prompt_n") != len(ids) or len(tokens) != count:
        raise ValueError(f"llama.cpp did not complete exact prompt/token count for {salt}")
    return reply, tokens, tops


def assess(url, package, prompt_ids, result_ids, gold_ids, tops, label):
    first = text_divergence(package.tok, result_ids, gold_ids)
    if first is None:
        return {"kind": "match", "prefix_tokens": min(len(result_ids), len(gold_ids))}
    k, token, fragment = first
    gold = gold_ids[k]
    top = top_logprobs(tops, k)
    views = []
    if not near_tie(top, gold, token):
        _, _, view = record(url, package, prompt_ids + gold_ids[:k], 1, f"{label}-forced", n_probs=10)
        views.append(top_logprobs(view, 0))
    return {"kind": "near-tie" if near_tie(top, gold, token, views=views) else "reference-gap",
            "prefix_tokens": k, "candidate": fragment[:48], "reference": package.tok.decode([gold]),
            "gaps_nats": exact_gaps((top, *views), gold, token), "reference_top": top, "forced_views": views}


def compare_pair(url, package, prompt_ids, old_ids, new_ids, case):
    first = text_divergence(package.tok, new_ids, old_ids)
    if first is None:
        return {"kind": "match", "prefix_tokens": min(len(old_ids), len(new_ids))}
    k, new_token, fragment = first
    old_token = old_ids[k]
    _, _, tops = record(url, package, prompt_ids + old_ids[:k], 1, f"{case}-paired-prefix", n_probs=10)
    top = top_logprobs(tops, 0)
    if not top:
        raise ValueError(f"{case}: no teacher-forced reference probabilities")
    greedy = max(top, key=top.get)
    labels = {name: {"token": token, "text": package.tok.decode([token]),
                     "relative_to_reference": exact_gaps((top,), greedy, token),
                     "reference_near_tie": near_tie(top, greedy, token)}
              for name, token in (("frozen", old_token), ("repaired", new_token))}
    return {"kind": "divergent", "prefix_tokens": k, "candidate_fragment": fragment[:48],
            "arms": labels, "reference_top": top, "reference_top_text": package.tok.decode([greedy])}


def reference(url, package, out):
    frozen = json.loads((out / "long_fun.json").read_text())
    repaired = json.loads((out / "long_staging.json").read_text())
    rows = {}
    for case, prompt in prompts():
        old, new = frozen[case], repaired[case]
        messages = [{"role": "user", "content": prompt}]
        ids = package.ids(package.render_messages(messages, enable_thinking=False))
        assert len(ids) == old["prompt_tokens"] == new["prompt_tokens"]
        assert old["prompt_sha256"] == new["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        reply, gold, top = record(url, package, ids, 128, f"{case}-greedy")
        old_ids, new_ids = package.ids(old["text"]), package.ids(new["text"])
        rows[case] = {
            "prompt_sha256": old["prompt_sha256"], "reference": reply["content"],
            "frozen": assess(url, package, ids, old_ids, gold, top, f"{case}-frozen"),
            "repaired": assess(url, package, ids, new_ids, gold, top, f"{case}-repaired"),
            "paired_first_divergence": compare_pair(url, package, ids, old_ids, new_ids, case),
        }
        write(out / "long-reference.json", rows)
        row = rows[case]
        print(case, row["frozen"]["kind"], row["repaired"]["kind"],
              row["paired_first_divergence"]["kind"], flush=True)


if __name__ == "__main__":
    mode, endpoint, root, output = sys.argv[1:]
    {"capture": capture, "reference": reference}[mode](endpoint, Package(root), Path(output))
