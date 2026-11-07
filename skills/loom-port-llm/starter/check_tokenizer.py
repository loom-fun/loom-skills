#!/usr/bin/env python3
"""Does a tokenizer.json tokenize like the GGUF it is meant for?

    check_tokenizer.py MODEL.gguf tokenizer.json [--quiet]

Prints `MISMATCH n` (0 = consistent) and the first few differences; exit 0 iff n == 0.

The candidate tokenizer repos port.sh tries come from the GGUF's metadata and its repo's card, and they can name a
different model with a different tokenizer: a GGUF of an instruct model often records its pretraining base
(`general.base_model` = Qwen/Qwen3-1.7B-Base), whose tokenizer.json lacks the chat model's added tokens
(`<think>`, `</think>`, `<tool_response>`). Loom then splits `<think>` into pieces where llama.cpp (the GGUF's own
vocabulary) sees one token, and greedy output diverges at the first generated token. The GGUF's token list is the
truth: every control or user-defined token (types 3 and 4) must be an added token of tokenizer.json with the same id,
and every other token string must map to the same id in the vocabulary.
"""
import json
import sys

from gguf import GGUFReader

NORMAL, CONTROL, USER_DEFINED = 1, 3, 4      # gguf token types; unknown (2), unused (5) and byte (6) are not checked


def mismatches(gguf_path, tok_path, limit=8):
    r = GGUFReader(gguf_path)
    toks = r.fields["tokenizer.ggml.tokens"].contents()
    types = r.fields["tokenizer.ggml.token_type"].contents() if "tokenizer.ggml.token_type" in r.fields else []
    tj = json.load(open(tok_path, encoding="utf-8"))
    added = {a["content"]: a["id"] for a in tj.get("added_tokens") or []}
    vocab = (tj.get("model") or {}).get("vocab") or {}
    if isinstance(vocab, list):          # unigram: [[piece, score], ...], the id is the index
        vocab = {p[0]: i for i, p in enumerate(vocab)}
    bad, examples = 0, []
    for i, t in enumerate(toks):
        ty = types[i] if i < len(types) else NORMAL
        if ty not in (NORMAL, CONTROL, USER_DEFINED):
            continue
        if ty in (CONTROL, USER_DEFINED):
            if t.startswith("[PAD") or t.startswith("<|reserved") or t.startswith("<unused"):
                ok = added.get(t, vocab.get(t, i)) == i      # padding/reserved slots: present or absent, never moved
            else:
                ok = added.get(t) == i
        else:
            ok = vocab.get(t, added.get(t)) == i
        if not ok:
            bad += 1
            if len(examples) < limit:
                examples.append((i, t, ty, added.get(t), vocab.get(t)))
    return bad, examples


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) != 2:
        sys.exit(__doc__)
    bad, ex = mismatches(*a)
    print(f"MISMATCH {bad}")
    if "--quiet" not in sys.argv:
        for i, t, ty, ad, vo in ex:
            print(f"  id {i} {t!r} (gguf type {ty}): added id {ad}, vocab id {vo}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
