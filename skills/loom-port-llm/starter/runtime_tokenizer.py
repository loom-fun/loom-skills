"""tokenizer.json in a form loom-serve's tokenizer loads, tokenizing exactly as HF tokenizers does (build.py applies it).

Two rewrites, each a no-op unless the file has that shape:
- a pre-tokenizer regex `(?:X)+(?!X)` becomes `(?:X)+`: the negative lookahead of exactly the group a greedy repetition
  just stopped on can never fail (the repetition stops only where X does not match), so both match the same text. The
  runtime's regex engine rejects a group inside a lookahead (Laguna's newline split `(?:\\r?\\n)+(?!\\r?\\n)`). X holds no
  parentheses and does not end in a backslash.
- a BPE model's `unk_token` that is neither in the vocabulary nor an added token is dropped: HF never emits it (it has
  no id), and the runtime refuses to load it (Laguna: "[UNK]").

usage: python runtime_tokenizer.py   (self-check)
"""
import re

_REDUNDANT_LOOKAHEAD = re.compile(r"\(\?:([^()]*[^()\\])\)\+\(\?!\1\)")


def runtime_regex(pattern):
    return _REDUNDANT_LOOKAHEAD.sub(lambda m: f"(?:{m.group(1)})+", pattern)


def rewrite_tokenizer(tok):
    """Rewrites a tokenizer.json dict in place; returns the number of changes."""
    n = 0
    def walk(node):
        nonlocal n
        if isinstance(node, dict):
            pat = node.get("pattern")
            if isinstance(pat, dict) and isinstance(pat.get("Regex"), str):
                new = runtime_regex(pat["Regex"])
                n += new != pat["Regex"]
                pat["Regex"] = new
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(tok.get("pre_tokenizer"))
    model = tok.get("model") or {}
    unk = model.get("unk_token")
    if unk is not None and unk not in model.get("vocab", {}) and unk not in {t["content"] for t in tok.get("added_tokens", [])}:
        model["unk_token"] = None
        n += 1
    return n


if __name__ == "__main__":
    texts = ["a\n\nb", "\r\n\r\nx\n", "x\r\ry\n\r\n\nz", "\n", "no newline", "a\r\n\rb\n\n\n", "abab aab ab", "aaab"]
    cases = {r"(?:\r?\n)+(?!\r?\n)": r"(?:\r?\n)+", r"(?:ab|a)+(?!ab|a)": r"(?:ab|a)+",
             r"\s+(?!\S)|\s+": None, r"(?:a)+(?!b)": None, r"(?:\r?\n)+?(?!\r?\n)": None}
    for src, want in cases.items():
        got = runtime_regex(src)
        assert got == (want or src), (src, got)
        for t in texts:
            assert [m.span() for m in re.finditer(src, t)] == [m.span() for m in re.finditer(got, t)], (src, t)
    split = lambda r: {"type": "Split", "pattern": {"Regex": r}}
    tok = {"pre_tokenizer": {"type": "Sequence", "pretokenizers": [split(r"(?:\r?\n)+(?!\r?\n)"), split(r"\s+(?!\S)|\s+")]},
           "model": {"type": "BPE", "unk_token": "[UNK]", "vocab": {"a": 0}}, "added_tokens": [{"id": 1, "content": "<unk>"}]}
    assert rewrite_tokenizer(tok) == 2 and tok["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == r"(?:\r?\n)+"
    assert tok["model"]["unk_token"] is None
    for unk in ("a", "<unk>"):   # an unk_token the tokenizer can emit stays
        t2 = {"model": {"unk_token": unk, "vocab": {"a": 0}}, "added_tokens": [{"id": 1, "content": "<unk>"}]}
        assert rewrite_tokenizer(t2) == 0 and t2["model"]["unk_token"] == unk
    print("runtime_tokenizer OK")
