"""Tokenizer pre-tokenizer regexes in a form loom-serve's regex engine parses (build.py applies it to tokenizer.json).

One rewrite only: a negative lookahead of exactly the group a greedy repetition just stopped on, `(?:X)+(?!X)`, can
never fail (the repetition stops only where X does not match), so `(?:X)+` matches the same text. Laguna's newline
split `(?:\\r?\\n)+(?!\\r?\\n)` needs it (the engine rejects a group inside a lookahead). X holds no parentheses and does
not end in a backslash; nothing else is changed.

usage: python runtime_regex.py   (self-check: rewritten patterns split sample texts exactly like the originals)
"""
import re

_REDUNDANT_LOOKAHEAD = re.compile(r"\(\?:([^()]*[^()\\])\)\+\(\?!\1\)")


def runtime_regex(pattern):
    return _REDUNDANT_LOOKAHEAD.sub(lambda m: f"(?:{m.group(1)})+", pattern)


def rewrite_tokenizer(tok):
    """Rewrites every pre_tokenizer Split regex of a tokenizer.json dict in place; returns the number changed."""
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
    return n


if __name__ == "__main__":
    texts = ["a\n\nb", "\r\n\r\nx\n", "x\r\ry\n\r\n\nz", "\n", "no newline", "a\r\n\rb\n\n\n"]
    cases = {r"(?:\r?\n)+(?!\r?\n)": r"(?:\r?\n)+", r"(?:ab|a)+(?!ab|a)": r"(?:ab|a)+",
             r"\s+(?!\S)|\s+": None, r"(?:a)+(?!b)": None, r"(?:\r?\n)+?(?!\r?\n)": None}
    for src, want in cases.items():
        got = runtime_regex(src)
        assert got == (want or src), (src, got)
        for t in texts + ["abab aab ab", "aaab"]:
            assert [m.span() for m in re.finditer(src, t)] == [m.span() for m in re.finditer(got, t)], (src, t)
    tok = {"pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": r"(?:\r?\n)+(?!\r?\n)"}},
                                                                    {"type": "Split", "pattern": {"Regex": r"\s+(?!\S)|\s+"}}]}}
    assert rewrite_tokenizer(tok) == 1 and tok["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == r"(?:\r?\n)+"
    print("runtime_regex OK")
