"""Unit tests of the near-tie rule with llama.cpp's own views (bench_common.near_tie): python test_near_tie.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import near_tie  # noqa: E402

G, L = 1, 2   # the golden's token, Loom's token
FILL = {10: -30.0, 11: -31.0}   # other listed tokens (a view's last listed log-prob is far below)


def view(gap=None, gold=-0.1):
    """A top list with gold at `gold` and Loom's token `gap` below it; gap None: Loom's token unlisted."""
    return {**FILL, G: gold, **({} if gap is None else {L: gold - gap})}


def loom_preferred(lp):
    """A top list where llama.cpp ranks Loom's token (at lp) and not gold."""
    return {**FILL, L: lp}


CASES = [   # (name, golden top, views, expected)
    # Constructed: a view ranking Loom's token worse, or not at all, must never widen the bar.
    ("3.0 / 3.0 / 5.0", view(3.0), [view(3.0), view(5.0)], False),
    ("2.0 / 2.0 / 4.0", view(2.0), [view(2.0), view(4.0)], False),
    ("2.0, batched unlisted (>= 4.5), micro 2.1", view(2.0), [view(None), view(2.1)], False),
    ("2.5, other view unlisted (>= 6)", view(2.5), [view(None)], False),
    # Trigger cases (DevLab runs in PR #110).
    ("gemma-3-1b compare: 1.364 / 1.284 / 1.061", view(1.364), [view(1.284), view(1.061)], True),
    ("Qwen3-Coder compare: unlisted / 1.685 / 1.207", view(None), [view(1.685), view(1.207)], True),
    ("Ornith-122B think_off: 1.128 / 1.75 / 1.74", view(1.128), [view(1.75), view(1.74)], True),
    ("Ornith-122B agent: 1.502 / 1.23 / 1.57", view(1.502), [view(1.23), view(1.57)], True),
    ("Ornith-122B compare: 1.549 / -1.168 / 0.253", view(1.549), [view(-1.168), view(0.253)], True),
    ("gemma-4-26B agent: `_` p 1.0, batched ranks `/` only", view(None, gold=0.0), [loom_preferred(-1.24), view(None, gold=0.0)], True),
    ("Laguna 'Based': 1.30 / 0.72", view(1.30), [view(0.72)], True),
    ("Laguna 'Let': 2.17 / 1.72", view(2.17), [view(1.72)], False),
    ("RoPE 10000 bug: 1.348 / 1.467 / 1.336", view(1.348), [view(1.467), view(1.336)], False),
    ("RoPE 10000 bug: 2.777 / 2.705 / 2.49", view(2.777), [view(2.705), view(2.49)], False),
    ("no views: the old rule", view(1.2), [], False),
]

bad = 0
for name, top, views, want in CASES:
    got = near_tie(top, G, L, views=views)
    bad += got != want
    print(f"{'ok  ' if got == want else 'FAIL'} {name} -> {got}")
print("all ok" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
