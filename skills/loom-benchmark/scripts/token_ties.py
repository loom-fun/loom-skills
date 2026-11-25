"""Log-probability checks shared by the online reference and offline replay tools."""
import math


NOISE_CAP = 0.5   # nats: the most llama.cpp's own spread may widen the bar


def exact_gaps(tops, gold_id, loom_id):
    """logp(gold) - logp(loom) in each of llama.cpp's views of one position that lists both tokens."""
    return [v[gold_id] - v[loom_id] for v in tops if gold_id in v and loom_id in v]


def gap_noise(tops, gold_id, loom_id):
    """llama.cpp's own noise at one position: how far the exact gap moves across its views, capped at NOISE_CAP."""
    exact = exact_gaps(tops, gold_id, loom_id)
    return min(max(exact) - min(exact), NOISE_CAP) if exact else 0.0


def near_tie(top, gold_id, loom_id, max_gap=1.0, views=()):
    """Is Loom's token at a divergence a near-tie of llama.cpp's greedy token? top: {id: logprob} from
    llama.cpp at that position. Yes when the log-prob gap is <= max_gap, or when llama.cpp itself is unsure
    there: Loom picked its second choice and its first has probability < 0.5 (flint: p 0.38 vs 0.13, gap
    1.03, while llama.cpp's own re-recording of the same prompt sat at gap 1.05). A token llama.cpp is
    confident against, or ranks lower, is a bug.
    With `views` (llama_views: llama.cpp's own other numerics there): yes when the smallest gap among top and
    the views is <= max_gap plus their spread (max - min), capped at NOISE_CAP. Only views listing both
    tokens count: one without Loom's token never widens the bar. One without gold that lists Loom's token is
    llama.cpp itself ranking Loom's token higher, which passes (gemma-4-26B: `_` at p 1.0 token by token,
    unlisted batched, where `/` is top-1)."""
    if gold_id not in top:
        return False
    tops = (top, *views)
    exact = exact_gaps(tops, gold_id, loom_id)
    if any(loom_id in v and gold_id not in v for v in tops) or exact and min(exact) <= max_gap + gap_noise(tops, gold_id, loom_id):
        return True
    ranked = sorted(top, key=top.get, reverse=True)
    return ranked[:2] == [gold_id, loom_id] and math.exp(top[gold_id]) < 0.5
