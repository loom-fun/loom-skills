"""Speculative-decoding counters of loom-serve's /metrics (standard library only): what perf.py, c4_steady.py,
spec_check.py and the card scripts (make_info.py) read. bench_common re-exports them."""


# This SDK defined proposed_tokens as verification_passes × configured draft depth.
DEPTH_COUNTED_PROPOSAL_SDKS = {"0.2.0"}

DRAFT_COUNTERS = ("draft_proposed_tokens", "draft_accepted_inputs", "decode_tokens")


def draft_counters(text):
    """loom-serve's cumulative speculative decoding counters {draft_proposed_tokens, draft_accepted_inputs,
    decode_tokens} from its /metrics text ({} when it has no drafting counter: no drafting, or llama-server)."""
    vals = {}
    for line in (text or "").splitlines():
        for k in DRAFT_COUNTERS:
            if f'field="{k}"' in line:
                vals[k] = float(line.split()[-1])
    return vals if any(k.startswith("draft_") for k in vals) else {}


def acceptance_between(before, after):
    """Accepted / proposed drafted tokens between two draft_counters() snapshots, or None (nothing proposed, or the
    server reports no proposal count: proposals_unreported())."""
    prop = after.get("draft_proposed_tokens", 0) - before.get("draft_proposed_tokens", 0)
    acc = after.get("draft_accepted_inputs", 0) - before.get("draft_accepted_inputs", 0)
    return round(acc / prop, 3) if prop > 0 else None


def proposals_unreported(vals):
    """Whether a draft_counters() snapshot counts accepted drafted tokens but no proposals. hrx-loom 0.3.0 serving a
    service ABI 1 controller does not know how many tokens the drafter proposed (its publisher passes no proposal
    counts: proposed_tokens is null in the telemetry events and /metrics omits draft_proposed_tokens); SDK 0.2.0
    counted verifications x the configured draft depth. The acceptance is then not measurable, and it is never
    derived from the configured depth."""
    return "draft_accepted_inputs" in vals and "draft_proposed_tokens" not in vals


def accepted_between(before, after):
    """Drafted tokens accepted between two draft_counters() snapshots (int), or None when the server counts none."""
    if "draft_accepted_inputs" not in after:
        return None
    return int(after["draft_accepted_inputs"] - before.get("draft_accepted_inputs", 0))


def per_pass_between(before, after):
    """Drafted tokens accepted per verification pass between two draft_counters() snapshots, from the counters the SDK
    reports without a proposal count: each decode position is a verified anchor or an accepted drafted token
    (hrx-loom's text statistics: accepted = consumed - known - verifications, decode_tokens = consumed of decode spans),
    so verification passes = decode_tokens - draft_accepted_inputs. None without both counters or without a pass.
    No draft depth is involved: the depth is a ceiling, not a count of what the drafter presented."""
    if "decode_tokens" not in after or "draft_accepted_inputs" not in after:
        return None
    acc = after["draft_accepted_inputs"] - before.get("draft_accepted_inputs", 0)
    passes = after["decode_tokens"] - before.get("decode_tokens", 0) - acc
    return round(acc / passes, 3) if passes > 0 else None
