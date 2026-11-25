#!/usr/bin/env python3
"""Per-kernel difference of two profiles of the same scenario (profile_stats.py --json of devlab_perf.sh step profile:V), e.g. one package on
two SDKs: device time per decode verify pass (decode scenarios with a pass count) or per token, per family and per kernel, new minus old.
The question it answers: is a slowdown in the kernels' device time (which kernels), or between them (device-busy unchanged, wall time up:
the runtime / host side), or neither (the same per-pass cost: the row moved with the drafted-token acceptance)?

usage: profile_diff.py OLD.json NEW.json [--old-req OLD.req.json --new-req NEW.req.json] [--top N] [--md OUT.md]
The req files (profile_requests.py output) add the request wall time (decode: after the first token) and the draft counters (passes = decode_tokens -
draft_accepted_inputs). Device busy of a dec2k / dec32k profile includes its prompt's prefill: compare the decode kernels' rows, not only the total."""
import argparse
import json
import sys


def last_json(path):
    return json.loads(open(path).read().strip().splitlines()[-1])


def unit(prof, req):
    """(divisor, name, extra) of one profile: verify passes for a drafted decode, else tokens."""
    d = (req or {}).get("draft") or {}
    if prof.get("scenario", "").startswith("dec") and d.get("decode_tokens"):
        passes = d["decode_tokens"] - d.get("draft_accepted_inputs", 0)
        if passes > 0:
            return passes, "pass", {"tokens": d["decode_tokens"], "passes": passes, "tokens_per_pass": round(d["decode_tokens"] / passes, 3)}
    if prof.get("steps"):
        return prof["steps"], "pass", {"passes": prof["steps"]}
    return prof["tokens"], "token", {"tokens": prof["tokens"]}


def wall_ms(req):
    """Request wall time of the measured requests: a decode scenario's decode part (total - time to first token), a prefill's time to first token."""
    if not req:
        return None
    dec = req.get("scenario", "").startswith("dec")
    return sum((r.get("total_s", 0) - r.get("ttft_s", 0)) if dec else r.get("ttft_s", 0) for r in req.get("requests", [])) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--old-req"); ap.add_argument("--new-req")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--md")
    a = ap.parse_args()
    po, pn = json.load(open(a.old)), json.load(open(a.new))
    ro = last_json(a.old_req) if a.old_req else None
    rn = last_json(a.new_req) if a.new_req else None
    uo, name, xo = unit(po, ro)
    un, name_n, xn = unit(pn, rn)
    if name != name_n:
        sys.exit(f"the profiles count different units ({name} vs {name_n}): pass both req files or neither")
    wo, wn = wall_ms(ro), wall_ms(rn)
    L = [f"### {po['scenario']}: device busy per {name}, old -> new", ""]
    L.append(f"- old: {xo}, busy {po['busy_ms'] / uo:.3f} ms/{name}" + (f", wall {wo / uo:.3f} ms/{name}" if wo else ""))
    L.append(f"- new: {xn}, busy {pn['busy_ms'] / un:.3f} ms/{name}" + (f", wall {wn / un:.3f} ms/{name}" if wn else ""))
    db = pn["busy_ms"] / un / (po["busy_ms"] / uo) - 1
    L.append(f"- busy per {name} {db:+.1%}" + (f"; wall per {name} {wn / un / (wo / uo) - 1:+.1%}; wall minus busy per {name} "
                                              f"{(wo - po['busy_ms']) / uo:.3f} -> {(wn - pn['busy_ms']) / un:.3f} ms" if wo and wn else ""))
    L += ["", f"| family | old ms/{name} | new ms/{name} | delta ms | delta % |", "|---|---|---|---|---|"]
    fams = sorted(set(po["families"]) | set(pn["families"]),
                  key=lambda f: -abs(pn["families"].get(f, {}).get("ms", 0) / un - po["families"].get(f, {}).get("ms", 0) / uo))
    for f in fams:
        o = po["families"].get(f, {}).get("ms", 0) / uo; n = pn["families"].get(f, {}).get("ms", 0) / un
        L.append(f"| {f} | {o:.4f} | {n:.4f} | {n - o:+.4f} | {(n / o - 1) * 100 if o else float('nan'):+.1f} |")
    L += ["", f"| kernel (top {a.top} by |delta|) | old count | new count | old ms/{name} | new ms/{name} | delta ms | delta % |",
          "|---|---|---|---|---|---|---|"]
    ks = sorted(set(po["kernels"]) | set(pn["kernels"]),
                key=lambda k: -abs(pn["kernels"].get(k, {}).get("ms", 0) / un - po["kernels"].get(k, {}).get("ms", 0) / uo))
    for k in ks[:a.top]:
        o, n = po["kernels"].get(k, {}), pn["kernels"].get(k, {})
        om, nm = o.get("ms", 0) / uo, n.get("ms", 0) / un
        L.append(f"| {k} | {o.get('count', 0)} | {n.get('count', 0)} | {om:.4f} | {nm:.4f} | {nm - om:+.4f} | {(nm / om - 1) * 100 if om else float('nan'):+.1f} |")
    print("\n".join(L))
    if a.md:
        open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
